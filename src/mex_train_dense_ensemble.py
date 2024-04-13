from collections import defaultdict
import argparse
from mex_trainer import train_epoch_dense_ensemble,eval_model_dense_ensemble
from mex_dataloader import create_dense_ensemble_dataloader
from mex_load_dense_ensemble import MexClassifierDenseEnsemble
from utils import plot_accuracy_loss,save_training_history
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from mex_eval import get_confusion_matrix,get_predictions,get_scores,get_classification_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="number of epochs for training")
    parser.add_argument("--learning_rate",type=float,help="learning rate")
    parser.add_argument("--batch_size",type=int,help="batch size for training")
    args = parser.parse_args()

    EPOCHS = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\033[96m' + 'Device : ',device + '\033[0m')
    print()

    # embeddings are hosted on drive hence loading it from there
    bert_df = pd.read_csv('/content/drive/MyDrive/smote_bert_track_1_embeds.csv')
    roberta_df = pd.read_csv('/content/drive/MyDrive/smote_roberta_track_1_embeds.csv')
    deberta_df = pd.read_csv('/content/drive/MyDrive/smote_deberta_track_1_embeds.csv')
    y = bert_df['label'].values
    # Label to index
    tags = bert_df.label.unique().tolist()
    num_classes = len(tags)
    class_to_index = {tag: i for i, tag in enumerate(tags)}

    bert_df = bert_df.drop('label',axis = 1)
    roberta_df = roberta_df.drop('label',axis = 1)
    deberta_df = deberta_df.drop('label',axis = 1)
    X_train_bert, X_val_bert, y_train, y_val = train_test_split(bert_df, y, test_size=0.2, random_state=42,stratify=y)
    X_train_roberta, X_val_roberta, y_train, y_val = train_test_split(roberta_df, y, test_size=0.2, random_state=42,stratify=y)
    X_train_deberta, X_val_deberta, y_train, y_val = train_test_split(deberta_df, y, test_size=0.2, random_state=42,stratify=y)
    
    # test_df = pd.read_csv('/content/homo-mex-2024/data/public_data_test_phase/track_1_test.csv')
    print('\033[96m' + 'Loaded Training, validation and test dataframes'+ '\033[0m')
    print()
    y_train_df = pd.DataFrame({'label':y_train})
    y_val_df = pd.DataFrame({'label':y_val})
    # Encode labels
    y_train_df["label"] = y_train_df["label"].map(class_to_index)
    y_val_df["label"] = y_val_df["label"].map(class_to_index)
    class_names = ['P', 'NR', 'NP']

    model = MexClassifierDenseEnsemble().to(device)
    print('\033[96m' + 'Model Loaded'+ '\033[0m')
    print()

    train_data_loader = create_dense_ensemble_dataloader(X_train_bert,X_train_roberta,X_train_deberta,batch_size=batch_size)
    val_data_loader = create_dense_ensemble_dataloader(X_val_bert,X_val_roberta,X_val_deberta,batch_size=batch_size)
    # test_data_loader = None
    print('\033[96m' + 'Dataloaders created')
    print()

    total_steps = len(train_data_loader) * EPOCHS

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer,EPOCHS)
    print('\033[96m' + 'Loss function,Optimizer and Learning Rate Schedule set'+ '\033[0m')
    print()

    history = defaultdict(list)
    best_acc = 0
    print('\033[96m' + 'Starting training...'+ '\033[0m')
    print()

    for epoch in range(EPOCHS):
        print('\033[96m' + f'Epoch {epoch + 1}/{EPOCHS}'+ '\033[0m')
        print('\033[96m' + '-' * 10+ '\033[0m')

        train_acc, train_loss = train_epoch_dense_ensemble(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(X_train_bert)
        )

        print('\033[96m' + f'Train loss {train_loss} accuracy {train_acc}'+ '\033[0m')

        val_acc, val_loss = eval_model_dense_ensemble(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(X_val_bert)
        )

        print('\033[96m' + f'Val   loss {val_loss} accuracy {val_acc}'+ '\033[0m')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_acc:
            torch.save(model.state_dict(), '/content/homo-mex-2024/artifacts/best_model_state_dense_ensemble.bin')
            torch.save(obj=model.state_dict(),f='/content/homo-mex-2024/artifacts/best_model_state_dense_ensemble.pth')
            best_acc = val_acc
    print()
    print('\033[96m' + 'Training finished'+ '\033[0m')
    print()

    plot_accuracy_loss(history)

    history_csv_file_path = "/content/homo-mex-2024/artifacts/history.csv"
    save_training_history(history=history,path=history_csv_file_path)
    print('\033[96m' + 'Training History saved'+ '\033[0m')
    print()
    # test_acc, _ = eval_model_dense_ensemble(
    #     model,
    #     test_data_loader,
    #     loss_fn,
    #     device,
    #     len(test_df)
    #     )

    # print('Test Accuracy',test_acc.item())
    # print()

    print('\033[96m' + 'Getting Predictions...'+ '\033[0m')
    print()
    # y_review_texts_test, y_pred_test, y_pred_probs_test, y_test = get_predictions(model,test_data_loader)
    y_review_texts_val, y_pred_val, y_pred_probs_val, y_val = get_predictions(model,val_data_loader)
    y_review_texts_train, y_pred_train, y_pred_probs_train, y_train = get_predictions(model,train_data_loader)

    # print('Test Data Classification Report : ')
    # print()
    # get_classification_report(y_test,y_pred_test)
    # get_scores(y_test,y_pred_test)
    # get_confusion_matrix(y_test,y_pred_test)
    print('\033[96m' + 'Val Data Classification Report : '+ '\033[0m')
    print()
    get_classification_report(y_val,y_pred_val)
    get_scores(y_val,y_pred_val)
    get_confusion_matrix(y_val,y_pred_val)
    print('\033[96m' + 'Train Data Classification Report : '+ '\033[0m')
    print()
    get_classification_report(y_train,y_pred_train)
    get_scores(y_train,y_pred_train)
    get_confusion_matrix(y_train,y_pred_train)
