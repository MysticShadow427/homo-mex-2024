from collections import defaultdict
import argparse
from mex_trainer import train_epoch_ensemble,eval_model_ensemble
from mex_dataloader import create_data_loader_ensemble
from mex_load_ensemble import MexClassifierEnsemble
from utils import plot_accuracy_loss,save_training_history
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from mex_preprocess import remove_chars_except_punctuations,remove_newline_pattern,remove_numbers_and_urls,remove_pattern
from mex_eval import get_confusion_matrix,get_predictions_ensemble,get_scores,get_classification_report

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

    df = pd.read_csv('/content/homo-mex-2024/data/public_data_dev_phase/randomly_oversampled_text.csv')

    df['content'] = df['content'].apply(remove_pattern)
    df['content'] = df['content'].apply(remove_numbers_and_urls)
    df['content'] = df['content'].apply(remove_chars_except_punctuations)
    df['content'] = df['content'].apply(remove_newline_pattern)
    X = df['content']
    y = df['label']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

    train_df = pd.DataFrame({'content': X_train, 'label': y_train})
    val_df = pd.DataFrame({'content': X_val, 'label': y_val})
    
    # test_df = pd.read_csv('/content/homo-mex-2024/data/public_data_test_phase/track_1_test.csv')
    print('\033[96m' + 'Loaded Training, validation and test dataframes'+ '\033[0m')
    print()


    # test_df['content'] = test_df['content'].apply(remove_pattern)
    # test_df['content'] = test_df['content'].apply(remove_numbers_and_urls)
    # test_df['content'] = test_df['content'].apply(remove_chars_except_punctuations)
    # test_df['content'] = test_df['content'].apply(remove_newline_pattern)

    print('\033[96m' + 'Preprocessing of Data done'+ '\033[0m')
    print()

    # Label to index
    tags = train_df.label.unique().tolist()
    num_classes = len(tags)
    class_to_index = {tag: i for i, tag in enumerate(tags)}
    # Encode labels
    train_df["label"] = train_df["label"].map(class_to_index)
    val_df["label"] = val_df["label"].map(class_to_index)
    class_names = ['P', 'NR', 'NP']

    bert_checkpoint = 'dccuchile/bert-base-spanish-wwm-uncased'
    roberta_checkpoint = 'JonatanGk/roberta-base-bne-finetuned-cyberbullying-spanish'
    deberta_checkpoint = 'microsoft/mdeberta-v3-base'
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)
    roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_checkpoint)
    deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_checkpoint)
    print('\033[96m' + 'Tokenizers Loaded'+ '\033[0m')
    print()
    model = MexClassifierEnsemble().to(device)
    print('\033[96m' + 'Model Loaded'+ '\033[0m')
    print()

    train_data_loader = create_data_loader_ensemble(train_df,bert_tokenizer=bert_tokenizer,roberta_tokenizer=roberta_tokenizer,deberta_tokenizer=deberta_tokenizer,max_len=100,batch_size=batch_size)
    val_data_loader = create_data_loader_ensemble(val_df,bert_tokenizer=bert_tokenizer,roberta_tokenizer=roberta_tokenizer,deberta_tokenizer=deberta_tokenizer,max_len=100,batch_size=batch_size)
    # test_data_loader = None
    print('\033[96m' + 'Dataloaders created')
    print()

    total_steps = len(train_data_loader) * EPOCHS

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
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

        train_acc, train_loss = train_epoch_ensemble(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_df)
        )

        print('\033[96m' + f'Train loss {train_loss} accuracy {train_acc}'+ '\033[0m')

        val_acc, val_loss = eval_model_ensemble(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(val_df)
        )

        print('\033[96m' + f'Val   loss {val_loss} accuracy {val_acc}'+ '\033[0m')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_acc:
            torch.save(model.state_dict(), '/content/homo-mex-2024/artifacts/best_model_state_ensemble.bin')
            torch.save(obj=model.state_dict(),f='/content/homo-mex-2024/artifacts/best_model_state_ensemble.pth')
            best_acc = val_acc
    print()
    print('\033[96m' + 'Training finished'+ '\033[0m')
    print()

    plot_accuracy_loss(history)

    history_csv_file_path = "/content/homo-mex-2024/artifacts/history.csv"
    save_training_history(history=history,path=history_csv_file_path)
    print('\033[96m' + 'Training History saved'+ '\033[0m')
    print()
    # test_acc, _ = eval_model(
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
    y_review_texts_val, y_pred_val, y_pred_probs_val, y_val = get_predictions_ensemble(model,val_data_loader)
    y_review_texts_train, y_pred_train, y_pred_probs_train, y_train = get_predictions_ensemble(model,train_data_loader)

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

# we need to now change the paths and other stuff according to val data and training data we have gotten and accordingly the augmentations and sentence embeddings we need to generatea