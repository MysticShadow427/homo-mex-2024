from collections import defaultdict
import argparse
from mex_trainer import train_epoch,eval_model
from mex_dataloader import create_data_loader
from load_llm import MexSpanClassifier
from utils import plot_accuracy,plot_loss,save_training_history
from transformers import AutoModel,AutoTokenizer,AdamW,get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from mex_preprocess import remove_chars_except_punctuations,remove_newline_pattern,remove_numbers_and_urls,remove_pattern
from mex_eval import get_confusion_matrix,get_predictions,get_scores,get_classification_report
from load_lora_llm import MexSpanClassifierLoRA


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="number of epochs for training")
    parser.add_argument("--learning_rate",type=float,help="learning rate")
    parser.add_argument("--batch_size",type=int,help="batch size forr training")
    args = parser.parse_args()

    EPOCHS = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device : ',device)
    print()

    df = pd.read_csv('/content/homo-mex-2024/data/public_data_dev_phase/track_1_dev.csv')

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
    print('Loaded Training, validation and test dataframes')
    print()


    # test_df['content'] = test_df['content'].apply(remove_pattern)
    # test_df['content'] = test_df['content'].apply(remove_numbers_and_urls)
    # test_df['content'] = test_df['content'].apply(remove_chars_except_punctuations)
    # test_df['content'] = test_df['content'].apply(remove_newline_pattern)

    print('Preprocessing of Data done')
    print()

    # Label to index
    tags = train_df.label.unique().tolist()
    num_classes = len(tags)
    class_to_index = {tag: i for i, tag in enumerate(tags)}
    # Encode labels
    train_df["label"] = train_df["label"].map(class_to_index)
    val_df["label"] = val_df["label"].map(class_to_index)
    class_names = ['P', 'NR', 'NP']

    checkpoint = 'dccuchile/bert-base-spanish-wwm-uncased'

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print('Tokenizer Loaded')
    print()
    model = MexSpanClassifier(n_classes=3).to(device)
    print('Model Spanish BERT Loaded')
    print()

    train_data_loader = create_data_loader(train_df,tokenizer=tokenizer,max_len=100,batch_size=batch_size)
    val_data_loader = create_data_loader(val_df,tokenizer=tokenizer,max_len=100,batch_size=batch_size)
    # test_data_loader = create_data_loader(test_df,tokenizer=tokenizer,max_len=100,batch_size=batch_size)
    print('Dataloaders created')
    print()

    total_steps = len(train_data_loader) * EPOCHS

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
                )
    print('Loss function,Optimizer and Learning Rate Schedule set')
    print()

    history = defaultdict(list)
    best_acc = 0
    print('Starting training...')
    print()

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_df)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(val_df)
        )

        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_acc:
            torch.save(model.state_dict(), '/content/homo-mex-2024/artifacts/best_model_state_full_fine_tune.bin')
            torch.save(obj=model.state_dict(),f='/content/homo-mex-2024/artifacts/best_model_full_fine_tune.pth')
            best_acc = val_acc
    print()
    print('Training finished')
    print()

    plot_accuracy(history)
    plot_loss(history)

    history_csv_file_path = "/content/homo-mex-2024/artifacts/history.csv"
    save_training_history(history=history,path=history_csv_file_path)
    print('Training History saved')
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

    print('Getting Predictions...')
    print()
    # y_review_texts_test, y_pred_test, y_pred_probs_test, y_test = get_predictions(model,test_data_loader)
    y_review_texts_val, y_pred_val, y_pred_probs_val, y_val = get_predictions(model,val_data_loader)
    y_review_texts_train, y_pred_train, y_pred_probs_train, y_train = get_predictions(model,train_data_loader)

    # print('Test Data Classification Report : ')
    # print()
    # get_classification_report(y_test,y_pred_test)
    # get_scores(y_test,y_pred_test)
    # get_confusion_matrix(y_test,y_pred_test)
    print('Val Data Classification Report : ')
    print()
    get_classification_report(y_val,y_pred_val)
    get_scores(y_val,y_pred_val)
    get_confusion_matrix(y_val,y_pred_val)
    print('Train Data Classification Report : ')
    print()
    get_classification_report(y_train,y_pred_train)
    get_scores(y_train,y_pred_train)
    get_confusion_matrix(y_train,y_pred_train)
    


