import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class_names = ['P', 'NR', 'NP']

def get_peft_predictions(model, data_loader):
    model = model.eval()

    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs.logits, dim=1)

            probs = F.softmax(outputs.logits, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return predictions, prediction_probs, real_values

def get_predictions_test(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            # targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            # real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    return review_texts, predictions, prediction_probs, real_values

def get_predictions(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values

def get_predictions_dense_ensemble(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            bert_embeds = d[0].to(device)
            roberta_embeds = d[1].to(device)
            deberta_embeds = d[2].to(device)
            targets = d[3].to(device)

            outputs = model(
                bert_embeds,
                roberta_embeds,
                deberta_embeds
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)


            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values

def get_predictions_ensemble(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            bert_input_ids = d["bert_input_ids"].to(device)
            bert_attention_mask = d["bert_attention_mask"].to(device)
            roberta_input_ids = d["roberta_input_ids"].to(device)
            roberta_attention_mask = d["roberta_attention_mask"].to(device)
            deberta_input_ids = d["deberta_input_ids"].to(device)
            deberta_attention_mask = d["deberta_attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
            bert_input_ids,
            bert_attention_mask,
            roberta_input_ids,
            roberta_attention_mask,
            deberta_input_ids,
            deberta_attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values

def get_predictions_ensemble_test(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            bert_input_ids = d["bert_input_ids"].to(device)
            bert_attention_mask = d["bert_attention_mask"].to(device)
            roberta_input_ids = d["roberta_input_ids"].to(device)
            roberta_attention_mask = d["roberta_attention_mask"].to(device)
            deberta_input_ids = d["deberta_input_ids"].to(device)
            deberta_attention_mask = d["deberta_attention_mask"].to(device)
            # targets = d["targets"].to(device)

            outputs = model(
            bert_input_ids,
            bert_attention_mask,
            roberta_input_ids,
            roberta_attention_mask,
            deberta_input_ids,
            deberta_attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(probs)
            # real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    # real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values

def get_classification_report(y_test, y_pred):
    print(classification_report(y_test, y_pred))

def get_confusion_matrix(y_test, y_pred):
  cm = confusion_matrix(y_test, y_pred)
  print('\033[96m' + 'Confusion Matrix : \n'+'\033[0m',cm)
  df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
  hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True Labels')
  plt.xlabel('Predicted Labels')
  plt.show()

def get_scores(y_test,y_pred):
    print('Accuracy : ',accuracy_score(y_test,y_pred))
    print()
    print('Precision : ',precision_score(y_test,y_pred,average='weighted'))
    print()
    print('Recall : ',recall_score(y_test,y_pred,average='weighted'))
    print()
    print('F-1 : ',f1_score(y_test,y_pred,average='weighted'))

def generate_submission_track_1(model, data_loader):
    # we need to write code according to the class names {'NR': 0, 'P': 1, 'NP': 2}
    class_mapping = {'NR': 0, 'P': 1, 'NP': 2}
    review_texts, predictions, prediction_probs, real_values = get_predictions_test(model,data_loader)
    data = [('{}_Track1'.format(i), list(class_mapping.keys())[label.item()]) for i, label in enumerate(predictions)]
    df = pd.DataFrame(data, columns=['sub_id', 'label'])
    df.to_csv('/content/drive/MyDrive/homo_mex_track_1_full_fine_tune_sub.csv',index = False)
    print('Submission CSV Generated')
    

def generate_submission_lora_track_1(model, data_loader):
    class_mapping = {'NP': 0, 'NR': 1, 'P': 2}
    predictions, prediction_probs, real_values = get_peft_predictions(model, data_loader)
    data = [('{}_Track1'.format(i), list(class_mapping.keys())[label.item()]) for i, label in enumerate(predictions)]
    df = pd.DataFrame(data, columns=['sub_id', 'label'])
    df.to_csv('/content/drive/MyDrive/homo_mex_track_1_lora_sub.csv',index = False)
    print('Submission CSV Generated')

def generate_submission_track_3(model, data_loader):
    # we need to write code according to the class names {'NR': 0, 'P': 1, 'NP': 2}
    class_mapping = {'P': 0, 'NP': 1}
    review_texts, predictions, prediction_probs, real_values = get_predictions(model,data_loader)
    data = [('{}_Track3'.format(i), list(class_mapping.keys())[label.item()]) for i, label in enumerate(predictions)]
    df = pd.DataFrame(data, columns=['sub_id', 'label'])
    df.to_csv('/content/drive/MyDrive/homo_mex_track_3_sub.csv',index = False)
    print('Submission CSV Generated')
    

def generate_submission_lora_track_3(model, data_loader):
    class_mapping = {'NP': 0, 'P': 1}
    predictions, prediction_probs, real_values = get_peft_predictions(model, data_loader)
    data = [('{}_Track3'.format(i), list(class_mapping.keys())[label.item()]) for i, label in enumerate(predictions)]
    df = pd.DataFrame(data, columns=['sub_id', 'label'])
    df.to_csv('/content/drive/MyDrive/homo_mex_track_3_lora_sub.csv',index = False)
    print('Submission CSV Generated')

def generate_submission_xgboost_track_1(model,X_test,le):
    predictions = model.predict(X_test)
    data = [
    ('{}_Track1'.format(i), le.inverse_transform([label])[0]) 
    for i, label in enumerate(predictions)
    ]
    df = pd.DataFrame(data, columns=['sub_id', 'label'])
    df.to_csv('/content/drive/MyDrive/homo_mex_track_1_sub_xgboost.csv',index = False)
    print('Submission CSV Generated')

def generate_submission_track_1_ensemble(model,data_loader):
    class_mapping = {'NR': 0, 'P': 1, 'NP': 2}
    review_texts, predictions, prediction_probs, real_values = get_predictions_ensemble_test(model,data_loader)
    data = [('{}_Track1'.format(i), list(class_mapping.keys())[label.item()]) for i, label in enumerate(predictions)]
    df = pd.DataFrame(data, columns=['sub_id', 'label'])
    df.to_csv('/content/drive/MyDrive/homo_mex_track_1_ensemble_sub.csv',index = False)
    print('Submission CSV Generated')

    
