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

def get_classification_report(y_test, y_pred, target_names=class_names):
    print(classification_report(y_test, y_pred, target_names))

def get_confusion_matrix(y_test, y_pred):
  cm = confusion_matrix(y_test, y_pred)
  df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
  hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True Labels')
  plt.xlabel('Predicted Labels');

def get_scores(y_test,y_pred):
    print('Accuracy : ',accuracy_score(y_test,y_pred))
    print()
    print('Precision : ',precision_score(y_test,y_pred))
    print()
    print('Recall : ',recall_score(y_test,y_pred))
    print()
    print('F-1 : ',f1_score(y_test,y_pred))