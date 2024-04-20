# totally different and independent script from the 'load_lora_llm.py'
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from peft import AutoPeftModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from mex_eval import get_confusion_matrix,get_classification_report,get_peft_predictions,get_scores
from mex_augment_data import random_oversample
import pandas as pd

peft_model_name = 'bert-base-spanish-wwm-uncased-peft-homo-mex'
modified_base = 'bert-base-spanish-wwm-uncased-modified-homo-mex'
base_model = 'dccuchile/bert-base-spanish-wwm-uncased'

train_df = pd.read_csv('/content/homo-mex-2024/data/public_data_train_phase/track_1_train.csv')
train_df = random_oversample(train_df)
train_df.to_csv('/content/homo-mex-2024/data/public_data_train_phase/track_1_train.csv',index = False)
train_dataset = load_dataset("csv", data_files='/content/homo-mex-2024/data/public_data_train_phase/track_1_train.csv')
val_dataset = load_dataset("csv", data_files='/content/homo-mex-2024/data/public_data_dev_phase/track_1_dev.csv')
tokenizer = AutoTokenizer.from_pretrained(base_model)

num_labels = 3
class_names = ['NP','NR','P']

id2label = {i: label for i, label in enumerate(class_names)}
label2id = {label: i for i,label in enumerate(class_names)}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

def preprocess(examples):
    examples["label"] = [label2id[label] for label in examples["label"]]
    tokenized = tokenizer(examples['content'], truncation=True, padding=True)
    return tokenized

tokenized_dataset_train = train_dataset.map(preprocess, batched=True,  remove_columns=["content"])
tokenized_dataset_val = val_dataset.map(preprocess, batched=True,  remove_columns=["content"])
# tokenized_dataset_val = tokenized_dataset['train'].train_test_split(test_size=0.2,seed = 42) #stratify_by_column="label")
train_dataset=tokenized_dataset_train['train']
test_dataset=tokenized_dataset_val['train']
print('\033[96m' + 'Datasets ready'+ '\033[0m')
print()

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=32,
)
print('\033[96m' + 'Training arguments set.'+ '\033[0m')
print()

def get_trainer(model):
      return  Trainer(
          model=model,
          args=training_args,
          train_dataset=train_dataset,
          eval_dataset=test_dataset,
          data_collator=data_collator,
      )

model = AutoModelForSequenceClassification.from_pretrained(base_model, id2label=id2label)
print('\033[96m' + 'Loaded pretrained spanish BERT'+ '\033[0m')
print()

peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1,target_modules='all-linear')
peft_model = get_peft_model(model, peft_config)

print('PEFT Model')
peft_model.print_trainable_parameters()

peft_lora_finetuning_trainer = get_trainer(peft_model)
print('\033[96m' + 'Training the peft model...'+ '\033[0m')
print()
peft_lora_finetuning_trainer.train()
peft_lora_finetuning_trainer.evaluate()
print('\033[96m' + 'Saving fine tuned peft model...'+ '\033[0m')
print()
tokenizer.save_pretrained(modified_base)
peft_model.save_pretrained(peft_model_name)
print('\033[96m' + 'Saved the peft model'+ '\033[0m')
print()
inference_model = AutoPeftModelForSequenceClassification.from_pretrained(peft_model_name, id2label=id2label).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(modified_base)
print('\033[96m' + 'Loaded Trained Model for inference'+ '\033[0m')
print()

val_data_loader = DataLoader(test_dataset, batch_size=16, collate_fn=data_collator)
train_data_loader = DataLoader(train_dataset, batch_size=16, collate_fn=data_collator)
print('\033[96m' + 'Getting Predictions...'+ '\033[0m')
print()
# y_review_texts_test, y_pred_test, y_pred_probs_test, y_test = get_predictions(model,test_data_loader)
y_pred_val, y_pred_probs_val, y_val = get_peft_predictions(inference_model,val_data_loader)
y_pred_train, y_pred_probs_train, y_train = get_peft_predictions(inference_model,train_data_loader)

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