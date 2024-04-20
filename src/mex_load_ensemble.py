import torch
import torch.nn as nn
from transformers import AutoModel

model_1 = 'dccuchile/bert-base-spanish-wwm-uncased'
model_2 = 'JonatanGk/roberta-base-bne-finetuned-cyberbullying-spanish'
model_3 = 'microsoft/mdeberta-v3-base'

class MexClassifierEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.4
        self.hidden_1 = 1000
        self.hidden_2 = 500
        self.hidden_3 = 100
        self.hidden_4 = 3 * self.hidden_3
        self.hidden_5 = 50
        self.bert = AutoModel.from_pretrained(model_1,return_dict=False)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.roberta = AutoModel.from_pretrained(model_2,return_dict=False)
        for param in self.roberta.parameters():
            param.requires_grad = False
        self.deberta = AutoModel.from_pretrained(model_3,return_dict=False)
        for param in self.deberta.parameters():
            param.requires_grad = False
            
        self.bert_hidden_1 = nn.Linear(in_features=768,out_features=self.hidden_1)
        self.bert_hidden_2 = nn.Linear(in_features=self.hidden_1,out_features=self.hidden_2)
        self.bert_hidden_3 = nn.Linear(in_features=self.hidden_2,out_features=self.hidden_3)
        self.bert_dropout = nn.Dropout(self.dropout)

        self.roberta_hidden_1 = nn.Linear(in_features=768,out_features=self.hidden_1)
        self.roberta_hidden_2 = nn.Linear(in_features=self.hidden_1,out_features=self.hidden_2)
        self.roberta_hidden_3 = nn.Linear(in_features=self.hidden_2,out_features=self.hidden_3)
        self.roberta_dropout = nn.Dropout(self.dropout)

        self.deberta_hidden_1 = nn.Linear(in_features=768,out_features=self.hidden_1)
        self.deberta_hidden_2 = nn.Linear(in_features=self.hidden_1,out_features=self.hidden_2)
        self.deberta_hidden_3 = nn.Linear(in_features=self.hidden_2,out_features=self.hidden_3)
        self.deberta_dropout = nn.Dropout(self.dropout)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_4,out_features=self.hidden_3),
            nn.Linear(in_features=self.hidden_3,out_features=self.hidden_5),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=self.hidden_5,out_features=3)
        )
    
    def forward(self,bert_input_ids, bert_attention_mask,roberta_input_ids, roberta_attention_mask,deberta_input_ids, deberta_attention_mask):
        _, bert_pooled_output = self.bert(
      input_ids=bert_input_ids,
      attention_mask=bert_attention_mask
    )
        _, roberta_pooled_output = self.roberta(
      input_ids=roberta_input_ids,
      attention_mask=roberta_attention_mask
    ) # rather than this we can have simple sentence transformers but we then also need to change the dataloaders too
        deberta_output = self.deberta(
      input_ids=deberta_input_ids,
      attention_mask=deberta_attention_mask
    )
        bert_embeds = self.bert_hidden_1(bert_pooled_output)
        bert_embeds = self.bert_hidden_2(bert_embeds)
        bert_embeds = self.bert_hidden_3(bert_embeds)

        roberta_embeds = self.roberta_hidden_1(roberta_pooled_output)
        roberta_embeds = self.roberta_hidden_2(roberta_embeds)
        roberta_embeds = self.roberta_hidden_3(roberta_embeds)

        deberta_pooled_output = torch.mean(deberta_output[0],dim = 1)
        deberta_embeds = self.deberta_hidden_1(deberta_pooled_output)
        deberta_embeds = self.deberta_hidden_2(deberta_embeds)
        deberta_embeds = self.deberta_hidden_3(deberta_embeds)

        stacked_tensor = torch.stack((bert_embeds, roberta_embeds, deberta_embeds), dim=1)
        modified_tensor = torch.mean(stacked_tensor,dim=1) #torch.flatten(stacked_tensor,start_dim=1)
        logits = self.classifier(modified_tensor)
        return logits
        

        

