import torch
import torch.nn as nn

class MexClassifierDenseEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.4
        self.hidden_1 = 1000
        self.hidden_2 = 500
        self.hidden_3 = 100
        self.hidden_4 = 3 * self.hidden_3
        self.hidden_5 = 50

        self.bert_features = nn.Sequential(
            nn.Linear(in_features=768,out_features=self.hidden_1),
            nn.Linear(in_features=self.hidden_1,out_features=self.hidden_2),
            nn.Linear(in_features=self.hidden_2,out_features=self.hidden_3),
            nn.Dropout(self.dropout)
        )

        self.roberta_features = nn.Sequential(
            nn.Linear(in_features=768,out_features=self.hidden_1),
            nn.Linear(in_features=self.hidden_1,out_features=self.hidden_2),
            nn.Linear(in_features=self.hidden_2,out_features=self.hidden_3),
            nn.Dropout(self.dropout)
        )

        self.deberta_features = nn.Sequential(
            nn.Linear(in_features=768,out_features=self.hidden_1),
            nn.Linear(in_features=self.hidden_1,out_features=self.hidden_2),
            nn.Linear(in_features=self.hidden_2,out_features=self.hidden_3),
            nn.Dropout(self.dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_4,out_features=self.hidden_3),
            nn.Linear(in_features=self.hidden_3,out_features=self.hidden_5),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=self.hidden_5,out_features=3)
        )
    
    def forward(self,bert_embeds,roberta_embeds,deberta_embeds):
        bert_hidden_embeds = self.bert_features(bert_embeds)
        roberta_hidden_embeds = self.roberta_features(roberta_embeds)
        deberta_hidden_embeds = self.deberta_features(deberta_embeds)

        stacked_tensor = torch.stack((bert_hidden_embeds,roberta_hidden_embeds,deberta_hidden_embeds),axis = 1)
        logits = self.classifier(torch.flatten(stacked_tensor,start_dim=1))
        
        return logits
