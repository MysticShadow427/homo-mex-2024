import torch
import torch.nn as nn
from transformers import AutoModel

checkpoint = 'dccuchile/bert-base-spanish-wwm-uncased'

class MexSpanClassifierLSTM(nn.Module):
    def __init__(self,dropout,bidirectional,num_layers,hidden_size=192):
        super().__init__()
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.feature_extractor = AutoModel.from_pretrained(checkpoint,return_dict=False)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.lstm = nn.LSTM(input_size = 768,hidden_size = self.hidden_size,batch_first = True,dropout = self.dropout,bidirectional = self.bidirectional,num_layers = self.num_layers)
        if self.bidirectional:
            self.out = nn.Linear(in_features = 2*self.hidden_size,out_features=3)
        else:
            self.out = nn.Linear(in_features=self.hidden_size,out_features=3)
        
    def forward(self, input_ids, attention_mask,hidden):
        seq_output,_ = self.feature_extractor(input_ids=input_ids,attention_mask=attention_mask)
        outputs, (hn, cn) = self.lstm(seq_output,hidden)
        logits = self.out(torch.mean(outputs,dim=1))
        return logits
        