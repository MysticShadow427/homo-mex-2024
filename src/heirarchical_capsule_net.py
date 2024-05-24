import torch
from torch import nn
from transformers import AutoModel
from capsule_net import CapsNet
from heirarchical_net import HierAttNet

class HCN(nn.Module):
    def __init__(self,num_capsules, in_channels, out_channels, num_routing_iterations,word_hidden_size, sent_hidden_size, batch_size,max_sent_length, max_word_length,num_route_nodes=768,hidden_size = 384,num_classes = 3,dropout = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = True
        self.hieratt_net = HierAttNet(word_hidden_size, sent_hidden_size, batch_size,
                 max_sent_length, max_word_length)
        self.capsule_net = CapsNet(num_capsules, num_route_nodes, in_channels, out_channels, num_routing_iterations)
        self.bilstm = nn.LSTM(input_size = 768,hidden_size = self.hidden_size,batch_first = True,dropout = self.dropout,bidirectional = True,num_layers = 1)
        self.bert = AutoModel.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased',return_dict = False)
        self.linear_1 = nn.Linear(in_features=None,out_features=256)
        self.linear_2 = nn.Linear(in_features=256,out_features=128)
        self.out = nn.Linear(in_features=128,out_features=self.num_classes)
        self.dropout_layer = nn.Dropout(self.dropout) # after BiLSTM

    def forward(self,input_ids,attention_mask):
        seq_output,_ = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        print(seq_output) # for in_channels
        batch_size = seq_output.size(0)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).to('cuda')
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).to('cuda')
        f_outs, h_outs = self.lstm(seq_output, (h0, c0))
        f_outs = self.dropout_layer(f_outs)
        f_han = self.hieratt_net(f_outs)
        f_cap = self.capsule_net(f_outs)
        feats = torch.cat((f_han, f_cap,f_aux), dim=1)
        feats = self.linear_1(feats)
        feats = self.linear_2(feats)
        
        return self.out(feats)
