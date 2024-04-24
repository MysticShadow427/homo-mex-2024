from transformers import AutoModel
import torch
import torch.nn as nn

checkpoint = 'dccuchile/bert-base-spanish-wwm-uncased'

class MexSpanClassifier(nn.Module):

  def __init__(self, n_classes):
    super(MexSpanClassifier, self).__init__()
    self.bert = AutoModel.from_pretrained(checkpoint,return_dict=False)
    modules = [self.bert.embeddings, *self.bert.encoder.layer[:5]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)
  # https://colab.research.google.com/drive/1FklMjfL6-fMhoPo73BYQ7EvSVYWLykku#scrollTo=IDGwprtLC5cw&uniqifier=1