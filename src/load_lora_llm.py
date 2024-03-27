# use lora fine tuning here as full fine tuning can lead to catastrophic forgetting and worse results on the classification task
import torch
import torch.nn as nn
from functools import partial
from transformers import AutoModel

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = True
lora_value = True
lora_projection = True
lora_mlp = True
lora_head = True

layers = []

assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)

checkpoint = 'dccuchile/bert-base-spanish-wwm-uncased'
bert_model = AutoModel.from_pretrained(checkpoint,return_dict=False)
for param in bert_model.parameters():
    param.requires_grad = False

for layer in bert_model.encoder.layer:
    if lora_query:
        layer.attention.self.query= assign_lora(layer.attention.self.query)
    if lora_key:
        layer.attention.self.key = assign_lora(layer.attention.self.key)
    if lora_value:
        layer.attention.self.value = assign_lora(layer.attention.self.value)
    if lora_projection:
        layer.attention.output.dense = assign_lora(layer.attention.output.dense)
    if lora_mlp:
        layer.intermediate.dense = assign_lora(layer.intermediate.dense)
    if lora_head:
        layer.output.dense = assign_lora(layer.output.dense)

class MexSpanClassifierLoRA(nn.Module):

  def __init__(self, n_classes):
    super(MexSpanClassifierLoRA, self).__init__()
    self.bert = bert_model
    self.drop = nn.Dropout(p=0.1)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)