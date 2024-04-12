import torch
from torch.utils.data import Dataset, DataLoader

class MexSpanDataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      review,
      truncation = True,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }
  
class MexSpanEnsembleDataset(Dataset):

  def __init__(self, reviews, targets, bert_tokenizer,roberta_tokenizer,deberta_tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.bert_tokenizer = bert_tokenizer
    self.roberta_tokenizer = roberta_tokenizer
    self.deberta_tokenizer = deberta_tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    bert_encoding = self.bert_tokenizer.encode_plus(
      review,
      truncation = True,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    roberta_encoding = self.roberta_tokenizer.encode_plus(
      review,
      truncation = True,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    deberta_encoding = self.deberta_tokenizer.encode_plus(
      review,
      truncation = True,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,
      'bert_input_ids': bert_encoding['input_ids'].flatten(),
      'bert_attention_mask': bert_encoding['attention_mask'].flatten(),
      'roberta_input_ids': roberta_encoding['input_ids'].flatten(),
      'roberta_attention_mask': roberta_encoding['attention_mask'].flatten(),
      'deberta_input_ids': deberta_encoding['input_ids'].flatten(),
      'deberta_attention_mask': deberta_encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = MexSpanDataset(
    reviews=df.content.to_numpy(),
    targets=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True
  )

# make a dataloader for smote augmented embeddings from 3 different models we will use simple dense neural nets