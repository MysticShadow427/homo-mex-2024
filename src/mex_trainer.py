import torch.nn as nn
import numpy as np 
import torch 
from tqdm import tqdm

def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0

  for d in tqdm(data_loader):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

def train_epoch_dense_ensemble(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0

  for d in tqdm(data_loader):
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
    loss = loss_fn(outputs, targets.squeeze())

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)


def train_epoch_ensemble(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0

  for d in tqdm(data_loader):
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
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

def train_epoch_lstm(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples,
  num_layers,
  hidden_size
):
  model = model.train()

  losses = []
  correct_predictions = 0

  for d in tqdm(data_loader):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)


    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds.indices == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in tqdm(data_loader):
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model_dense_ensemble(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in tqdm(data_loader):
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

      loss = loss_fn(outputs, targets.squeeze())

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model_ensemble(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in tqdm(data_loader):
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

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model_lstm(model, data_loader, loss_fn, device, n_examples,num_layers,hidden_size):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in tqdm(data_loader):
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
      )
      preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds.indices == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

