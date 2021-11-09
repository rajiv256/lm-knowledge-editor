# Building models
import os

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from transformers import BertModel

import configs
# Building datasets
import src.preprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BertForFever(nn.Module):
    def __init__(self):
        """Downloads a BERT base uncased model and adds a linear layer on top of it"""
        super(BertForFever, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, 1)

    def forward(self, ids, token_ids, mask):
        """
        inputs: ids, token_ids, and mask each of dim = [bsz x seqlen]
        returns: logits of a positive label, dim = [bsz]
        """
        sequence_output = self.bert(input_ids=ids, attention_mask=mask)[0]
        # print(sequence_output.size())
        sequence_output = nn.ReLU()(sequence_output)
        # print(sequence_output.size())
        linear_output = self.classifier(sequence_output[:, 0, :])
        # print(linear_output.size())

        return linear_output.squeeze()


def finetune_bert(model, train_loader, val_loader, configs=None):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                                 eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                                 )
    epochs = 4
    best_accurate_preds = 0
    loss = nn.BCELoss()

    for epoch in range(epochs):
        for x, y in tqdm.tqdm(train_loader):
            model.zero_grad()
            ids, mask = x["input_ids"], x["attention_mask"]
            ids, mask = ids.squeeze(1), mask.squeeze(1)
            y = y.type(torch.FloatTensor)
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            output = model(ids, None, mask)
            # print(output)
            # print(y)
            losses = loss(torch.sigmoid(output), y)
            losses.backward()
            optimizer.step()

        with torch.no_grad():
            val_loss = 0
            accurate_preds = 0
            total_preds = 0
            for x, y in tqdm.tqdm(val_loader):
                ids, mask = x["input_ids"], x["attention_mask"]
                ids, mask = ids.squeeze(1), mask.squeeze(1)
                y = y.type(torch.FloatTensor)
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                output = model(ids, None, mask)
                val_loss += loss(torch.sigmoid(output), y)
                probs = torch.sigmoid(output)  # maps logits to [0,1]
                preds = torch.round(probs).squeeze()  # rounds to closest integer, 0 or 1
                accurate_preds += torch.sum(preds == y).item()
                total_preds += len(y).item()

            print("Validation loss: ", val_loss, "|| Accuracy: ", accurate_preds / total_preds)
            if accurate_preds > best_accurate_preds:
                torch.save(model.state_dict(), "models")
    model.load_state_dict(torch.load("models"))
    return model


# Build the datasets
batch_size = 16
train_dataset = src.preprocess.FeverDataset(os.path.join(configs.DATA_DIR, 'fever', 'train.jsonl'))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

dev_dataset = src.preprocess.FeverDataset(os.path.join(configs.DATA_DIR, 'fever', 'dev.jsonl'))
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

model = BertForFever()
model = model.to(device)
finetune_bert(model, train_loader, dev_loader)
