from os import path

import torch
from torch import nn
from transformers import BertModel


class BertFF(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.layers = nn.Sequential(
            nn.Linear(768, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        return self.layers(pooled_output)

    def save(self, epoch_num: int) -> None:
        torch.save(self.state_dict(), path.join('models', f'BertFF_{epoch_num}.pt'))

    def load(self, epoch_num: int) -> None:
        self.load_state_dict(torch.load(path.join('models', f'BertFF_{epoch_num}.pt')))


class BertAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.encoder = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 768)
        )

    def forward(self, input_id, mask, decode=True):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        encoded = self.encoder(pooled_output)
        return self.decoder(encoded), pooled_output if decode else encoded

    def save(self, epoch_num: int) -> None:
        torch.save(self.encoder.state_dict(), path.join('models', f'BertAE_{epoch_num}.pt'))

    def load(self, epoch_num: int) -> None:
        self.encoder.load_state_dict(torch.load(path.join('models', f'BertAE_{epoch_num}.pt')))
