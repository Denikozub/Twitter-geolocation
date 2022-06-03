from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):

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
