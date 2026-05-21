'''
LSTM variant of the stage 4 text classifier (task 4-5).
'''

import torch
import torch.nn.functional as F
from local_code.stage_4_code.Method_RNN import Method_RNN
from torch import nn


class Method_RNN_LSTM(Method_RNN):
    embedding_dim = 180
    hidden_dim = 128
    learning_rate = 1e-3
    max_epoch = 12
    dropout_p = 0.3
    weight_decay = 1e-4

    def _build_network(self, num_classes):
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_index)
        self.recurrent = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(self.hidden_dim * 2, 1)
        self.dropout = nn.Dropout(self.dropout_p)
        self.classifier = nn.Linear(self.hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.recurrent(embedded)
        mask = x != self.pad_index
        scores = self.attention(output).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(output * weights.unsqueeze(-1), dim=1)
        return self.classifier(self.dropout(pooled))
