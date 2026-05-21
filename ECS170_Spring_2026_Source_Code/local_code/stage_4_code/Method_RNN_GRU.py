'''
GRU variant of the stage 4 text classifier (task 4-5).
'''

from local_code.stage_4_code.Method_RNN import Method_RNN
from torch import nn


class Method_RNN_GRU(Method_RNN):
    # Keep ablation baseline (Runs 3–5); tuned settings live on Method_RNN / Method_RNN_LSTM only.
    embedding_dim = 128
    learning_rate = 1e-3
    max_epoch = 10
    dropout_p = 0.0
    weight_decay = 0.0

    def _build_network(self, num_classes):
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_index)
        self.recurrent = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
