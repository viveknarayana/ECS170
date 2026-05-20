'''
LSTM variant of the stage 4 text classifier (task 4-5).
'''

from local_code.stage_4_code.Method_RNN import Method_RNN
from torch import nn


class Method_RNN_LSTM(Method_RNN):

    def _build_network(self, num_classes):
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_index)
        self.recurrent = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def _last_hidden(self, hidden):
        # LSTM returns (h_n, c_n); we only want h_n's top layer
        h_n, _ = hidden
        return h_n[-1]
