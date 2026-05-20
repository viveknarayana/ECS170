'''
GRU variant of the joke generator (task 4-5).
'''

from local_code.stage_4_code.Method_RNN_Generator import Method_RNN_Generator
from local_code.stage_4_code.Dataset_Loader_Generation import PAD_ID
from torch import nn


class Method_RNN_Generator_GRU(Method_RNN_Generator):

    def _build_network(self):
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=PAD_ID)
        self.recurrent = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.head = nn.Linear(self.hidden_dim, self.vocab_size)
