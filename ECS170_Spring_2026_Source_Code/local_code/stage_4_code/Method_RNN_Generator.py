'''
Concrete MethodModule class for RNN joke generator (stage 4 baseline).
'''

from local_code.base_class.method import method
from local_code.stage_4_code.Dataset_Loader_Generation import (
    PAD_ID, UNK_ID, BOS_ID, EOS_ID, tokenize_joke,
)
import os
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Method_RNN_Generator(method, nn.Module):
    data = None
    max_epoch = 30
    learning_rate = 1e-3
    batch_size = 64
    embedding_dim = 128
    hidden_dim = 256
    training_curve_folder_path = '../../result/stage_4_result/plots/'
    training_curve_file_name_prefix = 'gen_train_loss_vs_epoch'

    # set by Setting before run()
    vocab_size = None
    id_to_token = None
    vocab = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.embedding = None
        self.recurrent = None
        self.head = None

    def _build_network(self):
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=PAD_ID)
        self.recurrent = nn.RNN(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.head = nn.Linear(self.hidden_dim, self.vocab_size)

    def _run_recurrent(self, embedded, hidden=None):
        # returns (sequence_output, new_hidden); split so LSTM subclass can override
        return self.recurrent(embedded, hidden)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, new_hidden = self._run_recurrent(embedded, hidden)
        logits = self.head(output)
        return logits, new_hidden

    def train(self, X, y):
        # input is tokens[:-1], target is tokens[1:] (next-token prediction with teacher forcing)
        X_arr = np.asarray(X, dtype=np.int64)
        inputs = torch.LongTensor(X_arr[:, :-1])
        targets = torch.LongTensor(X_arr[:, 1:])

        if self.embedding is None:
            self._build_network()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        train_loader = DataLoader(
            TensorDataset(inputs, targets),
            batch_size=self.batch_size,
            shuffle=True
        )

        loss_history = []
        for epoch in range(self.max_epoch):
            epoch_loss = 0.0
            total = 0
            for batch_X, batch_y in train_loader:
                logits, _ = self.forward(batch_X)
                # CE expects (N, C); flatten time dimension into batch dimension
                loss = loss_function(logits.reshape(-1, self.vocab_size), batch_y.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                # vanilla RNN explodes easily on long generations; cheap insurance
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
                total += batch_X.size(0)
            epoch_loss = epoch_loss / max(total, 1)
            loss_history.append(epoch_loss)
            print('Epoch:', epoch, 'Loss:', epoch_loss)
        return loss_history

    def _encode_seed(self, seed_words):
        # accept either a list of words or a single string; lowercase via the tokenizer
        if isinstance(seed_words, str):
            tokens = tokenize_joke(seed_words)
        else:
            tokens = tokenize_joke(' '.join(seed_words))
        return [BOS_ID] + [self.vocab.get(t, UNK_ID) for t in tokens]

    def generate(self, seed_words, max_length=30):
        # greedy decoding: feed seed, then sample argmax until EOS or max_length.
        # Skipping nn.Module.eval() because train(X,y) overrides its train(mode) sibling
        # and we have no dropout/batchnorm to toggle anyway.
        with torch.no_grad():
            ids = self._encode_seed(seed_words)
            input_ids = torch.LongTensor([ids])
            logits, hidden = self.forward(input_ids)
            generated = list(ids)
            next_id = int(logits[0, -1].argmax().item())
            generated.append(next_id)

            steps = 0
            while next_id != EOS_ID and steps < max_length:
                input_ids = torch.LongTensor([[next_id]])
                logits, hidden = self.forward(input_ids, hidden)
                next_id = int(logits[0, -1].argmax().item())
                generated.append(next_id)
                steps += 1

        # drop BOS, stop at EOS, detokenize
        words = []
        for i in generated[1:]:
            if i == EOS_ID:
                break
            words.append(self.id_to_token.get(i, '<UNK>'))
        return ' '.join(words)

    def run(self):
        print('method running...')
        print('--start training...')
        loss_history = self.train(self.data['train']['X'], self.data['train']['y'])

        os.makedirs(self.training_curve_folder_path, exist_ok=True)
        run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = self.method_name.replace(' ', '_')
        curve_file_name = f"{self.training_curve_file_name_prefix}_{safe_name}_ep{self.max_epoch}_lr{self.learning_rate}_{run_tag}.png"
        curve_path = os.path.join(self.training_curve_folder_path, curve_file_name)
        plt.figure()
        plt.plot(range(1, len(loss_history) + 1), loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Convergence ({self.method_name})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(curve_path)
        plt.close()
        print('saved training convergence plot to:', curve_path)

        return {
            'train_loss_history': loss_history,
            'train_loss_curve_path': curve_path
        }
