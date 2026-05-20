from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import os
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Method_RNN(method, nn.Module):
    data = None
    max_epoch = 5
    learning_rate = 1e-3
    batch_size = 64
    embedding_dim = 128
    hidden_dim = 128
    training_curve_folder_path = '../../result/stage_4_result/plots/'
    training_curve_file_name_prefix = 'train_loss_vs_epoch'

    # set by the script before run() so the embedding layer is sized right
    vocab_size = None
    pad_index = 0

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.embedding = None
        self.recurrent = None
        self.classifier = None

    def _build_network(self, num_classes):
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_index)
        self.recurrent = nn.RNN(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def _last_hidden(self, hidden):
        # vanilla RNN hidden shape is (num_layers, batch, hidden); take the top layer
        return hidden[-1]

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.recurrent(embedded)
        return self.classifier(self._last_hidden(hidden))

    def _to_tensor(self, X, y=None):
        X_tensor = torch.LongTensor(np.asarray(X, dtype=np.int64))
        if y is None:
            return X_tensor
        return X_tensor, torch.LongTensor(np.asarray(y, dtype=np.int64))

    def train(self, X, y):
        X_tensor, y_tensor = self._to_tensor(X, y)
        num_classes = int(y_tensor.max().item()) + 1
        if self.embedding is None:
            self._build_network(num_classes)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        train_loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True
        )

        loss_history = []
        for epoch in range(self.max_epoch):
            epoch_loss = 0.0
            total = 0
            for batch_X, batch_y in train_loader:
                y_pred = self.forward(batch_X)
                train_loss = loss_function(y_pred, batch_y)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_loss += train_loss.item() * batch_X.size(0)
                total += batch_X.size(0)

            epoch_loss = epoch_loss / max(total, 1)
            loss_history.append(epoch_loss)

            # batched full-set pass to avoid OOM on 25k x 200 tokens
            with torch.no_grad():
                preds = []
                for batch_X, _ in DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=256):
                    preds.append(self.forward(batch_X).max(1)[1])
                all_pred = torch.cat(preds)
            accuracy_evaluator.data = {'true_y': y_tensor, 'pred_y': all_pred}
            print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', epoch_loss)
        return loss_history

    def test(self, X):
        X_tensor = self._to_tensor(X)
        with torch.no_grad():
            preds = []
            for (batch_X,) in DataLoader(TensorDataset(X_tensor), batch_size=256):
                preds.append(self.forward(batch_X).max(1)[1])
        return torch.cat(preds)

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

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y'],
            'train_loss_history': loss_history,
            'train_loss_curve_path': curve_path
        }
