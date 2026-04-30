'''
Concrete MethodModule class for a simple CNN classifier.
'''

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


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 10
    learning_rate = 1e-3
    batch_size = 128
    training_curve_folder_path = '../../result/stage_3_result/plots/'
    training_curve_file_name_prefix = 'train_loss_vs_epoch'

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.network = None

    def _prepare_input(self, X):
        arr = np.asarray(X, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

        if arr.ndim == 3:
            # N x H x W -> N x 1 x H x W
            arr = np.expand_dims(arr, axis=1)
        elif arr.ndim == 4:
            # N x H x W x C -> N x C x H x W
            arr = np.transpose(arr, (0, 3, 1, 2))
        else:
            raise ValueError('Unexpected input shape: ' + str(arr.shape))
        return arr

    def _build_network(self, input_shape, num_classes):
        c, h, w = input_shape
        self.network = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * (h // 4) * (w // 4), num_classes)
        )

    def forward(self, x):
        return self.network(x)

    def train(self, X, y):
        X_tensor = torch.FloatTensor(self._prepare_input(X))
        y_tensor = torch.LongTensor(np.asarray(y, dtype=np.int64))

        num_classes = int(np.max(y_tensor.numpy())) + 1
        if self.network is None:
            self._build_network(X_tensor.shape[1:], num_classes)

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

            with torch.no_grad():
                all_pred = self.forward(X_tensor).max(1)[1]
            accuracy_evaluator.data = {'true_y': y_tensor, 'pred_y': all_pred}
            print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', epoch_loss)
        return loss_history

    def test(self, X):
        X_tensor = torch.FloatTensor(self._prepare_input(X))
        with torch.no_grad():
            y_pred = self.forward(X_tensor)
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        loss_history = self.train(self.data['train']['X'], self.data['train']['y'])

        os.makedirs(self.training_curve_folder_path, exist_ok=True)
        run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        curve_file_name = f"{self.training_curve_file_name_prefix}_ep{self.max_epoch}_lr{self.learning_rate}_{run_tag}.png"
        curve_path = os.path.join(self.training_curve_folder_path, curve_file_name)
        plt.figure()
        plt.plot(range(1, len(loss_history) + 1), loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Convergence')
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
