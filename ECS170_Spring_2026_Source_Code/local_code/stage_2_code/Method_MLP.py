'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Method_MLP(method, nn.Module):
    data = None
    hidden_dim = 256
    # it defines the max rounds to train the model
    max_epoch = 20
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    # location to save convergence plot
    training_curve_folder_path = '../../result/stage_2_result/plots/'
    training_curve_file_name_prefix = 'train_loss_vs_epoch'

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # Stage 2: 784 input features, 256 hidden units
        self.fc_layer_1 = nn.Linear(784, self.hidden_dim)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # Last layer outputs logits (per-class scores). No Softmax: nn.CrossEntropyLoss expects logits.
        self.fc_layer_2 = nn.Linear(self.hidden_dim, 10)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))
        # output layer result (logits); self.fc_layer_2(h) will be an N x 10 tensor
        # n (denotes the input instance number): 0th dimension; 10 (denotes the class number): 1st dimension
        # With softmax removed: we pass raw scores to CrossEntropyLoss (see PyTorch docs)
        logits = self.fc_layer_2(h)
        return logits

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        loss_history = []
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.asarray(X, dtype=np.float32)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)
            loss_history.append(train_loss.item())

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch%100 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
        return loss_history
    
    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.asarray(X, dtype=np.float32)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        loss_history = self.train(self.data['train']['X'], self.data['train']['y'])
        os.makedirs(self.training_curve_folder_path, exist_ok=True)
        run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        curve_file_name = (
            f"{self.training_curve_file_name_prefix}_"
            f"ep{self.max_epoch}_lr{self.learning_rate}_h{self.hidden_dim}_{run_tag}.png"
        )
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
            
