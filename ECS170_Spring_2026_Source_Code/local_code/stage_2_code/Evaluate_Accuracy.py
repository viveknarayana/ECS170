'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score
import numpy as np


def _labels_to_numpy(y):
    if hasattr(y, 'detach'):
        y = y.detach().cpu().numpy()
    return np.asarray(y).astype(int).ravel()


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        ty = _labels_to_numpy(self.data['true_y'])
        py = _labels_to_numpy(self.data['pred_y'])
        return accuracy_score(ty, py)
        
