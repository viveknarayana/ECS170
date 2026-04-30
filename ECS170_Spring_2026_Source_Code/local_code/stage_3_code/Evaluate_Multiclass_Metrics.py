'''
Multiclass Precision / Recall / F1 using sklearn.
'''

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


def _labels_to_numpy(y):
    if hasattr(y, 'detach'):
        y = y.detach().cpu().numpy()
    return np.asarray(y).astype(int).ravel()


class Evaluate_Multiclass_Metrics(evaluate):
    data = None

    def evaluate(self):
        print('evaluating multiclass precision / recall / F1...')
        ty = _labels_to_numpy(self.data['true_y'])
        py = _labels_to_numpy(self.data['pred_y'])
        results = {}
        for avg in ('macro', 'weighted', 'micro'):
            results[f'precision_{avg}'] = float(precision_score(ty, py, average=avg, zero_division=0))
            results[f'recall_{avg}'] = float(recall_score(ty, py, average=avg, zero_division=0))
            results[f'f1_{avg}'] = float(f1_score(ty, py, average=avg, zero_division=0))
        for k in sorted(results.keys()):
            print(' ', k + ':', results[k])
        return results
