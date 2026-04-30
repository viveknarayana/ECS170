# Stage 3 Training Log

This folder stores informal run logs for Stage 3 experiments.

## MNIST - Training Run 1

Run context:
- Dataset: `MNIST`
- Epochs: `10`
- Learning rate: `0.001`
- Batch size: `128`
- Device: CPU

Model architecture used in this run:
1. `Conv2d(in_channels, 32, kernel_size=3, padding=1)`
2. `ReLU`
3. `MaxPool2d(2)`
4. `Conv2d(32, 64, kernel_size=3, padding=1)`
5. `ReLU`
6. `MaxPool2d(2)`
7. `Flatten`
8. `Linear(flat_features, num_classes)`

Observed training progress snapshot:
- Epoch 6: train accuracy `0.9937333333333334`, loss `0.02412305560503155`
- Epoch 7: train accuracy `0.9959666666666667`, loss `0.019869066911439102`
- Epoch 8: train accuracy `0.9936833333333334`, loss `0.017234346013888718`
- Epoch 9: train accuracy `0.9971166666666667`, loss `0.014455708391265944`

Saved learning curve:
- `../../result/stage_3_result/plots/train_loss_vs_epoch_ep10_lr0.001_20260430_140630.png`
- `../result/stage_3_result/plots/train_loss_vs_epoch_ep10_lr0.001_20260430_140630.png` (path relative to this log folder)

Evaluation results (test set):
- Accuracy: `0.9897`
- F1 macro: `0.9895997432460863`
- F1 micro: `0.9897`
- F1 weighted: `0.9897044209335761`
- Precision macro: `0.9895993675861394`
- Precision micro: `0.9897`
- Precision weighted: `0.9897380903776551`
- Recall macro: `0.9896299175022062`
- Recall micro: `0.9897`
- Recall weighted: `0.9897`

Quick inference sanity check:
- `test_index=0, pred=7, true=7`

## MNIST - Training Run 2

Run context:
- Dataset: `MNIST`
- Epochs: `10` (kept fixed for fair comparison)
- Learning rate: `0.001`
- Batch size: `64` (changed from 128)
- Device: CPU

Exact model changes from Run 1:
- `kernel_size`: `3 -> 5` in both conv layers
- `padding`: `1 -> 2` in both conv layers (to preserve spatial size with 5x5 kernels)
- `batch_size`: `128 -> 64`
- unchanged: conv channels (`32, 64`), pooling layout, classifier head, optimizer, epochs

Model architecture used in this run:
1. `Conv2d(in_channels, 32, kernel_size=5, padding=2)`
2. `ReLU`
3. `MaxPool2d(2)`
4. `Conv2d(32, 64, kernel_size=5, padding=2)`
5. `ReLU`
6. `MaxPool2d(2)`
7. `Flatten`
8. `Linear(flat_features, num_classes)`

Observed training progress snapshot:
- Epoch 0: train accuracy `0.9853666666666666`, loss `0.14638993615383902`
- Epoch 1: train accuracy `0.9906333333333334`, loss `0.04448867419815312`
- Epoch 2: train accuracy `0.9893166666666666`, loss `0.030728234968163695`
- Epoch 3: train accuracy `0.9960166666666667`, loss `0.02359672035177549`
- Epoch 4: train accuracy `0.9947166666666667`, loss `0.017034269976204573`
- Epoch 5: train accuracy `0.9956166666666667`, loss `0.01418289322082807`
- Epoch 6: train accuracy `0.9976666666666667`, loss `0.010378085317866256`
- Epoch 7: train accuracy `0.9955333333333334`, loss `0.009692463995938306`
- Epoch 8: train accuracy `0.9971333333333333`, loss `0.008063491814293229`
- Epoch 9: train accuracy `0.9991166666666667`, loss `0.006000412640573147`

Saved learning curve:
- `../../result/stage_3_result/plots/train_loss_vs_epoch_ep10_lr0.001_20260430_150829.png`
- `../result/stage_3_result/plots/train_loss_vs_epoch_ep10_lr0.001_20260430_150829.png` (path relative to this log folder)

Evaluation results (test set):
- Accuracy: `0.9928`
- F1 macro: `0.9927411732610212`
- F1 micro: `0.9928`
- F1 weighted: `0.9927992570704081`
- Precision macro: `0.992749768733457`
- Precision micro: `0.9928`
- Precision weighted: `0.9928101805899674`
- Recall macro: `0.9927440374693214`
- Recall micro: `0.9928`
- Recall weighted: `0.9928`

Quick inference sanity check:
- `test_index=0, pred=7, true=7`
