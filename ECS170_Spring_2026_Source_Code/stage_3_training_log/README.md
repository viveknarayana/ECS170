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
