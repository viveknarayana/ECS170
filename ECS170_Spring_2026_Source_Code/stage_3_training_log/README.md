# Stage 3 Training Log

This folder stores informal run logs for Stage 3 experiments.

## Summary table (all training runs)

Precision, Recall, and F1 are **macro** averages on the test split (micro and weighted values appear in each run section below). **Bold** marks what changed versus the previous run for that dataset (Run 1 per dataset is the baseline).

| Run | Dataset | Layers | LR | Epochs | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | MNIST | 2× Conv3×3 (32, 64) + ReLU + pool each; Linear | 0.001 | 10 | 0.9897 | 0.9896 | 0.9896 | 0.9896 |
| 2 | MNIST | 2× **Conv5×5** (32, 64) + ReLU + pool each; Linear; **batch 64** (Run 1 used 128) | 0.001 | 10 | 0.9928 | 0.9927 | 0.9927 | 0.9927 |
| 3 | ORL (simple) | Conv5×5 (16) + ReLU + pool; Linear | 0.001 | 10 | 0.9500 | 0.9250 | 0.9500 | 0.9333 |
| 4 | ORL (deep) | **2×** Conv5×5 **(16, 32)** + ReLU + pool each; **Dropout 0.3**; Linear | 0.001 | **30** | 0.9750 | 0.9625 | 0.9750 | 0.9667 |
| 5 | CIFAR-10 (simple) | 2× Conv3×3 (32, 64) + ReLU + pool each; Linear | 0.001 | 10 | 0.6485 | 0.6566 | 0.6485 | 0.6480 |
| 6 | CIFAR-10 (medium) | **3×** Conv3×3 (32, 64, **128**) + ReLU + pool each; Linear | 0.001 | 10 | 0.6842 | 0.6933 | 0.6842 | 0.6841 |
| 7 | CIFAR-10 (deep) | **5×** Conv3×3 + **BN** + ReLU, 3 pools; **Dropout**; Linear **2048→256→10** | 0.001 | 10 | 0.7663 | 0.7830 | 0.7663 | 0.7641 |

## Experimental procedure (by dataset)

Across datasets we changed only what the benchmark called for and held the rest fixed when comparing runs. On **MNIST** we used a two-block 3×3 CNN with batch 128 as a baseline, then switched to **5×5** convolutions with the same channel layout and **batch 64** to widen the receptive field and alter gradient noise while keeping learning rate, optimizer, and epochs the same. On **ORL** we started with a single conv block, then moved to a **second block**, **dropout** before the linear layer, and **more epochs** so added capacity and memorization risk were balanced on a small face dataset while leaving kernel and pooling choices comparable to the shallow run. On **CIFAR-10** we ran a short ablation chain, a **two-block** floor, a **third** conv stage for richer features, then a **deep** stack with **batch normalization**, **dropout**, and a **two-layer** classifier head under the same epoch budget so later gains track architecture rather than simply training longer.

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

## ORL - Training Run 1 (variant: simple)

Run context:
- Dataset: `ORL`
- Variant: `simple`
- Epochs: `10`
- Learning rate: `0.001`
- Batch size: `32`
- Device: CPU

Model architecture used in this run:
1. `Conv2d(in_channels, 16, kernel_size=5, padding=2)`
2. `ReLU`
3. `MaxPool2d(2)`
4. `Flatten`
5. `Linear(flat_features, num_classes)`

Observed training progress:
- Epoch 0: train accuracy `0.6666666666666666`, loss `3.621510214275784`
- Epoch 1: train accuracy `0.9166666666666666`, loss `1.254526150226593`
- Epoch 2: train accuracy `0.9972222222222222`, loss `0.36153461866908604`
- Epoch 3: train accuracy `1.0`, loss `0.08514447758595149`
- Epoch 4: train accuracy `1.0`, loss `0.0260486568013827`
- Epoch 5: train accuracy `1.0`, loss `0.012939365456501643`
- Epoch 6: train accuracy `1.0`, loss `0.007611856526798673`
- Epoch 7: train accuracy `1.0`, loss `0.005424019176926878`
- Epoch 8: train accuracy `1.0`, loss `0.00433216556492779`
- Epoch 9: train accuracy `1.0`, loss `0.003537168519364463`

Saved learning curve:
- `../../result/stage_3_result/plots/train_loss_vs_epoch_ORL_simple_ep10_lr0.001_20260507_033334.png`

Evaluation results (test set):
- Accuracy: `0.95`
- F1 macro: `0.9333333333333332`
- F1 micro: `0.95`
- F1 weighted: `0.9333333333333332`
- Precision macro: `0.925`
- Precision micro: `0.95`
- Precision weighted: `0.925`
- Recall macro: `0.95`
- Recall micro: `0.95`
- Recall weighted: `0.95`

Quick inference sanity check:
- `test_index=0, pred=1, true=1`

## ORL - Training Run 2 (variant: deep)

Run context:
- Dataset: `ORL`
- Variant: `deep`
- Epochs: `30`
- Learning rate: `0.001`
- Batch size: `32`
- Device: CPU

Exact model changes from Run 1 (simple):
- added a second conv block (`16 -> 32` channels)
- added `Dropout(0.3)` before the classifier
- epochs `10 -> 30`
- unchanged: kernel size (5x5), padding (2), stride (1), pooling, optimizer (Adam), batch size (32), learning rate (1e-3)

Model architecture used in this run:
1. `Conv2d(in_channels, 16, kernel_size=5, padding=2)`
2. `ReLU`
3. `MaxPool2d(2)`
4. `Conv2d(16, 32, kernel_size=5, padding=2)`
5. `ReLU`
6. `MaxPool2d(2)`
7. `Flatten`
8. `Dropout(0.3)`
9. `Linear(flat_features, num_classes)`

Observed training progress (selected epochs):
- Epoch 0: train accuracy `0.3388888888888889`, loss `3.645611317952474`
- Epoch 1: train accuracy `0.8222222222222222`, loss `2.345067420270708`
- Epoch 2: train accuracy `0.9694444444444444`, loss `0.5224661058849759`
- Epoch 3: train accuracy `0.9972222222222222`, loss `0.11510597036944495`
- Epoch 6: train accuracy `1.0`, loss `0.017561120837409464`
- Epoch 9: train accuracy `1.0`, loss `0.007076167842994133`
- Epoch 14: train accuracy `1.0`, loss `0.00029459446409924164`
- Epoch 19: train accuracy `1.0`, loss `0.00019447675311110087`
- Epoch 29: train accuracy `1.0`, loss `0.00011925064047520411`

Saved learning curve:
- `../../result/stage_3_result/plots/train_loss_vs_epoch_ORL_deep_ep30_lr0.001_20260507_033612.png`

Evaluation results (test set):
- Accuracy: `0.975`
- F1 macro: `0.9666666666666666`
- F1 micro: `0.975`
- F1 weighted: `0.9666666666666666`
- Precision macro: `0.9625`
- Precision micro: `0.975`
- Precision weighted: `0.9625`
- Recall macro: `0.975`
- Recall micro: `0.975`
- Recall weighted: `0.975`

Quick inference sanity check:
- `test_index=0, pred=1, true=1`

Comparison vs simple variant:
- simple: 95.0% test acc (1 conv, 10 epochs)
- deep:   97.5% test acc (2 conv + dropout, 30 epochs)
- gain from added depth/regularization/epochs: +2.5 percentage points (1 fewer misclassification on the 40-image test set).

## CIFAR - Training Run 1 (variant: simple)

Run context:
- Dataset: `CIFAR-10`
- Variant: `simple`
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

Observed training progress:
- Epoch 0: train accuracy `0.39484`, loss `1.7220909796905517`
- Epoch 1: train accuracy `0.53404`, loss `1.3580733214569092`
- Epoch 2: train accuracy `0.57986`, loss `1.225619200553894`
- Epoch 3: train accuracy `0.60646`, loss `1.1518175605010987`
- Epoch 4: train accuracy `0.62428`, loss `1.0991495751571656`
- Epoch 5: train accuracy `0.63858`, loss `1.0621258233833313`
- Epoch 6: train accuracy `0.65178`, loss `1.0261795722198486`
- Epoch 7: train accuracy `0.6647`, loss `0.9881911171340942`
- Epoch 8: train accuracy `0.67308`, loss `0.9621131206893921`
- Epoch 9: train accuracy `0.6817`, loss `0.9402069597244262`

Saved learning curve:
- `../../result/stage_3_result/plots/train_loss_vs_epoch_CIFAR_simple_ep10_lr0.001_20260507_034324.png`

Evaluation results (test set):
- Accuracy: `0.6485`
- F1 macro: `0.6480175296704095`
- F1 micro: `0.6485`
- F1 weighted: `0.6480175296704096`
- Precision macro: `0.6565728965908215`
- Precision micro: `0.6485`
- Precision weighted: `0.6565728965908215`
- Recall macro: `0.6485000000000001`
- Recall micro: `0.6485`
- Recall weighted: `0.6485`

Quick inference sanity check:
- `test_index=0, pred=8, true=3` (wrong — model predicted class 8, actual was 3)

Notes:
- Below the 70% target as expected for a shallow 2-conv baseline. Establishes the floor for the ablation series; expect medium/deep variants to clear 70%.
- Train-test gap is small (train ~68%, test 65%), suggesting the model is capacity-limited rather than overfitting. More layers / BatchNorm / dropout in the deeper variants should help.

## CIFAR - Training Run 2 (variant: medium)

Run context:
- Dataset: `CIFAR-10`
- Variant: `medium`
- Epochs: `10`
- Learning rate: `0.001`
- Batch size: `128`
- Device: CPU

Exact model changes from Run 1 (simple):
- added a third conv block (`64 -> 128` channels) with another `MaxPool2d(2)`
- spatial reduction: `32x32 -> 16x16 -> 8x8 -> 4x4` (was `32 -> 16 -> 8`)
- unchanged: kernel size (3x3), padding (1), stride (1), no BatchNorm, no Dropout, optimizer (Adam), batch size (128), learning rate (1e-3), epochs (10)

Model architecture used in this run:
1. `Conv2d(in_channels, 32, kernel_size=3, padding=1)`
2. `ReLU`
3. `MaxPool2d(2)`
4. `Conv2d(32, 64, kernel_size=3, padding=1)`
5. `ReLU`
6. `MaxPool2d(2)`
7. `Conv2d(64, 128, kernel_size=3, padding=1)`
8. `ReLU`
9. `MaxPool2d(2)`
10. `Flatten`
11. `Linear(flat_features, num_classes)`

Observed training progress:
- Epoch 0: train accuracy `0.39648`, loss `1.6914559703063965`
- Epoch 1: train accuracy `0.55232`, loss `1.293464029121399`
- Epoch 2: train accuracy `0.60982`, loss `1.1353431934738158`
- Epoch 3: train accuracy `0.64554`, loss `1.0341495520401`
- Epoch 4: train accuracy `0.6717`, loss `0.9619465403556824`
- Epoch 5: train accuracy `0.6951`, loss `0.8955190141677857`
- Epoch 6: train accuracy `0.71452`, loss `0.8427418274307251`
- Epoch 7: train accuracy `0.7291`, loss `0.7981300364494324`
- Epoch 8: train accuracy `0.74048`, loss `0.7599060789108276`
- Epoch 9: train accuracy `0.7563`, loss `0.7181942917823791`

Saved learning curve:
- `../../result/stage_3_result/plots/train_loss_vs_epoch_CIFAR_medium_ep10_lr0.001_20260507_035127.png`

Evaluation results (test set):
- Accuracy: `0.6842`
- F1 macro: `0.6841086749428625`
- F1 micro: `0.6842`
- F1 weighted: `0.6841086749428626`
- Precision macro: `0.693344328475281`
- Precision micro: `0.6842`
- Precision weighted: `0.693344328475281`
- Recall macro: `0.6842`
- Recall micro: `0.6842`
- Recall weighted: `0.6842`

Quick inference sanity check:
- `test_index=0, pred=3, true=3` (correct — same image that simple variant got wrong)

Comparison vs simple variant:
- simple: 64.85% test acc (2 conv blocks)
- medium: 68.42% test acc (3 conv blocks)
- gain from added depth: +3.57 percentage points
- still below the 70% target — the deep variant adds BatchNorm + Dropout + a wider FC head to push past 70%.
- training loss still decreasing at epoch 9 (train acc 75.6% vs test 68.4%), suggesting the model would benefit from more epochs or stronger regularization rather than just more capacity.

## CIFAR - Training Run 3 (variant: deep)

Run context:
- Dataset: `CIFAR-10`
- Variant: `deep`
- Epochs: `10`
- Learning rate: `0.001`
- Batch size: `128`
- Device: CPU

Exact model changes from Run 2 (medium):
- doubled-up 3x3 convs: each of the first two blocks now has two stacked conv layers instead of one
- added `BatchNorm2d` after every conv
- added a fully connected hidden layer `Linear(2048 -> 256) + ReLU` between flatten and the classifier
- added `Dropout(0.3)` before the FC hidden layer and before the output layer
- spatial layout unchanged (`32x32 -> 16x16 -> 8x8 -> 4x4`); other hyperparameters unchanged

Model architecture used in this run:
1. `Conv2d(in_channels, 32, kernel_size=3, padding=1) + BN + ReLU`
2. `Conv2d(32, 32, kernel_size=3, padding=1) + BN + ReLU`
3. `MaxPool2d(2)`
4. `Conv2d(32, 64, kernel_size=3, padding=1) + BN + ReLU`
5. `Conv2d(64, 64, kernel_size=3, padding=1) + BN + ReLU`
6. `MaxPool2d(2)`
7. `Conv2d(64, 128, kernel_size=3, padding=1) + BN + ReLU`
8. `MaxPool2d(2)`
9. `Flatten`
10. `Dropout(0.3)`
11. `Linear(2048 -> 256) + ReLU`
12. `Dropout(0.3)`
13. `Linear(256 -> num_classes)`

Observed training progress:
- Epoch 0: train accuracy `0.49888`, loss `1.3983327325439454`
- Epoch 1: train accuracy `0.66494`, loss `0.9622204212760925`
- Epoch 2: train accuracy `0.71432`, loss `0.8248789291572571`
- Epoch 3: train accuracy `0.74338`, loss `0.7401256809806823`
- Epoch 4: train accuracy `0.76598`, loss `0.6768600988769531`
- Epoch 5: train accuracy `0.78378`, loss `0.6260377000045776`
- Epoch 6: train accuracy `0.79736`, loss `0.587485109539032`
- Epoch 7: train accuracy `0.81332`, loss `0.5454869079208374`
- Epoch 8: train accuracy `0.8249`, loss `0.5084948580932617`
- Epoch 9: train accuracy `0.8314`, loss `0.4821077932739258`

Saved learning curve:
- `../../result/stage_3_result/plots/train_loss_vs_epoch_CIFAR_deep_ep10_lr0.001_20260507_041103.png`

Evaluation results (test set):
- Accuracy: `0.7663`
- F1 macro: `0.7641409999681933`
- F1 micro: `0.7663`
- F1 weighted: `0.7641409999681932`
- Precision macro: `0.7830173124139689`
- Precision micro: `0.7663`
- Precision weighted: `0.7830173124139689`
- Recall macro: `0.7663`
- Recall micro: `0.7663`
- Recall weighted: `0.7663`

Quick inference sanity check:
- `test_index=0, pred=3, true=3` (correct)

Comparison across all CIFAR variants:
- simple (2 conv, no BN, no dropout):                64.85% test acc
- medium (3 conv, no BN, no dropout):                68.42% test acc
- deep   (5 conv + BN + dropout + FC hidden layer):  76.63% test acc

Cumulative gain from simple to deep: +11.78 percentage points; clears the 70% target with margin to spare. Train accuracy still climbing at epoch 9 (83.14%), so additional epochs would likely improve test accuracy further, but training time on CPU was the bottleneck.
