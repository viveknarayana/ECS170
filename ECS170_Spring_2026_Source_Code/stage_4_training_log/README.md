# Stage 4 Training Log

This folder stores informal run logs for Stage 4 experiments (RNN text classification on IMDb and RNN text generation on a short-jokes corpus).

## Summary table (all training runs)

For classification, Precision/Recall/F1 are **macro** averages on the 25k test split (micro and weighted values appear in each run section below). For generation there is no test-set accuracy — runs report final training loss (cross-entropy over the vocabulary) and sample generations. **Bold** marks what changed versus the previous run within the same subtask.

| Run | Subtask | Cell | Pooling | LR | Epochs | Accuracy | Precision | Recall | F1 | Final Train Loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | IMDb | Vanilla RNN | last hidden | 0.001 | 5 | 0.5057 | 0.5057 | 0.5057 | 0.5056 | 0.6914 |
| 2 | IMDb | LSTM | last hidden | 0.001 | 5 | 0.5167 | 0.5655 | 0.5167 | 0.4060 | 0.6282 |
| 3 | IMDb | Vanilla RNN | **mean over non-PAD** | 0.001 | **10** | 0.8322 | 0.8330 | 0.8322 | 0.8321 | 0.0205 |
| 4 | IMDb | LSTM | mean over non-PAD | 0.001 | 10 | 0.8296 | 0.8334 | 0.8296 | 0.8291 | 0.0101 |
| 5 | IMDb | GRU | mean over non-PAD | 0.001 | 10 | 0.8278 | 0.8349 | 0.8278 | 0.8269 | 0.0107 |
| 6 | Jokes | Vanilla RNN | n/a (next-token) | 0.001 | 30 | n/a | n/a | n/a | n/a | 1.673 |
| 7 | Jokes | LSTM | n/a (next-token) | 0.001 | 30 | n/a | n/a | n/a | n/a | 2.391 |
| 8 | Jokes | GRU | n/a (next-token) | 0.001 | 30 | n/a | n/a | n/a | n/a | 1.468 |
| 9 | IMDb | LSTM | **mean non-PAD, BiLSTM** | 0.001 | **12** | 0.8456 | 0.8473 | 0.8456 | 0.8455 | 0.0725 |
| 10 | IMDb | LSTM | **attention over non-PAD, BiLSTM** | 0.001 | 12 | **0.8499** | **0.8505** | **0.8499** | **0.8499** | 0.0518 |

## Experimental procedure (by subtask)

On **IMDb classification** the assignment-required swap is the recurrent cell (vanilla RNN → LSTM → GRU). We held everything else fixed across cells: top-10,000 vocab, sequence length 200 with post-padding, `Embedding(10000, 128)`, hidden size 128, 1 layer, Adam at lr=1e-3, batch 64, CrossEntropyLoss. Runs 1 and 2 read the final hidden state of the recurrent layer as the classifier input; because sequences are post-padded, that state is computed after the long `<PAD>` tail rather than after the review content, and both runs sat at chance (~50%). Runs 3–5 changed the pooling step in `forward()` to mean across non-PAD positions and bumped epochs from 5 to 10; test accuracy jumped to about 83% on the vanilla RNN. With that pooling fixed, we swapped only the recurrent cell (RNN, LSTM, GRU) while keeping embedding size 128, hidden size 128, batch 64, and learning rate 0.001; all three cells scored within about 0.5% on the same test split. **Run 9** added **bidirectional** LSTM, **embedding size 180**, **dropout 0.3**, **weight decay 1e-4**, and **12 epochs** with mean pooling (**84.56%** test). **Run 10** kept that setup but replaced mean pooling with **masked attention pooling** over non-PAD timesteps, reaching **84.99%** test accuracy (**0.84992** raw) — our best IMDb result and **0.008 percentage points** below the stated 85% bar.

On **joke generation** the model is `Embedding(4732, 128) → cell (256 hidden, 1 layer) → Linear(256, 4732)`. Training is per-position next-token prediction with teacher forcing: each joke is wrapped with `<BOS>`/`<EOS>`, padded to length 40, and supplies `input = tokens[:-1]`, `target = tokens[1:]`. Loss is CrossEntropy at every position with `ignore_index=PAD_ID`. Gradient norm is clipped at 5.0 as insurance against the loss spikes vanilla RNNs can produce on long sequences. At inference time we greedy-decode from the 3 seed words until `<EOS>` or 30 generated tokens. Runs 5, 6, 7 are the three cells in the same configuration.

## IMDb - Training Run 1 (vanilla RNN, last hidden)

Run context:
- Dataset: `IMDb` (25k train / 25k test)
- Cell: `nn.RNN` (1 layer, hidden 128)
- Pooling: final hidden state
- Epochs: `5`
- Learning rate: `0.001`
- Batch size: `64`
- Device: CPU

Model architecture used in this run:
1. `Embedding(10000, 128, padding_idx=0)`
2. `RNN(128, 128, batch_first=True)`
3. take `hidden[-1]`
4. `Linear(128, 2)`

Observed training progress:
- Epoch 0: train accuracy `0.5256`, loss `0.6957113693618775`
- Epoch 1: train accuracy `0.51116`, loss `0.6937031488418579`
- Epoch 2: train accuracy `0.51476`, loss `0.6936171920585632`
- Epoch 3: train accuracy `0.52164`, loss `0.6907985713195801`
- Epoch 4: train accuracy `0.52448`, loss `0.6913536590194702`

Saved learning curve:
- `../result/stage_4_result/plots/train_loss_vs_epoch_RNN_IMDB_ep5_lr0.001_20260520_003000.png`

Evaluation results (test set):
- Accuracy: `0.50572`
- F1 macro: `0.5055953237974214`
- F1 micro: `0.50572`
- F1 weighted: `0.5055953237974214`
- Precision macro: `0.5057257755760818`
- Precision micro: `0.50572`
- Precision weighted: `0.5057257755760817`
- Recall macro: `0.50572`
- Recall micro: `0.50572`
- Recall weighted: `0.50572`

Notes:
- Loss never moves off `ln(2) ≈ 0.693`; model is predicting essentially at chance.
- `hidden[-1]` is the RNN state after the long `<PAD>` tail rather than after the review content. Run 3 changes the pooling step.

## IMDb - Training Run 2 (LSTM, last hidden)

Run context:
- Dataset: `IMDb`
- Cell: `nn.LSTM` (1 layer, hidden 128)
- Pooling: final hidden state (`h_n[-1]`)
- Epochs: `5`
- Learning rate: `0.001`
- Batch size: `64`
- Device: CPU

Exact model changes from Run 1:
- `nn.RNN -> nn.LSTM` (same hidden size)
- `_last_hidden` returns `h_n[-1]` instead of `hidden[-1]` to handle the `(h, c)` tuple
- unchanged: embedding, vocab, sequence length, pooling strategy, optimizer, batch size, learning rate, epochs

Model architecture used in this run:
1. `Embedding(10000, 128, padding_idx=0)`
2. `LSTM(128, 128, batch_first=True)`
3. take `h_n[-1]`
4. `Linear(128, 2)`

Observed training progress:
- Epoch 0: train accuracy `0.53428`, loss `0.6927513120269775`
- Epoch 1: train accuracy `0.54952`, loss `0.6869376020431519`
- Epoch 2: train accuracy `0.53688`, loss `0.6610974114227295`
- Epoch 3: train accuracy `0.67428`, loss `0.6301270512008667`
- Epoch 4: train accuracy `0.5662`, loss `0.6282185446548462`

Saved learning curve:
- `../result/stage_4_result/plots/train_loss_vs_epoch_RNN_IMDB_LSTM_ep5_lr0.001_20260520_004129.png`

Evaluation results (test set):
- Accuracy: `0.51668`
- F1 macro: `0.40601232019032363`
- F1 micro: `0.51668`
- F1 weighted: `0.4060123201903236`
- Precision macro: `0.5654765629830271`
- Precision micro: `0.51668`
- Precision weighted: `0.5654765629830271`
- Recall macro: `0.51668`
- Recall micro: `0.51668`
- Recall weighted: `0.51668`

Notes:
- Train accuracy bounces (67% at epoch 3 then back to 57% at epoch 4) and loss only drifts down slowly. Same root cause as Run 1.
- F1 macro (0.41) well below accuracy (0.52) on a balanced test set indicates the classifier is predicting one class disproportionately.

## IMDb - Training Run 3 (vanilla RNN, mean over non-PAD)

Run context:
- Dataset: `IMDb`
- Cell: `nn.RNN` (1 layer, hidden 128)
- Pooling: mean over non-PAD timesteps
- Epochs: `10`
- Learning rate: `0.001`
- Batch size: `64`
- Device: CPU

Exact model changes from Run 1:
- pooling: `hidden[-1]` → mean of `output` over positions where `x != PAD`
- epochs: `5 -> 10`
- unchanged: embedding, vocab, sequence length, hidden size, optimizer, batch size, learning rate

Model architecture used in this run:
1. `Embedding(10000, 128, padding_idx=0)`
2. `RNN(128, 128, batch_first=True)` (keep per-timestep outputs)
3. `mask = (x != 0).unsqueeze(-1); pooled = (output * mask).sum(1) / mask.sum(1)`
4. `Linear(128, 2)`

Observed training progress:
- Epoch 0: train accuracy `0.8578`, loss `0.4824680353355408`
- Epoch 1: train accuracy `0.9024`, loss `0.31341489780902865`
- Epoch 2: train accuracy `0.93192`, loss `0.24589118203163146`
- Epoch 3: train accuracy `0.94928`, loss `0.19837085830688478`
- Epoch 4: train accuracy `0.96596`, loss `0.1588528996038437`
- Epoch 5: train accuracy `0.97948`, loss `0.12142492518424988`
- Epoch 6: train accuracy `0.9866`, loss `0.08840324820578098`
- Epoch 7: train accuracy `0.9902`, loss `0.05725204108119011`
- Epoch 8: train accuracy `0.99652`, loss `0.03866833661735058`
- Epoch 9: train accuracy `0.99728`, loss `0.02047051647990942`

Saved learning curve:
- `../result/stage_4_result/plots/train_loss_vs_epoch_RNN_IMDB_ep10_lr0.001_20260520_010308.png`

Evaluation results (test set):
- Accuracy: `0.83216`
- F1 macro: `0.8320573738887441`
- F1 micro: `0.83216`
- F1 weighted: `0.832057373888744`
- Precision macro: `0.8329738926141618`
- Precision micro: `0.83216`
- Precision weighted: `0.8329738926141619`
- Recall macro: `0.83216`
- Recall micro: `0.83216`
- Recall weighted: `0.83216`

Notes:
- Train accuracy reaches 99.7% by epoch 9; test accuracy 83.2% (train/test gap ~16%) indicates overfitting.

Comparison vs Run 1 (same cell, last-hidden pooling):
- Run 1: 50.6% test acc, loss stuck at 0.69
- Run 3: 83.2% test acc, loss reaches 0.02
- The change is the pooling step; +32.6 percentage points from one line in `forward()`.

## IMDb - Training Run 4 (LSTM, mean over non-PAD)

Run context:
- Dataset: `IMDb`
- Cell: `nn.LSTM` (1 layer, hidden 128)
- Pooling: mean over non-PAD timesteps
- Epochs: `10`
- Learning rate: `0.001`
- Batch size: `64`
- Device: CPU

Exact model changes from Run 3:
- `nn.RNN -> nn.LSTM` (same hidden size)
- unchanged: embedding, vocab, sequence length, pooling, optimizer, batch size, learning rate, epochs

Model architecture used in this run:
1. `Embedding(10000, 128, padding_idx=0)`
2. `LSTM(128, 128, batch_first=True)`
3. mean of `output` over non-PAD positions (same mask as Run 3)
4. `Linear(128, 2)`

Observed training progress:
- Epoch 0: train accuracy `0.87344`, loss `0.4612877783918381`
- Epoch 1: train accuracy `0.91652`, loss `0.29615185505867003`
- Epoch 2: train accuracy `0.94276`, loss `0.2238802589082718`
- Epoch 3: train accuracy `0.97092`, loss `0.16277887214899064`
- Epoch 4: train accuracy `0.9852`, loss `0.10748365723848342`
- Epoch 5: train accuracy `0.9942`, loss `0.05909671436667442`
- Epoch 6: train accuracy `0.99396`, loss `0.032983105410933494`
- Epoch 7: train accuracy `0.99912`, loss `0.017130227185636757`
- Epoch 8: train accuracy `0.99576`, loss `0.012850887437537312`
- Epoch 9: train accuracy `0.99844`, loss `0.010062658505626022`

Saved learning curve:
- `../result/stage_4_result/plots/train_loss_vs_epoch_RNN_IMDB_LSTM_ep10_lr0.001_20260520_011224.png`

Evaluation results (test set):
- Accuracy: `0.82956`
- F1 macro: `0.8290667141598844`
- F1 micro: `0.82956`
- F1 weighted: `0.8290667141598844`
- Precision macro: `0.8334086539862635`
- Precision micro: `0.82956`
- Precision weighted: `0.8334086539862634`
- Recall macro: `0.82956`
- Recall micro: `0.82956`
- Recall weighted: `0.82956`

Notes:
- Train accuracy reaches 99.84% by epoch 9; final training loss (0.010) is half of Run 3's (0.020), so the LSTM fits the training set more tightly under the same epoch budget.

Comparison vs Run 3 (vanilla RNN, same pooling):
- Run 3: 83.2% test acc, final train loss 0.020
- Run 4: 83.0% test acc, final train loss 0.010
- LSTM fits training data more tightly but generalizes equivalently. Once mean-pooling restores gradient flow through every non-PAD position, the gated cell's main advantage on long sequences is largely neutralized at this sequence length / hidden size.

## IMDb - Training Run 5 (GRU, mean over non-PAD)

Run context:
- Dataset: `IMDb`
- Cell: `nn.GRU` (1 layer, hidden 128)
- Pooling: mean over non-PAD timesteps
- Epochs: `10`
- Learning rate: `0.001`
- Batch size: `64`
- Device: CPU

Exact model changes from Run 3:
- `nn.RNN -> nn.GRU` (same hidden size)
- unchanged: embedding, vocab, sequence length, pooling, optimizer, batch size, learning rate, epochs

Model architecture used in this run:
1. `Embedding(10000, 128, padding_idx=0)`
2. `GRU(128, 128, batch_first=True)`
3. mean of `output` over non-PAD positions (same mask as Run 3)
4. `Linear(128, 2)`

Observed training progress:
- Epoch 0: train accuracy `0.87604`, loss `0.45018126670837405`
- Epoch 1: train accuracy `0.9166`, loss `0.29117559309005736`
- Epoch 2: train accuracy `0.9476`, loss `0.22364340370178223`
- Epoch 3: train accuracy `0.96816`, loss `0.161746485517025`
- Epoch 4: train accuracy `0.99056`, loss `0.10264753176927567`
- Epoch 5: train accuracy `0.99668`, loss `0.04830418703436851`
- Epoch 6: train accuracy `0.99868`, loss `0.024128602760732174`
- Epoch 7: train accuracy `0.99868`, loss `0.009162776311486959`
- Epoch 8: train accuracy `0.99712`, loss `0.011862530796043574`
- Epoch 9: train accuracy `0.99504`, loss `0.010725021382980048`

Saved learning curve:
- `../result/stage_4_result/plots/train_loss_vs_epoch_RNN_IMDB_GRU_ep10_lr0.001_20260520_012531.png`

Evaluation results (test set):
- Accuracy: `0.8278`
- F1 macro: `0.8268814910825132`
- F1 micro: `0.8278`
- F1 weighted: `0.8268814910825132`
- Precision macro: `0.8349076316006441`
- Precision micro: `0.8278`
- Precision weighted: `0.8349076316006441`
- Recall macro: `0.8278`
- Recall micro: `0.8278`
- Recall weighted: `0.8278`

Notes:
- Train accuracy crosses 99% at epoch 4 (earliest of the three cells), then oscillates between 99.5% and 99.9% as the optimizer chases tiny gradient signals.
- Same overfitting pattern as Run 4: low training loss (0.011) with no corresponding improvement in test accuracy.

Comparison across all post-fix IMDb cells (Runs 3-5, same pooling and hyperparameters):
- Run 3 (vanilla RNN): 83.22% test acc, final train loss 0.020
- Run 4 (LSTM):        82.96% test acc, final train loss 0.010
- Run 5 (GRU):         82.78% test acc, final train loss 0.011

Test accuracy across the three cells falls inside a 0.5-point band. With mean-pooling providing parallel gradient paths through every non-PAD position, the gated cells' long-sequence advantage does not materialize at sequence length 200 / hidden size 128. The bottleneck on this configuration is generalization, not gradient flow — adding capacity (LSTM, GRU) lowers training loss further but does not move test accuracy. A meaningful gap between cells would likely require longer sequences, larger hidden size, or regularization (dropout) — none of which were varied here.

## IMDb - Training Run 9 (LSTM, BiLSTM + regularization, embed 180)

Run context:
- Dataset: `IMDb`
- Cell: `nn.LSTM` (1 layer, hidden 128, **bidirectional=True**)
- Pooling: mean over non-PAD timesteps
- **Embedding dim: 180** (vs 128 in Runs 3–5)
- **Dropout: 0.3** after pooling
- **Weight decay: 1e-4** (Adam)
- Epochs: `12`
- Learning rate: `0.001`
- Batch size: `64`
- Device: CPU

Exact model changes from Run 4:
- unidirectional → **bidirectional** LSTM (`classifier` input `hidden_dim * 2`)
- **embedding_dim 128 → 180**
- **dropout 0.3** on pooled features
- **weight_decay 1e-4**
- epochs **10 → 12**
- unchanged: mean non-PAD pooling, vocab 10k, sequence length 200, hidden 128, batch size, learning rate

Model architecture used in this run:
1. `Embedding(10000, 180, padding_idx=0)`
2. `LSTM(180, 128, batch_first=True, bidirectional=True)`
3. mean of `output` over non-PAD positions
4. `Dropout(0.3)`
5. `Linear(256, 2)`

Observed training progress:
- Epoch 0: train accuracy `0.83076`, loss `0.4721860939693451`
- Epoch 1: train accuracy `0.89268`, loss `0.340502436876297`
- Epoch 2: train accuracy `0.91244`, loss `0.28533978717803954`
- Epoch 3: train accuracy `0.9332`, loss `0.25625092436790464`
- Epoch 4: train accuracy `0.94696`, loss `0.21812300235748291`
- Epoch 5: train accuracy `0.961`, loss `0.18611047837257386`
- Epoch 6: train accuracy `0.97164`, loss `0.15753731080532074`
- Epoch 7: train accuracy `0.95772`, loss `0.1321261991596222`
- Epoch 8: train accuracy `0.9806`, loss `0.12090864715576172`
- Epoch 9: train accuracy `0.98396`, loss `0.0981909324836731`
- Epoch 10: train accuracy `0.987`, loss `0.08293312027215957`
- Epoch 11: train accuracy `0.97696`, loss `0.07251750641942024`

Saved learning curve:
- `../result/stage_4_result/plots/train_loss_vs_epoch_RNN_IMDB_LSTM_ep12_lr0.001_20260520_234805.png`

Evaluation results (test set):
- Accuracy: `0.84564`
- F1 macro: `0.8454600885913598`
- F1 micro: `0.84564`
- F1 weighted: `0.8454600885913597`
- Precision macro: `0.8472570711698999`
- Precision micro: `0.84564`
- Precision weighted: `0.8472570711698999`
- Recall macro: `0.84564`
- Recall micro: `0.84564`
- Recall weighted: `0.84564`

Notes:
- Best IMDb test score until Run 10; superseded by attention-pooling run at **84.99%**.
- **+1.2 points** over Run 4 (unidirectional LSTM, embed 128).
- Still **0.44 percentage points** below 85% on its own.

Comparison vs Runs 3–5 (shared mean-pool baseline, embed 128):
- Run 3 (vanilla RNN): 83.22% test acc
- Run 4 (LSTM): 82.96% test acc
- Run 5 (GRU): 82.78% test acc
- Run 9 (LSTM tuned, mean pool): **84.56%** test acc

## IMDb - Training Run 10 (LSTM, BiLSTM + attention pooling, embed 180)

Run context:
- Dataset: `IMDb`
- Cell: `nn.LSTM` (1 layer, hidden 128, **bidirectional=True**)
- Pooling: **masked attention** over non-PAD timesteps (`Linear(256, 1)` + softmax)
- **Embedding dim: 180**
- **Dropout: 0.3** after pooling
- **Weight decay: 1e-4** (Adam)
- Epochs: `12`
- Learning rate: `0.001`
- Batch size: `64`
- Device: CPU

Exact model changes from Run 9:
- mean pool → **attention pool** over non-PAD positions
- unchanged: bidirectional LSTM, embedding 180, dropout, weight decay, epochs, hidden 128, batch size, learning rate

Model architecture used in this run:
1. `Embedding(10000, 180, padding_idx=0)`
2. `LSTM(180, 128, batch_first=True, bidirectional=True)`
3. `attention = Linear(256, 1)`; softmax weights over non-PAD outputs
4. `Dropout(0.3)`
5. `Linear(256, 2)`

Observed training progress:
- Epoch 0: train accuracy `0.86172`, loss `0.4578816579055786`
- Epoch 1: train accuracy `0.89544`, loss `0.3280862643623352`
- Epoch 2: train accuracy `0.91952`, loss `0.2756608881473541`
- Epoch 3: train accuracy `0.93936`, loss `0.25000179077625273`
- Epoch 4: train accuracy `0.95436`, loss `0.20161202676296233`
- Epoch 5: train accuracy `0.96572`, loss `0.16869770679473878`
- Epoch 6: train accuracy `0.97872`, loss `0.1293191934633255`
- Epoch 7: train accuracy `0.98108`, loss `0.1009162006855011`
- Epoch 8: train accuracy `0.98828`, loss `0.07784723918437958`
- Epoch 9: train accuracy `0.98288`, loss `0.06250999860405922`
- Epoch 10: train accuracy `0.9914`, loss `0.05381414687156677`
- Epoch 11: train accuracy `0.99228`, loss `0.05177496106982231`

Saved learning curve:
- `../result/stage_4_result/plots/train_loss_vs_epoch_RNN_IMDB_LSTM_ep12_lr0.001_20260521_023628.png`

Evaluation results (test set):
- Accuracy: `0.84992`
- F1 macro: `0.8498613772336874`
- F1 micro: `0.84992`
- F1 weighted: `0.8498613772336874`
- Precision macro: `0.8504673705936012`
- Precision micro: `0.84992`
- Precision weighted: `0.8504673705936011`
- Recall macro: `0.84992`
- Recall micro: `0.84992`
- Recall weighted: `0.84992`

Notes:
- **Best IMDb test score in this log** (+0.43 points vs Run 9 mean pool at 84.56%).
- Raw accuracy **0.84992** rounds to **85.0%** at one decimal but is **8e-5** below 0.85 as a float threshold.
- Attention pooling was the only architectural change vs Run 9.

## Jokes - Training Run 1 (vanilla RNN)

Run context:
- Dataset: short jokes (1,622 examples)
- Cell: `nn.RNN` (1 layer, hidden 256)
- Epochs: `30`
- Learning rate: `0.001`
- Batch size: `64`
- Gradient clipping: `max_norm=5.0`
- Device: CPU

Model architecture used in this run:
1. `Embedding(4732, 128, padding_idx=0)`
2. `RNN(128, 256, batch_first=True)` (returns per-position outputs)
3. `Linear(256, 4732)` (one logit per vocab token at every position)
4. CrossEntropyLoss at every non-PAD position (teacher forcing)

Observed training progress (selected epochs):
- Epoch 0: loss `7.1460337697651175`
- Epoch 5: loss `4.670979770867363`
- Epoch 10: loss `3.7665591345762355`
- Epoch 15: loss `3.0377716988729344`
- Epoch 20: loss `2.446427703344719`
- Epoch 25: loss `1.9775396081287087`
- Epoch 29: loss `1.6726209723405567`

Saved learning curve:
- `../result/stage_4_result/plots/gen_train_loss_vs_epoch_GEN_RNN_ep30_lr0.001_20260520_005327.png`

Sample generations (seeds drawn from the dataset):
- `what does a` → `what does a nosey pepper ? a penguin .`
- `gravity makes a` → `gravity makes a lot of problems .`
- `did you hear` → `did you hear about the ointment that does n't have nightmares nightmares ?`

Sample generations (freestyle seeds):
- `my dog said` → ``my dog said `` i 'm getting mighty fed up with this joke , i was a little ... wiping ! ! ! ! ! ! ! ! ! ! ! ! ! !``
- `the computer is` → `the computer is the loneliest ? prov-alone !`
- `a horse walks` → ``a horse walks into a bar ... and says , `` i 'm a wholesaler .``

Notes:
- Smooth monotonic loss decay; gradient clipping kept the run stable.
- Greedy decoding gets stuck in a `! ! ! ! !` loop on the `my dog said` seed (argmax can't escape a token that frequently follows itself).

## Jokes - Training Run 2 (LSTM)

Run context:
- Dataset: short jokes (1,622 examples)
- Cell: `nn.LSTM` (1 layer, hidden 256)
- Epochs: `30`
- Learning rate: `0.001`
- Batch size: `64`
- Gradient clipping: `max_norm=5.0`
- Device: CPU

Exact model changes from Run 1:
- `nn.RNN -> nn.LSTM`
- unchanged: embedding, vocab, sequence length, hidden size, optimizer, batch size, learning rate, epochs, gradient clipping

Observed training progress (selected epochs):
- Epoch 0: loss `7.362810940983557`
- Epoch 5: loss `4.987308559817539`
- Epoch 10: loss `4.262689602213341`
- Epoch 15: loss `3.6849812361814824`
- Epoch 20: loss `3.177349705878374`
- Epoch 25: loss `2.718132356239159`
- Epoch 29: loss `2.3905915055704177`

Saved learning curve:
- `../result/stage_4_result/plots/gen_train_loss_vs_epoch_GEN_LSTM_ep30_lr0.001_20260520_005523.png`

Sample generations (seeds drawn from the dataset):
- `what does a` → `what does a nosey pepper ? it 's a little meteor .`
- `gravity makes a` → `gravity makes a joke about the bar ... i 'm a little forced .`
- `did you hear` → `did you hear about the kidnapping recently ? the goatherd was a little meteor .`

Sample generations (freestyle seeds):
- `my dog said` → `my dog said he was a job ... ... but it was a little crabby !`
- `the computer is` → `the computer is a balloon 's favorite genre of the alphabet ? i 'll be a little`
- `a horse walks` → `a horse walks into a bar ... and a mop .`

Notes:
- Higher final loss than Run 1 (2.39 vs 1.67). LSTM has 4 gates per cell vs the vanilla RNN's 1, so more parameters to fit under the same 30-epoch budget.
- Outputs avoid the `! ! ! ! !` repetition loop. Several generations end with high-frequency training-set tail phrases (`a little crabby`, `a little meteor`).

## Jokes - Training Run 3 (GRU)

Run context:
- Dataset: short jokes (1,622 examples)
- Cell: `nn.GRU` (1 layer, hidden 256)
- Epochs: `30`
- Learning rate: `0.001`
- Batch size: `64`
- Gradient clipping: `max_norm=5.0`
- Device: CPU

Exact model changes from Run 1:
- `nn.RNN -> nn.GRU`
- unchanged: embedding, vocab, sequence length, hidden size, optimizer, batch size, learning rate, epochs, gradient clipping

Observed training progress (selected epochs):
- Epoch 0: loss `7.209637497268093`
- Epoch 5: loss `4.711469249866453`
- Epoch 10: loss `3.7482368446307763`
- Epoch 15: loss `2.969544898420021`
- Epoch 20: loss `2.3182957040102354`
- Epoch 25: loss `1.8012345761758919`
- Epoch 29: loss `1.4677279673140087`

Saved learning curve:
- `../result/stage_4_result/plots/gen_train_loss_vs_epoch_GEN_GRU_ep30_lr0.001_20260520_005744.png`

Sample generations (seeds drawn from the dataset):
- `what does a` → `what does a nosey pepper do ? a penguin !`
- `gravity makes a` → `gravity makes a lot of conflict in the world 's greatest composers ? because they are n't really care .`
- `did you hear` → `did you hear about the guy who invented a knife that fell into the bar ? he was a little crabby !`

Sample generations (freestyle seeds):
- `my dog said` → `my dog said , i was watching a really poorly done scary .`
- `the computer is` → `the computer is a joke about a tornado . i 'm a tortoise ...`
- `a horse walks` → ``a horse walks into a bar ... and asks , `` is a mop of a tree ? a pool table .``

Notes:
- Lowest final loss of the three cells (1.47). GRU has 3 gates vs LSTM's 4, fewer parameters to fit under the same epoch budget.
- Outputs are the longest and most varied of the three cells; less reliance on the `a little crabby` / `a little meteor` tail phrases seen in Run 2.

Comparison across all jokes runs:
- vanilla RNN: final loss 1.673; produces shortest outputs, occasional repetition loops
- LSTM:        final loss 2.391; produces middle-length outputs, leans on memorized tail phrases
- GRU:         final loss 1.468; produces longest and most varied outputs

Final training loss does not align with qualitative output quality. The vanilla RNN reached a lower loss than the LSTM but produces more pathological greedy-decoding loops; the GRU edged out both on loss with the most varied outputs. This is a property of greedy `argmax` decoding on a 1,622-joke corpus rather than of the cells themselves.
