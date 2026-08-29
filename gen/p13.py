# Generator script for "Practice13_RNN_for_Time_Series.ipynb"
# 개념과 데이터(KOSPI 일별 시세)는 이 과목 구버전 저장소의 "Chapter3_Deep Learning_3_RNN.ipynb" 에서 가져왔고,
# 코드는 이 과목의 CLAUDE.md / code-patterns.md 규약에 맞춰 새로 작성했다. 구버전에서 고친 것:
# 분할 전 fit_transform (data leakage) -> 시간순 70/15/15 분할 후 train 통계로만 표준화,
# split=200 (test 가 train 보다 큼) -> val 분할 신설, num_layers 2 -> 1 (Step 4 재귀식과 1:1 대응),
# 회귀 출력단 nn.Sigmoid 제거 (z-score 타깃은 [0,1] 을 벗어난다), lr -> rho, MSE 를 "rmse" 로 부르던 오표기.
#
# train/evaluate 시그니처: code-patterns.md §7(b) 는 분류 전제라 evaluate 가 (loss, accuracy, y_pred) 를
# 반환한다. 이 회차는 회귀이므로 accuracy 자리를 RMSE(지수 포인트 환산)로 바꿨다. §7 의 나머지 규칙
# (두 함수만, criterion 은 함수 내부 생성, model.train()/model.eval(), per-batch mean 누적)은 그대로 유지한다.
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip("\n")))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip("\n")))


# ---------------------------------------------------------------- Cell 0 - title
md(r"""
# Practice 13 — RNN for Time Series

A recurrent network reads five consecutive trading days and predicts the closing index of the next one.

Two models share one recurrent layer and differ only in what their linear readout is allowed to read.

| Model | Readout input |
|:---|:---|
| A | the last hidden state |
| B | all hidden states, laid end to end |

**Dataset.** KOSPI daily quotes, 431 trading days from 2019-01-30 to 2020-10-30.

| Step | What it produces |
|:---|:---|
| Windows | the input tensor and the next-day target |
| Split | chronological train / validation / test, standardized on training statistics |
| `train` / `evaluate` | the epoch loop and the no-grad pass |
| Comparison | learning curves, predictions in index points, RMSE |
""")

# ---------------------------------------------------------------- Cell 1 - Step 0 md
md(r"""
---
## Step 0. Imports and Setup
""")

# ---------------------------------------------------------------- Cell 2 - imports
code(r"""
import math
import os
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)

plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
""")

# ---------------------------------------------------------------- Cell 3 - Step 1 md
md(r"""
---
## Step 1. The Series

Daily KOSPI quotes, one row per trading day.

| Column | Use |
|:---|:---|
| `Open`, `High`, `Low`, `Volume` | model input |
| `Close` | prediction target |
| `Date`, `Adj Close` | not used |
""")

# ---------------------------------------------------------------- Cell 4 - load csv
code(r"""
DATA_PATH = './data/kospi.csv'
DATA_URL = 'https://raw.githubusercontent.com/jongmoonha/AI-ME-Practice/main/data/kospi.csv'

if not os.path.exists(DATA_PATH):
    os.makedirs('./data', exist_ok=True)
    print('downloading kospi.csv ...')
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

df = pd.read_csv(DATA_PATH)

first_date = df['Date'].iloc[0]
last_date = df['Date'].iloc[-1]
print(f'rows: {len(df)}, from {first_date} to {last_date}')
print(f'missing values: {df.isna().sum().sum()}')

df.head(10)
""")

# ---------------------------------------------------------------- Cell 5 - series plot
code(r"""
fig, ax = plt.subplots(1, 1, figsize=(12, 4))

ax.plot(df['Close'].values, 'k-', linewidth=1)
ax.set_xlabel('Trading day')
ax.set_ylabel('Close')
ax.set_title('KOSPI closing index')
ax.grid(alpha=0.3)

plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- Cell 6 - Step 2 md
md(r"""
---
## Step 2. Sliding Windows

The model reads a window of $T$ consecutive days and predicts the `Close` of the day that follows it.

| Tensor | Shape | Meaning |
|:---|:---|:---|
| `X_seq` | $(N, T, d)$ | $N$ windows of $T$ days, $d$ features per day |
| `y_seq` | $(N, 1)$ | the `Close` of the day after each window |

With $T = 5$ and 431 trading days the series yields $N = 431 - 5 = 426$ windows.

Each window overlaps its neighbour by four days, so consecutive windows are far from independent.
""")

# ---------------------------------------------------------------- Cell 7 - to_sequences
code(r"""
sequence_length = 5

X_all = df[['Open', 'High', 'Low', 'Volume']].values.astype(np.float32)
y_all = df[['Close']].values.astype(np.float32)


def to_sequences(X, y, sequence_length):
    # window i covers days i .. i+sequence_length-1 and targets the Close of day i+sequence_length
    X_windows = []
    y_targets = []
    for i in range(len(X) - sequence_length):
        X_windows.append(X[i:i + sequence_length])
        y_targets.append(y[i + sequence_length])
    return np.array(X_windows), np.array(y_targets)


X_seq, y_seq = to_sequences(X_all, y_all, sequence_length)

print(f'X_seq {tuple(X_seq.shape)}   (N, T, d)')
print(f'y_seq {tuple(y_seq.shape)}   (N, 1)')
""")

# ---------------------------------------------------------------- Cell 8 - Step 3 md
md(r"""
---
## Step 3. Chronological Split and Scaling

A shuffled split would drop later days into the training set and let the model interpolate between days it
has already been shown.
Cutting the windows in time order keeps every validation and test window after every training window.

| Split | Windows | Share |
|:---|:---|:---|
| Train | 298 | 70% |
| Validation | 63 | 15% |
| Test | 65 | 15% |

The standardization statistics are computed on the training windows alone and then applied to all three
splits.
""")

# ---------------------------------------------------------------- Cell 9 - loaders md
md(r"""
Four loaders over three splits.
`train_loader` shuffles because the update order should not follow the calendar, while `train_eval_loader`
holds the same windows in time order so the predictions can be drawn as a curve.

| Loader | Windows | `shuffle` | Used for |
|:---|:---|:---|:---|
| `train_loader` | train | `True` | the update loop |
| `train_eval_loader` | train | `False` | predictions in time order |
| `val_loader` | validation | `False` | monitoring after each epoch |
| `test_loader` | test | `False` | read once, at the end |
""")

# ---------------------------------------------------------------- Cell 10 - split, scale, tensors, loaders
code(r"""
n_train = int(len(X_seq) * 0.70)
n_val = int(len(X_seq) * 0.15)

X_train, y_train = X_seq[:n_train], y_seq[:n_train]
X_val, y_val = X_seq[n_train:n_train + n_val], y_seq[n_train:n_train + n_val]
X_test, y_test = X_seq[n_train + n_val:], y_seq[n_train + n_val:]

input_size = X_seq.shape[2]

# every day inside a training window counts once, so the windows are flattened before averaging
input_train_mean = X_train.reshape(-1, input_size).mean(axis=0)
input_train_std = X_train.reshape(-1, input_size).std(axis=0)
input_train_std[input_train_std == 0] = 1.0   # leave zero-variance features unscaled

close_train_mean = y_train.mean()
close_train_std = y_train.std()

X_train = (X_train - input_train_mean) / input_train_std
X_val = (X_val - input_train_mean) / input_train_std
X_test = (X_test - input_train_mean) / input_train_std

y_train = (y_train - close_train_mean) / close_train_std
y_val = (y_val - close_train_mean) / close_train_std
y_test = (y_test - close_train_mean) / close_train_std

print(f'windows - train {len(X_train)}, val {len(X_val)}, test {len(X_test)}')
print(f'close_train_mean {close_train_mean:.2f}, close_train_std {close_train_std:.2f}')
print(f'standardized target range - train [{y_train.min():.3f}, {y_train.max():.3f}]')
print(f'standardized target range - val   [{y_val.min():.3f}, {y_val.max():.3f}]')
print(f'standardized target range - test  [{y_test.min():.3f}, {y_test.max():.3f}]')
print(f'test windows above the training maximum: {(y_test > y_train.max()).sum()} of {len(y_test)}')

batch_size = 20

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

train_set = TensorDataset(X_train_tensor, y_train_tensor)
val_set = TensorDataset(X_val_tensor, y_val_tensor)
test_set = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
train_eval_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

print(f'batches per epoch - train {len(train_loader)}, val {len(val_loader)}, test {len(test_loader)}')

X_batch, y_batch = next(iter(train_loader))
print(f'one batch - X {tuple(X_batch.shape)}, y {tuple(y_batch.shape)}')
""")

# ---------------------------------------------------------------- Cell 11 - Step 4 md
md(r"""
---
## Step 4. What a Recurrent Layer Computes

The layer carries a hidden state $\mathbf{h}_t \in \mathbb{R}^{H}$ and rewrites it once per time step.

$$\mathbf{h}_t = \tanh\!\left(\mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_{xh}
+ \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_{hh}\right),
\qquad t = 1, \dots, T, \qquad \mathbf{h}_0 = \mathbf{0}$$

The same two matrices are reused at every step, which is what makes the layer recurrent.

| Symbol | Shape | Parameter |
|:---|:---|:---|
| $\mathbf{W}_{xh}$ | $(H, d)$ | `weight_ih_l0` |
| $\mathbf{W}_{hh}$ | $(H, H)$ | `weight_hh_l0` |
| $\mathbf{b}_{xh}$, $\mathbf{b}_{hh}$ | $(H,)$ each | `bias_ih_l0`, `bias_hh_l0` |

A textbook writes one bias where PyTorch keeps two; the unrolled loop below adds both.
""")

# ---------------------------------------------------------------- Cell 12 - hand vs nn.RNN
code(r"""
hidden_size = 8
num_layers = 1

torch.manual_seed(42)
demo_rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

X_window = X_train_tensor[0:1]                          # a single window, (1, T, d)
h0 = torch.zeros(num_layers, 1, hidden_size)            # the initial hidden state is all zeros

with torch.no_grad():
    out, h_n = demo_rnn(X_window, h0)

# the same recurrence written out one time step at a time
weight_ih = demo_rnn.weight_ih_l0
weight_hh = demo_rnn.weight_hh_l0
bias_ih = demo_rnn.bias_ih_l0
bias_hh = demo_rnn.bias_hh_l0

h = torch.zeros(hidden_size)
with torch.no_grad():
    for t in range(sequence_length):
        x_t = X_window[0, t]
        h = torch.tanh(weight_ih @ x_t + bias_ih + weight_hh @ h + bias_hh)

print(f'out {tuple(out.shape)} holds h_1 .. h_T, h_n {tuple(h_n.shape)} holds h_T alone')
print('by hand :', h.numpy())
print('nn.RNN  :', out[0, -1].numpy())
""")

# ---------------------------------------------------------------- Cell 13 - Step 5 md
md(r"""
---
## Step 5. `train` and `evaluate`

| Function | Returns |
|:---|:---|
| `evaluate(model, loader, device)` | `(loss, rmse, y_pred)` |
| `train(model, train_loader, val_loader, optimizer, epochs, device)` | `(train_losses, train_rmses, val_losses, val_rmses)` |

`loss` averages one mean squared error per batch, which is what the learning curves plot, and `rmse` is
its square root in index points.
The last batch of a split is shorter than the others, so Step 8 recomputes the reported figures from
`y_pred` over each split as a whole.

Monitoring runs on `val_loader`; `test_loader` inside the loop would let the test set steer the choice
between the two models.
""")

# ---------------------------------------------------------------- Cell 14 - evaluate
code(r"""
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    loss_sum = 0.0
    preds = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_hat = model(X_batch)
            loss_sum += criterion(y_hat, y_batch).item()
            preds.append(y_hat.cpu())

    loss = loss_sum / len(loader)
    rmse = math.sqrt(loss) * close_train_std      # standardized error back into index points
    y_pred = torch.cat(preds).numpy()
    return loss, rmse, y_pred
""")

# ---------------------------------------------------------------- Cell 15 - train
code(r"""
def train(model, train_loader, val_loader, optimizer, epochs, device):
    criterion = nn.MSELoss()
    train_losses, train_rmses, val_losses, val_rmses = [], [], [], []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_hat = model(X_batch)
            loss = criterion(y_hat, y_batch)

            optimizer.zero_grad()                 # clear the accumulated gradients
            loss.backward()                       # autograd computes the gradients
            optimizer.step()                      # the optimizer updates the parameters

            loss_sum += loss.item()

        train_loss = loss_sum / len(train_loader)
        val_loss, val_rmse, _ = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        train_rmses.append(math.sqrt(train_loss) * close_train_std)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)

        if (epoch + 1) % 50 == 0:
            print(f'  Epoch {epoch+1:3d}/{epochs}  '
                  f'train MSE={train_loss:.5f}  val MSE={val_loss:.5f}  '
                  f'val RMSE={val_rmse:.1f} points')
    return train_losses, train_rmses, val_losses, val_rmses
""")

# ---------------------------------------------------------------- Cell 16 - shared settings
code(r"""
rho = 1e-3
epochs = 200

print(f'Sequence length : {sequence_length}')
print(f'Hidden size     : {hidden_size}')
print(f'Recurrent layers: {num_layers}')
print(f'Batch size      : {batch_size}')
print(f'Learning rate   : {rho}')
print(f'Epochs          : {epochs}')
print(f'Optimizer       : Adam')
print(f'Loss            : {nn.MSELoss()}')
""")

# ---------------------------------------------------------------- Cell 17 - Step 6 md
md(r"""
---
## Step 6. Model A — Last Hidden State

The readout sees only the final hidden state.

$$\hat{y} = \mathbf{w}^{\top}\mathbf{h}_T + b, \qquad \mathbf{w} \in \mathbb{R}^{H}$$

$\mathbf{h}_T$ is the state after the layer has read all $T$ days, so the earlier days reach the readout
only through what the recurrence carried forward.
""")

# ---------------------------------------------------------------- Cell 18 - class A
code(r"""
class RNNLastState(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, h_n = self.rnn(x, h0)      # out holds h_1 .. h_T for every window in the batch
        y_hat = self.fc(out[:, -1, :])  # the last time step only
        return y_hat
""")

# ---------------------------------------------------------------- Cell 19 - train A
code(r"""
torch.manual_seed(42)
model_last_state = RNNLastState(input_size, hidden_size, num_layers).to(device)
optimizer_last_state = optim.Adam(model_last_state.parameters(), lr=rho)

torch.manual_seed(42)   # train_loader shuffles from the same state for both models
print('Model A (last hidden state)')
train_losses_last_state, train_rmses_last_state, val_losses_last_state, val_rmses_last_state = train(
    model_last_state, train_loader, val_loader, optimizer_last_state, epochs, device)
""")

# ---------------------------------------------------------------- Cell 20 - Step 7 md
md(r"""
---
## Step 7. Model B — All Hidden States

The readout sees every hidden state, laid end to end.

$$\hat{y} = \mathbf{w}^{\top}\left[\mathbf{h}_1; \mathbf{h}_2; \dots; \mathbf{h}_T\right] + b,
\qquad \mathbf{w} \in \mathbb{R}^{HT}$$

The recurrent layer is untouched, so the only thing that changes is what the linear layer is allowed to
read.
""")

# ---------------------------------------------------------------- Cell 21 - class B
code(r"""
class RNNAllStates(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, h_n = self.rnn(x, h0)             # out holds h_1 .. h_T for every window in the batch
        y_hat = self.fc(out.reshape(out.size(0), -1))   # all time steps, laid end to end
        return y_hat
""")

# ---------------------------------------------------------------- Cell 22 - train B
code(r"""
torch.manual_seed(42)
model_all_states = RNNAllStates(input_size, hidden_size, num_layers, sequence_length).to(device)
optimizer_all_states = optim.Adam(model_all_states.parameters(), lr=rho)

torch.manual_seed(42)   # train_loader shuffles from the same state for both models
print('Model B (all hidden states)')
train_losses_all_states, train_rmses_all_states, val_losses_all_states, val_rmses_all_states = train(
    model_all_states, train_loader, val_loader, optimizer_all_states, epochs, device)
""")

# ---------------------------------------------------------------- Cell 23 - Step 8 md
md(r"""
---
## Step 8. Comparison

| Item | A (last state) | B (all states) |
|:---|:---|:---|
| Readout input | `out[:, -1, :]` | `out.reshape(out.size(0), -1)` |
| Readout layer | `nn.Linear(hidden_size, 1)` | `nn.Linear(hidden_size * sequence_length, 1)` |
| Recurrent layer | `nn.RNN`, $H = 8$, one layer, $\tanh$ | same |
| Data, split, scaling, loaders | shared objects from Step 3 | same |
| $\rho$, epochs, batch size, optimizer, loss | shared values from Step 5 | same |
| Seed before construction and before training | 42 | 42 |
""")

# ---------------------------------------------------------------- Cell 24 - loss curves
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses_last_state, 'b-', linewidth=2, label='A (last state)')
axes[0].plot(train_losses_all_states, 'r--', linewidth=2, label='B (all states)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE (standardized)')
axes[0].set_title('Training loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(val_losses_last_state, 'b-', linewidth=2, label='A (last state)')
axes[1].plot(val_losses_all_states, 'r--', linewidth=2, label='B (all states)')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE (standardized)')
axes[1].set_title('Validation loss'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- Cell 25 - reading md
md(r"""
The readout of B carries $HT + 1 = 41$ weights against $H + 1 = 9$ in A.
That is a consequence of what B reads, not a second change on top of it.

The test window is mostly an extrapolation.
Step 3 counted 58 of its 65 targets above the training maximum of 1.350, so both prediction curves sit
below the actual line there.
The validation targets stay inside the training range.
""")

# ---------------------------------------------------------------- Cell 26 - prediction plot
code(r"""
_, _, y_pred_train_last_state = evaluate(model_last_state, train_eval_loader, device)
_, _, y_pred_val_last_state = evaluate(model_last_state, val_loader, device)
_, _, y_pred_test_last_state = evaluate(model_last_state, test_loader, device)

_, _, y_pred_train_all_states = evaluate(model_all_states, train_eval_loader, device)
_, _, y_pred_val_all_states = evaluate(model_all_states, val_loader, device)
_, _, y_pred_test_all_states = evaluate(model_all_states, test_loader, device)

# the three splits are already in time order, so concatenating them rebuilds the whole series
y_pred_last_state = np.concatenate(
    [y_pred_train_last_state, y_pred_val_last_state, y_pred_test_last_state])
y_pred_all_states = np.concatenate(
    [y_pred_train_all_states, y_pred_val_all_states, y_pred_test_all_states])

close_pred_last_state = y_pred_last_state[:, 0] * close_train_std + close_train_mean
close_pred_all_states = y_pred_all_states[:, 0] * close_train_std + close_train_mean
close_actual = y_seq[:, 0]      # y_seq was never standardized, so it is still in index points

fig, ax = plt.subplots(1, 1, figsize=(14, 5))

ax.plot(close_actual, 'k--', linewidth=1.5, label='actual')
ax.plot(close_pred_last_state, 'b-', linewidth=1, label='A (last state)')
ax.plot(close_pred_all_states, 'r-', linewidth=1, label='B (all states)')
ax.axvline(n_train, color='gray', linestyle=':', label='train / validation')
ax.axvline(n_train + n_val, color='green', linestyle=':', label='validation / test')

ax.set_xlabel('Window index'); ax.set_ylabel('KOSPI Close')
ax.set_title('Prediction vs actual (index points)')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- Cell 27 - rmse print
code(r"""
# RMSE over each split as a whole
train_rmse_last_state = math.sqrt(((y_pred_train_last_state - y_train) ** 2).mean())
val_rmse_last_state = math.sqrt(((y_pred_val_last_state - y_val) ** 2).mean())
test_rmse_last_state = math.sqrt(((y_pred_test_last_state - y_test) ** 2).mean())

train_rmse_all_states = math.sqrt(((y_pred_train_all_states - y_train) ** 2).mean())
val_rmse_all_states = math.sqrt(((y_pred_val_all_states - y_val) ** 2).mean())
test_rmse_all_states = math.sqrt(((y_pred_test_all_states - y_test) ** 2).mean())

print('RMSE on standardized targets, with the same error in index points in brackets')
print(f'Model A (last state)  '
      f'train {train_rmse_last_state:.4f} [{train_rmse_last_state * close_train_std:6.1f}]  '
      f'val {val_rmse_last_state:.4f} [{val_rmse_last_state * close_train_std:6.1f}]  '
      f'test {test_rmse_last_state:.4f} [{test_rmse_last_state * close_train_std:6.1f}]')
print(f'Model B (all states)  '
      f'train {train_rmse_all_states:.4f} [{train_rmse_all_states * close_train_std:6.1f}]  '
      f'val {val_rmse_all_states:.4f} [{val_rmse_all_states * close_train_std:6.1f}]  '
      f'test {test_rmse_all_states:.4f} [{test_rmse_all_states * close_train_std:6.1f}]')
""")

# ---------------------------------------------------------------- Cell 28 - reading the numbers
md(r"""
The validation column is where the two readouts can be compared, because it is the only held-out stretch
that stays inside the range the models were fitted on.
The test column measures something else: how far the forecast drifts once the series leaves that range.
""")

# ---------------------------------------------------------------- Cell 29 - summary
md(r"""
---
## Summary

- A recurrent layer keeps one hidden state and reuses the same two weight matrices at every step,
  so window length leaves its parameter count unchanged — only the readout of B grows with it.
- The two models shared the recurrent layer, the loaders and every hyperparameter.
  What separates their curves is the readout input.
- Errors are reported in index points as well as standardized units, because a standardized RMSE says
  nothing about the size of the miss.
""")

# ---------------------------------------------------------------- Cell 30 - exercises
md(r"""
---
## Exercises

1. Set `num_layers` to 2 in both classes and retrain.
   Both classes read the same variable.
2. Set `sequence_length` to 40 and rerun from Step 2.
   Report what happens to the training loss.
3. Add `Close` to the input columns and retrain.
   Does yesterday's level change the test error?
4. Predict the day-to-day change in `Close` instead of its level.
""")

# ---------------------------------------------------------------- Cell 31 - blank
code(r"""
# Write your code here
""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}

with open('Practice13_RNN_for_Time_Series.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'{len(cells)} cells written')
