# Code Patterns — 01_AI-ME_Graduate

이 과목의 노트북에서 반복 사용하는 코드 패턴. `notebook-profile.json` 의 `code_patterns` 가 이 파일을 가리킨다.

규약의 **이유와 서술**은 `CLAUDE.md` 에 있다. 이 문서는 그 규약을 실제 코드로 옮긴 형태이며,
둘이 어긋나면 `CLAUDE.md` 가 우선한다.

대상 독자는 **기계공학 대학원생**이다. 아래 패턴이 더 짧게 쓸 수 있는데도 풀어 쓰는 이유는
Python 이 어려워서가 아니라, **수식의 각 항이 코드의 어느 줄인지 한눈에 보이게** 하기 위해서다.

**영어 강의다.** 아래 코드 블록의 주석·`print` 문자열·차트 텍스트가 전부 영어인 것은 그 때문이며,
노트북에 옮길 때도 그대로 영어로 둔다. 이 문서 자체의 설명문만 한국어다.

## 목차

1. [선형 모델 학습 골격](#1-선형-모델-학습-골격)
2. [손으로 유도한 기울기 ↔ autograd 대조](#2-손으로-유도한-기울기--autograd-대조)
3. [PyTorch 학습 4단계](#3-pytorch-학습-4단계)
4. [같은 문제를 여러 방법으로 풀 때](#4-같은-문제를-여러-방법으로-풀-때)
5. [시각화](#5-시각화)
6. [표 데이터 파이프라인](#6-표-데이터-파이프라인)
7. [train / evaluate 시그니처 — 두 가지 형태](#7-train--evaluate-시그니처)
8. [재현성](#8-재현성)

---

## 1. 선형 모델 학습 골격

강의 PDF 의 4줄 골격을 그대로 쓴다. 이 4줄이 선형회귀·퍼셉트론·로지스틱회귀에서 문자 그대로 동일하다.

```python
y_hat = X @ w          # prediction (classification: sigma(X @ w))
E     = y - y_hat      # residual
dJ    = -X.T @ E       # gradient
w     = w - rho * dJ   # update
```

- 중간변수 `E` 를 반드시 명시한다. `dJ = -X.T @ (y - X @ w)` 처럼 인라인하지 않는다 —
  수식의 $\mathbf{E}$ 가 코드에서 사라지면 대응이 끊긴다
- 손실은 $J = \tfrac{1}{2}\|\mathbf{E}\|_2^2$ 정의를 쓴다: `loss = (E ** 2).sum() / 2`
- augmented 행렬: `X = np.hstack([x**0, x])`
- 행렬곱은 `@`, 역행렬은 `np.linalg.inv`. `np.asmatrix` 와 `*` 행렬곱은 쓰지 않는다

### 정규방정식 (closed form)

```python
X = np.hstack([x**0, x])
w_least_squares = np.linalg.inv(X.T @ X) @ X.T @ y
```

---

## 2. 손으로 유도한 기울기 ↔ autograd 대조

이 과목의 핵심 셀 형태다. **같은 $\mathbf{w}$ 에서** 두 값을 계산해 나란히 `print` 한다.

```python
w_check = np.array([[1.0], [1.0]])

# gradient from the formula
y_hat = X @ w_check
E     = y - y_hat
dJ_formula = -X.T @ E

# gradient from autograd
w_tensor = torch.tensor(w_check, dtype=torch.float32, requires_grad=True)
loss = ((X_tensor @ w_tensor - y_tensor) ** 2).sum() / 2
loss.backward()

print('formula  :', dJ_formula.flatten())
print('autograd :', w_tensor.grad.numpy().flatten())
```

- **`/ 2` 가 두 값을 일치시키는 핵심이다.** 빠뜨리면 autograd 쪽이 정확히 2배가 된다
- `np.allclose` / `assert` 로 검증하지 않는다. 두 줄을 나란히 출력해 **눈으로** 확인한다
- 비교하는 두 계산은 반드시 **같은 `w`** 에서 출발한다

---

## 3. PyTorch 학습 4단계

같은 학습을 네 단계로 보여준다. 뒤로 갈수록 PyTorch 가 대신해 주는 범위가 넓어진다 —
기울기 → 갱신 → 손실 정의 → 파라미터와 순전파.

```python
# Stage 1 - autograd computes the gradient, the update is written by hand
w_autograd = torch.tensor(w_init, dtype=torch.float32, requires_grad=True)
for i in range(n_iter):
    y_hat = X_tensor @ w_autograd
    loss  = ((y_hat - y_tensor) ** 2).sum() / 2
    loss.backward()
    with torch.no_grad():
        w_autograd -= rho * w_autograd.grad
    w_autograd.grad.zero_()

# Stage 2 - the optimizer performs the update as well
w_optimizer = torch.tensor(w_init, dtype=torch.float32, requires_grad=True)
optimizer = torch.optim.SGD([w_optimizer], lr=rho)
for i in range(n_iter):
    y_hat = X_tensor @ w_optimizer
    loss  = ((y_hat - y_tensor) ** 2).sum() / 2
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Stage 3 - the loss definition moves into an nn module
mse_sum = torch.nn.MSELoss(reduction='sum')
loss = mse_sum(y_hat, y_tensor) / 2      # / 2 to match the definition J = 1/2 ||E||^2

# Stage 4 - the parameters and the forward pass move into an nn.Module class
class LinearRegressor(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat

model = LinearRegressor(in_features=1)
optimizer = torch.optim.SGD(model.parameters(), lr=rho)
```

- `backward()` 는 기울기를 **누적**한다. 매 반복 `grad.zero_()` 또는 `optimizer.zero_grad()` 를 부른다
- optimizer 인자는 API 이름 그대로 `lr` 이며, `lr=rho` 로 넘겨 **같은 학습률임을 코드로 드러낸다**
- `nn.MSELoss(reduction='sum')` 는 $\|\mathbf{E}\|_2^2$ 를 준다. 이 과목의 $J$ 정의와 맞추려면 `/ 2` 를 곱한다
- `nn.Linear` 를 쓰면 bias 가 모듈 안에 있으므로 **augmented 열 없는 raw 입력**을 넣는다.
  augmented `X` 를 그대로 넣으면 bias 가 두 번 들어간다
- `nn.Module` 클래스는 `super().__init__()` 을 반드시 호출한다. 빠뜨리면 파라미터가 추적되지 않아
  `model.parameters()` 가 비고, optimizer 가 아무것도 갱신하지 않는다
- `forward()` 를 직접 부르지 않고 `model(x)` 로 호출한다
- 앞 단계와 결과를 대조하려면 `nn.Linear` 의 랜덤 초기화를 덮어써 같은 초기값에서 출발시킨다:
  `with torch.no_grad(): model.linear.weight.fill_(...); model.linear.bias.fill_(...)`.
  float64 로 맞출 때는 `.double()` 을 **먼저** 부르고 값을 채운다 (순서를 바꾸면 float32 로 반올림된다)
- 분류에서는 `forward` 가 **logit 을 그대로 반환**한다. `BCEWithLogitsLoss` / `CrossEntropyLoss` 가
  sigmoid · softmax 를 내장하므로, `forward` 에서 또 씌우면 두 번 적용된다.
  확률이나 레이블이 필요한 시점(학습 후 평가·시각화)에만 밖에서 `torch.sigmoid` / `argmax` 를 쓴다

---

## 4. 같은 문제를 여러 방법으로 풀 때

**공통 조건을 위에서 한 번만 정의하고 모든 방법이 재사용한다.**

```python
w_init = np.random.randn(2, 1)   # every method starts from the same initial weights
rho    = 0.00002
n_iter = 3000
```

방법별 결과는 **방법 이름을 풀어 쓴 변수**에 담는다.

```python
w_least_squares, w_numpy, w_autograd, w_optimizer, w_sklearn
loss_history_numpy, loss_history_autograd, loss_history_optimizer
```

`w_ls2`, `w_t`, `w2`, `lh_np` 같은 축약은 쓰지 않는다.

마지막에 표 대신 `print` 로 나란히 출력한다.

```python
print(f'True values   : w0 = {w0:.6f}, w1 = {w1:.6f}')
print(f'Least Squares : w0 = {w_least_squares[0, 0]:.6f}, w1 = {w_least_squares[1, 0]:.6f}')
print(f'NumPy GD      : w0 = {w_numpy[0, 0]:.6f}, w1 = {w_numpy[1, 0]:.6f}')
```

---

## 5. 시각화

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(x, y, c='r', s=30, alpha=0.7, label='data')
axes[0].plot(x, y_hat, 'b-', linewidth=2, label='prediction')
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
axes[0].set_title('Regression Line'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(loss_history_numpy,     'b-',  linewidth=3, label='NumPy (formula)')
axes[1].plot(loss_history_autograd,  'r--', linewidth=2, label='torch (autograd)')
axes[1].plot(loss_history_optimizer, 'g:',  linewidth=2, label='torch (optimizer)')
axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('J')
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()
```

- **차트 텍스트는 전부 영어.** 한글은 두부 문자로 렌더링된다
- 여러 방법의 곡선을 겹칠 때 선 스타일을 달리한다(`'-'`, `'--'`, `':'`).
  뒤 곡선이 앞 곡선을 가려도 스타일이 다르면 포개진 것이 보이고, **완전히 포개지는 것이 곧
  "같은 결과" 라는 증거**가 된다
- 회귀: 왼쪽 scatter + 회귀선, 오른쪽 loss. 분류: 왼쪽 Loss, 오른쪽 Accuracy
- `gridspec` 금지 — 독립 셀 + `plt.subplots` 로 분리한다

---

## 6. 표 데이터 파이프라인

분할 → 표준화 → (필요하면) Tensor/DataLoader. **표준화 블록은 sklearn 회차든 torch 회차든 같은
형태를 쓴다** — 규약과 이유는 `CLAUDE.md` "데이터 분할과 스케일링" 에 있다.

```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

train_mean = X_train.mean(axis=0)
train_std  = X_train.std(axis=0)
train_std[train_std == 0] = 1.0   # leave zero-variance features unscaled
X_train = (X_train - train_mean) / train_std
X_val   = (X_val   - train_mean) / train_std
X_test  = (X_test  - train_mean) / train_std

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
    batch_size=16, shuffle=True)
```

- 표준화 통계는 **train set 에서만** 계산해 val·test 에 동일 적용한다
- `+ 1e-8` epsilon trick 을 쓰지 않는다. train 에서만 상수인 특징을 `1e-8` 로 나누면
  val/test 입력이 `1e8` 규모로 폭발한다. sklearn `StandardScaler` 와 동일하게 상수 특징은 그대로 둔다
- 통계량 이름은 `train_mean`, `train_std`. `mean`/`std` 는 메서드를 가리고 `mu`/`sigma` 는 약어다

---

## 7. train / evaluate 시그니처

**두 가지 형태가 있고, 회차의 단계에 따라 쓰는 것이 다르다.** 섞어 쓰지 않는다.

### (a) 전배치 형태 — DataLoader 도입 전

전체 데이터를 한 번에 넣어 `n_iter` 번 갱신한다. 모델 정의만 바꿔 끼우면 같은 함수로 회귀도 분류도
학습된다는 것을 보여주는 것이 목적이다.

```python
def train(model, X, y, optimizer, loss_fn, n_iter):
    # run n_iter full-batch updates and return the loss at each iteration
    loss_history = []
    for i in range(n_iter):
        z = model(X)
        loss = loss_fn(z, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_history.append(loss.item())
    return loss_history
```

- `loss_fn` 을 인자로 받는다. 이 단계에서는 회귀(`MSELoss`)와 분류(`BCEWithLogitsLoss`,
  `CrossEntropyLoss`) 가 같은 함수를 쓰기 때문이다
- `evaluate` 를 따로 두지 않는다. 학습 후 `torch.no_grad()` 블록에서 정확도를 직접 계산한다
- 반환은 loss 이력 하나뿐이다

### (b) 미니배치 형태 — DataLoader 도입 후

미니배치 학습을 도입하는 회차부터 아래 두 함수를 정의하고 이후 회차에서 그대로 재사용한다.
`loss_fn` 을 인자로 받지 않는 것이 (a) 와의 차이다 — 이 단계부터는 분류로 고정되기 때문이다.

```python
def evaluate(model, loader, device):
    # model.eval() + torch.no_grad()
    # returns (loss, accuracy, y_pred)

def train(model, train_loader, val_loader, optimizer, epochs, device):
    # epoch loop calls model.train() and then evaluate(val_loader)
    # returns (train_losses, train_accs, val_losses, val_accs)
```

- **두 함수만.** `fit`, `make_sgd`, scheduler wrapper 등을 추가하지 않는다
- `criterion` 은 함수 내부에서 생성한다 (인자로 받지 않음)
- 학습 중 모니터링은 `val_loader` 로만 한다. `test_loader` 가 학습 루프에 들어가면 data leakage 다
- loss 누적은 per-batch mean 을 더해 배치 개수로 나눈다. `loss.item() * len(batch)` 트릭은 쓰지 않는다
- `model.to(device)` → optimizer 생성 순서를 지킨다
- 코드 셀에 docstring 대신 `#` 주석을 쓴다

---

## 8. 재현성

```python
np.random.seed(42)
torch.manual_seed(42)
```

- 노트북 상단에서 한 번 설정한다
- 비교 대상을 생성하기 직전마다 재설정한다. 그러지 않으면 격리하려던 변수 외에 초기값 차이가 섞인다
