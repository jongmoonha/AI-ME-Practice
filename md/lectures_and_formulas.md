# Lectures & Formulas Reference

각 강의(Lec) 챕터별 주요 개념, 수식, notation 정리.

---

## Notation 총괄표

| 기호 | 의미 |
|------|------|
| $\mathbf{x}, \mathbf{X}$ | 입력 벡터 / 설계행렬 (augmented: bias 포함) |
| $\mathbf{w}, \mathbf{W}$ | 가중치 벡터 (이진) / 가중치 행렬 (다중클래스) |
| $\mathbf{y}, \mathbf{Y}$ | 정답 벡터 / one-hot 행렬 |
| $\hat{y}$ | 예측값 |
| $z, \mathbf{z}$ | pre-activation (선형 결합값) |
| $\sigma(z)$ | 활성함수 (step / sigmoid / softmax 등 문맥에 따라 다름) |
| $o_k, \mathbf{O}$ | softmax 출력 (클래스 $k$) / 출력 행렬 |
| $J$ | 손실함수 (Loss / Cost) |
| $\partial J / \partial \mathbf{w}$ | 그래디언트 — 강의 표기. $\nabla J$ 는 쓰지 않음 |
| $\rho$ | 학습률 (learning rate) — $\alpha$ 대신 $\rho$ 로 통일 |
| $N$ | 샘플 수 |
| $d$ | 입력 특징 수 |
| $C$ | 클래스 수 |
| $p$ | 은닉 뉴런 수 |
| $\lambda$ | 규제 강도 (regularization) |
| $\mu, \sigma$ | 평균, 표준편차 (표준화용) |
| $\epsilon$ | 수치 안정 상수 |
| $\gamma, \beta$ | Batch Normalization 학습 파라미터 |
| $\odot$ | 원소별 곱 (Hadamard product) |

---

## 선형 모델 수식/코드 컨벤션

모든 선형 모델에 일관되게 적용되는 수식/코드 표기 규칙.
**단일 출처: `0_ref/강의자료/Lec 03-04_선형회귀와 분류 정리.pdf` (최종 Notation 정리 슬라이드).**

### 1. 행렬 shape

| 대상 | shape | 의미 |
|------|-------|------|
| $\mathbf{X}$ | $(N, d+1)$ | augmented 설계행렬 (0번 열 = 1, 행=샘플, 열=특징) |
| $\mathbf{y}$ | $(N, 1)$ or $(N,)$ | 정답 벡터 (binary/regression) |
| $\mathbf{Y}$ (one-hot) | $(N, C)$ | 다중클래스 one-hot 타겟 |
| $\mathbf{w}$ | $(d+1, 1)$ or $(d+1,)$ | 단일 출력 가중치 벡터 |
| $\mathbf{W}$ | $(d+1, C)$ | **다중클래스 가중치 (행=입력, 열=클래스)** |
| $\mathbf{z}, \hat{\mathbf{y}}$ | $(N, 1)$ or $(N,)$ | pre-activation / 예측값 |
| $\mathbf{Z}, \mathbf{O}$ | $(N, C)$ | 다중클래스 pre-activation / softmax 출력 |

**bias 흡수**: 입력에 $x_0 = 1$ 열을 붙여 $b$ 를 $w_0$ 로 흡수한다.
$\hat{y} = w_1 x_1 + w_2 x_2 + b$ 와 $\hat{y} = w_0 x_0 + w_1 x_1 + w_2 x_2$ 는 같은 모델이며,
후자만 $\hat{\mathbf{y}} = \mathbf{Xw}$ 한 줄로 묶인다.

**1D vs 2D 배열**: 벡터를 2D 열배열 `(N,1)`/`(d+1,1)` 로 두든 1D `(N,)`/`(d+1,)` 로 두든
`X @ w` 와 `-X.T @ E` 의 **값은 동일**하다. 회귀 노트북은 2D, 분류 노트북은 1D 를 쓰고
차이는 shape 출력으로만 보여준다.

**$\mathbf{w}$ 가 행벡터 $(1, d+1)$ 이면** $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}^T$,
$\partial J / \partial \mathbf{w} = -(\mathbf{y}-\mathbf{Xw})^T\mathbf{X}$ 로 전치가 붙는다.
우리 노트북은 **열벡터 $(d+1, 1)$ 기준**으로 통일한다.

> 다중클래스 $\mathbf{W}$는 $(d+1) \times C$ 형태. $(C, d+1)$ 아님.

### 2. 순전파

| 모델 | pre-activation | 예측 |
|------|----------------|------|
| Linear Regression | — | $\hat{\mathbf{y}} = \mathbf{Xw}$ |
| Perceptron | $\mathbf{z} = \mathbf{Xw}$ | $\hat{\mathbf{y}} = \sigma(\mathbf{z})$ &nbsp;($\sigma$ = 계단함수) |
| Logistic (binary) | $\mathbf{z} = \mathbf{Xw}$ | $\hat{\mathbf{y}} = \sigma(\mathbf{z})$ &nbsp;($\sigma$ = sigmoid) |
| Logistic (multi) | $\mathbf{Z} = \mathbf{XW}$ | $\mathbf{O} = \text{softmax}(\mathbf{Z})$ |

- **매트릭스 표기가 기준**: $\mathbf{z} = \mathbf{Xw}$ (binary), $\mathbf{Z} = \mathbf{XW}$ (multi).
  교과서 표기($\mathbf{w}^T\mathbf{x}$, $\mathbf{x}$는 열벡터)는 우리 컨벤션에 맞춰야 함.
  - **예외**: 퍼셉트론 손실 정의 $J(\mathbf{w}) = \sum_{\mathbf{x}_k \in Y} -y_k(\mathbf{w}^T \mathbf{x}_k)$
    만 강의 표기 그대로 둔다 (오분류 집합 $Y$ 를 샘플 단위로 도는 식이라 이 형태가 자연스러움).
- $\mathbf{x}_n = \mathbf{X}[n]$ 는 **행벡터** ($1 \times (d+1)$). 개별 샘플: $z_n = \mathbf{x}_n \mathbf{w}$ (전치 없음).
- 다중클래스 클래스 $k$: $z_k = \mathbf{x}\,\mathbf{w}_k$ where $\mathbf{w}_k = \mathbf{W}[:, k]$
- 활성함수는 **계단함수든 sigmoid든 모두 $\sigma$**. 코드도 `sigma()` 하나로 통일
  (`step()`, `sigmoid()` 같은 별칭 금지).
- 코드: `z = X @ w`, `Z = X @ W` (NOT `X @ W.T`)

### 3. 잔차/오차 신호

$$\mathbf{E} = \mathbf{y} - \hat{\mathbf{y}} \quad \text{또는} \quad \mathbf{E} = \mathbf{Y} - \mathbf{O}$$

통계학 잔차 $r = y - \hat{y}$ 형태. 코드는 반드시 E 중간변수 명시 (인라인 금지):
```python
E = y - y_hat            # binary/regression
E = y_onehot - O         # multi-class
```

### 4. 그래디언트 — 4개 모델 완전 동일

$$\boxed{\dfrac{\partial J}{\partial \mathbf{w}} = -\mathbf{X}^T \mathbf{E}}$$

| 모델 | $\partial J / \partial \mathbf{w}$ |
|------|-----------|
| Linear Regression | $-\mathbf{X}^T \mathbf{E}$ |
| Perceptron | $-\mathbf{X}^T \mathbf{E}$ |
| Logistic (binary) | $-\mathbf{X}^T \mathbf{E}$ |
| Logistic (multi) | $-\mathbf{X}^T \mathbf{E}$ |

**선형회귀 유도**:

$$\frac{\partial J}{\partial \mathbf{w}} = \mathbf{X}^T\mathbf{Xw} - \mathbf{X}^T\mathbf{y} = \mathbf{X}^T(\mathbf{Xw}-\mathbf{y}) = -\mathbf{X}^T(\mathbf{y}-\mathbf{Xw}) = -\mathbf{X}^T(\mathbf{y}-\hat{\mathbf{y}}) = -\mathbf{X}^T\mathbf{E}$$

**선형회귀의 상수 2**: $J = \|\mathbf{y}-\hat{\mathbf{y}}\|_2^2$ 를 그대로 미분하면 $-2\mathbf{X}^T\mathbf{E}$ 다.
강의 정리 슬라이드는 이 **상수 2를 학습률 $\rho$ 에 흡수**시켜 네 모델의 기울기를 한 형태로 통일한다.
우리도 이를 따르며 수식·코드 어디에도 2를 쓰지 않는다.
(2를 붙이든 안 붙이든 $\rho$ 가 두 배 차이일 뿐 수렴 결과는 같다 — 이 한 줄은 선형회귀 노트북
마크다운에 남겨 학생이 의아해하지 않게 한다.)

> **autograd 로 손실을 직접 미분하는 노트북**(코드의 기울기를 `loss.backward()` 결과와 눈으로
> 대조하는 경우)에서는 상수 2를 $\rho$ 에 흡수시킬 수 없다 — autograd 는 코드에 적힌 $J$ 를
> 그대로 미분하기 때문이다. 이럴 때는 $J = \tfrac{1}{2}\|\mathbf{E}\|_2^2$ 로 정의한다.
> 그러면 미분값이 $-\mathbf{X}^T\mathbf{E}$ 가 되어 상수 2 없이도 손으로 유도한 기울기와
> autograd 결과가 정확히 일치한다. (`nn.MSELoss(reduction='sum')` 를 쓰면 $\|\mathbf{E}\|_2^2$
> 이므로 이 경우 $\tfrac{1}{2}$ 를 곱해 맞춘다.)

**개별 샘플 기여 (벡터 형태):**
- Binary/regression: $\dfrac{\partial \ell^{(n)}}{\partial \mathbf{w}} = -E_n\,\mathbf{x}_n^T$ &nbsp; (shape: $(d+1) \times 1$)
- Multi-class (outer product): $\dfrac{\partial \ell^{(n)}}{\partial \mathbf{W}} = -\mathbf{x}_n^T\,\mathbf{E}_{n,:}$ &nbsp; (shape: $(d+1) \times C$)

**배치 합산**: $\sum_n -E_n\,\mathbf{x}_n^T = -\mathbf{X}^T \mathbf{E}$ (binary) / $\sum_n -\mathbf{x}_n^T\,\mathbf{E}_{n,:} = -\mathbf{X}^T \mathbf{E}$ (multi)

### 5. 가중치 갱신

$$\mathbf{w} \leftarrow \mathbf{w} - \rho\,\frac{\partial J}{\partial \mathbf{w}} = \mathbf{w} + \rho\,\mathbf{X}^T \mathbf{E}$$

- **학습률 기호는 $\rho$, 코드 변수는 `rho`** ($\alpha$ / `alpha` 사용 금지).
- 네 모델 모두 동일. 선형회귀도 예외 없음.

### 6. 변수명 규칙 (코드)

| 역할 | 변수명 |
|------|--------|
| 활성함수 | `sigma` (계단함수·sigmoid 공통), `softmax` |
| 연속 예측 | `y_hat` (= `X @ w` 또는 `sigma(X @ w)`), `O` (softmax 행렬) |
| 클래스 예측 | `y_pred` (= `(y_hat >= 0.5).astype(int)` 또는 `np.argmax(O, axis=1)`) |
| One-hot 타겟 | `y_onehot` (NOT `Y_oh`) |
| 잔차/오차 | `E = y - y_hat` 또는 `E = y_onehot - O` |
| 그래디언트 | `dJ` (벡터), `dW` (행렬) |
| 학습률 | `rho` (수식 $\rho$) — 단, PyTorch 사용 시 `lr` (아래 참조) |
| 가중치 | `w` / `W` (`theta` 금지) |

**학습률 표기 — 두 층위로 나눈다:**

| 구간 | 표기 | 예 |
|------|------|-----|
| 손으로 유도한 갱신식을 직접 구현 (numpy, autograd 수동 갱신 포함) | 수식 $\rho$ / 코드 `rho` | `w = w - rho * dJ`, `w_t -= rho * w_t.grad` |
| PyTorch optimizer 사용 | `lr` | `torch.optim.SGD(model.parameters(), lr=0.001)` |

- `lr` 은 PyTorch API 인자명이므로 그대로 쓴다. `learning_rate` 라는 변수명은 쓰지 않는다
  (예외: Keras 는 API 자체가 `learning_rate=` 라 불가피).
- 두 구간을 한 노트북에서 비교할 때는 `lr=rho` 로 넘겨 **같은 학습률임을 코드로 드러낸다.**

- 행렬곱은 **항상 `@`**. `np.matrix` / `np.asmatrix` / `*` 를 이용한 행렬곱은 쓰지 않는다
  (`*` 가 원소곱인지 행렬곱인지 학생이 헷갈림). 역행렬은 `np.linalg.inv`.
- augmented 행렬: `X = np.hstack([np.ones((N, 1)), x])` 또는 `np.column_stack([np.ones(N), x_raw])`

### 7. 4개 모델 공통 학습 코드 골격

```python
y_hat = sigma(X @ w)      # 선형회귀는 sigma 없이 X @ w
E     = y - y_hat
dJ    = -X.T @ E
w     = w - rho * dJ
```

### 8. 수식 표기 패턴 — 개별 샘플 ↔ 매트릭스

노트북 "사용할 수식" 섹션 및 Step 5/6/7 등에서 각 공식을 **두 줄로 라벨 명시** (혼동 방지):

```
**모델:**
- 개별 샘플: $\hat{y}_n = \sigma(\mathbf{x}_n \mathbf{w})$
- 매트릭스: $\hat{\mathbf{y}} = \sigma(\mathbf{Xw})$

**잔차:**
- 개별: $E_n = y_n - \hat{y}_n$
- 매트릭스: $\mathbf{E} = \mathbf{y} - \hat{\mathbf{y}}$

**기울기:**
- 개별 샘플 기여: $\partial \ell^{(n)} / \partial \mathbf{w} = -E_n\,\mathbf{x}_n^T$
- 매트릭스: $\partial J / \partial \mathbf{w} = -\mathbf{X}^T \mathbf{E}$
```

**규칙:**
- 그래디언트 기호는 **$\partial J / \partial \mathbf{w}$ 로 통일** (강의 표기). $\nabla J$ 는 쓰지 않는다.
- 스칼라 표기엔 반드시 샘플 인덱스 $n$ 명시 (`z_n`, `y_n`, `E_n`). 인덱스 없는 스칼라 금지.
- 매트릭스 표기엔 인덱스 없음 (`z = Xw`, `∂J/∂w = -X^T E`).
- 합산 $\sum_n$ 은 유도 과정에서만, 최종 결과는 매트릭스로 귀결.
- Binary/Regression: 소문자 bold (`y`, `ŷ`, `E`). Multi-class: 대문자 bold (`Y`, `O`, `E`).

### 9. 샘플 index vs 특징 index

강의 자료는 **특징이 여러 개일 때만 특징 첨자를 붙이는** 규칙을 따른다.
입력 특징이 1개인 구간에서는 특징 첨자가 필요 없어 남은 첨자 하나를 샘플에 쓰고,
특징이 2개 이상으로 늘어나는 순간 **아래첨자=특징 / 위첨자=샘플** 로 확장된다.
이는 표기 변경이 아니라 축약의 해제다.

| 대상 | 표기 | 비고 |
|------|------|------|
| $n$번째 샘플 행벡터 | $\mathbf{x}_n = \mathbf{X}[n]$ | **볼드 $\mathbf{x}$ 의 아래첨자는 항상 샘플** |
| 특징이 1개일 때의 스칼라 | $x_n$ | 특징 첨자 생략 (샘플만 남음) |
| 특징이 2개 이상일 때의 스칼라 | $x_i^{(n)}$ | 아래=특징 $i$, 위=샘플 $n$ |
| $\mathbf{X}$ 의 $j$번째 열 | $\mathbf{X}_{:,j}$ | 볼드 소문자 $\mathbf{x}_j$ 로 쓰지 않는다 (샘플과 충돌) |
| 샘플 단위 스칼라 (x 외) | $y_n,\ \hat{y}_n,\ z_n,\ E_n$ | 특징 첨자가 없으므로 항상 아래첨자 |
| 다중클래스 | $y_k^{(n)},\ o_k^{(n)}$ | 아래=클래스 $k$, 위=샘플 $n$ |

**금지**: $x_{ni}$ / $x_{ki}$ 같은 혼합 아래첨자 (어느 쪽이 샘플인지 학생이 추측해야 함).
반드시 $x_i^{(n)}$ 형태로 쓴다.

> 문헌마다 갈리는 지점이다 — Bishop(PRML)·ESL·Murphy·오일석은 "샘플=아래첨자",
> Goodfellow(DL)·Ng(CS229)·Raschka는 "샘플=위첨자 $(n)$". 우리는 강의 자료를 따라
> **볼드 벡터는 아래첨자=샘플, 스칼라는 특징 첨자가 붙는 순간 위첨자=샘플** 로 절충한다.

### 10. 강의 컨벤션을 따르는 이유
- 통계/수학 잔차 표기 $r = y - \hat{y}$ 와 일치
- $\partial J / \partial \mathbf{w} = -\mathbf{X}^T \mathbf{E}$ 로 **4개 모델이 문자 그대로 동일**
- DL 코드 컨벤션 $\delta = o - y$ 와 부호만 다르고 수치 동일

---

## Lec 02 — Python 기초

수식 없음. Python 문법, NumPy, Matplotlib, 데이터 로딩 기초.

**핵심 개념:**
- 변수, 자료형, 자료구조 (list, tuple, dict)
- 제어문 (if/for/while), 함수, 클래스/OOP
- NumPy 배열 연산, reshape, 인덱싱, 행렬 곱 (`@`)
- Matplotlib 시각화 (line, scatter, subplot, imshow)
- scikit-learn / UCI 데이터셋 로딩

---

## Lec 03 — Linear Regression & Perceptron

### 3-1. Linear Regression

**모델:**

$$\hat{y}_n = w_0 + w_1 x_n, \qquad \hat{\mathbf{y}} = \mathbf{Xw}$$

$$\mathbf{X} = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_N \end{bmatrix}, \quad \mathbf{w} = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix}$$

**손실함수 (MSE):**

$$J(\mathbf{w}) = \sum_{n=1}^{N}(\hat{y}_n - y_n)^2 = \|\hat{\mathbf{y}} - \mathbf{y}\|_2^2 = \|\mathbf{Xw} - \mathbf{y}\|_2^2$$

**풀이 1 — 정규방정식 (Least Squares):**

$$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

**풀이 2 — 경사하강법 (Gradient Descent):**

$$\frac{\partial J}{\partial \mathbf{w}} = \mathbf{X}^T\mathbf{Xw} - \mathbf{X}^T\mathbf{y} = \mathbf{X}^T(\mathbf{Xw} - \mathbf{y}) = -\mathbf{X}^T(\mathbf{y} - \hat{\mathbf{y}})$$

$$\mathbf{w} \leftarrow \mathbf{w} - \rho \frac{\partial J}{\partial \mathbf{w}} = \mathbf{w} + \rho\,\mathbf{X}^T(\mathbf{y} - \hat{\mathbf{y}})$$

> 엄밀한 미분값은 $-2\mathbf{X}^T(\mathbf{y}-\hat{\mathbf{y}})$ 이나, 상수 2는 $\rho$ 에 흡수시켜
> 퍼셉트론·로지스틱과 같은 형태로 통일한다.

---

### 3-2. Perceptron

**활성함수 (Unit Step):**

$$\sigma(z) = \begin{cases} 1 & z \geq 0 \\ 0 & z < 0 \end{cases}$$

**순전파:**

$$\mathbf{z} = \mathbf{Xw}, \qquad \hat{\mathbf{y}} = \sigma(\mathbf{z})$$

**손실함수** (오분류 집합 $Y$ 에 대해서만 합산):

$$J(\mathbf{w}) = \sum_{\mathbf{x}_k \in Y} -y_k\,(\mathbf{w}^T \mathbf{x}_k)$$

**그래디언트:**

$$\frac{\partial J}{\partial \mathbf{w}} = -\mathbf{X}^T(\mathbf{y} - \sigma(\mathbf{z})) = -\mathbf{X}^T(\mathbf{y} - \hat{\mathbf{y}})$$

**가중치 갱신:**

$$\mathbf{w} \leftarrow \mathbf{w} + \rho\,\mathbf{X}^T(\mathbf{y} - \hat{\mathbf{y}})$$

**결정경계 (2D):**

$$w_0 + w_1 x_1 + w_2 x_2 = 0 \;\Rightarrow\; x_2 = -\frac{w_0 + w_1 x_1}{w_2}$$

---

## Lec 04 — Logistic Regression

### 4-1. Binary Logistic Regression

**활성함수 (Sigmoid):**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**순전파:**

$$\mathbf{z} = \mathbf{Xw}, \qquad \hat{\mathbf{y}} = \sigma(\mathbf{z}), \qquad P(y=1 \mid \mathbf{x}_n) = \sigma(z_n)$$

$$\hat{y}_{c,n} = \begin{cases} 1 & \sigma(z_n) > 0.5 \\ 0 & \text{else} \end{cases}$$

**손실함수 (Binary Cross-Entropy):**

$$J = -\sum_{n=1}^{N} \bigl[ y_n \ln \sigma(z_n) + (1-y_n) \ln(1-\sigma(z_n)) \bigr]$$

**그래디언트:**

$$\frac{\partial J}{\partial w_i} = -\sum_{n=1}^{N} (y_n - \hat{y}_n)\,x_i^{(n)}, \quad \hat{y}_n = \sigma(z_n)$$

$$\frac{\partial J}{\partial \mathbf{w}} = -\mathbf{X}^T(\mathbf{y} - \sigma(\mathbf{z})) = -\mathbf{X}^T(\mathbf{y} - \hat{\mathbf{y}})$$

**가중치 갱신:**

$$\mathbf{w} \leftarrow \mathbf{w} - \rho\,\frac{\partial J}{\partial \mathbf{w}} = \mathbf{w} + \rho\,\mathbf{X}^T(\mathbf{y} - \hat{\mathbf{y}})$$

---

### 4-2. Multi-class Logistic Regression (Softmax)

**활성함수 (Softmax):**

$$o_k = \frac{e^{z_k}}{\sum_{q=1}^{C} e^{z_q}}$$

**순전파** ($\mathbf{W}$ shape $(d+1, C)$, $\mathbf{Z} = \mathbf{X}\mathbf{W}$):

$$z_k = w_{k0} + w_{k1}x_1 + w_{k2}x_2, \qquad \hat{y} = \arg\max_k\, o_k$$

**손실함수 (Cross-Entropy):**

$$\ell^{(n)} = -\sum_{k=1}^{C} y_k^{(n)} \log o_k^{(n)} = -\log o_{y^{(n)}}^{(n)}$$

$$J = \sum_{n=1}^{N} \ell^{(n)}$$

**그래디언트:**

$$\frac{\partial J}{\partial w_{ki}} = -\sum_{n=1}^{N} (y_k^{(n)} - o_k^{(n)})\,x_i^{(n)}$$

$$\frac{\partial J}{\partial \mathbf{W}} = -\mathbf{X}^T (\mathbf{Y} - \mathbf{O})$$

**가중치 갱신:**

$$\mathbf{W} \leftarrow \mathbf{W} - \rho\,\frac{\partial J}{\partial \mathbf{W}} = \mathbf{W} + \rho\,\mathbf{X}^T (\mathbf{Y} - \mathbf{O})$$

**결정경계 (class $a$ vs $b$):**

$$z_a = z_b \;\Rightarrow\; x_2 = -\frac{(w_{a0}-w_{b0}) + (w_{a1}-w_{b1})x_1}{w_{a2}-w_{b2}}$$

---

### Lec 03-04 비교 요약

| 모델 | 활성함수 | $\hat{\mathbf{y}}=$ | 손실함수 | 출력 공간 | 그래디언트 $\partial J/\partial \mathbf{w}$ |
|------|---------|---------------------|---------|----------|--------------|
| Linear Regression | identity | $\mathbf{Xw}$ | MSE $\|\hat{\mathbf{y}}-\mathbf{y}\|_2^2$ | $\mathbb{R}$ | $-\mathbf{X}^T(\mathbf{y}-\hat{\mathbf{y}})$ |
| Perceptron | 계단함수 $\sigma$ | $\sigma(\mathbf{Xw})$ | $\sum_{\mathbf{x}_k \in Y} -y_k(\mathbf{w}^T\mathbf{x}_k)$ | $\{0,1\}$ | $-\mathbf{X}^T(\mathbf{y}-\hat{\mathbf{y}})$ |
| Logistic (Binary) | sigmoid $\sigma$ | $\sigma(\mathbf{Xw})$ | BCE | $(0,1)$ | $-\mathbf{X}^T(\mathbf{y}-\hat{\mathbf{y}})$ |
| Logistic (Multi) | softmax | $\text{softmax}(\mathbf{XW})=\mathbf{O}$ | CE | $(0,1)^C$ | $-\mathbf{X}^T(\mathbf{Y}-\mathbf{O})$ |

→ 공통 형태: $\dfrac{\partial J}{\partial \mathbf{w}} = -\mathbf{X}^T(\text{target}-\text{pred}) = -\mathbf{X}^T\mathbf{E}$ &nbsp;— 네 모델이 **문자 그대로 동일**

---

## Lec 05 — PyTorch 기초

### 5-1. Autograd & Gradient Descent

**Autograd 예시:**

$$y = \theta^2 + 2\theta + 1 \;\Rightarrow\; \frac{dy}{d\theta} = 2\theta + 2$$

**Chain Rule:**

$$\frac{df}{d\theta} = \frac{d\hat{y}}{d\theta} \cdot \frac{df}{d\hat{y}}$$

**PyTorch로 Linear Regression:**

$$\hat{\mathbf{y}} = \mathbf{Xw}, \quad J = \|\mathbf{Xw}-\mathbf{y}\|_2^2, \quad \mathbf{w} \leftarrow \mathbf{w} - \rho\,\frac{\partial J}{\partial \mathbf{w}}$$

3가지 구현 비교: (1) NumPy 수동, (2) PyTorch 수동 업데이트, (3) `torch.optim` 활용

---

### 5-2. Tensor Manipulation

**핵심 연산:**

| 연산 | 설명 |
|------|------|
| 원소별 곱 | $a_{ij} \times b_{ij}$ |
| 행렬 곱 | $\sum_k a_{ik} b_{kj}$ (`@` 연산자) |
| `view(-1, n)` | reshape (자동 차원 계산) |
| `squeeze()` / `unsqueeze(dim)` | 차원 제거 / 추가 |
| `cat(dim)` / `stack(dim)` | 기존 축 결합 / 새 축 생성 |
| `scatter_()` | one-hot 인코딩 |

---

### 5-3. PyTorch Linear Models

Lec 03-04 수식을 `torch.nn`으로 구현:

| 모델 | PyTorch Loss |
|------|-------------|
| Linear Regression | `nn.MSELoss()` |
| Binary Classification | `nn.BCEWithLogitsLoss()` (sigmoid 내장) |
| Multi-class Classification | `nn.CrossEntropyLoss()` (softmax+NLL 내장) |

---

## Lec 06 — Multilayer Perceptron (MLP)

### 6-1. Backpropagation

**활성함수 (Sigmoid):**

$$\tau(x) = \sigma(x) = \frac{1}{1+e^{-x}}, \qquad \tau'(x) = \sigma(x)(1-\sigma(x))$$

**출력층 에러 신호:**

$$\boldsymbol{\delta} = (\mathbf{y} - \mathbf{o}) \odot \tau'(\mathbf{osum}), \quad \text{shape: } (c, 1)$$

**출력층 가중치 그래디언트:**

$$\Delta \mathbf{U}^2 = -\boldsymbol{\delta}\,\mathbf{z}^T, \quad \text{shape: } (c, p+1)$$

**은닉층 에러 신호:**

$$\boldsymbol{\eta} = \tau'(\mathbf{zsum}) \odot (\boldsymbol{\delta}^T \tilde{\mathbf{U}}^2)^T, \quad \text{shape: } (p, 1)$$

**은닉층 가중치 그래디언트:**

$$\Delta \mathbf{U}^1 = -\boldsymbol{\eta}\,\mathbf{x}^T, \quad \text{shape: } (p, d+1)$$

**가중치 갱신:**

$$\mathbf{U} \leftarrow \mathbf{U} - \rho\,\Delta\mathbf{U}$$

**Notation (Lec 06 고유):**

| 기호 | 의미 |
|------|------|
| $\mathbf{U}^1$ | 은닉층 가중치 $(p \times (d+1))$ |
| $\mathbf{U}^2$ | 출력층 가중치 $(c \times (p+1))$ |
| $\tau, \tau'$ | 활성함수 및 미분 |
| $\boldsymbol{\delta}$ | 출력층 에러 신호 |
| $\boldsymbol{\eta}$ | 은닉층 에러 신호 |
| $\mathbf{osum}$ | 출력층 pre-activation |
| $\mathbf{zsum}$ | 은닉층 pre-activation |

### 6-2. Loss 함수 선택

| 출력 | 활성함수 | PyTorch Loss |
|------|---------|-------------|
| 이진 분류 | sigmoid | `BCEWithLogitsLoss` |
| 다중 분류 | softmax (내장) | `CrossEntropyLoss` |

---

## Lec 07 — Training Pipeline, Optimization, Regularization

### 7-1. Training Pipeline

**데이터 파이프라인:**

```
train_test_split(stratify=y)
  → 표준화 (train 기준 μ, σ)
  → FloatTensor(X), LongTensor(y)
  → TensorDataset → DataLoader
```

**표준화:**

$$x_i^{\text{new}} = \frac{x_i - \mu_i}{\sigma_i} \qquad (\mu, \sigma \text{는 train set에서만 계산})$$

**Epoch / Iteration / Batch:**

| 용어 | 정의 |
|------|------|
| Batch Size | 한 번에 처리하는 샘플 수 |
| Iteration | 파라미터 업데이트 1회 (= 미니배치 1개) |
| Epoch | 전체 훈련 데이터 1회 순회 |

**훈련 루프 핵심:**
- `model.train()` → 미니배치 순회 → loss.backward() → optimizer.step()
- `model.eval()` → `torch.no_grad()` → 검증 loss/accuracy 계산

---

### 7-2. Optimization

**SGD:**

$$\theta \leftarrow \theta - \rho\,\frac{\partial J}{\partial \theta}$$

**Adam:** 1차/2차 모멘트 적응형 학습률

**가중치 초기화:**

| 방법 | 분포 | 적합한 활성함수 |
|------|------|--------------|
| He (Kaiming) | $w \sim \mathcal{N}(0,\, 2/n_{\text{in}})$ | ReLU |
| Xavier | $w \sim \mathcal{N}(0,\, 1/n_{\text{in}})$ | Sigmoid, Tanh |

**Batch Normalization:**

$$\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad z'_i = \gamma\,\hat{z}_i + \beta$$

**활성함수 비교:**

| 함수 | 수식 | 특징 |
|------|------|------|
| Sigmoid | $\frac{1}{1+e^{-z}}$ | gradient vanishing 문제 |
| Tanh | $\tanh(z)$ | 0 중심, 여전히 vanishing |
| ReLU | $\max(0, z)$ | gradient vanishing 해결 |
| Leaky ReLU | $z$ if $z>0$, $0.01z$ otherwise | dead neuron 방지 |

---

### 7-3. Regularization

**L2 규제 (Weight Decay):**

$$J_{\text{reg}} = J(\theta) + \lambda\|\theta\|_2^2$$

**L1 규제 (Lasso):**

$$J_{\text{reg}} = J(\theta) + \lambda\|\theta\|_1$$

**Dropout:**
- Train: 확률 $p$로 뉴런 비활성화, 활성 뉴런을 $\frac{1}{1-p}$로 스케일
- Eval: 모든 뉴런 활성, 스케일 없음

**하이퍼파라미터 탐색:** Grid Search로 $\lambda$, $p$ 등 탐색

---

## Lec 08 — Convolutional Neural Network (CNN) 기초

### 8-1. MLP의 한계 (이미지 입력)

| 문제 | 설명 |
|------|------|
| 공간 정보 소실 | 28×28 → 784 평탄화 시 인접 픽셀 관계 사라짐 |
| 위치 변화에 약함 | 1픽셀 시프트만으로 입력 벡터가 완전히 달라짐 |
| 파라미터 폭증 | $784 \times 128$만 해도 100k+ 파라미터 |

→ **해결책**: 입력의 2D 구조를 보존하면서 가중치를 공유하는 합성곱 연산.

---

### 8-2. Convolution 연산

**2D 합성곱 (단일 채널):**

$$y[i,j] = \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} x[i+m,\,j+n]\,\cdot\,w[m,n] + b$$

- $x$: 입력 feature map, $w$: kernel (filter), $K$: kernel size
- 같은 kernel이 모든 위치에 슬라이딩 → **가중치 공유 (weight sharing)**
- 학습 파라미터: $K \times K + 1$ (단일 채널 단일 커널)

**다채널 입력 → 다채널 출력:**

$$y_{c_o}[i,j] = \sum_{c_i=1}^{C_{in}} \sum_{m,n} x_{c_i}[i+m,\,j+n]\,\cdot\,w_{c_o, c_i}[m,n] + b_{c_o}$$

- 입력 채널 $C_{in}$, 출력 채널 $C_{out}$
- 학습 파라미터: $C_{out} \times C_{in} \times K \times K + C_{out}$

---

### 8-3. 출력 shape 공식

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2P - K}{S} \right\rfloor + 1$$

| 기호 | 의미 |
|------|------|
| $K$ | kernel size |
| $P$ | padding (입력 가장자리에 0 추가) |
| $S$ | stride (커널 이동 폭) |

**자주 쓰는 조합:**
- $K=3, P=1, S=1$ → 출력 크기 **유지** (28→28)
- $K=3, P=0, S=1$ → 2씩 줄어듦 (28→26)
- $K=2, P=0, S=2$ (MaxPool) → **절반** (28→14)

---

### 8-4. Pooling

| 종류 | 동작 |
|------|------|
| MaxPool | 영역 내 **최댓값** (가장 강한 특징) |
| AvgPool | 영역 내 **평균** |

**효과:**
- 다운샘플링 (해상도 절반) → 연산량 감소
- 작은 위치 변화에 강건 (translation invariance)
- 학습 파라미터 **없음**

---

### 8-5. 기본 CNN 구조

```
Input → [Conv → ReLU → Pool] × N → Flatten → FC → Output
```

**PyTorch API:**

| 모듈 | 인자 |
|------|------|
| `nn.Conv2d` | `(in_channels, out_channels, kernel_size, stride, padding)` |
| `nn.MaxPool2d` | `(kernel_size, stride)` |
| `nn.AvgPool2d` | `(kernel_size, stride)` |

**텐서 shape (PyTorch 관례):** $(N, C, H, W)$
- $N$: 배치, $C$: 채널, $H$: 높이, $W$: 너비

---

## Lec 09 — Advanced CNN

### 9-1. BatchNorm2d

P13a 의 BatchNorm (선택 심화 노트북)을 **채널별**로 적용 (4D 텐서 $(N,C,H,W)$):

$$\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}, \quad y_{n,c,h,w} = \gamma_c\,\hat{x}_{n,c,h,w} + \beta_c$$

- 통계 $\mu_c, \sigma_c$ 는 미니배치 내 같은 채널 $c$의 모든 $(n,h,w)$에서 계산
- $\gamma_c, \beta_c$: 채널별 학습 파라미터

---

### 9-2. Conv-BN-ReLU 블록

모던 CNN의 표준 빌딩 블록:

```
Conv2d → BatchNorm2d → ReLU
```

- BatchNorm이 학습 안정화 + 약한 규제 효과
- bias=False로 둠 (BN의 $\beta$가 흡수)

---

### 9-3. 데이터 증강 (Data Augmentation)

학습 시 매 epoch마다 입력 이미지에 **랜덤 변환** 적용 → 사실상 데이터 양 증가.

| 변환 | 설명 |
|------|------|
| `RandomCrop` | 랜덤 위치에서 잘라내기 |
| `RandomHorizontalFlip` | 좌우 반전 (확률 $p$) |
| `RandomRotation` | 랜덤 각도 회전 |
| `ColorJitter` | 밝기/대비/채도 변화 |

**주의:** 학습 데이터에만 적용. 테스트 데이터에는 적용 X (재현성).

---

### 9-4. 전이학습 (Transfer Learning)

큰 데이터셋(ImageNet)에서 **사전 학습한 모델**을 가져와 새 문제에 적용.

| 전략 | 동작 | 적합 상황 |
|------|------|----------|
| Feature Extraction | backbone 동결, 마지막 FC만 학습 | 데이터 적음, 도메인 유사 |
| Fine-tuning | 일부/전체 다시 학습 (작은 학습률) | 데이터 충분, 도메인 다름 |

**PyTorch:**
```python
model = torchvision.models.resnet18(weights='DEFAULT')
for p in model.parameters(): p.requires_grad = False  # 동결
model.fc = nn.Linear(512, num_new_classes)            # FC 교체
```

---

### 9-5. 대표 CNN 아키텍처

| 모델 | 깊이 | 핵심 특징 |
|------|------|----------|
| LeNet-5 | 5 | 최초의 CNN (1998, 손글씨) |
| AlexNet | 8 | ReLU + Dropout + GPU (2012, ImageNet 우승) |
| VGG | 13~19 | 3x3 conv 누적, 단순한 구조 |
| ResNet | 18~152 | Skip connection: $y = F(x) + x$ → 매우 깊은 학습 가능 |
| Inception | — | 다양한 kernel 크기 병렬 |

---

### 9-6. Receptive Field (수용 영역)

출력 픽셀 1개에 영향을 미치는 **입력 영역의 크기**.

- 깊어질수록 receptive field 넓어짐 → 더 큰 패턴 인식
- 3x3 conv 두 번 = 5x5 conv 한 번 (receptive field 같음, 파라미터는 적음)
