# Spec — Practice13_RNN_for_Time_Series.ipynb

> **PDF 원문 대조 미수행 — 이 과목에 RNN 강의 슬라이드가 없다.**
> `lecture_notes/` 에는 `Ch1-ML 1_Linear Regression.pdf` 한 편뿐이고,
> `md/lectures_and_formulas.md` · `md/practice_outline_ref.md` 에도 RNN/recurrent/sequence 항목이 없다
> (`grep -i "RNN|Recurrent|LSTM|GRU|sequence"` → 0 hit).
> 따라서 **내용 소스는 구버전 저장소의 노트북 하나뿐**이며,
> 아래 모든 수식·기호는 이 과목의 `CLAUDE.md` "Notation" 을 기준으로 **새로 표기한 것**이다.
> 소스 노트북에는 수식이 한 줄도 없다 (§6-1 참조).

---

## 0. 메타

| 항목 | 값 |
|------|-----|
| 대상 과목 | `D:\Main\00_Research\00_Python\00_Lecture\01_AI-ME_Graduate` |
| 대상 노트북 | `Practice13_RNN_for_Time_Series.ipynb` |
| generator | `gen/p13.py` |
| 작업 유형 | **신규** |
| 대응 강의 | **없음** — 이 과목 `lecture_notes/` 에 RNN 슬라이드 부재 |
| 내용 소스 | `D:\Main\00_Research\00_Python\00_Lecture\_archive\01_AI-ME_Graduate_backup\Chapter3_Deep Learning_3_RNN.ipynb` (28 셀, 한국어) |
| 데이터셋 | KOSPI 일별 시세 `kospi.csv` — 431 행 × 7 열, 2019-01-30 ~ 2020-10-30 |
| 라이브러리 | `numpy`, `pandas`, `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`, `matplotlib` |

### 파일명 결정 근거

소스 제목은 "Vanila RNN을 이용한 Kospi 예측" 이다.
회차 번호 13 은 `Practice12_Fine_Tuning_on_a_New_Class.ipynb` 다음 번호이며,
`{Name}` 은 형제 노트북(`Practice09_CNN_Pipeline`, `Practice10_CNN_Visualization`)처럼
**기법 + 적용 대상**을 붙인 `RNN_for_Time_Series` 로 한다.
대안 `Practice13_Recurrent_Neural_Networks.ipynb` 는 노트북 후반이 전부 시계열 회귀이므로 내용을 덜 드러낸다.
노트북 첫 셀 제목: `# Practice 13 — RNN for Time Series` (CLAUDE.md "노트북 제목은 자기 파일명과 일치시킨다").

### 데이터 조달

- 원본: `_archive\01_AI-ME_Graduate_backup\data\kospi.csv` (동일 파일이 `AI-ME-Practice-1\data\` 에도 있음, 내용 동일)
- 소스 노트북은 GitHub raw URL 에서 직접 읽는다:
  `https://raw.githubusercontent.com/jongmoonha/AI-ME-Practice/refs/heads/main/data/kospi.csv`
- **이 과목의 `data/` 는 `.gitignore` 에 있다** (`.gitignore:9`). 따라서 csv 를 커밋해 두는 방식은 성립하지 않는다.
- **결정:** `gen/p08.py:283-292` 의 "없으면 내려받는다" 패턴을 그대로 따른다.
  `data/kospi.csv` 가 없으면 위 URL 에서 `urllib.request.urlretrieve` 로 받고, 있으면 그대로 읽는다.
  경로는 노트북 실행 위치(과목 루트) 기준 `./data/kospi.csv`.

### 데이터 사실 (실측)

| 항목 | 값 |
|------|-----|
| 열 | `Date, Open, High, Low, Close, Adj Close, Volume` |
| 행 수 | 431 (2019-01-30 ~ 2020-10-30, 거래일) |
| 결측 | 없음 (`isna().sum()` 전 열 0) |
| `Close` 범위 | 1457.64 ~ 2443.58 |
| `Volume` 범위 | 0 ~ 1,984,200 (`Volume == 0` 인 행 1개 존재) |
| 사용 입력 | `Open, High, Low, Volume` (4열) — 소스와 동일 |
| 사용 타깃 | `Close` |
| 미사용 | `Adj Close` (소스도 미사용), `Date` (플롯 축으로만 쓰지 않음 — §5 참조) |

---

## 1. Notation

기존 기호는 `CLAUDE.md` "Notation" 표에서 그대로 가져왔다.
RNN 고유 기호는 **이 과목에 선례가 없어 이번에 새로 도입**하며, 해당 행에 `NEW` 를 붙였다 (§6-2 에 목록).

| 기호 | 의미 | shape | 코드 변수명 | 출처 |
|------|------|-------|-----------|------|
| $N$ | 윈도우(샘플) 수 | scalar | `len(x_seq)` | CLAUDE.md Notation |
| $d$ | 입력 특징 수 = 4 | scalar | `input_size` | CLAUDE.md Notation |
| $T$ `NEW` | 시퀀스 길이(윈도우 폭) = 5 | scalar | `sequence_length` | 신규 — 소스 `sequence_length` |
| $H$ `NEW` | hidden state 차원 = 8 | scalar | `hidden_size` | 신규 — 소스 `hidden_size` |
| $t$ `NEW` | 시퀀스 내 시점 인덱스, $t = 1 \dots T$ | scalar | (루프 변수 `t`) | 신규 |
| $\mathbf{X}$ | 윈도우로 자른 입력 텐서 | $(N, T, d)$ = (426, 5, 4) | `x_seq` | CLAUDE.md Notation ($\mathbf{X}$ 를 설계행렬로 사용) |
| $\mathbf{x}_t$ `NEW` | 한 윈도우의 $t$ 번째 시점 입력 | $(d,)$ = (4,) | `x_seq[i, t]` | 신규 |
| $\mathbf{y}$ | 타깃 (윈도우 다음 날 `Close`) | $(N, 1)$ = (426, 1) | `y_seq` | CLAUDE.md Notation |
| $\hat{\mathbf{y}}$ | 예측 | $(N, 1)$ | `y_hat` | CLAUDE.md Notation |
| $\mathbf{E}$ | 잔차 $\mathbf{y} - \hat{\mathbf{y}}$ | $(N, 1)$ | `E` | CLAUDE.md Notation |
| $\mathbf{h}_t$ `NEW` | 시점 $t$ 의 hidden state | $(H,)$ = (8,) / 배치는 $(B, H)$ | `h` | 신규 |
| $\mathbf{h}_0$ `NEW` | 초기 hidden state = $\mathbf{0}$ | $(\text{num\_layers}, B, H)$ = (1, B, 8) | `h0` | 신규 — 소스 `torch.zeros(...)` |
| $\mathbf{W}_{xh}$ `NEW` | 입력→hidden 가중치 | $(H, d)$ = (8, 4) | `rnn.weight_ih_l0` | 신규 (PyTorch `nn.RNN` 파라미터) |
| $\mathbf{W}_{hh}$ `NEW` | hidden→hidden 가중치 | $(H, H)$ = (8, 8) | `rnn.weight_hh_l0` | 신규 |
| $\mathbf{b}_{xh}, \mathbf{b}_{hh}$ `NEW` | PyTorch 의 **두 개** bias | $(H,)$ = (8,) 각각 | `rnn.bias_ih_l0`, `rnn.bias_hh_l0` | 신규 — §2 주의 참조 |
| $\mathbf{w}$ | readout 선형층 가중치 | A: $(1, H)$ / B: $(1, HT)$ | `self.fc.weight` | CLAUDE.md Notation (`theta` 금지) |
| $b$ | readout bias | $(1,)$ | `self.fc.bias` | CLAUDE.md Notation |
| $\rho$ | 학습률 = 1e-3 | scalar | `rho` (optimizer 에는 `lr=rho`) | CLAUDE.md Notation, `alpha`/`eta`/`lr=` 리터럴 금지 |
| $B$ | 미니배치 크기 = 20 | scalar | `batch_size` | 소스 `batch_size = 20` |

**금지 확인:** `theta`, `alpha`, `eta`, `step_size`, `lr = 1e-3` (리터럴 직접 대입) 사용 금지.
소스의 `lr = 1e-3` 은 반드시 `rho = 1e-3` → `optim.Adam(model.parameters(), lr=rho)` 로 바꾼다.
(profile 체크포인트 `lr-symbol` 이 `^\s*(alpha|eta|step_size|learning_rate)\s*=` 로 발화한다.)

---

## 2. 수식

소스 노트북에는 수식이 없다. 셀 14 가 유일한 개념 설명인데 **텍스트 없이 base64 PNG 한 장**이다
(`![image.png](data:image/png;base64,...)`, 97KB, 캡션 없음).
따라서 아래 수식은 **표준 vanilla RNN 정의를 이 과목 notation 으로 새로 적은 것**이며, 강의 출처가 없다 (§6-1).

### 2.1 Hidden state 재귀식 (핵심)

$$\mathbf{h}_t = \tanh\!\left(\mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_{xh} + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_{hh}\right), \qquad t = 1, \dots, T, \qquad \mathbf{h}_0 = \mathbf{0}$$

- 출처: 없음 (표준 정의). PyTorch `nn.RNN` 기본 `nonlinearity='tanh'` 와 일치.
- 코드 대응 (수동 계산, Step 4):
  ```python
  h = torch.tanh(weight_ih @ x_t + bias_ih + weight_hh @ h + bias_hh)
  ```
- 코드 대응 (라이브러리):
  ```python
  out, h_n = self.rnn(x, h0)
  ```

> **주의 — bias 가 두 개다.** 교과서 표기는 보통 $\mathbf{b}_h$ 하나지만
> PyTorch `nn.RNN` 은 `bias_ih_l0` 와 `bias_hh_l0` 를 따로 갖는다 (합은 수학적으로 동일).
> Step 4 의 수동 계산이 `nn.RNN` 출력과 같은 값이 되려면 **둘 다 더해야 한다.**
> 마크다운 수식도 위처럼 두 개로 적어 코드와 1:1 대응시킨다. 하나로 합쳐 적지 않는다.

### 2.2 Readout — 두 전략

**A. 마지막 hidden state 만 사용 (many-to-one)**

$$\hat{y} = \mathbf{w}^{\top}\mathbf{h}_T + b, \qquad \mathbf{w} \in \mathbb{R}^{H}$$

**B. 모든 hidden state 를 이어 붙여 사용**

$$\hat{y} = \mathbf{w}^{\top}\left[\mathbf{h}_1; \mathbf{h}_2; \dots; \mathbf{h}_T\right] + b, \qquad \mathbf{w} \in \mathbb{R}^{HT}$$

- 출처: 소스 셀 16 코드 및 그 안의 한국어 주석.
- B 가 소스가 실제로 구현한 것(`out.reshape(out.shape[0], -1)` + `nn.Linear(hidden_size * sequence_length, 1)`).
- **소스의 용어는 틀렸다.** 소스는 B 를 "many to many 전략" 이라 부르지만,
  출력이 스칼라 하나뿐이므로 B 도 many-to-one 이다. 진짜 many-to-many 는 시점마다 출력을 낸다.
  **노트북에서 "many to many" 라는 표현을 쓰지 않는다.** (§6-4)

### 2.3 손실과 갱신

$$J = \frac{1}{B}\sum_{n=1}^{B}\left(\hat{y}_n - y_n\right)^2 \quad (\text{미니배치 평균}), \qquad \mathbf{w} \leftarrow \mathbf{w} - \rho\,\frac{\partial J}{\partial \mathbf{w}}$$

- 코드 대응: `criterion = nn.MSELoss()` (기본 `reduction='mean'`), `optimizer = optim.Adam(model.parameters(), lr=rho)`
- **`/ 2` 를 붙이지 않는다.** CLAUDE.md 의 $1/2$ 규약은 "손으로 유도한 기울기와 `loss.backward()` 를
  눈으로 대조" 하기 위한 것인데, 이 회차에는 손 유도 기울기 셀이 없다.
  `gen/p09.py:179, 203` 도 `nn.CrossEntropyLoss()` 를 그대로 쓴다 (배수 보정 없음).
  → 이 판단은 §6-3 에 올려 리더 확인을 받는다.
- 갱신식의 $\rho$ 는 Adam 의 step size 이며 위 식은 plain SGD 형태의 개념 표기다.
  마크다운에 "Adam adapts this step size per parameter" 한 줄만 덧붙인다.

### 2.4 평가 지표

$$\text{RMSE}_{\text{scaled}} = \sqrt{J}, \qquad \text{RMSE}_{\text{points}} = \text{RMSE}_{\text{scaled}} \times \sigma_{\text{close,train}}$$

- 코드 대응: `rmse_points = math.sqrt(mse) * close_train_std`
- **소스의 버그를 고친 것이다.** 소스 셀 24 는 `mean_squared_error(...)` 값을
  `'... train rmse {:.5f}, test rmse{:.5f}'` 라고 출력한다 — MSE 를 RMSE 라 부른다. 그대로 옮기지 않는다.
- 지수 포인트 환산을 함께 출력해야 기계공학 대학원생이 "이 오차가 얼마나 큰가" 를 물리량으로 읽는다.

---

## 3. 섹션 구성

셀 유형과 순서. `Step N` 헤더 체계는 `gen/p09.py`, `gen/p11.py`, `gen/p12.py` 와 동일하다.
모든 마크다운 셀은 `CLAUDE.md` "설명 분량" 상한을 지킨다
(첫 셀 150단어, Summary 100단어, 그 외 120단어 / 산문 80단어 / 문단 3문장 / **한 줄 1문장**).

| § | 제목 | 내용 | 셀 유형 | 비고 |
|---|------|------|---------|------|
| — | `# Practice 13 — RNN for Time Series` | 제목 + 개요 3~4문장 + 데이터 표 | md | 150단어 이내 |
| 0 | `## Step 0. Imports and Setup` | import, `np.random.seed(42)`, `torch.manual_seed(42)`, `plt.rcParams['axes.unicode_minus'] = False`, `device` | md + code | `gen/p09.py:51-72` 와 동일 형태 |
| 1 | `## Step 1. The Series` | csv 다운로드-if-absent → `pd.read_csv` → `df.head(10)` | md + code | CLAUDE.md "데이터 로드 후 `head(10)`" |
| 1 | 〃 | `Close` 시계열 라인 플롯 1장 (x = 행 인덱스) | code | 비정상성이 눈에 보여야 §8 논의가 성립 |
| 2 | `## Step 2. Sliding Windows` | 윈도우 개념 설명 + shape 표 | md | $(N, T, d)$ 표기 |
| 2 | 〃 | `make_sequences(...)` 정의 + 호출 + shape 출력 | code | **함수명 주의 — §6-5** |
| 3 | `## Step 3. Chronological Split and Scaling` | 왜 셔플 분할이 아니라 시간순 분할인가 (2문장) + 분할 표 | md | |
| 3 | 〃 | 시간순 70/15/15 분할 → **train 통계로만** 표준화 → tensor → DataLoader 4개 | code | 분할이 먼저. `fit_transform` 금지 |
| 4 | `## Step 4. What a Recurrent Layer Computes` | §2.1 수식 + 언롤 설명 + $\mathbf{h}_0 = \mathbf{0}$ | md | 새 기호 도입 지점 |
| 4 | 〃 | `nn.RNN` 하나 만들어 **수식 손계산 vs `nn.RNN` 출력** 나란히 `print` | code | 이 과목의 서명 패턴 (code-patterns §2). `allclose`/`assert` 금지 |
| 5 | `` ## Step 5. `train` and `evaluate` `` | 두 함수 반환값 표 | md | `gen/p09.py:157-170` 형태 |
| 5 | 〃 | `def evaluate(model, loader, device)` | code | 회귀판 — 반환 `(loss, y_pred)` |
| 5 | 〃 | `def train(model, train_loader, val_loader, optimizer, epochs, device)` | code | 반환 `(train_losses, val_losses)` |
| 6 | `## Step 6. Model A — Last Hidden State` | readout A 수식(§2.2 A) + 왜 $\mathbf{h}_T$ 만 쓰나 | md | |
| 6 | 〃 | `class RNNLastState(nn.Module)` 정의 | code | 명시적 클래스. factory 금지 |
| 6 | 〃 | seed 재설정 → 모델 생성 → optimizer → `train(...)` 호출 | code | |
| 7 | `## Step 7. Model B — All Hidden States` | readout B 수식(§2.2 B) + 차이는 readout 입력뿐 | md | |
| 7 | 〃 | `class RNNAllStates(nn.Module)` 정의 | code | **코드 반복.** for 문 순회 금지 |
| 7 | 〃 | seed 재설정 → 모델 생성 → optimizer → `train(...)` 호출 | code | 하이퍼파라미터는 Step 5 에서 정의한 것 재사용 |
| 8 | `## Step 8. Comparison` | §4 통제표 그대로 | md | |
| 8 | 〃 | loss 곡선 겹쳐 그리기 (1×2: train / val) | code | 선 스타일 `'-'`, `'--'` |
| 8 | 〃 | 두 모델 test 예측 vs 실제 (지수 포인트로 역변환, 경계선 2개) | code | §5 참조 |
| 8 | 〃 | train/val/test RMSE 를 scaled 와 points 로 `print` | code | 표 대신 `print` 나란히 (code-patterns §4) |
| — | `## Summary` | 4~5줄 요약 | md | 100단어. 새 내용 금지 |
| — | `## Exercises` | 연습문제 3~4개 + `# Write your code here` 빈 셀 | md + code | `gen/p09.py:554-566` 형태 |

**총 셀 수 목표: 마크다운 14 ± 2, 코드 15 ± 2.**

### 3.1 분할과 스케일링 — 확정 수치 (실측)

`sequence_length = 5` → 윈도우 $N = 431 - 5 = 426$.

| 분할 | 윈도우 수 | 마지막 타깃 날짜 |
|------|-----------|------------------|
| train | 298 (70%) | 2020-04-23 |
| val | 63 (15%) | 2020-07-24 |
| test | 65 (15%) | 2020-10-30 |

```python
n_train = int(len(x_seq) * 0.70)
n_val   = int(len(x_seq) * 0.15)
# 시간 순서를 그대로 자른다. train_test_split 을 쓰지 않는다 (미래가 train 에 섞인다)
```

표준화는 `CLAUDE.md` "데이터 분할과 스케일링" 의 형태를 그대로 쓴다 (**sklearn 스케일러 사용 금지**):

```python
input_train_mean = x_train.reshape(-1, input_size).mean(axis=0)
input_train_std  = x_train.reshape(-1, input_size).std(axis=0)
input_train_std[input_train_std == 0] = 1.0   # leave zero-variance features unscaled

close_train_mean = y_train.mean()
close_train_std  = y_train.std()
```

- 실측 통계(참고, 저자가 재계산할 것): `input_train_mean ≈ [2080.8, 2092.5, 2066.6, 550111.2]`,
  `input_train_std ≈ [139.7, 133.7, 145.9, 210279.4]`, `close_train_mean ≈ 2077.7`, `close_train_std ≈ 140.5`
- `+ 1e-8` 금지 (`epsilon-trick` 체크포인트)
- 이름은 `mean`/`std`/`mu`/`sigma` 금지 (`cryptic-var` 체크포인트가
  `^\s*(mean|std|min|max|...)\s*=` 로 발화)

> **저자가 반드시 알아야 할 사실 — test 구간은 외삽이다.**
> train 통계로 표준화하면 test 타깃의 표준화 값 범위가 **0.998 ~ 2.605** 다
> (원 지수 2218 ~ 2444, 2020년 8~10월 상승장). 즉 모델이 학습 중 본 적 없는 수준이다.
> 예측선이 실제선 아래에 깔리는 것이 **정상 결과**이며, 이를 Step 8 마크다운에 한 문장으로 정직하게 적는다.
> 이 사실을 모르면 저자가 "결과가 이상하다" 며 분할이나 스케일링을 임의로 되돌리게 된다.
> (소스의 `split = 200` 은 더 나쁘다 — test 타깃이 -1.334 까지 내려가 COVID 급락 구간이 통째로 외삽이 된다.)

### 3.2 하이퍼파라미터 — Step 5 에서 한 번만 정의

| 이름 | 값 | 출처 |
|------|-----|------|
| `sequence_length` | 5 | 소스 셀 10 |
| `input_size` | 4 | 소스 셀 13 (`x_seq.size(2)`) |
| `hidden_size` | 8 | 소스 셀 13 |
| `num_layers` | **1** | 소스는 2 — **변경, §6-6** |
| `batch_size` | 20 | 소스 셀 12 |
| `rho` | 1e-3 | 소스 셀 19 `lr = 1e-3` |
| `epochs` | 200 | 소스 셀 19 `num_epochs = 200` |
| optimizer | `optim.Adam` | 소스 셀 19 |
| `criterion` | `nn.MSELoss()` | 소스 셀 19 (`train`/`evaluate` **내부**에서 생성 — code-patterns §7b) |

`epochs = 200`, train 배치 15개/epoch → CPU 에서도 수십 초. `cell_time_warning_sec = 300` 를 넘지 않는다.

### 3.3 `train` / `evaluate` — 회귀판 시그니처

code-patterns §7(b) 는 분류 고정형(`(loss, accuracy, y_pred)`)이다.
**이 회차는 회귀라 accuracy 가 없다.** 시그니처는 유지하고 반환에서 accuracy 만 뺀다.

```python
def evaluate(model, loader, device):
    # one pass over a loader with the gradient turned off
    # returns (loss, y_pred)

def train(model, train_loader, val_loader, optimizer, epochs, device):
    # epoch loop; calls evaluate(val_loader) after each epoch
    # returns (train_losses, val_losses)
```

- **두 함수만.** `fit`, `make_*`, `build_*` 금지 (`helper-factory` 체크포인트)
- `criterion` 은 함수 내부에서 생성한다 (인자로 받지 않는다)
- `test_loader` 를 `train(...)` 인자로 넘기지 않는다 (`test-leak` 체크포인트가
  `train\s*\([^)]*test_loader` 로 발화)
- loss 누적은 per-batch mean 을 더해 `len(loader)` 로 나눈다. `loss.item() * len(batch)` 금지
- 코드 셀에 docstring(`"""`) 금지 → `#` 주석 (`docstring` 체크포인트)
- epoch 로그는 `if (epoch + 1) % 50 == 0` 로 솎아 출력한다 (200 줄 출력 방지)

**DataLoader 4개:**

| loader | dataset | `shuffle` | 용도 |
|--------|---------|-----------|------|
| `train_loader` | train 윈도우 | `True` | 학습 |
| `train_eval_loader` | train 윈도우 (동일) | `False` | 시간 순서 예측 플롯 |
| `val_loader` | val 윈도우 | `False` | epoch 마다 모니터링 |
| `test_loader` | test 윈도우 | `False` | Step 8 에서 **1회만** |

> 소스는 train 도 `shuffle=False` 였다. 예측을 시간순으로 이어 붙이기 위해서다.
> 학습에는 셔플이 맞으므로 **플롯용 로더를 따로 둔다.**
> `gen/p09.py:112-122` 도 같은 이유로 augment 안 한 train dataset 을 별도로 만든다.

---

## 4. 비교 실험 통제표

**격리 변수: 선형 readout 이 무엇을 보는가 — 마지막 hidden state 하나 vs 전체 hidden state.**

| 항목 | A `RNNLastState` | B `RNNAllStates` |
|------|------------------|------------------|
| **readout 입력 (격리 변수)** | `out[:, -1, :]` → $(B, H)$ | `out.reshape(B, -1)` → $(B, HT)$ |
| **readout 층 (격리 변수의 귀결)** | `nn.Linear(hidden_size, 1)` | `nn.Linear(hidden_size * sequence_length, 1)` |
| 데이터셋·분할 | 동일 (시간순 298/63/65) | 동일 |
| 표준화 통계 | 동일 (`input_train_mean/std`, `close_train_*`) | 동일 |
| `sequence_length` | 5 | 5 |
| `hidden_size` | 8 | 8 |
| `num_layers` | 1 | 1 |
| RNN 활성 | `tanh` (기본) | `tanh` (기본) |
| 출력 활성 | 없음 | 없음 |
| `rho` | 1e-3 | 1e-3 |
| `epochs` | 200 | 200 |
| `batch_size` | 20 | 20 |
| optimizer | `optim.Adam` | `optim.Adam` |
| 손실 | `nn.MSELoss()` | `nn.MSELoss()` |
| seed | 모델 생성 **직전** `torch.manual_seed(42)` 재설정 | 동일 |

**정직하게 적을 것 (CLAUDE.md "의도된 다중 차이라면 솔직히 적는다"):**
readout 파라미터 수가 A 는 $H + 1 = 9$, B 는 $HT + 1 = 41$ 로 다르다.
이는 별개의 변수가 아니라 **격리 변수의 직접적 귀결**이다.
Step 8 마크다운에 한 문장으로 명시한다 — "B has more readout parameters; that is a consequence of what it reads, not a separate change."

**하이퍼파라미터는 Step 5 에서 한 번만 정의하고 두 모델이 그대로 재사용한다.**
모델마다 다시 정의하면 비교가 성립하지 않는다 (CLAUDE.md "비교 실험 통제 변수").

---

## 5. 시각화 명세

`gridspec` 금지. 각 그림은 독립 코드 셀 + `plt.subplots`. 모든 텍스트 영어.

### 5.1 Step 1 — 시리즈 개관 (1장)

- `plt.subplots(1, 1, figsize=(12, 4))`
- `Close` 를 행 인덱스에 대해 라인 플롯
- title `'KOSPI closing index'`, xlabel `'Trading day'`, ylabel `'Close'`, `grid(alpha=0.3)`
- **목적:** 2020년 3월 급락과 8~10월 상승이 보여야 Step 3·8 의 외삽 논의가 근거를 갖는다

### 5.2 Step 8 — 학습 곡선 (1장, 1×2)

- `plt.subplots(1, 2, figsize=(12, 4))`
- 좌 `axes[0]`: train loss — A `'b-'`, B `'r--'`
- 우 `axes[1]`: validation loss — A `'b-'`, B `'r--'`
- xlabel `'Epoch'`, ylabel `'MSE (standardized)'`, `legend()`, `grid(alpha=0.3)`
- 선 스타일을 달리하는 이유는 code-patterns §5 — 겹쳐도 둘 다 보이게

### 5.3 Step 8 — 예측 vs 실제 (1장, 1×1)

- `plt.subplots(1, 1, figsize=(14, 5))`
- x 축: 전체 윈도우 인덱스 0 ~ 425 (train → val → test 가 시간순으로 이어짐)
- **지수 포인트로 역변환해서 그린다:** `y * close_train_std + close_train_mean`
- 곡선 3개: actual `'k--'`, model A `'b-'` (linewidth 1), model B `'r-'` (linewidth 1)
- 수직 경계선 2개: `axvline(n_train)`, `axvline(n_train + n_val)` — `linestyle=':'`, 각각 라벨
- title `'Prediction vs actual (index points)'`, ylabel `'KOSPI Close'`, `legend()`
- **소스의 `plotting()` 함수를 옮기지 않는다.** 소스는 한 함수 안에서 예측·MSE 계산·플롯·반환을
  전부 하고 title 에 MSE 를 "rmse" 로 잘못 표기한다. 예측은 `evaluate` 로, 플롯은 플롯 셀로 분리한다.

### 5.4 Step 8 — 수치 출력

```python
print(f'Model A (last state)  train RMSE = ... , val RMSE = ... , test RMSE = ...  ({...:.1f} index points)')
print(f'Model B (all states)  train RMSE = ... , val RMSE = ... , test RMSE = ...  ({...:.1f} index points)')
```

표 대신 `print` 나란히 (code-patterns §4). `np.allclose` / `assert` 금지 (`assert-compare` 체크포인트).

---

## 6. 확인 필요 항목

### 6-1. `[근거 없음 — 확인 필요]` §2 의 모든 수식

이 과목에 RNN 강의자료가 없다.
소스 노트북의 유일한 개념 설명 셀(셀 14)은 **텍스트 없는 base64 PNG 한 장**이라 수식을 읽어낼 수 없다.
§2 의 재귀식·readout 식은 **표준 vanilla RNN 정의를 내가 이 과목 notation 으로 새로 적은 것**이며
강의 출처가 없다.
→ **리더/사용자 판단 필요:** 이 수식들을 그대로 쓸 것인지, 아니면 실제 강의에서 쓰는 표기를 받아올 것인지.
`consistency-verifier` 는 대조할 강의자료가 없으므로 §1·§2 자체를 기준선으로 삼을 수밖에 없다.

### 6-2. `[새로 도입, 선례 없음]` RNN 고유 기호 8종

$T$, $H$, $t$, $\mathbf{x}_t$, $\mathbf{h}_t$, $\mathbf{h}_0$, $\mathbf{W}_{xh}$, $\mathbf{W}_{hh}$, $\mathbf{b}_{xh}$, $\mathbf{b}_{hh}$.
이 과목 `CLAUDE.md` "Notation" 표에 없는 기호다. 관용적 딥러닝 표기를 따랐고,
기존 규약과의 정합성은 다음과 같이 맞췄다.

- 가중치는 $\mathbf{w}$ / 대문자 행렬은 $\mathbf{W}$ — `theta` 계열을 쓰지 않았다
- readout 가중치는 기존 규약 그대로 $\mathbf{w}$ / `w`
- 학습률은 $\rho$ / `rho` 고정
- $N$, $d$ 의 의미를 바꾸지 않았다 ($N$ = 샘플 수 = 윈도우 수, $d$ = 입력 특징 수)

→ **승인되면 `CLAUDE.md` "Notation" 표에 이 행들을 추가할 것을 제안한다** (§7 참조).
지금 기록하지 않으면 다음 시퀀스 회차에서 같은 항목이 다시 미결로 올라온다.

### 6-3. `[규약 해석 충돌]` 손실함수의 $1/2$

`CLAUDE.md` "손실함수에 $1/2$ 를 붙인다" 는 이 과목의 명시 규약이다.
그러나 그 근거는 "손으로 유도한 기울기를 `loss.backward()` 와 눈으로 대조" 하기 위함이고,
이 회차에는 손 유도 기울기 셀이 없다.
또 `gen/p09.py:179, 203` 은 `nn.CrossEntropyLoss()` 를 배수 보정 없이 그대로 쓴다 (선례).
→ **내 판정: `nn.MSELoss()` 를 그대로 쓴다 (`/ 2` 없음).** §2.3 에 반영했다.
리더가 반대하면 `criterion` 호출부 한 곳만 바꾸면 되므로 영향 범위는 작다.

### 6-4. `[소스 오류 — 수정함]` "many to many" 용어

소스 셀 16 은 전체 hidden state 를 flatten 해 스칼라 하나를 내는 구조를 "many to many 전략" 이라 부른다.
출력이 하나이므로 사실은 many-to-one 이다.
→ **노트북에서 "many to many" 를 쓰지 않는다.** Model A/B 를 "last hidden state" / "all hidden states"
로 부른다 (§2.2, §4). 리더 확인만 받고 진행한다.

### 6-5. `[체크포인트 저촉 가능]` 시퀀스 생성 함수 이름

소스는 `seq_data(x, y, sequence_length)` 다.
`cryptic-var` 는 변수만 보므로 함수명은 걸리지 않지만, `seq_data` 는 축약이라 CLAUDE.md
"암호 같은 약어 금지" 취지에 맞지 않는다.
`make_sequences` / `build_sequences` 는 **`helper-factory` 체크포인트가
`^\s*def\s+(make_|build_|create_|get_)\w*` 로 발화한다** (noise:true 이지만 감사에 올라온다).
→ **내 제안: `def to_sequences(x, y, sequence_length)`.** 두 패턴 어디에도 걸리지 않고 뜻이 분명하다.
리더가 다른 이름을 원하면 §3 표의 해당 행만 바꾸면 된다.

### 6-6. `[소스와 다름 — 판단 필요]` `num_layers` 2 → 1

소스는 `num_layers = 2` 다. 이 회차는 Step 4 에서 재귀식을 손계산과 대조하는데,
2층이면 §2.1 의 식이 1층만 설명하게 되어 수식과 코드가 어긋난다.
→ **내 판정: `num_layers = 1` 로 하고, 층 쌓기는 Exercises 로 보낸다.**
성능이 아니라 "수식과 코드가 1:1 로 대응하는가" 를 우선한 결정이다.

### 6-7. `[소스와 다름 — 판단 필요]` 출력단 `nn.Sigmoid()` 제거

소스는 readout 끝에 `nn.Sigmoid()` 를 붙인다. MinMaxScaler 로 타깃을 [0, 1] 로 눌렀기 때문에만 성립한다.
이 명세는 규약대로 **z-score 표준화**(train 통계)를 쓰므로 타깃이 [0, 1] 을 벗어나고,
Sigmoid 가 있으면 모델이 도달할 수 없는 구간이 생긴다.
또 code-patterns §3 은 "회귀의 `forward` 는 출력을 그대로 반환" 을 요구한다.
→ **내 판정: `nn.Sigmoid()` 제거.**

### 6-8. `[소스와 다름 — 판단 필요]` 스케일러와 분할

소스 셀 6 은 `scaler.fit_transform(df[...])` 를 **분할 전 전체 데이터**에 적용한다 — 명백한 data leakage 이고
`CLAUDE.md` "분할이 먼저다" 정면 위반이다. 소스 셀 11 의 `split = 200` 도 train 200 / test 226 으로
test 가 train 보다 크다.
→ **내 판정: 시간순 70/15/15 + train 통계로만 표준화.** §3.1 에 수치를 실측해 넣었다.
val 분할은 소스에 없지만 code-patterns §7(b) 가 val 모니터링을 요구하므로 신설한다.

### 6-9. `[범위 판단 필요]` BPTT / vanishing gradient 를 다룰 것인가

**소스에는 BPTT 도 vanishing gradient 도 한 글자도 없다.** LSTM/GRU 언급도 없다.
`CLAUDE.md` 는 "슬라이드에 없는 내용을 임의로 추가하지 않는다" 고 규정한다.
→ **내 판정: 본문에 넣지 않는다.** 대신 Exercises 에 한 문항으로 남긴다 —
"Increase `sequence_length` to 40 and retrain. Report what happens to the training loss."
리더가 본문 설명을 원하면 Step 4 뒤에 마크다운 셀 1개(120단어 이내)를 추가하는 형태가 적절하며,
그 경우에도 실험 셀은 만들지 않는다.

### 6-10. `[확인 필요]` 데이터 출처 URL 의 수명

`data/` 가 `.gitignore` 에 있어 노트북은 외부 URL 에 의존한다
(`raw.githubusercontent.com/jongmoonha/AI-ME-Practice/...`).
이 저장소가 사라지면 노트북이 학생 환경에서 실행되지 않는다.
→ **리더 확인 필요:** 이 URL 을 계속 쓸 것인지, 아니면 `.gitignore` 에서 `data/kospi.csv` 만
예외 처리(`!data/kospi.csv`)해 커밋할 것인지. 431행 CSV 라 용량 문제는 없다.

### 6-11. `[미검증]` 실행 시간과 수렴

`epochs = 200`, hidden_size 8, 배치 15개/epoch 는 CPU 에서도 짧을 것으로 보이나
**실제 학습 결과(수렴 여부, 두 모델의 우열)는 아직 실행해 보지 않았다.**
저자가 실행한 뒤 loss 곡선이 평평하거나 발산하면 `epochs` 또는 `rho` 조정이 필요할 수 있다.
조정하면 **두 모델에 동시에 같은 값을 적용**해야 §4 통제표가 유지된다.

---

## 7. 하네스 갱신 제출

이번 작업에서 확인한 것으로, 다음 사람이 같은 탐색을 반복하지 않도록 남긴다.

### 7-1. 즉시 정정 대상 — 프로필의 강의자료 선언이 현실과 어긋난다

`notebook-profile.json` 의 `lecture_sources.slides_glob: "lecture_notes/*.pdf"` 와
`authority: "slides"` 는 **PDF 1편만 존재**하는 현 상태에서 Practice 02 이후 회차에 대해 전부 공허하다.

```
lecture_notes/  →  Ch1-ML 1_Linear Regression.pdf  (이것 하나뿐)
```

Practice 02 ~ 13 은 대조할 슬라이드가 없다.
→ **제안:** `lecture_sources` 에 `coverage` 필드를 추가해 "슬라이드가 실재하는 범위" 를 명시하고,
범위 밖 회차는 `authority` 가 자동으로 "없음(사람 판단)" 이 되게 한다.
지금은 분석자가 매 회차 `ls lecture_notes/` 를 하고 나서야 이 사실을 알게 된다.

### 7-2. 강의자료 공백 — 사용자에게 보고

- `lecture_notes/` 에 Ch2(ML)·Ch3(DL) 슬라이드가 전부 없다
- `md/lectures_and_formulas.md`, `md/practice_outline_ref.md` 는 학부 과목 요약본이라
  **RNN 을 포함해 딥러닝 후반부 항목이 아예 없다** (`grep -i rnn` 0 hit)
- → RNN 회차의 수식 표기는 이번에 새로 만든 것이며, 실제 강의 표기와 다를 수 있다.
  강의 슬라이드를 `lecture_notes/` 에 추가하는 것이 근본 해결이다

### 7-3. 결론이 나면 규약에 남길 것

§6-2 (RNN 기호 8종), §6-3 (`/ 2` 적용 범위), §6-5 (함수 이름 규칙) 는 리더 판단이 나는 즉시
`CLAUDE.md` 에 반영해야 한다. 특히:

- **§6-3 은 이 회차만의 문제가 아니다.** `CLAUDE.md` "손실함수에 $1/2$ 를 붙인다" 가
  "손 유도 기울기를 대조하는 셀이 있는 회차에 한한다" 는 단서를 달지 않으면,
  앞으로 모든 회차에서 같은 질문이 반복된다
- **§6-5 의 `make_`/`build_` 금지와 "약어 금지" 가 함수 이름에서 충돌**한다.
  허용되는 동사 접두사(`to_`, `compute_`, `split_` 등)를 CLAUDE.md 에 한 줄로 예시해 두면
  다음 저자가 헤매지 않는다

### 7-4. 소스 노트북에서 발견한 결함 (기록용)

`_archive\01_AI-ME_Graduate_backup\Chapter3_Deep Learning_3_RNN.ipynb` 를 다른 회차로 옮길 사람을 위해:

| 셀 | 결함 |
|----|------|
| 6 | 분할 전 전체 데이터에 `fit_transform` — data leakage |
| 9 | 코드 셀 docstring (이 과목 `docstring` 체크포인트 위반) |
| 11 | `split = 200` — test(226) 가 train(200) 보다 큼 |
| 14 | 개념 설명이 텍스트 없는 base64 PNG — 수식을 읽어낼 수 없음 |
| 16 | many-to-one 구조를 "many to many" 라 오기 |
| 16 | 회귀 출력에 `nn.Sigmoid()` — MinMax 스케일링에만 의존하는 설계 |
| 19 | `lr = 1e-3` (이 과목은 `rho`) |
| 24 | MSE 값을 title 에 "rmse" 로 출력 — 지표 오표기 |
| 24 | 예측·지표·플롯을 한 함수에 묶고 `model` 을 전역에서 캡처 |
| 전체 | 마크다운·주석이 한국어 (이 과목은 영어 전용) |
