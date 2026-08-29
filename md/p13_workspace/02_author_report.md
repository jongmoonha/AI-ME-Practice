# Author Report — Practice13_RNN_for_Time_Series.ipynb

- generator: `D:\Main\00_Research\00_Python\00_Lecture\01_AI-ME_Graduate\gen\p13.py`
- 노트북: `D:\Main\00_Research\00_Python\00_Lecture\01_AI-ME_Graduate\Practice13_RNN_for_Time_Series.ipynb`
- 라운드: **3** (auditor D-1 반영)
- nbconvert 실행: **성공** (전 32셀, 에러 0)
- 총 셀 수: **code 17개 / markdown 15개** (명세 §3 목표 code 15±2, md 14±2 — 둘 다 범위 내)
- 실행 시간: **15.3초** (전체 노트북. 최장 셀은 Model A 학습 ~4초 → `cell_time_warning_sec = 300` 대비 여유)
- 실행 환경: `KMP_DUPLICATE_LIB_OK=TRUE`, device = `cuda`
- 명세 대비 미구현: 없음
- 이번 라운드 반영한 지적: **auditor D-1** (verifier 는 라운드 2 에서 완전 PASS)
- 변경된 셀 범위: **코드 27 한 줄** (주석 1줄. 실행 결과·수치 변화 없음)

---

## 라운드 3 — 반영 내역

### [auditor D-1] 셀 27 첫 줄 주석 → 반영

```python
# 수정 전
# over a whole split at once, rather than averaging the per-batch means the loaders produce
# 수정 후
# RMSE over each split as a whole
```

"다른 방식을 안 쓴 이유" 를 설명하던 뒷절을 뗐다. 그 내용은 이미 셀 13 마크다운에 학생용 문장으로 들어가 있어 코드 주석에서 중복이었다. 주석은 이제 "이 줄이 무엇을 하는가" 만 말한다.

출력 수치는 라운드 2 와 동일하다 (주석만 바뀌었으므로).

```
Model A (last state)  train 0.1796 [  25.2]  val 0.1971 [  27.7]  test 0.4544 [  63.8]
Model B (all states)  train 0.1782 [  25.0]  val 0.2564 [  36.0]  test 0.2613 [  36.7]
```

자체 점검: `audit_notebook.py` 후보 없음 / `markdown_budget.py` 0 over cap, 9 above soft cap / nbconvert 32셀 에러 0, 15.3초.

### 다음 저자를 위한 주의 (auditor 확인 사항, 조치 불필요)

- **셀 29(Summary)는 산문 하드캡 80단어에 정확히 걸쳐 있다.** 이 셀에 문장을 더하려면 **먼저 줄이고** 더해야 한다. 라운드 2 에서 실제로 한 번 넘겼다가(86단어) 되돌린 자리다.
- 셀 12 의 `X_window`(3-D) / `x_t`(1-D 벡터) 대소문자 구분과 `forward(self, x)` 예외는 **최종 승인**됐다. 근거는 아래 라운드 2 기록 참조.

---

## 라운드 2 기록 — 반영 내역

### 1. [auditor B-1] 변수명 대소문자 → 반영

`X_all`, `X_seq`, `X_windows`, `X_window`, `X_train`, `X_val`, `X_test`, `X_train_tensor`, `X_val_tensor`, `X_test_tensor`, `X_batch` 로 전부 대문자화했다. `to_sequences(X, y, sequence_length)` 시그니처와 셀 11 의 `print('one batch - X ...')` 라벨, 셀 6 마크다운 표의 `` `X_seq` `` 까지 함께 고쳤다. y 계열은 지시대로 소문자 유지.

**두 곳만 소문자로 남겼다 (판단 필요하면 지적해 달라):**

| 남긴 이름 | 이유 |
|------|------|
| `x_t` (셀 12) | 셀 11 마크다운 재귀식의 $\mathbf{x}_t$ 와 1:1 대응하는 **한 시점의 벡터** $(d,)$ 다. 설계행렬이 아니라 벡터이므로 수식이 소문자 볼드를 쓰고, `X_t` 로 바꾸면 바로 위 수식과 기호가 어긋난다 |
| `forward(self, x)` (셀 18, 21) | `gen/p09.py:337` 의 `forward(self, x)` 선례를 따랐다. 데이터 변수가 아니라 메서드 파라미터다 |

### 2. [auditor B-2] 셀 29 Summary 첫 불릿 → 반영

`so window length leaves its parameter count unchanged — only the readout of B grows with it` 로 범위를 한정했다. 셀 25(readout 41 vs 9)·Exercise 2 와 더 이상 부딪히지 않는다.

### 3. [verifier V-1] 최종 보고 RMSE → 반영 (지적이 맞았다)

셀 27 에서 `evaluate` 가 반환한 `y_pred` 전체를 split 전체 타깃과 직접 맞춰 다시 계산한다.

```python
# over a whole split at once, rather than averaging the per-batch means the loaders produce
val_rmse_last_state = math.sqrt(((y_pred_val_last_state - y_val) ** 2).mean())
```

학습 곡선·epoch 로그는 지시대로 배치평균(`loss_sum / len(loader)`) 그대로 뒀다 — code-patterns §7(b) 규칙 유지. 셀 26 의 `evaluate` 호출은 이제 loss/rmse 를 안 쓰므로 `_, _, y_pred_... = evaluate(...)` 로 바꿨다.

**수정 전후 (배치평균 → 진짜 값):**

| | A val | A test | B val | B test |
|---|---|---|---|---|
| 수정 전 (배치평균) | 25.3 | 60.5 | 32.7 | 35.5 |
| 수정 후 (진짜) | **27.7** | **63.8** | **36.0** | **36.7** |

val 이 약 +10% 로 가장 크게 틀렸다 (63개 → [20,20,20,3]). 검증자 추정과 일치한다.
셀 13 마크다운의 정의 문장도 고쳤다 — `loss` 는 "배치별 MSE 의 평균, 학습 곡선이 그리는 값" 이고, "마지막 배치가 짧으므로 Step 8 이 `y_pred` 로 split 전체에 대해 다시 계산한다" 로 둘을 구분해 서술했다.

**결론은 바뀌지 않았다:** train 은 두 모델이 사실상 동일(25.2 vs 25.0), val 에서 A 가 낫고(27.7 vs 36.0), test 에서 순서가 뒤집힌다(63.8 vs 36.7).

### 4. [verifier V-2] 셀 25 사실 오류 → 반영 (지적이 맞았다)

"above everything the training window contained" 와 "above 2.6" 을 버리고, 셀 10 이 실제로 인쇄하는 값을 그대로 인용했다.

> Step 3 counted 58 of its 65 targets above the training maximum of 1.350, so both prediction curves sit below the actual line there.
> The validation targets stay inside the training range.

실측 재확인: train max **1.350**, test 중 1.350 초과 **58/65 (89.2%)**, 2.6 초과는 **1개뿐**. 검증자 수치와 일치한다.

### 5. [verifier V-3] 셀 10 인쇄에 val 범위 추가 → 반영

```
standardized target range - train [-4.414, 1.350]
standardized target range - val   [-1.343, 1.076]
standardized target range - test  [0.998, 2.605]
test windows above the training maximum: 58 of 65
```

val 범위는 검증자 실측(-1.343, 1.076)과 일치. 마지막 줄은 셀 25 의 "58 of 65" 주장이 인쇄된 값에 근거하도록 함께 넣었다 — 마크다운이 노트북이 실제로 찍는 숫자만 인용하게 된다.

### 선택 항목 — 둘 다 반영했다

**[O-1] `train()` 호출 직전 seed 재설정.** 셀 19·22 에 `torch.manual_seed(42)   # train_loader shuffles from the same state for both models` 한 줄씩 추가했다. 셀 23 통제표의 해당 행도 "Seed before construction and before training" 으로 고쳤다. 반영 후 두 번 연속 실행해 **모든 자릿수가 동일하게 재현**되는 것을 확인했다.

**[U-2] 다운로드 분기 실검증.** `data/kospi.csv` 를 잠시 옮기고 셀 4 소스를 **그대로 추출해 실행**했다. 다운로드 성공, 431행 정상 파싱. 받은 파일과 기존 로컬 파일의 SHA-256 이 **완전히 일치**한다 (`3c99f1bc...4783e4`). 원격 파일은 이미 push 되어 있다. 데이터는 원복 완료.

### 라운드 2 자체 점검

```
audit_notebook.py     → 후보 없음
markdown_budget.py    → 0 over cap, 9 above soft cap
nbconvert             → 32셀 전부 성공, 에러 0, 15.5초
```

V-2·B-2 수정으로 셀 25·29 가 일시적으로 hard cap(산문 80단어)을 넘어(89, 86) 두 셀과 셀 13 을 함께 다듬어 상한 안으로 되돌렸다. 예측 플롯을 다시 뽑아 test 구간에서 두 곡선이 실제선 아래에 있는 것을 눈으로 재확인했다 — 셀 25 문장의 근거다.

---

## 라운드 1 기록 (아래는 초회 저작 시점의 내용)

---

## 리더 §6 판정 반영 확인

| 항목 | 반영 |
|------|------|
| 6-1 | 명세 §1·§2 를 기준선으로 그대로 사용 |
| 6-2 | $T, H, t, \mathbf{x}_t, \mathbf{h}_t, \mathbf{h}_0, \mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{b}_{xh}, \mathbf{b}_{hh}$ 명세대로 도입 (셀 11 표) |
| 6-3 | `nn.MSELoss()` 를 보정 없이 사용 (셀 14, 15) |
| 6-4 | "many to many" 표현 미사용. Model A/B 는 "last hidden state" / "all hidden states" |
| 6-5 | `def to_sequences(x, y, sequence_length)` (셀 7) |
| 6-6 | `num_layers = 1`. 층 쌓기는 Exercises 1번 |
| 6-7 | 출력단 활성 없음. `forward` 가 `y_hat` 을 그대로 반환 |
| 6-8 | 시간순 298/63/65 분할 후 train 통계로만 표준화, `input_train_std[input_train_std == 0] = 1.0` 가드 (셀 10) |
| 6-9 | BPTT/vanishing 본문 제외. Exercises 2번 (`sequence_length` 40) |
| 6-10 | 로컬 `data/kospi.csv` 우선, 없으면 GitHub raw URL 다운로드 (셀 4). 로컬 파일 존재로 실행 검증 통과 |
| 6-11 | 아래 참조 — **조정 불필요** |

## 6-11 수렴 결과 (실측)

`epochs = 200`, `rho = 1e-3` 그대로 유지했다. 조정 없음 → §4 통제표 유지.

| epoch | A train MSE | A val MSE | B train MSE | B val MSE |
|------|------|------|------|------|
| 50 | 0.05320 | 0.04391 | 0.05506 | 0.05695 |
| 100 | 0.03967 | 0.03063 | 0.04174 | 0.04938 |
| 150 | 0.03654 | 0.03059 | 0.03601 | 0.05212 |
| 200 | 0.03296 | 0.03245 | 0.03203 | 0.05435 |

초기 train MSE 는 A 0.711 / B 0.639 → 200 epoch 만에 약 1/20 로 내려간다. 발산·평탄 구간 없음.

최종 RMSE (표준화 단위 [지수 포인트]):

```
Model A (last state)  train 0.1796 [  25.2]  val 0.1971 [  27.7]  test 0.4544 [  63.8]
Model B (all states)  train 0.1782 [  25.0]  val 0.2564 [  36.0]  test 0.2613 [  36.7]
```

> 위 숫자는 **라운드 2 의 값**이다. 라운드 1 은 배치평균 근사라 val/test 가 약 10% 낮게 나왔다
> (A val 25.3, A test 60.5, B val 32.7, B test 35.5). verifier V-1 참조.
> 아래 수렴 표의 epoch 로그는 학습 곡선용 배치평균이므로 라운드 2 에서도 그대로다.

- 두 모델의 **train** 오차는 거의 같다 (25.2 vs 25.0 포인트). 격리 변수의 효과는 **val 에서 갈린다** — A 27.7, B 36.0.
- **test 에서는 순서가 뒤집힌다** (A 63.8, B 36.7). 이는 명세 §3.1 이 경고한 외삽 구간이기 때문이다: test 타깃 65개 중 **58개(89.2%)가 train 최댓값 1.350 을 넘는다** (test 범위 0.998 ~ 2.605, train 범위 -4.414 ~ 1.350). val 범위(-1.343 ~ 1.076)는 train 안에 완전히 들어간다.
  → 노트북은 이 뒤집힘을 근거로 우열을 단정하지 않는다. 셀 25·28 마크다운이 "val 이 비교 가능한 유일한 held-out 구간이고, test 는 범위를 벗어난 뒤의 표류를 재는 것" 이라고만 적는다.
- 예측 플롯에서 test 구간 두 곡선이 실제선 아래에 깔리는 것을 **눈으로 확인했다** (정상 결과, §3.1 대로).

## 자체 점검 (판정 아님 — 감사자·검증자 몫)

```
python ../.claude/skills/notebook-convention-audit/scripts/audit_notebook.py "Practice13_RNN_for_Time_Series.ipynb"
  → 후보 없음 (적용 체크포인트 16종, 프로필 정상 인식)

python .claude/scripts/markdown_budget.py "Practice13_RNN_for_Time_Series.ipynb"
  → 0 over cap, 8 above soft cap
```

soft cap 초과 8건(셀 6, 8, 9, 11, 13, 23, 25, 29)은 전부 표 마크업이 큰 셀이며 hard cap(120단어 / 산문 80단어) 이내다.

그림 3장을 PNG 로 뽑아 직접 확인했다 — 시리즈 개관, 학습곡선 1×2, 예측 vs 실제. 역변환 누락·축 오류 없음.

---

## 명세와 다르게 구현한 것 (저자 판단, 검증자 확인 요망)

### A-1. `evaluate` / `train` 반환값 — 명세 §3.3 과 다름

명세 §3.3 은 `evaluate → (loss, y_pred)`, `train → (train_losses, val_losses)` 를 적었다.
리더의 특별 지침("§7b 의 `accuracy` 자리를 이 과제에 맞는 지표로 바꿔 같은 두-함수 구조를 유지")을 우선해 다음으로 구현했다.

```
evaluate(model, loader, device)                                   -> (loss, rmse, y_pred)
train(model, train_loader, val_loader, optimizer, epochs, device) -> (train_losses, train_rmses, val_losses, val_rmses)
```

`rmse` 는 표준화 단위가 아니라 **지수 포인트 환산값**(`math.sqrt(loss) * close_train_std`)이다.
표준화 RMSE 로 두면 `sqrt(loss)` 와 동어반복이라 학생에게 아무것도 더 주지 않기 때문이고, 명세 §2.4 가 정의한 $\text{RMSE}_\text{points}$ 가 바로 이것이다.

- 함수가 모듈 레벨 상수 `close_train_std` 를 읽는다. 이 과목 선례가 있다 — `gen/p09.py:140-143` 의 `unnormalize()` 가 `IMAGENET_MEAN`/`IMAGENET_STD` 를 같은 방식으로 읽는다.
- §7 의 나머지 규칙은 유지: 두 함수만, `criterion` 함수 내부 생성, `model.train()`/`model.eval()`, per-batch mean 누적, `test_loader` 는 학습 루프 밖.
- 이 조정 사유는 `gen/p13.py` 첫 주석 블록(8~10행)에 남겼고 노트북 본문에는 쓰지 않았다.

### A-2. 하이퍼파라미터 정의 위치 — 명세 §3.2 는 "Step 5 에서 한 번만"

`sequence_length` 는 Step 2(윈도우를 만드는 셀)에서, `input_size`·`hidden_size`·`num_layers` 는 Step 4(`nn.RNN` 을 처음 만드는 셀)에서, `batch_size` 는 Step 3(loader 를 만드는 셀)에서 필요하다. 전부 Step 5 로 몰면 앞 셀들이 정의되지 않은 이름을 쓰게 된다.

→ **각 상수를 첫 사용 지점에서 한 번만 정의하고 이후 재정의하지 않는다.** Step 5(셀 16)에는 `rho`·`epochs` 정의와 함께 공유 설정 전체를 `print` 하는 셀을 두어, 두 모델이 같은 값을 쓴다는 것이 한 화면에서 보이게 했다. 통제표(§4)의 요구인 "두 모델이 동일한 값을 재사용"은 그대로 성립한다.

### A-3. 마크다운에서 기호 $B$ 를 쓰지 않음

명세 §1 은 미니배치 크기를 $B$ 로 정의하지만, 이 회차는 모델을 **A / B** 로 부른다. 마크다운에서 $(B, H)$ 라고 쓰면 "Model B" 와 충돌해 읽는 사람이 걸린다.
→ 배치 축은 마크다운에서 기호로 쓰지 않고 코드(`out.size(0)`, `x.size(0)`)로만 드러냈다. 파라미터 수는 $H+1$ / $HT+1$ 로 적어 $B$ 를 피했다.

### A-4. 명세에 없는 셀 2개 추가

| 셀 | 내용 | 이유 |
|----|------|------|
| 9 (md) | DataLoader 4개 표 | 명세 §3.3 의 loader 표를 학생에게 보여줄 자리가 §3 셀 목록에 없었다. Step 3 md(셀 8)에 합치면 단어 상한을 넘는다 |
| 28 (md) | 두 문장 — "val 이 비교 가능한 구간, test 는 범위 밖 표류" | RMSE 출력(셀 27)의 val/test 순서 뒤집힘을 학생이 잘못 읽는 것을 막는다. Summary 에 넣으면 "Summary 에 새 내용 금지" 위반 |

---

## 하네스 갱신 제출

### [트리거 3 — 프로필/문서 선언과 실제의 불일치] generator script 명명이 문서 안에서 충돌한다

- **무엇:** `CLAUDE.md` "Notebook Generator Scripts" 절은 `_gen_p{NN}.py` 라고 쓰는데, 같은 문서의 "디렉터리 레이아웃" 표와 프로필 `notebooks.generator_glob`(`gen/*.py`), 그리고 실제 `gen/` 디렉터리(`p01.py` ~ `p12.py`)는 전부 `gen/p{NN}.py` 다.
- **어디에:** `01_AI-ME_Graduate/CLAUDE.md` "Notebook Generator Scripts" 절.
- **왜:** 한 문서 안에서 두 규칙이 충돌하면 다음 저자가 `_gen_p13.py` 를 루트에 만든다 — 그러면 "루트에는 `Practice*.ipynb` 와 `CLAUDE.md` 만" 규칙까지 함께 깨진다. 나는 리더 지시 덕에 `gen/p13.py` 로 갔지만, 지시가 없었으면 CLAUDE.md 의 명시적 문장을 따랐을 것이다.
- **근거:** 라운드 1, `gen/` 디렉터리 실측 12개 파일 전부 접두사 `_gen_` 없음.

### [트리거 3 — 프로필 선언과 실제의 불일치] 프로필에 `notebooks.generator_prefix` 키가 없다

- **무엇:** 에이전트 정의와 `practice-notebook-authoring` 스킬이 "프로필의 `notebooks.generator_prefix` 를 따라" generator 를 만들라고 지시하는데, `notebook-profile.json` 에는 그 키가 없다 (`generator_glob` 만 있다).
- **어디에:** `01_AI-ME_Graduate/.claude/notebook-profile.json` 에 `generator_prefix` 추가, 또는 스킬/에이전트 쪽 문구를 `generator_glob` 으로 정정.
- **왜:** 선언된 키를 찾지 못하면 저자가 다른 과목의 관행이나 추측으로 파일명을 정하게 된다. 이번에는 리더가 경로를 직접 줘서 막혔다.
- **근거:** 라운드 1, 프로필 `notebooks` 블록 실측.

### [트리거 1 — 규약에 없거나 불명확] 회귀 회차의 `train`/`evaluate` 시그니처가 `code-patterns.md` 에 없다

- **무엇:** `code-patterns.md` §7(b) 는 "이 단계부터는 분류로 고정" 을 전제해 `(loss, accuracy, y_pred)` 하나만 제공한다. 회귀 회차가 오면 매번 저자가 즉석에서 변형하게 된다.
- **어디에:** `01_AI-ME_Graduate/.claude/references/code-patterns.md` §7 에 (c) 회귀판을 추가.
- **왜:** 이번에 명세(§3.3)와 리더 지침이 서로 다른 변형을 제시했다 — 문서에 정본이 없기 때문이다. 다음 시계열/회귀 회차에서 같은 질문이 다시 올라온다.
- **근거:** 라운드 1, 명세 §3.3 vs 리더 특별 지침. 제안하는 정본:
  ```python
  def evaluate(model, loader, device):
      # returns (loss, metric, y_pred)   metric: 회차의 물리 단위 지표 (예: RMSE in original units)

  def train(model, train_loader, val_loader, optimizer, epochs, device):
      # returns (train_losses, train_metrics, val_losses, val_metrics)
  ```
  분류의 `accuracy` 자리를 회귀의 물리 단위 지표가 그대로 채우므로 §7(b) 와 형태가 같다.

### [트리거 1 — 규약에 없거나 불명확] "하이퍼파라미터를 한 곳에서 정의" 와 "첫 사용 지점에서 정의" 의 관계

- **무엇:** `CLAUDE.md` "비교 실험 통제 변수" 는 "위에서 한 번만 정의하고 모든 방법이 재사용" 을 요구한다. 실제로 통제에 필요한 것은 **"한 번만 정의하고 재정의하지 않는다"** 이지 "한 셀에 모아 둔다" 가 아니다. 데이터 파이프라인 상수(`sequence_length`, `batch_size`)는 물리적으로 앞 셀에서 필요하다.
- **어디에:** `CLAUDE.md` "비교 실험 통제 변수" 절에 한 줄.
- **왜:** 명세서(§3.2)가 "Step 5 에서 한 번만 정의" 로 적었고 나는 그대로 따를 수 없었다. 규약이 의도를 명시하지 않으면 명세 작성자와 저자가 매 회차 어긋난다.
- **근거:** 라운드 1, 명세 §3.2 vs 노트북 셀 7·10·12·16.

### [트리거 4 — generator script 함정] 셀 병합 시 `# Cell N` 주석 재번호가 수작업이 된다

- **무엇:** 셀 하나를 합치거나 쪼개면 그 아래 모든 `# ---- Cell N ----` 주석이 어긋난다. 이번에 정규식 일괄 치환으로 20개를 옮겼다.
- **어디에:** `.claude/skills/practice-notebook-authoring/references/generator-script-mechanics.md` §1 "셀 구분 주석".
- **왜:** 주석이 어긋난 채로 두면 감사 리포트의 "셀 23" 을 script 에서 못 찾는다 — 그 주석을 두는 목적 자체가 무너진다.
- **근거:** 라운드 1, 셀 9/11 병합 시 Cell 12~32 → 11~31 재번호.
- **제안:** 번호를 손으로 적는 대신 `md()`/`code()` 안에서 `len(cells)` 를 찍게 하거나, 생성 직후 `# Cell N` 주석과 실제 인덱스가 맞는지 확인하는 한 줄을 script 끝에 두는 형태를 문서에 예시로 넣는다.
