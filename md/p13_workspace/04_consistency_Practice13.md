# Consistency Verification — Practice13_RNN_for_Time_Series.ipynb (라운드 2)

## 판정: PASS
- 라운드 1 의 **V-1 · V-2 · V-3 전부 해소 확인** (수치·출력·그림으로 직접 검증)
- 선택사항이던 **O-1 · U-2 도 해소**
- 남은 것: V-4 (S-2 승인 대기 — 저자가 올바르게 보류), V-5 · V-6 (명세 쪽 정정 대상, 노트북 무관)
- 신규 발견 1건 (**N-1** — 보고되지 않은 `x` → `X` 전면 개명. **결함 아님, 개선.** 명세 §1 정정 필요)
- 회귀 0건

---

# 라운드 2 검증 (변경 범위 한정)

리더 지시대로 전체 재검증이 아니라 변경 범위와 지정된 4개 확인 항목에 한정했다.
단, 회귀 여부를 확인하기 위한 기계 검사(정규식 스윕 · 하이퍼파라미터 전수 · 감사 스크립트)는 전 셀에 돌렸다.

## R2-0. 실제 변경된 셀 — 보고와 대조

라운드 1 덤프와 라운드 2 덤프를 셀 단위로 diff 했다.

```
실제 변경: 코드 7, 10, 12, 14, 15, 19, 22, 26, 27 / 마크다운 6, 13, 23, 25, 29
리더 보고: 코드 7, 10, 11, 12, 14, 15, 18, 19, 21, 22, 26, 27 / 마크다운 6, 13, 23, 25, 29
```

- **셀 11 · 18 · 21 은 실제로는 내용이 바뀌지 않았다** (보고에는 변경으로 적혔다). 무해한 과다 보고다.
- 마크다운 변경 셀은 보고와 정확히 일치한다.
- **보고에 없던 변경이 하나 있다 → N-1 (`x` → `X` 전면 개명).**

## R2-1. 확인 항목 (1) — V-1 의 새 RMSE 가 split 전체를 올바르게 반영하는가 → **확인됨**

**코드 (셀 27, 신규):**
```python
val_rmse_last_state = math.sqrt(((y_pred_val_last_state - y_val) ** 2).mean())
```

**구조 검증 — 브로드캐스팅 함정 확인.** 이 형태에서 가장 흔한 사고는 `y_pred` 가 `(N,)`, `y` 가 `(N,1)` 일 때
차가 `(N,N)` 으로 브로드캐스팅되어 **아무 에러 없이 완전히 틀린 값**이 나오는 것이다. 직접 확인했다:

```
y_pred shape (63, 1)   y_val shape (63, 1)   diff shape (63, 1)
```

`evaluate` 가 `torch.cat(preds)` 로 `(N,1)` 을 만들고 `y_val` 도 셀 10 에서 `(N,1)` 로 유지되므로 안전하다.
또 셀 10 이 `y_val` 을 **재바인딩**으로 표준화하므로 `y_pred`(표준화 공간)와 같은 공간에서 비교된다 — 단위 불일치 없음.

**수치 검증 (a) — 노트북 자체 출력의 내부 정합성.** 셀 27 이 인쇄한 표준화 RMSE 에
셀 10 이 인쇄한 `close_train_std = 140.46` 을 곱해 괄호 안 지수 포인트와 대조했다.

| | 표준화 RMSE × 140.46 | 셀 27 괄호 | |
|---|---|---|---|
| A train | 0.1796 → 25.23 | `[  25.2]` | 일치 |
| A val | 0.1971 → **27.68** | `[  27.7]` | 일치 |
| A test | 0.4544 → **63.83** | `[  63.8]` | 일치 |
| B train | 0.1782 → 25.03 | `[  25.0]` | 일치 |
| B val | 0.2564 → **36.01** | `[  36.0]` | 일치 |
| B test | 0.2613 → **36.70** | `[  36.7]` | 일치 |

**수치 검증 (b) — 배치평균과의 차이가 예측대로인가 (같은 실행 안에서).**
노트북은 이제 최종 배치평균 수치를 인쇄하지 않지만, **epoch 200 로그가 배치평균 val RMSE 를 남긴다.**
같은 모델·같은 split 이므로 직접 대조할 수 있다.

| | 배치평균 (셀 19·22 epoch 200 로그) | split 전체 (셀 27) | 차이 |
|---|---|---|---|
| A val | `val RMSE=24.7 points` | `[  27.7]` | **+12.1%** |
| B val | `val RMSE=32.1 points` | `[  36.0]` | **+12.1%** |

**수치 검증 (c) — 독립 재실행.** 라운드 2 코드(seed 재설정 포함)를 임시 스크립트로 재현해
두 수치를 같은 실행 안에서 동시에 측정했다.

```
A  train bm 0.1802[25.3] true 0.1799[25.3] -0.2% | val bm 0.1772[24.9] true 0.1984[27.9] +12.0% | test bm 0.4322[60.7] true 0.4612[64.8] +6.7%
B  train bm 0.1802[25.3] true 0.1797[25.2] -0.3% | val bm 0.2295[32.2] true 0.2583[36.3] +12.6% | test bm 0.2584[36.3] true 0.2762[38.8] +6.9%
```

- **노트북의 보정 폭(+12.1%)이 내 독립 측정(+12.0% / +12.6%)과 일치한다.**
- train 은 −0.2% (298 = 14×20 + 18 로 거의 균등) → 라운드 1 예측과 동일.
- test 는 +6.7~6.9% → 라운드 1 에서 내가 측정한 +5~6% 와 같은 방향·크기.
- **셀 27 이 인쇄하는 값은 진짜 split 전체 RMSE 다. 배치평균이 아니다.** 확인됨.

**마크다운 반쪽도 고쳐졌다.** 셀 13 이 이제
"`loss` averages one mean squared error per batch, which is what the learning curves plot" /
"The last batch of a split is shorter than the others, so Step 8 recomputes the reported figures from
`y_pred` over each split as a whole" 로 서술한다. **코드가 계산하는 것과 정의문이 일치한다** —
라운드 1 V-1 의 핵심 지적("마크다운 정의 ↔ 코드 불일치")이 해소됐다.
"마지막 배치가 더 짧다" 는 서술도 세 split 전부 참이다 (train 18, val 3, test 5).

**저자 보고의 수치 짝에 대한 주의 (결함 아님).** 저자가 적은 "25.3→27.7, 60.5→63.8, 32.7→36.0, 35.5→36.7" 에서
앞의 값들은 **라운드 1 실행**의 것이다. O-1 seed 변경으로 학습 자체가 달라졌으므로 이 짝은 같은 실행의
before/after 가 아니다. **효과 자체는 위 (b)(c) 로 같은 실행 안에서 확인했으므로 결론에 영향은 없다.**

## R2-2. 확인 항목 (2) — 셀 25 가 셀 10 의 실제 인쇄값과 맞는가 → **정확히 맞는다**

| 셀 25 문장 | 셀 10 이 실제로 인쇄하는 것 | 판정 |
|---|---|---|
| "Step 3 counted **58 of its 65** targets" | `test windows above the training maximum: 58 of 65` | **축자 일치** |
| "above the training maximum of **1.350**" | `standardized target range - train [-4.414, 1.350]` | **축자 일치** |
| "The test window is **mostly** an extrapolation." | 58/65 = 89.2% | 일치 ("everything" → "mostly") |
| "The validation targets stay inside the training range." | `val [-1.343, 1.076]` vs `train [-4.414, 1.350]` | 일치 (학생이 직접 대조 가능) |
| "so both prediction curves sit below the actual line there" | 셀 26 그림 — 직접 열어 확인 | 일치 |

- 라운드 1 의 거짓 전칭명제("run above **everything** the training window contained")가 사라졌다.
- 외삽 기준선을 2.6 이 아니라 **train 최대 1.350** 으로 바로잡았다 — 라운드 1 지적의 두 번째 절반도 해소.
- **이제 이 문단의 모든 수가 노트북이 스스로 인쇄한 값의 인용이다.** 학생이 검증할 수 있다.
- 내 독립 재계산(라운드 1): train max z = 1.350, z > 1.350 인 test 58/65 → **인쇄값과 일치.**

## R2-3. 확인 항목 (3) — 셀 10 의 val 범위가 내 실측과 일치하는가 → **완전 일치**

```
standardized target range - train [-4.414, 1.350]
standardized target range - val   [-1.343, 1.076]      ← 신규
standardized target range - test  [0.998, 2.605]
test windows above the training maximum: 58 of 65      ← 신규
```

| 항목 | 내 독립 실측 (라운드 1, CSV 에서 재계산) | 노트북 인쇄 | 판정 |
|---|---|---|---|
| val z 범위 | **[-1.343, 1.076]** | `[-1.343, 1.076]` | **완전 일치** |
| train z 범위 | [-4.414, 1.350] | `[-4.414, 1.350]` | 일치 |
| test z 범위 | [0.998, 2.605] | `[0.998, 2.605]` | 일치 |
| train max 초과 test | **58 / 65** | `58 of 65` | **완전 일치** |
| val 이 train 범위 밖 | 0 / 63 | (범위 인쇄로 학생이 확인 가능) | 일치 |

**V-3 해소.** 셀 28 의 결론("val 은 비교 가능한 유일한 held-out 구간")이 이제 학생이 직접 읽을 수 있는
숫자 위에 선다. 라운드 1 에서는 내가 CSV 를 재계산하고서야 참임을 확인할 수 있었다.

## R2-4. 확인 항목 (4) — 결론이 새 수치로도 성립하는가 → **성립한다**

**노트북 자체 출력 (셀 27):**

| | val | test |
|---|---|---|
| A (last state) | **27.7** | 63.8 |
| B (all states) | 36.0 | **36.7** |

- **val: A 우세 (27.7 < 36.0)** ✓
- **test: 뒤집힘, B 우세 (36.7 < 63.8)** ✓
- 보정 전(라운드 1)의 A val 25.3 / B val 32.7 / A test 60.5 / B test 35.5 와 **우열 방향이 동일**하다.
  RMSE 정의를 고쳤어도 §4 비교 실험의 결론은 바뀌지 않았다 — 라운드 1 에서 예측한 대로다.

**독립 재실행에서도 재현:** A val 27.9 < B val 36.3, A test 64.8 > B test 38.8. **방향 동일.**

**그림으로도 확인 (PNG 추출 후 직접 열어 봄):**
- 셀 24 우측 val 패널: A(파랑)가 epoch 40 이후 B(빨강) 아래에 안정적으로 깔린다 → 셀 27 의 val 수치와 일치.
- 셀 26: test 구간(초록선 우측)에서 두 곡선 모두 실제선 아래, **B(빨강)가 A(파랑)보다 실제선에 가깝다**
  → 셀 27 의 test 수치와 일치. **그림과 숫자가 같은 이야기를 한다.**

**셀 28(미변경)은 새 수치에서도 여전히 참이다** — "val 이 비교 가능한 구간, test 는 범위를 벗어난 뒤의
표류를 재는 것". 통제 실행 없는 인과를 인쇄하지 않는 자세도 유지됐다.

## R2-5. 신규 발견 — N-1. 보고되지 않은 `x` → `X` 전면 개명 (결함 아님, 개선)

**리더의 변경 보고에 없던 변경이다.** diff 로 발견했다.

```
x_seq → X_seq,  x_all → X_all,  x_train/x_val/x_test → X_train/X_val/X_test,
x_batch → X_batch,  x_windows → X_windows,  x_window → X_window
(셀 6 마크다운 표, 7, 10, 12, 14, 15, 18, 21)
```

**판정: 정합성이 오히려 좋아졌다.**
- 명세 §1 은 설계행렬을 $\mathbf{X}$ (대문자)로 정의한다. 개명 후 코드가 그 기호와 대소문자까지 맞는다.
- **`x_t` 는 소문자로 남았다** (셀 12: `x_t = X_window[0, t]`). 벡터 $\mathbf{x}_t$ 는 소문자, 행렬 $\mathbf{X}$ 는
  대문자라는 수학 관례와 정확히 일치한다. **개명이 기계적이지 않고 의미를 구분했다.**
- `gen/p09.py` 의 `X_batch, Y_batch` 선례와도 맞는다.
- **누락 확인:** 소문자 `x_seq|x_all|x_train|x_val|x_test|x_batch|x_windows|x_window` 정규식 스윕 → **0 hit.**
  마크다운 표(셀 6)도 `X_seq` 로 갱신됐다. **개명이 완결되었고 stale 참조가 없다.**

**단, 명세 §1 의 코드 변수명 열이 이제 어긋난다** (`x_seq` → `X_seq` 등) → **S-7 (아래).**

## R2-6. O-1 · U-2 · 회귀 확인

**O-1 (seed) — 해소.** 셀 19·22 가 각각
```python
torch.manual_seed(42)   # 모델 생성용
...
torch.manual_seed(42)   # train_loader shuffles from the same state for both models
```
구조가 되었다. 검증: A·B 모두 (i) 새 seed-42 스트림에서 `self.rnn` 을 먼저 생성하므로 **초기 recurrent
가중치 동일**, (ii) `train()` 직전 다시 42 로 리셋하므로 **셔플 순서 동일**.
라운드 1 에서 지적한 "B 의 `fc` 가 난수를 더 소비해 셔플이 갈린다" 가 사라졌다.
**셀 23 통제표도 "Seed before construction **and before training** | 42 | 42" 로 정직하게 갱신됐다** —
표와 코드가 어긋나지 않는다.

**U-2 (다운로드 경로) — 부분 해소.** 저자가 SHA-256 로 로컬 파일과 원격 파일의 동일성을 확인했다고 보고했다.
**나는 이 해시 대조를 재수행하지 않았다** (라운드 1 에서 URL 200 OK · 34124 bytes · 헤더/첫 행 일치는
직접 확인했다). 저자 보고를 신뢰하되 **내가 검증한 것과 구분해 기록한다.**

**회귀 스윕 — 0건.** 전 셀 정규식 검사:

```
many-to-many 0 / Sigmoid 0 / fit_transform 0 / sklearn 0 / assert·allclose 0 / 1e-8 0
gridspec 0 / docstring(""" ) 0 / train(...test_loader) 0 / 한글 0 / 이모지 0
```

하이퍼파라미터 대입 전수 재확인 — **전부 정확히 한 번씩만, 재정의 없음:**
`sequence_length`(7), `input_size`(10), `batch_size`(10), `hidden_size`(12), `num_layers`(12), `rho`(16), `epochs`(16).
→ §4 통제표 유지.

셀 12 의 by-hand vs `nn.RNN` 대조 출력은 라운드 1 과 **동일한 값**이다 (개명은 값에 영향 없음). 재귀식 정합성 유지.
셀 26 의 `close_actual = y_seq[:, 0]` 도 그대로 — `y_seq` 는 여전히 표준화되지 않는다 (제자리 연산 없음 재확인).

기계 검사 (참고, 감사자 영역):
```
audit_notebook.py      → 후보 없음 (체크포인트 16종)
markdown_budget.py     → 0 over cap, 9 above soft cap
```
저자의 "hard cap 초과 해소" 보고와 일치한다.

## R2-7. 남은 항목

| 항목 | 상태 |
|---|---|
| **V-4** ($J$·갱신식·"Adam adapts…" 부재, $\rho$ 가 셀 23 에 정의 없이 등장) | **미해소 — 의도적.** S-2 승인 대기 중이라는 라운드 1 권고를 저자가 정확히 따랐다. 셀 23 은 여전히 $\rho$ 를 정의 없이 쓴다 |
| **V-5** ($t$ off-by-one) | 미해소 — **노트북 무관.** 명세 §1 정정 대상 (S-3) |
| **V-6** (URL 형태) | 기능상 해소 (라운드 1 에서 200 OK 확인). 명세 §6-10 의 URL 수명 미결은 그대로 |
| **O-2** (`train_rmses`/`val_rmses` 미사용) | 유지. 셀 26 이 `_, _, y_pred` 로 `evaluate` 의 `rmse` 마저 버리게 되어 **오히려 더 유휴해졌다.** 결함은 아니다 — `evaluate` 의 `rmse` 는 epoch 로그에서 여전히 쓰인다 |
| **O-3** ($H$ ↔ `hidden_size` 마크다운 미연결) | 유지. V-4 와 함께 해소될 항목 |
| **O-4 (신규)** | epoch 200 로그는 `val RMSE=24.7 points`(배치평균), 셀 27 은 `[27.7]`(전체). 같은 모델·같은 split 인데 숫자가 다르다. **셀 13 이 그 이유를 두 곳보다 앞서 설명하므로 적절하다** — 조치 불필요. 기록만 남긴다 |

## R2-8. 명세 정정 요청 추가

라운드 1 의 S-1 ~ S-6 은 유효하다. **S-1 은 이번 수정으로 노트북이 먼저 움직인 상태이므로 승인이 시급하다** —
지금 명세 §3.3 과 노트북이 어긋나 있다 (노트북이 옳고 명세가 낡았다).

| # | 명세 위치 | 요청 |
|---|---|---|
| **S-7 (신규)** | **§1 Notation 표의 코드 변수명 열** | `x_seq` → `X_seq`, `x_seq[i, t]` → `X_seq[i, t]` 로 갱신. 노트북이 N-1 로 대문자 개명을 마쳤고 그것이 $\mathbf{X}$ 표기와 더 잘 맞으므로 **명세를 노트북에 맞추는 방향이 옳다.** 벡터 `x_t` 는 소문자 유지 |
| **S-1 (재확인, 시급)** | §3.3 | 노트북 셀 13·27 이 이미 "학습곡선은 배치평균 / 최종 보고는 split 전체" 로 분리했다. 명세가 아직 "per-batch mean 을 더해 `len(loader)` 로 나눈다" 만 말한다 → **명세를 갱신하지 않으면 다음 라운드 저자가 되돌린다** |
| **S-8 (신규)** | §4 통제표 seed 행 | "모델 생성 **직전** 재설정" → "모델 생성 직전 **및 `train()` 호출 직전**". O-1 수정이 반영된 형태를 정본으로 남길 것 |

---

# 라운드 1 기록 (이하 원문 보존)

## 판정: PASS
- 불일치 6건 (치명 0 / 경미 6) / 미검증 2건
- 관찰 3건 (불일치 아님)
- **PASS 이지만 학생 배포 전 V-1·V-2·V-3 은 고치는 것이 좋다.** 셋 다 수식·부호·전치의 문제가 아니라
  "마크다운의 단정문을 노트북 출력이 뒷받침하지 못한다" 는 문제이며, V-1 은 인쇄되는 숫자가 최대 11% 어긋난다.

> **라운드 2 갱신:** V-1 · V-2 · V-3 · O-1 · U-2 는 해소되었다. 아래 내용은 라운드 1 시점의 기록이다.

---

## 0. 이번 회차의 검증축 — 표준 축은 성립하지 않는다

`notebook-profile.json` 의 `lecture_sources` 를 실측했다 (분석자 진술을 그대로 믿지 않고 직접 확인).

```
lecture_notes/  →  "Ch1-ML 1_Linear Regression.pdf"  (PDF 는 이것 하나뿐)
md/lectures_and_formulas.md    → RNN|Recurrent|LSTM|GRU|hidden state  0 hit
md/practice_outline_ref.md     → 0 hit
```

**따라서 프로필이 `authority: "slides"` 로 지정한 원본이 이 회차에는 실재하지 않는다.**
"강의 PDF ↔ 노트북" 대조는 수행하지 않았다 (§미검증 U-1).
리더 지시에 따라 다음 두 축으로 검증했다.

| 축 | 왼쪽 (기준선) | 오른쪽 (구현) |
|---|---|---|
| 축 1 | `_workspace/01_spec_Practice13.md` §1 Notation · §2 수식 · §3~§5 | 노트북 셀 |
| 축 2 | `_archive/01_AI-ME_Graduate_backup/Chapter3_Deep Learning_3_RNN.ipynb` (28셀, 읽기 전용) | 노트북 셀 |

축 1 의 기준선은 강의자료에서 추출된 것이 아니라 **명세 작성자가 표준 vanilla RNN 정의를 이 과목
notation 으로 새로 세운 것**이다 (명세 §6-1 이 스스로 밝힘). 이 사실은 아래 모든 "일치" 판정의
효력 범위를 규정한다 — **"이 과목 강의와 일치한다" 가 아니라 "승인된 명세와 일치한다" 이다.**

---

## 1. 대조표 — 축 1 (명세 §1·§2 ↔ 노트북)

### 1-1. Notation (명세 §1)

| # | 기호 | 명세 §1 (shape) | 노트북 (셀) | 판정 |
|---|------|----------------|------------|------|
| 1 | $N$ | 윈도우 수, `len(x_seq)` | 셀 6 md "$N = 431 - 5 = 426$" / 셀 7 출력 `x_seq (426, 5, 4)` | 일치 |
| 2 | $d$ | 입력 특징 4, `input_size` | 셀 10 `input_size = x_seq.shape[2]` → 4 (셀 7 출력) | 일치 |
| 3 | $T$ | 시퀀스 길이 5, `sequence_length` | 셀 6 md "$T = 5$" / 셀 7 `sequence_length = 5` | 일치 |
| 4 | $H$ | hidden 차원 8, `hidden_size` | 셀 11 md "$\mathbf{h}_t \in \mathbb{R}^{H}$" / 셀 12 `hidden_size = 8` / 셀 23 "$H = 8$" | 일치 (O-3 참조) |
| 5 | $t$ | $t = 1 \dots T$, 루프 변수 `t` | 셀 12 `for t in range(sequence_length)` → `t = 0..4` | **V-5 경미** |
| 6 | $\mathbf{X}$ | $(N,T,d)$ = (426,5,4), `x_seq` | 셀 7 출력 `x_seq (426, 5, 4)   (N, T, d)` | 일치 |
| 7 | $\mathbf{x}_t$ | $(d,)$ = (4,), `x_seq[i, t]` | 셀 12 `x_t = x_window[0, t]` → (4,) | **V-5 경미** (1-based vs 0-based) |
| 8 | $\mathbf{y}$ | $(N,1)$ = (426,1), `y_seq` | 셀 7 출력 `y_seq (426, 1)   (N, 1)` | 일치 |
| 9 | $\hat{\mathbf{y}}$ | $(N,1)$, `y_hat` | 셀 14·15·18·21 `y_hat` | 일치 |
| 10 | $\mathbf{h}_t$ | $(H,)$ = (8,) / 배치는 $(B,H)$ | 셀 12 `h = torch.zeros(hidden_size)` → (8,) | 일치 |
| 11 | $\mathbf{h}_0$ | $(\text{num\_layers}, B, H)$ | 셀 12 `torch.zeros(num_layers, 1, hidden_size)` → (1,1,8); 셀 18·21 `torch.zeros(self.num_layers, x.size(0), self.hidden_size)` | 일치 |
| 12 | $\mathbf{W}_{xh}$ | $(H,d)$ = (8,4), `weight_ih_l0` | 셀 11 표 `(H, d) / weight_ih_l0`; 셀 12 `weight_ih = demo_rnn.weight_ih_l0` | 일치 |
| 13 | $\mathbf{W}_{hh}$ | $(H,H)$ = (8,8), `weight_hh_l0` | 셀 11 표 `(H, H)`; 셀 12 `weight_hh` | 일치 |
| 14 | $\mathbf{b}_{xh},\mathbf{b}_{hh}$ | $(H,)$ 각각, **두 개** | 셀 11 표 "$(H,)$ each" + "A textbook writes one bias where PyTorch keeps two; the unrolled loop below adds both"; 셀 12 가 실제로 둘 다 더함 | 일치 |
| 15 | $\mathbf{w}$ (readout) | A $(1,H)$ / B $(1,HT)$ | 셀 18 `nn.Linear(hidden_size, 1)` / 셀 21 `nn.Linear(hidden_size * sequence_length, 1)` | 일치 |
| 16 | $b$ (readout bias) | $(1,)$ | `nn.Linear(..., 1)` 기본 bias | 일치 |
| 17 | $\rho$ | 학습률 1e-3, `rho`, `lr=rho` | 셀 16 `rho = 1e-3`; 셀 19·22 `optim.Adam(..., lr=rho)` | 일치 (마크다운 정의 부재 → **V-4**) |
| 18 | $B$ | 배치 20, `batch_size` | 셀 10 `batch_size = 20`; 마크다운에서 $B$ 미사용 (저자 A-3) | 일치 (A-3 정당, §3 참조) |
| 19 | 금지 기호 | `theta`/`alpha`/`eta`/`step_size`/`lr=` 리터럴 | 전 셀 0 hit | 일치 |

### 1-2. 수식 (명세 §2)

| # | 항목 | 명세 §2 (출처: `01_spec_Practice13.md`) | 노트북 (셀) | 판정 |
|---|------|--------------------------------------|------------|------|
| 20 | 재귀식 (§2.1) | $\mathbf{h}_t = \tanh(\mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_{xh} + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_{hh})$, $\mathbf{h}_0=\mathbf{0}$ | 셀 11 md — **문자 단위로 동일** (항 순서·bias 2개 포함) | 일치 |
| 21 | 재귀식 → 코드 (§2.1) | `h = torch.tanh(weight_ih @ x_t + bias_ih + weight_hh @ h + bias_hh)` | 셀 12 — **문자 단위로 동일** | 일치 |
| 22 | 재귀식 → 라이브러리 | `out, h_n = self.rnn(x, h0)` | 셀 18·21 `out, h_n = self.rnn(x, h0)` | 일치 |
| 23 | 손계산 ↔ `nn.RNN` 대조 | §3 "수식 손계산 vs `nn.RNN` 출력 나란히 `print`", `allclose`/`assert` 금지 | 셀 12 `print('by hand :', ...)` / `print('nn.RNN  :', ...)`; assert·allclose 0 hit | 일치 |
| 24 | Readout A (§2.2 A) | $\hat{y} = \mathbf{w}^{\top}\mathbf{h}_T + b,\ \mathbf{w}\in\mathbb{R}^{H}$ | 셀 17 md 동일; 셀 18 `self.fc(out[:, -1, :])` | 일치 |
| 25 | Readout B (§2.2 B) | $\hat{y} = \mathbf{w}^{\top}[\mathbf{h}_1;\dots;\mathbf{h}_T] + b,\ \mathbf{w}\in\mathbb{R}^{HT}$ | 셀 20 md 동일; 셀 21 `self.fc(out.reshape(out.size(0), -1))` | 일치 (배열 순서 확인, 아래) |
| 26 | 손실 $J$ (§2.3) | $J = \frac{1}{B}\sum(\hat y_n - y_n)^2$, `/2` 없음 | 셀 14·15 `criterion = nn.MSELoss()` (함수 내부 생성) | 코드 일치 / **수식 마크다운 부재 → V-4** |
| 27 | 갱신식 (§2.3) | $\mathbf{w} \leftarrow \mathbf{w} - \rho\,\partial J/\partial\mathbf{w}$ + "Adam adapts this step size per parameter" 한 줄 | 노트북에 `leftarrow` 0 hit, "adapts" 0 hit | **V-4 경미 (미구현)** |
| 28 | RMSE (§2.4) | $\text{RMSE}_{\text{points}} = \sqrt{J}\times\sigma_{\text{close,train}}$, 코드 `math.sqrt(mse) * close_train_std` | 셀 14 `rmse = math.sqrt(loss) * close_train_std` | 식 일치 / **$J$ 의 정의가 셀 13 서술과 어긋남 → V-1** |

**#25 배열 순서 확인 (전치·축 검증).** `out` 은 `batch_first=True` 이므로 $(B, T, H)$ 다.
`out.reshape(out.size(0), -1)` 은 시간축이 바깥, hidden 축이 안쪽으로 펼쳐지므로
$[\mathbf{h}_1; \mathbf{h}_2; \dots; \mathbf{h}_T]$ 순서와 정확히 일치한다.
`(B, H, T)` 로 두고 폈다면 $[\;\cdot\;]$ 의 의미가 달라졌을 자리이며, 여기서는 어긋나지 않았다.

**#21 수치 확인 (실제 노트북 출력, 셀 12).**

```
by hand : [ 0.5033927  -0.50471514 -0.09632248  0.00760593 -0.31598416 -0.35692045  0.10617422 -0.13799968]
nn.RNN  : [ 0.5033927  -0.50471514 -0.09632247  0.0076059  -0.31598416 -0.35692045  0.10617425 -0.13799976]
```

float32 반올림 수준(~1e-7)까지 일치. **bias 두 개를 모두 더했다는 것의 실증이며, 하나만 더했다면
여기서 값이 갈렸을 것이다.** 명세 §2.1 의 "주의" 가 요구한 검증이 실제로 성립했다.

### 1-3. 하이퍼파라미터 · 분할 · 데이터 (명세 §3)

| # | 항목 | 명세 §3 | 노트북 (셀 / 실제 출력) | 판정 |
|---|------|---------|------------------------|------|
| 29 | `sequence_length` | 5 | 셀 7 `= 5` (전 노트북 유일 대입) | 일치 |
| 30 | `input_size` | 4 | 셀 10 `= x_seq.shape[2]` (유일) | 일치 |
| 31 | `hidden_size` | 8 | 셀 12 `= 8` (유일) | 일치 |
| 32 | `num_layers` | **1** (§6-6 판정) | 셀 12 `= 1` (유일) | 일치 |
| 33 | `batch_size` | 20 | 셀 10 `= 20` (유일) | 일치 |
| 34 | `rho` | 1e-3 | 셀 16 `= 1e-3` (유일) | 일치 |
| 35 | `epochs` | 200 | 셀 16 `= 200` (유일); 로그가 `Epoch 200/200` 까지 실제로 찍힘 | 일치 (인자↔로그 대조 완료) |
| 36 | optimizer | `optim.Adam` | 셀 19·22 | 일치 |
| 37 | criterion 생성 위치 | `train`/`evaluate` **내부** | 셀 14 `criterion = nn.MSELoss()` / 셀 15 동일 | 일치 |
| 38 | 데이터 431행 | 431, 2019-01-30 ~ 2020-10-30 | 셀 4 출력 `rows: 431, from 2019-01-30 to 2020-10-30` | 일치 |
| 39 | 결측 | 없음 | 셀 4 출력 `missing values: 0` | 일치 |
| 40 | 입력/타깃 열 | `Open, High, Low, Volume` / `Close` | 셀 7 | 일치 |
| 41 | 분할 298/63/65 | §3.1 표 | 셀 10 출력 `train 298, val 63, test 65` | 일치 (내가 독립 재계산: 298/63/65) |
| 42 | 표준화 통계 | train 통계만, sklearn 금지 | 셀 10 `input_train_mean/std`, `close_train_mean/std` 전부 `x_train`/`y_train` 에서 | 일치 |
| 43 | 분할 순서 | 분할 → 표준화 | 셀 10 이 slice 먼저, 통계 계산, 그다음 변환 | 일치 |
| 44 | 0분산 가드 | `input_train_std[... == 0] = 1.0`, `+1e-8` 금지 | 셀 10 그대로, `1e-8` 0 hit | 일치 |
| 45 | `close_train_std` 값 | ≈ 140.5 (§3.1 실측 참고) | 셀 10 출력 `close_train_std 140.46` | 일치 |
| 46 | test 표준화 범위 | 0.998 ~ 2.605 (§3.1 경고) | 셀 10 출력 `test [0.998, 2.605]` | 일치 (내가 독립 재계산: 동일) |
| 47 | DataLoader 4개 | train(True)/train_eval(False)/val(False)/test(False) | 셀 10 네 개 모두, `shuffle` 값 일치 | 일치 |
| 48 | `test_loader` 격리 | `train(...)` 인자로 금지 | 셀 19·22 는 `val_loader` 전달; `test_loader` 는 셀 26 에서만 | 일치 |
| 49 | epoch 로그 솎기 | `% 50 == 0` | 셀 15 그대로, 출력 4줄 | 일치 |
| 50 | 함수 이름 | `to_sequences` (§6-5 판정) | 셀 7 `def to_sequences(x, y, sequence_length)` | 일치 |
| 51 | `train`/`evaluate` 반환 | §3.3 은 `(loss, y_pred)` / `(train_losses, val_losses)` | 셀 14 `(loss, rmse, y_pred)` / 셀 15 4-tuple | **의도된 변경 (A-1) — 정당, §3 참조** |
| 52 | 하이퍼파라미터 정의 위치 | §3.2 "Step 5 에서 한 번만" | 셀 7·10·12·16 에 분산 | **의도된 변경 (A-2) — 정당, §3 참조** |

### 1-4. 시각화 (명세 §5) — PNG 를 뽑아 직접 확인

| # | 항목 | 명세 §5 | 노트북 (셀) | 판정 |
|---|------|---------|------------|------|
| 53 | 5.1 시리즈 | (12,4), `'KOSPI closing index'`, `'Trading day'`, `'Close'`, `grid(alpha=0.3)` | 셀 5 — 전부 일치 | 일치 |
| 54 | 5.2 학습곡선 | 1×2 (12,4), A `'b-'` / B `'r--'`, `'Epoch'`, `'MSE (standardized)'` | 셀 24 — 전부 일치 | 일치 |
| 55 | 5.3 예측 | (14,5), actual `'k--'`, A `'b-'` lw1, B `'r-'` lw1, `axvline` 2개 `':'`, `'Prediction vs actual (index points)'`, `'KOSPI Close'` | 셀 26 — 전부 일치 | 일치 |
| 56 | 5.3 역변환 | `y * close_train_std + close_train_mean` | 셀 26 그대로. `close_actual = y_seq[:, 0]` | 일치 (아래 확인) |
| 57 | 5.4 수치 출력 | scaled 와 points 를 `print` 나란히, 표 금지, assert 금지 | 셀 27 두 줄 `print` | 일치 |
| 58 | `gridspec` 금지 | — | 0 hit | 일치 |

**#56 별첨 — 별칭(aliasing) 확인.** 셀 10 은 `y_train = y_seq[:n_train]` 로 **뷰**를 만든 뒤
`y_train = (y_train - ...) / ...` 로 **재바인딩**한다 (`-=`, `/=` 같은 제자리 연산 없음).
따라서 `y_seq` 는 원 지수 단위로 남고, 셀 26 의 주석 "y_seq was never standardized" 는 참이다.
제자리 연산이었다면 셀 26 의 실제선이 표준화 값으로 그려지고 예측선만 지수 포인트가 되어
그림이 조용히 무너졌을 자리다 — 어긋나지 않았다.

**그림 3장 직접 확인 (PNG 추출 후 열어 봄, "생성되었다" 로 통과시키지 않음).**
- 셀 5: 2020-03 급락과 08~10 상승이 명확히 보인다. Step 3·8 의 외삽 논의가 그림으로 뒷받침된다.
- 셀 24: 두 곡선이 실선/파선으로 구분되고 겹쳐도 읽힌다. 우측 val 패널에서 A(파랑)가 아래, B(빨강)가
  위로 갈라지는 것이 명확 — 셀 27 의 val 수치와 그림이 같은 이야기를 한다.
- 셀 26: 범례 5항목이 좌하단에 겹침 없이 놓임. 경계선 2개가 각각 298·361 위치. **test 구간(초록선 우측)에서
  두 예측선이 실제선(검정 파선) 아래에 깔리는 것이 눈으로 확인된다** — 셀 25 의 마지막 주장은 그림과 일치한다.

---

## 2. 대조표 — 축 2 (구버전 소스 노트북 ↔ 새 노트북)

소스: `_archive/01_AI-ME_Graduate_backup/Chapter3_Deep Learning_3_RNN.ipynb` (28셀). 수정하지 않았다.

### 2-1. 핵심 개념이 왜곡 없이 이어졌는가

| # | 개념 | 소스 (셀) | 새 노트북 (셀) | 판정 |
|---|------|----------|---------------|------|
| 59 | KOSPI 시계열 예측 | 셀 1 "Vanila RNN을 이용한 Kospi 예측", 셀 4 동일 CSV(431행) | 셀 0 제목 + 셀 4 동일 CSV | 이어짐 |
| 60 | 입력/타깃 열 | 셀 7 `X = [Open,High,Low,Volume]`, `y = Close` | 셀 7 동일 | 이어짐 |
| 61 | sliding window | 셀 9 `for i in range(len(x)-sequence_length)`, `x[i:i+L]` → `y[i+L]` | 셀 7 `to_sequences` 동일 로직 | 이어짐 (윈도우 정의 동일) |
| 62 | `sequence_length = 5` | 셀 10 | 셀 7 | 이어짐 |
| 63 | `hidden_size = 8`, `batch_size = 20` | 셀 12·13 | 셀 10·12 | 이어짐 |
| 64 | VanillaRNN 구조 | 셀 16 `nn.RNN(..., batch_first=True)` + `h0 = torch.zeros(num_layers, x.size()[0], hidden_size)` | 셀 18·21 동일 구조 | 이어짐 |
| 65 | 전체 hidden state flatten readout | 셀 16 `out.reshape(out.shape[0], -1)` + `nn.Linear(hidden_size*sequence_length, 1)` | 셀 21 `RNNAllStates` — **동일** | 이어짐 (= Model B) |
| 66 | Adam, MSELoss, 200 epoch, lr 1e-3 | 셀 19 | 셀 14·15·16 | 이어짐 |
| 67 | 예측 vs 실제 + 분할 경계선 플롯 | 셀 24 `plotting()` | 셀 26 (계산은 `evaluate`, 플롯은 플롯 셀로 분리) | 이어짐 + 구조 개선 |
| 68 | Model A (`RNNLastState`) | 소스에 **없음** — 셀 16 주석이 "Many to One 전략" 을 말로만 언급 | 셀 18 로 실제 구현 | **추가** (소스 주석의 미구현 대안을 격리 변수로 승격 — 왜곡 아님) |

**판정: 개념적 왜곡 없음.** 소스가 실제로 구현한 유일한 모델이 새 노트북의 Model B 로 그대로 보존되었고,
소스가 주석으로만 언급하고 구현하지 않은 "마지막 hidden state 만 쓰는" 대안이 Model A 로 구현되어
통제된 비교 실험의 격리 변수가 되었다. 소스의 내용이 빠지거나 뒤집힌 곳은 없다.

### 2-2. 명세가 지적한 소스 결함이 실제로 고쳐졌는가

| # | 소스 결함 (명세 §6 / §7-4) | 소스 근거 (셀) | 새 노트북 | 판정 |
|---|--------------------------|--------------|----------|------|
| 69 | **§6-4** many-to-one 을 "many to many" 로 오기 | 셀 16 주석 3줄 + `# many to many 전략` 2회 | `many to many` 정규식 **0 hit**. 셀 0·17·20·23 이 "last hidden state" / "all hidden states" | **고쳐짐** |
| 70 | **§6-7** 회귀 출력단 `nn.Sigmoid()` | 셀 16 `nn.Sequential(nn.Linear(...), nn.Sigmoid())` | `Sigmoid` **0 hit**. 셀 18·21 `forward` 가 `y_hat` 을 그대로 반환 | **고쳐짐** |
| 71 | **§6-8a** 분할 전 전체 데이터 `fit_transform` (leakage) | 셀 6 `scaler.fit_transform(df[[...]])` | `fit_transform` **0 hit**, `sklearn` **0 hit**. 셀 10 이 분할 → train 통계 → 변환 순서 | **고쳐짐** |
| 72 | **§6-8b** `split = 200` (test 226 > train 200) | 셀 11 | 셀 10 시간순 298/63/65, val 신설 | **고쳐짐** |
| 73 | §6-6 `num_layers = 2` | 셀 13 | 셀 12 `num_layers = 1`; 2층은 Exercise 1 | **고쳐짐** |
| 74 | §7-4 `lr = 1e-3` | 셀 19 | 셀 16 `rho = 1e-3`, `lr=rho` | **고쳐짐** |
| 75 | §7-4 MSE 를 "rmse" 로 오표기 | 셀 24 title `train rmse {:.5f}` 에 `mean_squared_error` 값 | 셀 27 은 `math.sqrt(loss)` 를 RMSE 로 표기 (제곱근을 실제로 취함) | **고쳐짐** (단 V-1 참조) |
| 76 | §7-4 코드 셀 docstring | 셀 9 `'''...'''` | 셀 7 은 `#` 주석 | **고쳐짐** |
| 77 | §7-4 `plotting()` 이 예측·지표·플롯을 한 함수에 묶고 `model` 을 전역 캡처 | 셀 24 | 셀 26 이 `evaluate` 호출 → 역변환 → 플롯으로 분리, 모델은 인자 | **고쳐짐** |
| 78 | §7-4 전체 한국어 | 전 셀 | 노트북 한글 0 hit (감사 스크립트 `hangul` 후보 없음) | **고쳐짐** |

**소스 결함 10종 전부 새 노트북에서 해소되었다. 되살아난 것은 없다.**

---

## 3. 저자 리포트의 "명세와 다르게 구현한 것" 4건 판정

리더가 명시적으로 요청한 항목이다. **저자의 진술을 그대로 믿지 않고 노트북·명세·선례를 각각 확인했다.**

### A-1. `train`/`evaluate` 시그니처 + RMSE 를 지수 포인트로 — **정당 (조건부)**

- **RMSE 단위 주장의 검증:** 명세 §2.4 는 문자 그대로
  $\text{RMSE}_{\text{points}} = \text{RMSE}_{\text{scaled}} \times \sigma_{\text{close,train}}$ 을 정의하고
  코드 대응을 `rmse_points = math.sqrt(mse) * close_train_std` 로 적었다.
  노트북 셀 14 는 `rmse = math.sqrt(loss) * close_train_std` — **식이 §2.4 와 동일하다.**
  → **저자의 "명세 §2.4 가 정의한 것이 바로 이것" 이라는 주장은 참이다.** 임의 변경이 아니다.
- 변수명만 다르다 (`rmse_points` → `rmse`). 셀 13 md 와 셀 27 헤더가 단위를 명시하므로 오해 소지는 없다.
- **선례 주장의 검증:** `gen/p09.py:139-143` 의 `unnormalize()` 가 실제로 `IMAGENET_STD`/`IMAGENET_MEAN` 을
  모듈 레벨에서 읽는다. **저자가 인용한 선례는 실재한다.**
- 두 함수만 유지, `criterion` 내부 생성, `model.train()`/`model.eval()`, `test_loader` 학습 루프 밖 —
  §3.3 의 나머지 규칙은 전부 지켜졌다 (대조표 #37·#48).
- **조건:** 이 변경이 끌고 들어온 결함이 하나 있다 → **V-1**. 그리고 반환된 `train_rmses`/`val_rmses` 는
  이후 어디에서도 읽히지 않는다 → **O-2**.

### A-2. 하이퍼파라미터 정의 위치 — **정당**

- 명세 §3.2 의 문언("Step 5 에서 한 번만 정의")을 따르면 셀 7 의 `to_sequences(x, y, sequence_length)` 와
  셀 10 의 `DataLoader(..., batch_size=batch_size)` 가 정의되지 않은 이름을 참조한다. **물리적으로 불가능하다.**
- **§4 통제표가 실제로 성립하는지 직접 확인:** 정규식으로 각 이름의 대입문을 전수 조사했다.

  | 이름 | 대입 셀 | 재정의 |
  |------|--------|-------|
  | `sequence_length` | 7 | 없음 |
  | `input_size` | 10 | 없음 |
  | `batch_size` | 10 | 없음 |
  | `hidden_size` | 12 | 없음 |
  | `num_layers` | 12 | 없음 |
  | `rho` | 16 | 없음 |
  | `epochs` | 16 | 없음 |

  **전부 정확히 한 번씩만 대입된다.** 두 모델(셀 19·22)이 같은 이름을 그대로 재사용하므로
  §4 통제표의 실질("두 모델이 동일한 값을 쓴다")은 완전히 성립한다.
- 셀 16 이 7개 값을 한 화면에 `print` 하므로 "한 곳에서 보인다" 는 §3.2 의 의도도 달성된다.
- → 통제를 해치지 않는 정당한 조정. 저자의 하네스 갱신 제안(CLAUDE.md 에 "한 번만 정의" 와 "한 셀에 모음" 을
  구분해 적을 것)에 **동의하며 지지한다.**

### A-3. 마크다운에서 $B$ 미사용 — **정당**

- 셀 0·17·20·23 이 모델을 "A"/"B" 로 부른다. 같은 문서에서 $(B, H)$ 를 쓰면 충돌이 실재한다.
- 노트북 마크다운에서 $B$ 는 0 hit, 파라미터 수는 셀 25 가 $HT + 1 = 41$ / $H + 1 = 9$ 로 적어 $B$ 를 피했다.
- 배치 축은 셀 18·21 의 `x.size(0)` / `out.size(0)` 으로 드러난다. 명세 §4 의 `out.reshape(B, -1)` 이
  `out.reshape(out.size(0), -1)` 이 된 것도 같은 이유이며 수학적으로 동일하다.
- → 정당. 다만 이 조정과 **무관하게** $\rho$ 가 정의 없이 셀 23 에 등장한다 → **V-4**.

### A-4. 명세에 없는 셀 2개 추가 — **정당 (셀 28 은 조건부)**

- **셀 9 (DataLoader 표):** 명세 §3.3 이 이 표를 명시했으나 §3 셀 목록에 자리를 주지 않았다.
  명세 내부의 누락을 메운 것이며 내용은 §3.3 표와 일치한다. → 정당.
- **셀 28 (val/test 해석 2문장):** 학생의 오독을 막는다는 목적은 타당하고, Summary 로 옮기면
  "Summary 에 새 내용 금지" 를 위반한다는 판단도 옳다. **그러나 이 셀의 핵심 주장을 뒷받침하는
  출력이 노트북에 없다** → **V-3**.

---

## 4. val↔test 뒤집힘 — 저자 주장의 독립 검증 (리더 질의 2)

**저자의 진술을 믿지 않고, 노트북의 실제 출력과 내가 재계산·재실행한 결과로만 판정했다.**

### 4-1. 노트북이 실제로 인쇄한 수치 (셀 27)

```
Model A (last state)  train 0.1799 [  25.3]  val 0.1801 [  25.3]  test 0.4311 [  60.5]
Model B (all states)  train 0.1787 [  25.1]  val 0.2331 [  32.7]  test 0.2526 [  35.5]
```

val 은 A 우세(25.3 < 32.7), test 는 B 우세(35.5 < 60.5). **뒤집힘은 실재한다.**

### 4-2. 외삽 주장의 검증 — 내가 CSV 에서 독립 재계산

| 구간 | 표준화 타깃 범위 | 원 지수 범위 | train 최대(z=1.350) 초과 비율 |
|------|-----------------|-------------|---------------------------|
| train | **[-4.414, 1.350]** | 1457.64 ~ 2267.25 | — |
| val | **[-1.343, 1.076]** | 1889.01 ~ 2228.83 | **0 / 63 = 0%** |
| test | **[0.998, 2.605]** | 2217.86 ~ 2443.58 | **58 / 65 = 89.2%** |

- train·test 범위는 노트북 셀 10 출력(`train [-4.414, 1.350], test [0.998, 2.605]`)과 **정확히 일치**한다.
- **val 범위 [-1.343, 1.076] 은 train 범위 안에 완전히 들어간다 (0% 초과).**
  → 셀 28 의 "validation ... is the only held-out stretch that stays inside the range the models were
  fitted on" 은 **참이다.** 다만 노트북이 이 숫자를 인쇄하지 않는다 → **V-3**.
- test 는 89.2% 가 train 최대를 넘는다. → "test 구간은 외삽" 이라는 명세 §3.1 의 경고는 **참이다.**
- 그러나 **10.8% (7/65) 는 train 범위 안에 있다.** → 셀 25 의 "run above **everything** the training
  window contained" 는 **거짓이다** → **V-2**.

### 4-3. 뒤집힘이 재현되는가 — 독립 재실행

노트북과 동일한 파이프라인을 임시 스크립트로 재구성해 200 epoch × 2 모델을 다시 돌렸다
(같은 seed, 같은 device=cuda; cuDNN 비결정성 때문에 완전 동일값은 기대하지 않는다).

| | A val | B val | A test | B test |
|---|---|---|---|---|
| 노트북 (셀 27) | 0.1801 | 0.2331 | 0.4311 | 0.2526 |
| 내 재실행 | 0.1764 | 0.2327 | 0.4299 | 0.2431 |

**val 우세(A) / test 우세(B) 의 방향이 그대로 재현된다.** 우연한 한 번의 실행이 아니다.

### 4-4. 논리 판정

**노트북이 하는 주장과 하지 않는 주장을 구분해 보았다.**

- 셀 28 은 **어느 모델이 낫다고 단정하지 않는다.** "val 은 비교 가능한 구간, test 는 범위를 벗어난 뒤의
  표류를 재는 것" 이라고만 적는다. → 통제 실행 없이 인과를 인쇄하지 않았다. **적절하다.**
- 셀 25 의 "so both prediction curves sit below the actual line there" 는 그림(셀 26)으로 직접 확인했다.
  **참이며 근거가 있다.**
- **저자 리포트의 "외삽 때문이라고 주장" 은 리포트 안의 설명이지 노트북 본문의 단정이 아니다.**
  노트북 본문은 "test 에서 B 가 이긴 것은 외삽 때문" 이라고 쓰지 않았다. 만약 썼다면 그 변수만 바꾼
  통제 실행이 없으므로 검증 불가 주장이 되었을 것이다. **저자가 그 선을 넘지 않은 것은 옳은 판단이다.**
- 다만 **"test 는 외삽" 이라는 사실 자체는 셀 10 출력으로 뒷받침되고, "val 은 내삽" 이라는 짝은
  뒷받침되지 않는다.** 논증의 절반만 인쇄되어 있다 → **V-3**.

**결론: val/test 뒤집힘에 대한 노트북의 서술은 실제 출력·실제 데이터 범위·재실행 결과와
논리적으로 맞아떨어진다. 단, 근거 인쇄가 절반 빠졌고(V-3) 표현 하나가 과장되었다(V-2).**

---

## 5. 불일치 상세

### V-1. 셀 27 이 "RMSE" 로 인쇄하는 값은 split 전체 RMSE 가 아니다 (최대 11% 낮음) — 경미

- **명세:** §2.4 `RMSE_scaled = sqrt(J)`; 셀 13 마크다운은
  "`loss` is **the mean squared error on standardized targets**" 라고 서술한다.
- **노트북:** 셀 14 —
  ```python
  loss_sum += criterion(y_hat, y_batch).item()
  loss = loss_sum / len(loader)
  rmse = math.sqrt(loss) * close_train_std
  ```
- **무엇이 다른가 — 분모.** `loss` 는 split 전체의 MSE 가 아니라 **배치별 MSE 의 산술평균**이다.
  배치 크기가 균등할 때만 둘이 같다. 실제로는:

  | split | 배치 크기 | 마지막 배치의 실제 지분 | 코드가 주는 가중치 |
  |-------|----------|---------------------|-----------------|
  | val (63) | 20,20,20,**3** | 4.76% | **25%** (5.25배 과대) |
  | test (65) | 20,20,20,**5** | 7.69% | **25%** (3.25배 과대) |

- **수치 확인 (수행함).** 같은 파이프라인을 재실행해 배치평균 RMSE 와 split 전체 RMSE 를 동시에 계산했다.

  | | 인쇄되는 값 (배치평균) | 참값 (split 전체) | 오차 |
  |---|---|---|---|
  | A val | 0.1764 [24.8 pt] | 0.1978 [**27.8** pt] | **−10.8%** |
  | B val | 0.2327 [32.7 pt] | 0.2612 [**36.7** pt] | **−10.9%** |
  | A test | 0.4299 [60.4 pt] | 0.4589 [**64.5** pt] | **−6.4%** |
  | B test | 0.2431 [34.2 pt] | 0.2564 [**36.0** pt] | **−5.2%** |
  | A train | 0.1800 [25.3 pt] | 0.1797 [25.2 pt] | −0.2% (298 = 14×20+18, 거의 균등) |

- **학생에게 미치는 영향:** 셀 27 이 "val RMSE = 25.3 index points" 라고 인쇄하지만 실제 값은 약 28 포인트다.
  기계공학 대학원생이 "이 오차가 물리량으로 얼마나 큰가" 를 읽으라고 넣은 숫자가 11% 어긋난다.
  또 셀 13 의 정의문("the mean squared error on standardized targets")이 코드가 계산하는 것과 다르다 —
  **이 역할이 존재하는 이유인 "마크다운 정의 ↔ 코드" 불일치에 정확히 해당한다.**
  **다만 A/B 의 우열 순서는 val·test 모두 바뀌지 않으므로 §4 비교 실험의 결론은 무사하다.**
- **원인은 저자가 아니라 명세다.** §3.3 이 "loss 누적은 per-batch mean 을 더해 `len(loader)` 로 나눈다.
  `loss.item() * len(batch)` 금지" 를 명령했고 저자는 그대로 따랐다. → §6 명세 정정 요청 참조.
- **수정 방향 (저자):** epoch 모니터링(셀 15)은 §3.3 대로 두고, **최종 지표만** 셀 26 에 이미 있는
  `y_pred_*` 와 `y_*_tensor` 로 직접 계산한다. 새 셀 불필요:
  ```python
  # 셀 26 에서 y_pred_test_last_state 와 y_test 는 이미 손에 있다
  test_rmse_last_state = np.sqrt(((y_pred_test_last_state - y_test) ** 2).mean()) * close_train_std
  ```
  또는 셀 13 마크다운을 "the average of the per-batch mean squared errors" 로 정직하게 고친다
  (값은 그대로 두되 정의를 코드에 맞춘다). **전자를 권한다** — 후자는 학생에게 설명하기 어려운 지표를 남긴다.

### V-2. 셀 25 의 "run above everything the training window contained" 가 셀 10 출력과 모순 — 경미

- **노트북 (셀 25):** "The test window is an extrapolation. Its closing levels **run above everything the
  training window contained**, and a standardized target above 2.6 is a level the model was never fitted on"
- **노트북 (셀 10, 같은 노트북의 출력):** `standardized target range - train [-4.414, 1.350], test [0.998, 2.605]`
- **무엇이 다른가 — 값.** test 최소 0.998 < train 최대 1.350 이다. 두 범위는 겹친다.
  내 재계산: **test 65개 중 7개(10.8%)가 train 범위 안에 있다.** "everything" 은 거짓이다.
- **두 번째 문제 — 기준선이 잘못 잡혔다.** "above 2.6" 을 외삽의 기준으로 제시하는데, 실제 기준은
  train 최대인 **1.350** 이다. 2.6 을 넘는 것은 test 최대점 하나뿐이지만, **1.350 을 넘는 것은 58개(89.2%)** 다.
  즉 이 문장은 외삽의 범위를 실제보다 **훨씬 좁게** 들리게 한다.
- **학생에게 미치는 영향:** 셀 10 출력을 읽은 학생이 셀 25 의 "everything" 과 눈앞의 `0.998 < 1.350` 을
  대조하면 문장이 틀렸음을 즉시 안다. 그러면 같은 셀의 나머지(외삽 설명 전체)도 신뢰를 잃는다.
  반대로 대조하지 않은 학생은 "2.6 넘는 부분만 문제" 로 외삽을 축소해 배운다. **양쪽 다 손해다.**
- **수치 확인 (수행함):** 위 표. train max z = 1.350, test 중 z > 1.350 인 것 58/65 (89.2%),
  train 범위 안 7/65 (10.8%).
- **수정 방향:** 두 문장을 사실로 교체. 예 —
  "Almost nine test windows in ten sit above the highest level the training window ever reached
  (a standardized target of 1.35), so the model is asked to predict levels it was never fitted on."
  숫자는 셀 10 이 인쇄하는 `train [..., 1.350]` 과 직접 이어진다.

### V-3. 셀 28 의 val 범위 주장을 뒷받침하는 출력이 노트북에 없다 — 경미

- **노트북 (셀 28):** "The validation column is where the two readouts can be compared, because it is
  **the only held-out stretch that stays inside the range the models were fitted on.**"
- **노트북 (셀 10):** train 과 **test** 범위만 인쇄한다. **val 범위는 노트북 어디에도 인쇄되지 않는다.**
  ```python
  print(f'standardized target range - train [{y_train.min():.3f}, {y_train.max():.3f}], '
        f'test [{y_test.min():.3f}, {y_test.max():.3f}]')
  ```
- **무엇이 다른가:** 주장의 절반(test 는 밖)만 출력으로 뒷받침되고, 결정적인 절반(val 은 안)은 없다.
- **수치 확인 (수행함): 주장 자체는 참이다.** val z 범위 [-1.343, 1.076], train [-4.414, 1.350] →
  **0/63 이 train 범위 밖.** 완전히 내삽 구간이다.
- **학생에게 미치는 영향:** 셀 28 은 이 노트북에서 학생이 결론을 내리는 문장이다("val 로 비교하라, test 로
  비교하지 마라"). 그 근거가 인쇄되지 않으면 학생은 저자의 말을 믿는 것 외에 할 일이 없다.
  **셀 25 가 test 를 숫자로 설명한 것과 대비되어 더 눈에 띈다.**
- **수정 방향:** 셀 10 의 `print` 에 val 한 조각을 넣는다. **한 줄 변경이고 새 셀이 필요 없다.**
  ```python
  print(f'standardized target range - train [{y_train.min():.3f}, {y_train.max():.3f}], '
        f'val [{y_val.min():.3f}, {y_val.max():.3f}], '
        f'test [{y_test.min():.3f}, {y_test.max():.3f}]')
  ```
  그러면 셀 28 의 두 문장이 학생이 직접 읽을 수 있는 숫자 위에 서고, V-2 의 수정과 함께
  Step 3 → Step 8 의 논증이 완결된다.

### V-4. 명세 §2.3 의 손실식·갱신식·"Adam" 한 줄이 노트북에 없고, $\rho$ 가 정의 없이 쓰인다 — 경미

- **명세 §2.3:** $J = \frac{1}{B}\sum(\hat y_n - y_n)^2$, $\mathbf{w} \leftarrow \mathbf{w} - \rho\,\partial J/\partial\mathbf{w}$,
  그리고 "마크다운에 'Adam adapts this step size per parameter' **한 줄만 덧붙인다**".
- **노트북:** `leftarrow` 0 hit, "adapts" 0 hit, $J$ 수식 0 hit. 마크다운의 $\rho$ 는 **셀 23 표에 단 한 번**:
  `| $\rho$, epochs, batch size, optimizer, loss | shared values from Step 5 | same |`
- **무엇이 다른가 — 기호.** 노트북이 마크다운에서 정의한 적 없는 기호 $\rho$ 를 셀 23 에서 쓴다.
  $T$·$d$·$N$·$H$ 는 전부 도입 지점이 있는데(셀 6·11) $\rho$ 만 없다.
- **학생에게 미치는 영향:** 학생은 셀 16 의 `rho = 1e-3` 과 `Learning rate : 0.001` 출력에서 역추적해야
  $\rho$ = 학습률임을 안다. 이 과목 CLAUDE.md 를 아는 학생에게는 자명하지만, **노트북은 독립적으로
  읽혀야 한다**는 것이 이 과목 규약이다. 또 손실이 무엇인지 수식으로 한 번도 적히지 않아
  Step 5 의 `nn.MSELoss()` 가 §2.3 의 어느 식인지 대응시킬 자리가 없다.
- **원인의 일부는 명세다.** §2.3 은 마크다운 한 줄을 요구했지만 §3 셀 목록의 어느 행에도 그 자리가 없다.
  명세 내부의 §2 ↔ §3 불일치다. → §6 참조.
- **수정 방향:** 셀 13 (Step 5 마크다운)에 두 줄. 단어 예산 안에 들어간다.
  ```markdown
  $$J = \frac{1}{|\text{batch}|}\sum_n (\hat{y}_n - y_n)^2, \qquad
    \mathbf{w} \leftarrow \mathbf{w} - \rho\,\frac{\partial J}{\partial \mathbf{w}}$$

  $\rho$ is the learning rate; Adam adapts this step size per parameter.
  ```
  (V-1 을 함께 고친다면 $J$ 의 분모 표기를 그때 확정하는 것이 좋다.)

### V-5. 명세 §1 의 $t = 1 \dots T$ 와 코드 루프 `t` 의 off-by-one — 경미 (명세 쪽 문제)

- **명세 §1:** `| $t$ | 시퀀스 내 시점 인덱스, $t = 1 \dots T$ | (루프 변수 `t`) |`,
  `| $\mathbf{x}_t$ | 한 윈도우의 $t$ 번째 시점 입력 | `x_seq[i, t]` |`
- **노트북 (셀 12):** `for t in range(sequence_length)` → `t = 0..4`, `x_t = x_window[0, t]`
- **무엇이 다른가 — 인덱스 기준.** 수학의 $\mathbf{x}_1$ 은 코드의 `x_window[0, 0]` 이다.
  명세가 선언한 대응 `$\mathbf{x}_t$ ↔ x_seq[i, t]` 는 문자 그대로는 한 칸 어긋난다.
- **학생에게 미치는 영향: 작다.** 수학 1-based / Python 0-based 는 보편적이고, 셀 12 의 루프는
  올바른 결과를 낸다(#21 수치 대조 통과). 셀 11 의 수식과 셀 12 의 출력
  `out (1, 5, 8) holds h_1 .. h_T` 도 서로 모순되지 않는다.
- **수치 확인 (수행함):** 셀 12 의 by-hand 결과가 `nn.RNN` 과 1e-7 이내로 일치 → 인덱싱은 실제로 옳다.
- **수정 방향: 노트북은 손대지 않는다.** 명세 §1 의 대응 칸을 `x_seq[i, t-1]` 로 적거나
  "코드 인덱스는 0-based" 라는 단서를 붙이는 것으로 충분하다. → §6 명세 정정 요청.

### V-6. 데이터 URL 형태가 명세와 다르고, 다운로드 분기가 실행되지 않았다 — 경미 (내가 대신 검증함)

- **명세 §데이터 조달:** `https://raw.githubusercontent.com/jongmoonha/AI-ME-Practice/refs/heads/main/data/kospi.csv`
- **노트북 (셀 4):** `https://raw.githubusercontent.com/jongmoonha/AI-ME-Practice/main/data/kospi.csv`
  (`refs/heads/` 없음)
- **무엇이 다른가 — 값(경로).** GitHub raw 는 두 형태를 모두 허용하므로 기능상 동등하다.
- **실행 검증에서 이 줄은 한 번도 돌지 않았다.** 셀 4 출력에 `downloading kospi.csv ...` 가 없다 —
  로컬 `data/kospi.csv` 가 이미 있어 `if not os.path.exists(...)` 가 건너뛰어졌다.
  **저자의 "실행 검증 통과" 는 이 분기를 포함하지 않는다.** 학생은 반대로 이 분기만 탄다.
- **수치 확인 (수행함):** 노트북에 적힌 URL 을 직접 열었다 → `200 OK, 34124 bytes`,
  첫 줄 `Date,Open,High,Low,Close,Adj Close,Volume`, 둘째 줄이 셀 4 출력의 2019-01-30 행과 일치.
  **URL 은 살아 있고 올바른 파일을 준다.**
- **수정 방향:** 노트북 수정 불필요. **명세 §6-10 의 미결(URL 수명)은 여전히 열려 있다** —
  외부 저장소 의존이라는 사실은 그대로다. 리더 판단 대상.

---

## 6. 명세서 정정 요청

**리더에게 먼저 보고한다. 나는 명세를 직접 고치지 않았다.**
아래는 전부 "노트북만 고치면 다음 라운드에 같은 오류가 재생산되는" 항목이다.

| # | 명세 위치 | 문제 | 요청 |
|---|----------|------|------|
| S-1 | **§3.3** "loss 누적은 per-batch mean 을 더해 `len(loader)` 로 나눈다. `loss.item() * len(batch)` 금지" | 이 규칙을 **최종 지표 보고에까지** 적용하면 인쇄되는 RMSE 가 참값보다 5~11% 낮다 (V-1, 수치 확인 완료). 규칙 자체는 epoch 모니터링에는 타당하다 (배치 손실의 평균이라는 의미가 분명하고 구현이 단순하다) | "**epoch 로그와 학습 곡선에는 배치평균을, 최종 보고 지표에는 `y_pred` 로 계산한 split 전체 값을 쓴다**" 로 단서를 나눌 것. 이 회차뿐 아니라 균등하지 않은 마지막 배치가 생기는 **모든** 회차에 해당한다 |
| S-2 | **§2.3 ↔ §3** | §2.3 이 마크다운 한 줄("Adam adapts this step size per parameter")과 $J$·갱신식을 요구하는데, §3 셀 목록에는 그 수식을 놓을 행이 없다. 저자가 §3 표를 따르면 §2.3 이 자동으로 미구현된다 (V-4) | §2 의 모든 수식에 대해 **§3 표의 어느 행에 들어가는지** 매핑을 넣을 것. §2 에만 있고 §3 에 없는 수식은 구조적으로 빠진다 |
| S-3 | **§1** $t$ / $\mathbf{x}_t$ 행 | 수학은 $t = 1 \dots T$, 코드 대응은 `x_seq[i, t]` 로 적어 한 칸 어긋난다 (V-5) | `x_seq[i, t-1]` 로 고치거나 "코드 인덱스는 0-based" 단서를 붙일 것 |
| S-4 | **§3.1** 저자 경고 블록 | test 범위(0.998~2.605)만 실측해 경고하고 **val 범위를 적지 않았다.** 그래서 저자가 셀 10 에 train·test 만 인쇄했고, 정작 결론 셀(28)이 기대는 val 범위가 빠졌다 (V-3) | §3.1 표에 **val 표준화 범위 [-1.343, 1.076]** 를 추가하고, "세 구간 범위를 모두 인쇄한다" 를 §3 셀 명세에 명시할 것 |
| S-5 | **§2.4** 변수명 | 명세는 `rmse_points`, 노트북은 `rmse` (값은 동일) | 경미. 명세를 노트북에 맞추거나 반대로. 판정에 영향 없음 |
| S-6 | **§3.2** "Step 5 에서 한 번만 정의" | 문언대로 하면 셀 7·10 이 미정의 이름을 참조해 **실행 불가** (A-2) | "각 상수를 **첫 사용 지점에서 한 번만** 정의하고 재정의하지 않는다. 공유 설정을 한 셀에서 `print` 해 한 화면에 보인다" 로 정정. 저자의 하네스 갱신 제안과 동일한 취지이며 **지지한다** |

**§6 판정을 뒤집을 근거는 발견하지 못했다.** §6-1 ~ §6-11 의 리더 판정 11건은 전부 노트북에
정확히 반영되었고(대조표 #32·#50·#69~#74, 축 2 §2-2), 그중 어느 것도 노트북 출력이나 데이터와
모순되지 않는다. 위 S-1 ~ S-6 은 §6 판정이 아니라 §1·§2.3·§3 의 **서술 정밀도** 문제다.

---

## 7. 관찰 (불일치 아님 — 기록용)

### O-1. A/B 의 배치 셔플 순서가 서로 다르다 (명세 §4 를 그대로 따른 결과)

셀 19·22 는 명세 §4 대로 **모델 생성 직전**에만 `torch.manual_seed(42)` 를 재설정한다.
그런데 B 의 생성이 A 보다 많은 난수를 소비한다 (`fc` 파라미터 41개 vs 9개).
그 결과 학습 루프에 들어갈 때 두 모델의 RNG 상태가 다르고, `train_loader(shuffle=True)` 의
배치 순서가 A 와 B 에서 갈린다.

- **격리 변수의 귀결이 아니라 통제되지 않은 잡음 변수다.** 다만 두 모델의 `nn.RNN` 초기 가중치는
  동일하게 유지되므로(생성 순서상 `self.rnn` 이 먼저) §4 의 핵심 통제는 성립한다.
- 영향은 작다 — 내 재실행에서 우열 방향이 그대로 재현되었다 (§4-3).
- 없애려면 `train(...)` 호출 직전에 한 번 더 `torch.manual_seed(42)` 를 두면 된다 (각 셀 한 줄).
- **명세 §4 가 "모델 생성 직전" 만 규정했으므로 저자의 잘못이 아니다.** 판정하지 않고 리더에게 올린다.
- 이 항목은 `convention-auditor` 의 "통제 안 된 비교 실험" 영역과 겹칠 수 있다. 감사자가 같은 것을
  올렸다면 **저자에게는 하나의 지시만 가야 한다** — 내 입장은 "지금 상태로도 §4 는 성립하며,
  한 줄로 더 좋아지므로 고치면 좋다(필수 아님)" 이다.

### O-2. `train_rmses` / `val_rmses` 가 반환되지만 이후 읽히지 않는다

셀 15 가 4-tuple 을 반환하고 셀 19·22 가 4개 이름에 바인딩하지만, `train_rmses_*` / `val_rmses_*` 는
그 뒤 어디에서도 사용되지 않는다 (셀 24 는 loss 만 그린다).
A-1 이 code-patterns §7(b) 의 `accuracy` 자리를 채우려고 도입한 값이며, 학생에게 보이는 효용은
셀 19·22 의 epoch 로그 `val RMSE=25.3 points` 한 조각뿐이다.
→ 결함은 아니다. 저자의 code-patterns §7(c) 신설 제안이 채택되면 이 형태가 정본이 되므로 그대로 두어도 좋고,
학습 곡선에 RMSE 패널을 추가하면 값이 쓰이게 된다. **판정하지 않는다.**

### O-3. $H$ 와 `hidden_size` 의 대응이 마크다운에 명시되지 않는다

셀 6 은 $T$·$d$·$N$ 을 말로 정의하고 셀 7 이 값을 인쇄한다. $H$ 는 셀 11 이
"a hidden state $\mathbf{h}_t \in \mathbb{R}^{H}$" 로 도입하고 셀 23 이 "$H = 8$" 을 준다.
그러나 코드 이름 `hidden_size` 와 기호 $H$ 를 잇는 문장은 없다 (셀 11 표는 $H$ 를
PyTorch 파라미터 이름에만 잇는다).
바로 다음 셀(12)에 `hidden_size = 8` 이 있어 추론 가능하며, **명세 §1 도 이 대응을 표로만 두었다.**
→ 결함으로 판정하지 않는다. V-4 를 고칠 때 한 단어 덧붙이면 함께 해소된다.

---

## 8. 미검증

| # | 항목 | 사유 |
|---|------|------|
| U-1 | **강의 슬라이드 ↔ 노트북 대조 (이 검증의 표준 축)** | **이 과목에 RNN 강의자료가 존재하지 않는다.** 내가 직접 실측: `lecture_notes/` 에 PDF 1편(`Ch1-ML 1_Linear Regression.pdf`)뿐, `md/lectures_and_formulas.md`·`md/practice_outline_ref.md` 에 `RNN|Recurrent|LSTM|GRU|hidden state` **0 hit**. 프로필의 `authority: "slides"` 가 이 회차에는 공허하다. 따라서 §1 의 모든 "일치" 판정은 **"승인된 명세와 일치" 이지 "이 과목 강의와 일치" 가 아니다.** 명세 §2 의 재귀식·readout 식은 표준 vanilla RNN 정의이며 수학적으로는 옳지만, **실제 강의에서 쓰는 표기와 다를 수 있다.** 이것은 노트북의 결함이 아니라 자료의 부재다 |
| U-2 | 셀 4 의 CSV 다운로드 분기 실행 | 실행 검증 시 로컬 `data/kospi.csv` 가 존재해 `if not os.path.exists(...)` 가 건너뛰어졌다. **분기 자체는 노트북 안에서 검증되지 않았다.** URL 유효성만 내가 별도로 확인했다 (200 OK, 34124 bytes, 헤더·첫 행 일치 — V-6). 로컬 파일을 잠시 치우고 재실행하면 완전 검증이 되지만, 그것은 저자 영역이므로 요청만 남긴다 |

**수치 확인은 모두 수행했다.** 실패한 검증 스크립트는 없다.
사용한 임시 스크립트는 노트북과 generator 에 넣지 않았고 scratchpad 에만 두었다.

---

## 9. 하네스 갱신 제출

`.claude/skills/practice-notebook-harness/references/harness-feedback-loop.md` 규칙에 따라 제출한다.
**나는 파일을 수정하지 않았다. 아래는 전부 승인 대상이다.**

### H-1. [트리거 1 — 규약에 없어 판단 기준이 없음] "마크다운의 범위·분포 단정문은 그 숫자를 인쇄하는 셀이 있어야 한다"

- **무엇:** 이번에 셀 28 이 "val 은 학습 범위 안에 있는 유일한 held-out 구간" 이라고 단정하는데
  그 범위를 인쇄하는 셀이 없었다 (V-3). 반대로 셀 25 는 test 범위를 인쇄된 숫자로 설명했다.
  **같은 노트북 안에서 근거 제시 기준이 달랐다.** 규약에 기준이 없어서다.
- **어디에:** `.claude/skills/lecture-notebook-consistency/SKILL.md` §3 중점 확인 항목, 그리고
  `practice-notebook-authoring` 쪽에도 같은 취지로.
- **왜:** 스킬은 이미 "인자와 로그의 일치" 와 "통제 실행 없는 인과 금지" 를 다루지만,
  **"통계적 사실 주장에 대한 출력 근거"** 항목은 없다. 인과가 아니라 단순 사실이라 기존 항목에 안 걸린다.
  정해두지 않으면 회차마다 절반만 인쇄된 논증이 통과한다.
- **근거:** 라운드 1, 셀 10 출력(train·test 만) vs 셀 28 주장(val 에 관한 것). 주장은 참이었으나
  검증자가 CSV 를 다시 계산하고서야 확인되었다 — **학생은 그럴 수 없다.**
- **제안 문구:** "마크다운이 어떤 구간·분포·범위를 단정하면, 그 수를 인쇄하는 셀이 노트북 안에 있어야 한다.
  세 분할 중 둘만 인쇄하고 셋을 논하지 않는다."

### H-2. [트리거 1 — 규약 불명확] 배치평균 누적 지표를 "RMSE" 라고 부를 수 있는가

- **무엇:** `code-patterns.md` §7(b) 와 명세 §3.3 이 `loss_sum / len(loader)` 를 정본으로 정했는데,
  이 값은 마지막 배치가 작을 때 split 전체 MSE 가 아니다. 그것을 그대로 `sqrt` 해 "RMSE (index points)"
  라는 **물리 단위 지표**로 인쇄하면 최대 11% 어긋난다 (V-1, 수치 확인 완료).
- **어디에:** `01_AI-ME_Graduate/.claude/references/code-patterns.md` §7, 그리고
  `lecture-spec-extraction` 스킬(명세가 §3.3 에 이 규칙을 복사해 넣는 지점).
- **왜:** 분류 회차에서는 `accuracy` 를 같은 방식으로 누적해도 오차가 눈에 안 띄어 지금까지 문제가 없었다.
  **회귀 회차에서 물리 단위로 환산하는 순간 드러난다.** 저자의 §7(c) 신설 제안이 채택되면
  이 함정이 정본에 그대로 새겨진다 — **채택 전에 단서를 붙여야 한다.**
- **근거:** 라운드 1, 재실행으로 측정. val 마지막 배치 3/63 이 25% 가중치를 받아 A val 24.8 vs 참값 27.8 pt.
- **제안:** §7(c) 를 신설하되 "epoch 로그·학습 곡선은 배치평균, **최종 보고 지표는 `y_pred` 전체로 계산**" 을
  함께 적는다. 저자의 제안(A-1 관련)에 이 한 줄을 추가하는 형태를 지지한다.

### H-3. [트리거 3 — 프로필 선언과 실제의 불일치] `lecture_sources` 에 `coverage` 가 없다 — **명세 §7-1 을 재확인하고 지지한다**

- **무엇:** 프로필이 `slides_glob: "lecture_notes/*.pdf"`, `authority: "slides"` 를 선언하지만
  실제 PDF 는 1편뿐이다. **나는 이 사실을 직접 실측해 확인했다** (§8 U-1).
  Practice 02~13 은 대조할 원본이 없다.
- **왜 검증자에게 특히 문제인가:** 이 역할의 존재 이유가 "강의 ↔ 노트북 대조" 인데,
  프로필만 보면 대조 가능한 것처럼 보인다. **`ls lecture_notes/` 를 하고 나서야 검증축이
  성립하지 않는다는 것을 안다.** 이번에는 리더가 미리 알려주어 막혔지만, 알려주지 않았다면
  검증자가 "일치" 판정을 내리고 그것이 "강의와 일치" 로 읽혔을 위험이 있었다.
- **제안:** `lecture_sources.coverage` 를 추가해 슬라이드가 실재하는 범위를 명시하고,
  범위 밖 회차에서는 `authority` 가 "없음(사람 판단)" 이 되게 한다.
  명세 §7-1 의 제안과 동일하며 **독립적으로 같은 결론에 도달했다.**

### H-4. [사용자 보고 — 강의자료 자체의 공백] RNN 강의 슬라이드가 없다

- **하네스 갱신이 아니라 별도 보고 항목이다.** 규칙대로 임의로 처리하지 않는다.
- `lecture_notes/` 에 Ch2(ML)·Ch3(DL) 슬라이드가 전부 없고, `md/` 의 두 요약본은 학부 과목
  요약이라 딥러닝 후반부가 아예 없다 (RNN 관련 0 hit — 내가 직접 확인).
- **결과:** Practice13 의 수식 표기는 이번에 새로 만든 것이며, 실제 강의 표기와 다를 수 있다.
  학생이 강의와 노트북을 나란히 놓았을 때 어긋날 가능성이 남아 있고, **나는 그것을 검증할 수 없다.**
- **근본 해결은 슬라이드를 `lecture_notes/` 에 추가하는 것이다.** 사람이 결정할 사항이다.
- 슬라이드가 들어오면 **§1 의 대조표를 그 기준으로 다시 돌려야 한다** — 지금의 PASS 는
  그때까지의 잠정 판정이다.

### H-5. [요약본 ↔ 원본 불일치] 해당 없음

이 회차는 대조할 원본 슬라이드가 없어 요약본과의 충돌을 판정할 기회 자체가 없었다.
프로필의 `authority_note` 가 경고하는 "손실함수 상수 2 처리" 충돌도 이 회차에서는 발화하지 않았다
(§6-3 판정대로 `nn.MSELoss()` 를 보정 없이 쓰며, 손 유도 기울기 셀이 없어 대조 대상이 아니다).

---

## 10. `notebook-author` 에게 보내는 수정 요청 (합의된 단일 지시)

우선순위 순. **셋 다 generator `gen/p13.py` 수정 → 재생성 → 재실행이며, 새 셀은 필요 없다.**

1. **V-3 (한 줄):** 셀 10 의 `print` 에 val 표준화 범위를 추가한다.
   → 셀 28 의 결론 문장이 학생이 읽을 수 있는 숫자 위에 선다.
2. **V-2 (두 문장 교체):** 셀 25 의 "run above everything the training window contained" 와
   "above 2.6" 을 실측에 맞게 고친다 — 기준선은 **train 최대 z = 1.350**, 초과 비율 **89.2%**.
   1번을 먼저 하면 이 문장이 셀 10 출력을 직접 인용할 수 있다.
3. **V-1 (셀 26 에 두 줄, 또는 셀 13 마크다운 한 문장):** 최종 보고 RMSE 를 `y_pred` 전체로 계산한다.
   epoch 로그(셀 15)는 §3.3 대로 배치평균을 유지한다.
   **S-1 이 리더 승인을 받은 뒤에 착수할 것** — 지금 고치면 명세 §3.3 과 노트북이 어긋난 상태가 된다.
4. **V-4 (선택, 두 줄):** 셀 13 에 $J$·갱신식과 "$\rho$ is the learning rate; Adam adapts this step size
   per parameter." 를 넣는다. **S-2 승인 후.** O-3 도 함께 해소된다.
5. **U-2 (실행 확인만):** 로컬 `data/kospi.csv` 를 잠시 옮기고 셀 4 를 한 번 돌려 다운로드 분기를 확인한다.
   URL 자체는 내가 200 OK 로 확인했다 (V-6). 노트북 수정 불필요.

**V-5·V-6 은 노트북을 고치지 않는다** — 명세 쪽 정정 대상이다 (S-3, §6-10 미결).
**O-1 은 필수가 아니다** — 감사자와 겹치면 "고치면 좋으나 §4 는 지금도 성립" 으로 하나의 지시를 보낸다.
