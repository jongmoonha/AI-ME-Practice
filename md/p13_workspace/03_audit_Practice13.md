# Convention Audit — Practice13_RNN_for_Time_Series.ipynb (라운드 2)

## 판정: FAIL
- BLOCK 0건 / FIX 1건 / NOTE 3건

라운드 1 의 FIX 2건(B-1, B-2)은 **둘 다 해소**됐다. 남은 FIX 1건은 라운드 2 수정 과정에서
**새로 들어온 코드 주석 한 줄**이며, 주석이라 재실행이 필요 없다 (generator 재생성만 하면 출력 불변).

> **셀 번호 체계 정정.** 라운드 1 리포트는 1-based 로 셀 번호를 적었으나,
> `audit_notebook.py`·`markdown_budget.py`·저자 리포트·리더 메시지는 **전부 0-based** 다
> (두 스크립트 모두 `enumerate(cells)` 에 `start=1` 이 없다).
> 라운드 1 의 모든 셀 번호는 실제보다 1 컸다. **이 리포트부터 0-based 로 통일한다.**
> 라운드 1 에서 저자가 수정을 정확히 찾아간 것은 내가 항상 원문을 함께 인용했기 때문이며,
> 번호만 믿었다면 어긋났을 것이다. 하네스 갱신에 올린다.

이번 라운드는 리더 지시대로 **변경 범위 한정 재확인**이다
(code 7,10,11,12,14,15,18,19,21,22,26,27 / markdown 6,13,23,25,29).
범위 밖 셀은 라운드 1 판정을 그대로 유지한다.

---

## 해소 확인 — 라운드 1 지적

### B-1. 입력 텐서 변수명 대문자화 — **해소**

`X_seq`, `X_all`, `X_train/X_val/X_test`, `X_train_tensor` 계열, `X_batch`, `X_window`,
`to_sequences(X, y, sequence_length)` 까지 일괄 반영됐다. 마크다운 셀 6 의 표도 `X_seq` 로 함께 고쳤다
(코드만 고치고 마크다운을 놓치는 것이 흔한 누락인데 걸리지 않았다).

전수 검색 결과 소문자로 남은 것은 **승인된 두 곳뿐**이다.

| 위치 | 원문 | 판정 |
|------|------|------|
| 셀 12 | `x_t = X_window[0, t]` | **타당.** 마크다운 셀 11 의 $\mathbf{x}_t$ 와 1:1 대응한다. `CLAUDE.md` "Notation" 이 행렬은 대문자($\mathbf{X}$), 벡터는 소문자로 두는 체계이므로, $(d,)$ 벡터인 `x_t` 는 소문자가 **오히려 규약에 맞다** |
| 셀 18, 21 | `def forward(self, x):` | **타당.** 선례 확인함 — `gen/p07.py:171`, `gen/p07.py:383`, `gen/p09.py:337` 전부 `def forward(self, x)`. 메서드 지역 파라미터라 호출부의 `X_batch` 와 충돌하지 않는다 |

부수 효과로 셀 12 가 더 좋아졌다 — `X_window`(3-D 텐서, 대문자) 에서 `x_t`(1-D 벡터, 소문자) 를 꺼내는
구조라 대소문자가 수식의 차원 구분을 그대로 드러낸다.

### B-2. Summary 첫 불릿 — **해소**

- 수정 후 (셀 29): `so window length leaves its parameter count unchanged — only the readout of B grows with it.`
- 셀 25 의 `The readout of B carries $HT + 1 = 41$ weights` 와 더 이상 부딪히지 않는다.
  오히려 뒷절이 셀 25 를 가리켜 두 셀이 서로를 보강한다.
- 셀 30 Exercise 2 (`Set sequence_length to 40`) 를 Model B 로 수행한 학생이 `nn.Linear(320, 1)` 을
  보더라도 Summary 가 이미 그렇게 예고한 상태다.
- "Summary 에 새 내용 금지" 도 함께 해소 — 결론이 셀 25 본문에 근거를 갖게 됐다.
- `its` 의 선행사는 주어인 "A recurrent layer" 로 자연스럽게 읽힌다.

### C-1(라운드 1 NOTE). seed 재설정 — **반영됐고, 실제로 작동한다**

셀 19·22 에 `torch.manual_seed(42)   # train_loader shuffles from the same state for both models` 가
학습 호출 **직전**에 추가됐다. 위치가 정확하다 — 모델·optimizer 생성이 끝난 뒤라
readout 크기가 소비한 난수 차이가 여기서 상쇄된다. 재현 실험으로 확인했다.

```
A first batch [210, 35, 137, 191, 124, 14, 34, 68]
B first batch [210, 35, 137, 191, 124, 14, 34, 68]
identical order? True      (라운드 1: False)
```

셀 23 통제표도 `Seed before construction and before training` 으로 함께 갱신됐다 — 표가 코드와 일치한다.
이제 A/B 차이는 readout 입력 **하나뿐**이며, 셀 29 의 `What separates their curves is the readout input`
가 조건 없이 참이 됐다. 주석 문구는 "이 줄이 무엇을 하는가" 를 설명하므로 `meta-comment` 위반이 아니다.

### C-5(라운드 1 NOTE). 배치 평균 편향 — **해소** (verifier V-1)

셀 27 이 `y_pred` 로부터 분할 전체에 대해 RMSE 를 다시 계산한다. 짧은 마지막 배치(val 3개, test 5개)가
20개짜리와 같은 가중치를 받던 문제가 사라졌다. `evaluate`/`train` 의 누적 방식은
`code-patterns.md` §7(b) 요구대로 per-batch mean 을 유지했고, **보고용 수치만** 분리해 다시 잰
구조라 규약과 정확성을 동시에 지켰다. 셀 13 마크다운이 이 이중 구조를 학생에게 설명한다.

**형상 검증(조용한 브로드캐스트 사고 확인).** `y_pred`(298,1) − `y_train`(298,1) = (298,1) 로
원소별 연산이 맞다. 만약 `y_pred` 가 1-D 였다면 (298,298) 로 퍼져 **에러 없이 틀린 RMSE** 가 나왔을
자리다. 실측으로 (298,1) 임을 확인했고, 출력값(train 0.1796 ≈ sqrt(train MSE 0.03303)=0.1817)도
정상 범위다.

### V-2 / V-3 (verifier 지적, 참고 확인)

내 감사 범위 밖이지만 같은 변경 범위라 사실만 대조했다 — **둘 다 노트북 출력과 일치**한다.

| 셀 25 서술 | 셀 10 실제 출력 |
|---|---|
| `58 of its 65 targets above the training maximum of 1.350` | `test windows above the training maximum: 58 of 65`, train 범위 `[-4.414, 1.350]` |
| `The validation targets stay inside the training range.` | val `[-1.343, 1.076]` ⊂ train `[-4.414, 1.350]` |

`mostly an extrapolation` 으로 한정어가 붙은 것도 65개 중 58개라는 실측과 맞다.

---

## FIX

### D-1. 셀 27 의 새 주석이 "다른 방식을 안 쓴 이유" 를 적는다 — 1건 (라운드 2 신규)

- 위치: 셀 27 (code), 첫 줄
- 원문: `# over a whole split at once, rather than averaging the per-batch means the loaders produce`
- 위반 규약: `CLAUDE.md` "Notebook Comments — 내부 지침 / 메타 코멘트 금지"
  > 코드 주석과 마크다운은 **"이 줄/셀이 무엇을 하는가"만** 설명한다.
  > 설계 결정, **다른 방식을 안 쓴 이유**, 참조 출처, diff 이력은 전부 금지
  > 금지 예: `# (not using asmatrix here)`
- 왜 문제인가: `rather than averaging the per-batch means the loaders produce` 는 채택하지 않은 대안을
  주석에 적은 것으로, 규약이 든 금지 예 `# (not using asmatrix here)` 와 같은 형태다.
  **게다가 이 설명은 이미 학생용 마크다운에 있다** — 셀 13 이
  `The last batch of a split is shorter than the others, so Step 8 recomputes the reported figures from
  y_pred over each split as a whole.` 라고 적는다. 규약이 요구하는 자리("비교·대안·근거는 마크다운 셀에")에
  이미 제대로 놓여 있으므로, 코드 주석의 뒷절은 중복이면서 금지 형태다.
- 왜 스크립트가 못 잡았나: 프로필 `meta-comment` 오버라이드 패턴에 `rather than` / `instead of using` /
  `not using` 계열 대안이 **없다**. 나는 수동 정규식으로 잡았다. 하네스 갱신에 올린다.
- 수정 방향: 뒷절만 떼고 "무엇을 하는가" 만 남긴다.
  `# RMSE over each split as a whole` (또는 `# RMSE recomputed over each split as a whole`)
- 재실행 불필요: 주석이라 출력이 바뀌지 않는다. `gen/p13.py` 수정 후 재생성만 하면 된다.

---

## NOTE

### N-1. `*_rmses_*`(지수 포인트) 와 `*_rmse_*`(표준화) 가 한 글자 차이다 — 라운드 2 신규

| 이름 | 정의 위치 | 단위 | 이후 사용 |
|------|-----------|------|-----------|
| `val_rmses_last_state` | 셀 19 (`train` 반환) | **지수 포인트** | **없음** |
| `val_rmse_last_state` | 셀 27 (재계산) | **표준화** | 셀 27 에서 출력 |

복수형 `s` 하나로 갈리는데 단위가 다르다. 셀 13 마크다운은 `rmse is its square root in index points`
라고 알려주므로, 그 설명을 읽은 학생이 셀 27 의 `val_rmse_last_state` 를 지수 포인트로 오해할 여지가 있다.

**등급을 NOTE 로 둔 이유:** 셀 27 의 print 가 같은 줄에서 `* close_train_std` 를 곱해 단위를 드러내고,
출력 헤더도 `RMSE on standardized targets, with the same error in index points in brackets` 라고
명시한다. 즉 **학생에게 잘못된 값이나 라벨이 전달되지는 않는다.** 코드를 가로질러 읽을 때만 걸린다.

고친다면 셀 27 쪽에 단위를 붙이는 것이 자연스럽다 — `val_rmse_scaled_last_state` 등.
`train()` 반환 4-tuple 은 리더 승인 시그니처이므로 그대로 둔다.
덧붙여 `train_rmses_*`·`val_rmses_*` 4개는 언패킹 후 한 번도 쓰이지 않는다(셀 19·22 에서만 등장).
시그니처를 학생에게 보여주는 값이 있으므로 제거를 권하지는 않는다.

### N-2. 마크다운 분량 — 캡은 지켰으나 여유가 없다

`markdown_budget.py` → **0 over cap** (통과). 다만 soft cap 초과가 8건 → 9건으로 늘었다.

| 셀 | 라운드 1 | 라운드 2 | 하드 캡 |
|----|---------|---------|--------|
| 13 (Step 5) | 85 단어 | **105 단어 / 산문 70** | 120 / 80 |
| 25 (extrapolation) | 산문 76 | 산문 74 | 80 |
| 29 (Summary) | 산문 75 | **산문 80** | **80** |

- 셀 13 은 리더 메시지에 "다시 다듬었다" 고 되어 있으나 라운드 1 대비 **20단어 늘었다**
  (V-1 설명이 들어간 결과). 하드 캡 안이므로 위반은 아니다.
- **셀 29(Summary)가 산문 하드 캡에 정확히 걸쳐 있다 (80/80).** 다음 개정에서 한 문장만 늘어도
  `markdown_budget.py` 가 exit 1 로 떨어진다. `CLAUDE.md` "왜 숫자로 박아 두는가" 가 경고하는
  "개정마다 조금씩 부푸는" 자리에 정확히 와 있다. 지금 손댈 필요는 없고, 다음에 Summary 를 만질 때는
  **먼저 지우고 넣어야** 한다는 점만 기록해 둔다.

### N-3. 라운드 1 NOTE 중 미해소분 (재론하지 않음, 상태만 기록)

| 항목 | 상태 |
|------|------|
| C-2 셀 26 이 예측 계산·역변환·플롯을 한 셀에 담음 | 그대로 (셀 27 이 RMSE 계산으로 분리되며 오히려 셀 26 은 순수 플롯에 가까워졌다) |
| C-3 셀 9 로더 표와 실제 로더 코드 사이 거리 | 그대로 |
| C-4 셀 16 의 `{nn.MSELoss()}` · 하드코딩 `'Adam'` 출력 | 그대로 |
| C-6 셀 30 Exercise 1 문구 | 그대로 |

전부 저자 재량이며 PASS/FAIL 에 영향이 없다.

---

## 검사한 체크포인트 (라운드 2 · 변경 범위 한정)

| 체크포인트 | 방법 | 결과 |
|-----------|------|------|
| 정규식 16종 전체 | `audit_notebook.py` (프로필 정상 인식) | **후보 0** |
| meta-comment (영어 확장) | **수동 정규식** — `rather than`/`instead of`/`not using`/`we chose` 등 | **후보 1 → D-1.** 셀 30 의 `instead of its level` 은 Exercise 문장이라 오탐으로 제외 |
| cross-ref (영어 확장) | 수동 정규식 | 후보 0 |
| B-1 대문자 일괄 반영 | 소문자 `x*` 식별자 전수 검색 + 선례 대조 | **해소.** 잔여 2건은 승인 예외이며 근거 확인 (`gen/p07.py:171,383`, `gen/p09.py:337`) |
| B-2 문장 | 셀 29 ↔ 셀 25 ↔ 셀 30 Exercise 2 삼자 대조 | **해소** |
| markdown budget | `markdown_budget.py` | **0 over cap.** 셀 29 가 80/80 으로 여유 없음 (N-2) |
| seed 재설정 실효성 | **재현 실험** (구성 후 재시드 → 배치 순서 비교) | **작동함.** A/B 배치 순서 완전 일치 (라운드 1 은 불일치) |
| RMSE 재계산 정확성 | **형상 검증 + 출력값 대조** | 통과. (298,1) 원소별 연산, 1-D 였다면 (298,298) 로 조용히 틀렸을 자리 |
| 셀 25 사실 주장 | 셀 10 출력과 수치 대조 | 통과. `58 of 65`, 최대 `1.350`, val ⊂ train 전부 일치 |
| 비교 실험 통제 | 셀 18~23 재확인 | **강화됨.** 격리 변수 1개(readout 입력), 배치 순서까지 동일해짐 |
| 실행 상태 | 출력 셀 전수 | 통과. 32셀, 에러 0, 그림 3장 |
| 변경 셀의 한글 유입 | 스크립트 `hangul` + 출력 수동 검사 | 후보 0 |

---

## 미검사 항목

| 항목 | 사유 |
|------|------|
| 범위 밖 셀 (code 0~6,8,9,13,16,17,20,23~25,28~31 중 미변경분) | 리더 지시에 따라 변경 범위 한정 재확인만 했다. 라운드 1 판정을 그대로 유지하며, 라운드 1 에서 통과한 항목을 다시 검사하지 않았다 |
| §2 수식의 옳고 그름 | 라운드 1 과 동일 — `consistency-verifier` 영역이고, 이 과목에 RNN 슬라이드가 없어 대조 권위가 없다. 노트북 내부 정합성(셀 12 손계산 ↔ `nn.RNN`)만 확인 |
| 그림 3장의 시각적 결함 | PNG 를 직접 렌더링하지 않았다. 셀 26 의 플롯 코드는 이번에 실질 변경이 없고(예측·역변환 로직 동일), 역변환 정확성은 수치로 검증했다. 축·범례의 순수 시각적 문제는 저자 육안 확인에 의존 |
| verifier V-1~V-3 의 판정 자체 | 내 영역이 아니다. 사실 대조만 했고 적정성 판단은 `consistency-verifier` 몫이다 |
| `gen/p13.py` 의 `# Cell N` 주석 정합성 | 이번에도 확인하지 않았다. 셀 병합이 없었으므로 어긋났을 가능성은 낮으나 미확인이다 |
| 재실행 재현성 | 저자의 nbconvert 를 재실행하지 않았다. 남은 출력 셀로만 확인했다. 라운드 1 대비 학습 수치가 달라진 것(A val 0.03245→0.03102)은 seed 재설정으로 배치 순서가 바뀐 결과이며 **예상된 변화**다 |

---

## 하네스 갱신 제출

### [승인 대상 · 트리거 1] `meta-comment` 패턴에 "채택하지 않은 대안" 계열이 없다

- **무엇:** `CLAUDE.md` "Notebook Comments" 는 금지 대상을 네 가지로 든다 — 설계 결정,
  **다른 방식을 안 쓴 이유**, 참조 출처, diff 이력. 그런데 프로필의 `meta-comment` 오버라이드 패턴은
  뒤 두 개(참조 출처·diff 이력)만 덮는다. 규약이 **금지 예로 가장 먼저 든** `# (not using asmatrix here)`
  형태를 잡는 대안이 정규식에 하나도 없다.
- **증거:** 이번 D-1(`rather than averaging the per-batch means...`)이 스크립트 후보 0 을 통과했고,
  내 수동 정규식으로만 잡혔다. 라운드 1 에서 같은 수동 스캔이 후보 0 이었던 것은
  그때는 이 주석이 존재하지 않았기 때문이다 — 즉 **정규식이 놓친 것이 아니라 아예 없다.**
- **어디에:** `01_AI-ME_Graduate/.claude/notebook-profile.json`
  → `checkpoints.overrides["meta-comment"].pattern`
- **제안 (기존 패턴에 OR 로 추가):**
  `|\brather\s+than\s+\w+ing\b|\binstead\s+of\s+\w+ing\b|\bnot\s+using\b|\bwe\s+(chose|opted|decided)\b|\bavoids?\s+using\b`
  `\w+ing` 로 좁힌 것은 `instead of its level`(셀 30 Exercise) 같은 정상 문장을 피하기 위해서다 —
  실제로 이번에 그 문장이 내 넓은 수동 패턴에 걸렸고 오탐으로 걸러냈다.
- **회귀 확인 필요:** 좁힌 패턴을 P13 에 돌리면 D-1 만 잡히고 Exercise 문장은 안 잡히는 것을 확인했다.
  **P01~P12 오탐 확인은 하지 않았다.** `enabled` 체크포인트의 패턴 변경이므로 확정 전 필수다.

### [승인 대상 · 트리거 3] 리포트 셀 번호 체계가 도구와 어긋난다 — 내 쪽 결함

- **무엇:** `audit_notebook.py`(L300)와 `markdown_budget.py`(L106) 는 둘 다
  `enumerate(cells)` 를 `start=1` 없이 쓴다 → **0-based**. 저자 리포트와 리더 메시지도 0-based 다.
  그런데 **라운드 1 감사 리포트만 1-based** 여서 모든 셀 번호가 1 컸다.
- **왜 사고가 안 났나:** 내가 모든 지적에 원문을 인용했고 저자가 그것으로 위치를 재확인했다.
  번호만 적었다면 저자가 인접 셀을 고쳤을 것이다.
- **어디에:** `.claude/skills/notebook-convention-audit/SKILL.md` 리포트 형식 절에
  "셀 번호는 `audit_notebook.py` 출력과 같은 0-based 를 쓴다" 한 줄. 겸해서
  "지적마다 원문을 반드시 인용한다" 의 근거로 이 사례를 남기면 규칙이 왜 있는지가 분명해진다.
- **근거:** 라운드 2 실측. 이번 리포트부터 0-based 로 전환했다.

### [상태 갱신] 라운드 1 제출분

| 항목 | 상태 |
|------|------|
| `cross-ref` 오버라이드의 한국어 대안이 영어 전용 과목에서 사문 | **미처리.** 라운드 2 에서도 수동 정규식으로 대신했다 (후보 0) |
| 회귀 회차 `train`/`evaluate` 정본이 `code-patterns.md` 에 없음 | **미처리.** 이번 B-1 이 정확히 이 공백에서 나왔다. (c) 회귀판을 추가할 때 **본문 루프 변수명(`X_batch, y_batch`)까지 포함한 완전한 코드 블록**으로 적을 것 |
| `CLAUDE.md` "디렉터리 레이아웃" 이 현실과 어긋남 (`_workspace/` 등) | **미처리** |

### 즉시 반영 대상 — 없음

이번 라운드에도 스크립트 후보가 0건이라 걸러낼 오탐이 없었다. `notebook_overrides`·`noise`·`enabled`
어느 것도 수정하지 않았다. (수동 스캔에서 나온 `instead of its level` 오탐 1건은 아직 프로필에 들어가지
않은 제안 패턴에서 나온 것이므로, 위 승인 항목의 패턴 설계에 반영해 두는 것으로 갈음한다.)
