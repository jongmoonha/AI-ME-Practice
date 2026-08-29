# Convention Audit — Practice12_Object_Detection_and_Segmentation.ipynb (라운드 6, 최종)

## 판정: PASS
- **BLOCK 0건 / FIX 1건 / NOTE 5건**
- **라운드 5 의 BLOCK A-1 해소 확인.** `patience=0` 수정으로 A-1 과 구 B-1 이 **한 수정으로 함께 닫혔다**
- FIX 1건은 **이번 수정이 새로 만든 것**이며 배포 상태에서는 거짓이 아니다 (아래 B-1)

### 미반영으로 남는 것 — 최종 보고용 (리더 요청)

| # | 항목 | 등급 | 상태 |
|---|---|---|---|
| B-1 | 셀 33 Exercise 2 가 `epochs` 를 바꾸라 하는데 실제 이름은 `finetune_epochs` 다 | **FIX** | **미반영 (신규).** 한 단어 수정 |
| C-1 | 셀 17 통제표 B 열이 `best of 120` 만 적어 셀 20 의 3행 출력을 2팔로 그린다 | NOTE | 미반영. 산문이 보완하므로 거짓 아님. **셀 17 이 116/120 이라 손대지 말 것 권고** |
| C-2 | 셀 32 `halved the score` 가 어느 체크포인트인지 말하지 않는다 | NOTE | 미반영. 다음 불릿이 예외를 처리한다. **Summary 산문 61/60 이라 여유 없음** |
| C-3 | 셀 3 `device` 가 계산·출력되고 어디에도 쓰이지 않는다 | NOTE | 미반영 (라운드 4 C-4 이월). 전 셀 검색 결과 셀 3 에만 등장 |
| C-4 | 셀 18·20·29 `val()` 이 학습 전용 인자 `seed`/`amp` 를 받는다 | NOTE | 미반영. **본질(통제표 오기재)은 해소**, 남은 인자는 무해 |
| C-5 | fuse 함정이 코드 주석에만 있고 마크다운에 없다 | NOTE | 미반영 (라운드 4 C-1 이월). 셀 28 에 한 문장 권고 |
| V-12 | `best.pt` 가 epoch 1 가중치라는 사실 미기재 | — | **리더 결정으로 미반영.** 사실이나 **노트북 출력만으로 epoch 1 을 특정할 수 없어** 적으면 근거 없는 주장이 된다. 판단에 동의한다 |

### 라운드 6 재검증 — 전부 독립 확인 (리더 보고를 받지 않고 다시 뽑았다)

| # | 확인 | 실측 | 판정 |
|---|---|---|---|
| 1 | 셀 19·29 로그 | **`120 epochs completed`** (0.019h / 0.023h), `EarlyStopping` **0건**, `patience=0` | **통과** |
| 2 | 통제표 ↔ 로그 | 셀 17 `+ 120 epochs` 가 이제 **로그와 일치**. 전 라운드의 표120 vs 로그101 해소 | **통과** |
| 3 | `finetune_epochs` 전파 | 정의 1곳(셀 19) + 사용 7곳 — 셀 19·29 `train()` **인자 양쪽**, 셀 20·22·23×2·29 **라벨 전부** f-string | **통과** |
| 4 | 코드 내 리터럴 `120` | **정의 그 자체 1건뿐.** 라벨·인자에 잔존 0 | **통과** |
| 5 | 셀 21 `about half` | 0.36998 / 0.68437 = **0.541** | **참** |
| 6 | 셀 24 orange·zebra | pretrained orange **0 → 6개**(5→6 증가), held-out **8건 → `zebra` 1건** | **참** |
| 7 | 셀 32 `halved` | mAP@50 0.541 / mAP@50-95 0.597 — `last.pt` 기준 성립 | **참** |
| 8 | 셀 17·21 마크다운의 `120` | **patience=0 으로 이제 전부 참이 됐다** (이것이 (b) 안을 고른 이유다) | **통과** |
| 9 | 셀 29 seg 수치 변동 | last 0.32036·0.22290·0.25711·0.16023 (미세 이동), best 불변. **어떤 마크다운도 seg 숫자를 단정하지 않음** | **파급 없음** |
| 10 | 셀 시간 | 학습 **68.4초 / 82.8초** — 프로필 `cell_time_warning_sec: 300` 대비 여유 | **통과** |
| 11 | 스크립트 2종 | `audit_notebook.py` 후보 없음 / `markdown_budget.py` **0 over cap** | **통과** |
| 12 | 구조 | 35셀, `execution_count` 1–20 연속, error 0, `execute_result` **0**, 한글·이모지 0, `def`/`gridspec`/`assert` 0, Step 0~7 연속, 그림 4장(12·16·23·27) | **통과** |

**`Closing dataloader mosaic` 이 새로 나타났다** — `close_mosaic=10` 기본값이 살아 있고 이번엔 120 epoch 을
완주하므로 epoch 111 에서 찍힌다 (101 에서 멈추던 때는 아예 없었다). **예고한 대로 결함으로 올리지 않는다.**
어떤 단정문도 여기 걸려 있지 않으며, 라운드 4 에서 내가 이 로그 줄을 인과로 잘못 지목했다가
리더의 통제 실험에 반증당한 자리다.

---

## FIX (라운드 6 신규)

### B-1. 셀 33 Exercise 2 가 `epochs` 를 바꾸라고 하는데, 학생이 그대로 하면 이번에 고친 결함이 되살아난다
- 위치: 셀 33 (markdown) — 유발 대상은 셀 19
- 원문: ``2. Change `epochs` in the detection fine-tuning call from 120 to 30.``
- 현재 셀 19:
  ```python
  finetune_epochs = 120
  finetune_model.train(..., epochs=finetune_epochs, ...)
  ```
- 위반 규약: `CLAUDE.md` "Notebook 편집 — 전체 맥락 우선"
- 왜 문제인가: **`train()` 호출부에 `120` 이라는 리터럴이 이제 없다.** 지시를 문자 그대로 따르는 학생은
  호출부를 `epochs=30` 으로 고치게 되고, 그러면 `finetune_epochs` 는 120 으로 남아
  **셀 20·22·23 라벨이 다시 `after 120 epochs` 를 인쇄한다** — 30 epoch 가중치에 120 이라는 이름이 붙는,
  **A-1 이 방금 닫은 바로 그 상태**다. 지시가 이번 수정의 불변식을 스스로 깨뜨린다.
  (한 줄 위의 변수를 알아채는 학생에게는 문제가 없다. 그래서 BLOCK 이 아니라 FIX 다.)
- 수정 방향 (한 단어):
  ``2. Change `finetune_epochs` from 120 to 30.``
  덧붙일 수 있는 정보 — **이 값은 segmentation 학습(셀 29)도 함께 짧게 만든다.** 두 `train()` 이
  같은 변수를 읽으며, 이는 셀 28 의 *"The training call is the one used for detection"* 을
  **코드로 보증하는 옳은 구조**다. 필요하면 괄호로 한 절만 덧붙인다.
- 분량 영향 없음 (Exercises 셀은 캡 여유가 있다).

---

> ### [라운드 5 이력] 판정 정정 — 나는 처음에 PASS 를 냈고, 그것이 틀렸다
>
> 초판에서 나는 이 항목을 **FIX(B-1)** 로 올리며 *"배포 상태의 노트북에서는 거짓이 아니고
> 학생이 Exercise 2 를 수행할 때만 발현한다"* 고 적었다. **그 전제가 사실이 아니다.**
> `verifier2` 가 셀 19·29 의 **학습 로그**를 읽어 반증했고, 나는 그것을 직접 확인해 채택한다.
>
> ```
> EarlyStopping: Training stopped early as no improvement observed in last 100 epochs.
>                Best results observed at epoch 1, best model saved as best.pt.
> To update EarlyStopping(patience=100) pass a new patience value ... or use `patience=0` to disable
> 101 epochs completed in 0.008 hours.
> ```
>
> 인자 로그 실측: `epochs=120`, **`patience=100`** (노트북이 넘기지 않은 ultralytics 기본값).
> best 가 epoch 1 이라 `1 + 100 = 101` 에서 멈췄다. 셀 29 도 동일하게 `101 epochs completed`.
> **학생이 아무것도 하지 않아도 `120 epochs` 라벨은 이미 거짓이다.**
>
> **내가 왜 못 잡았는가.** 나는 셀 18·20·22·26·29 의 **마지막 print 블록만** 뽑아 숫자를 대조했고,
> 셀 19·29 의 학습 로그 본문은 열지 않았다. 라운드 4 에서 학습 로그의 지표 변화를 두 번 잘못
> 해석한 뒤 *"학습 동역학에 주장을 걸지 않는다"* 는 원칙을 세웠는데, 그것을 **"로그를 읽지 않는다"**
> 로 잘못 실행했다. 로그에는 해석이 필요한 동역학만 있는 것이 아니라 **`101 epochs completed` 같은
> 해석이 필요 없는 사실**이 있고, 이번 결함은 정확히 그 종류였다. 자세한 것은 하네스 갱신 8번.

이 라운드는 **사용자 개정 3건 + 리더 그림 가독성 지시**에 대한 재감사다.
저자 보고(`02_author_report.md`)를 근거로 삼지 않고 노트북 실물로 전부 재확인했다.

| 확인 항목 | 방법 | 결과 |
|---|---|---|
| `audit_notebook.py` (16종) | 스크립트 | **후보 없음.** `lr-symbol` matplotlib 오탐도 미발화 |
| `markdown_budget.py` | 스크립트 | **0 over cap**, soft 6건 (셀 4·5·17·21·30·32) |
| Step 4 삭제 연쇄 | 삭제 식별자 10종 노트북 전체 정규식 | **잔재 0건** |
| Step 번호 연속성 | 헤더 전수 추출 | **Step 0 ~ Step 7 연속.** 옛 번호 잔재 0 |
| 통제표 ↔ 코드 | 셀 17 표 6행을 셀 18·19·20 인자와 한 줄씩 대조 | 6행이 **호출 인자와는** 전부 일치. 단 `weights evaluated`·`checkpoint` 두 행이 **실제로 학습된 epoch 수(101)와 어긋난다 → A-1** |
| 결과 해설 정직성 (숫자) | 셀 21·24·32 를 셀 20·22 출력 **숫자**와 대조 | 통과. 한 행만 보고 쓴 문장 없음 |
| 결과 해설 정직성 (epoch 수) | 셀 17·20·21·22·23·29 를 셀 19·29 **학습 로그**와 대조 | **BLOCK (A-1).** 8곳이 `120` 이라 적었으나 실측 **101** |
| 학습 로그 사실 대조 | 셀 19·29 로그의 `EarlyStopping`·`epochs completed`·`patience`·`optimizer` 행 | **초판에서 누락했다.** `verifier2` 반증을 확인해 채택 (하네스 갱신 8번) |
| 검증되지 않은 인과 | `because`/`due to`/`caused`/`therefore`/`thus`/`hence`/`mosaic`/`warmup` | **0건.** `because` 2건은 검증된 도메인 사실 |
| "fine-tuning 이 좋다/나쁘다" 일반화 | `improve`/`better`/`worse`/`help`/`harm` 전수 | **0건** |
| `execute_result` / error | JSON 전수 | **0건 / 0건.** `execution_count` 1–20 연속 |
| 한글·이모지 | 전 셀 non-ASCII 스캔 | **0.** source 의 non-ASCII 는 셀 0 제목의 `—` 하나 |
| 교차 참조 | 프로필 `cross-ref` 패턴 | 셀 0 의 `Practice 12` 자기 참조 1건뿐 |
| helper/docstring/gridspec/assert | 전 코드 셀 | `def` 0개, `"""` 0개, `gridspec` 0개, `assert`·`allclose` 0개 |
| 루트 오염 | `ls -A` | `.pt`/`runs/` 없음. **`_workspace/` 가 `md/p12_workspace/` 로 이동해 라운드 4 잔여 항목 해소** |

### 이번 개정의 핵심 3건 — 개별 판정

**1. Step 4 삭제의 연쇄 — 결함 없음.**
`ground_truth_box`·`predicted_box`·`best_iou`·`best_index`·`iou_threshold`·`verdict`·
`intersection`·`union`·`ground_truth_area`·`class_matches` 를 노트북 전체에서 검색해 **전부 0건**.
`IoU` 문자열도 0건. 남은 `precision`/`recall` 은 셀 18 의 `print` 라벨 2개이며 정의문이 아니다.
Exercises 1·3 이 사라진 IoU 루프를 참조하지 않는다 — 1번은 셀 14·15(`conf` → 목록), 3번은
셀 12 의 polygon 블록을 가리키며 **둘 다 실재하는 셀이다.** Summary 에 IoU/mAP 정의 잔재 없음.

**mAP 은 정의 없이 9회 등장하지만 틀린 주장이 하나도 없다.** 라운드 4 에서 BLOCK(A-3)을 받은
셀 32 의 잘못된 정의문(`mAP averages precision over a range of IoU thresholds`)은 삭제됐고,
현재 Summary 는 `score`·`number` 로만 쓴다. 정의를 뺀 것은 사용자 결정이므로 결함이 아니며,
**정의를 빼면서 서술도 함께 뺀 것이 옳은 처리다.**

**2. 통제 실험 (셀 17) — 통제 자체는 성립하나 두 행이 사실과 어긋난다 (→ A-1).**
표 6행을 코드와 한 줄씩 대조했다. **아래 표의 "코드 실측" 은 호출 인자이고, A-1 이 문제 삼는 것은
그 인자가 실제로 만들어낸 결과다.**

| 표의 행 | 표의 주장 | 코드 실측 | 판정 |
|---|---|---|---|
| weights evaluated | B = `+ 120 epochs (SGD, lr0=0.01)` | 인자는 `epochs=120` 이나 **실제 학습은 101 epoch** | **불일치 → A-1** |
| checkpoint | A 무선택 / B `best of 120` | best.pt 는 **101개 중 epoch 1** | **불일치 → A-1** |
| dataset | same | 셀 18·19·20 전부 `detection_yaml_path` | 일치 |
| evaluation split | `images/val` 양쪽 | yaml `val: images/val`, 세 호출 동일 | 일치 |
| model size | `yolo11n` 양쪽 | 셀 18·19 둘 다 `BUILD/'yolo11n.pt'` | 일치 |
| `imgsz` | 320 양쪽 | 셀 18·19·20 전부 `imgsz=320` | 일치 |

`20 epochs` 잔재 **0건**. `seed`·`amp`·`batch` 행 삭제와 `model size` 행 유지는 리더 지시대로이며,
**남은 6행이 전부 실제로 통제된 것**이다 — 즉 **통제 설계 자체는 유효하고, 깨진 것은 B 팔의 서술이다.** `checkpoint evaluated` 행과 `coco8` has no test split
서술 유지 확인 (셀 17 3번째 문단). 셀 22·23 의 시각 비교도 통제됐다 — 세 `predict()` 가
전부 `conf=0.25`, 같은 두 이미지, **가중치만 다르다.**

**3. 셀 24 결과 해설 — 결함 없음. 이번 개정에서 가장 잘 된 부분이다.**

| 셀 24 문장 | 대조 출력 | 판정 |
|---|---|---|
| `The training image gains the oranges the pretrained weights were missing` | 셀 22: pretrained `orange` **0개** → after 120 **5개** | 참 |
| `The held-out image loses almost everything those same pretrained weights found` | 셀 22: **8건 → `zebra` 1건** | 참 |
| `That is overfitting, and the last.pt row above is its number` | 셀 20 `B after 120 epochs 0.36912` | 참 |
| `Eight photographs are what this figure is about, not what fine-tuning does in general` | — | **일반화를 명시적으로 차단** |

- "fine-tuning 이 성능을 높인다" 로 읽히지 않는다 — `improve`/`better`/`help` **0건**
- "fine-tuning 은 해롭다" 로도 읽히지 않는다 — 마지막 문장이 8장짜리 데이터셋으로 범위를 못 박는다
- **한 행만 보고 쓴 문장 없음 (지시받은 최우선 확인 항목).** 셀 21 의
  `it keeps about half of the pretrained mAP@50` 은 **`last.pt` 와 `mAP@50` 으로 이중 한정**돼 있고,
  실측은 0.36912/0.68437 = **0.539**. 나머지 열(mAP@50-95)도 0.22711/0.39986 = **0.568** 로 같은 방향이라
  한정어를 빼도 참이 되는 자리다. 저자가 초안에서 `The validation rows ... went down` 을 썼다가
  잡았다고 보고했는데, **실물에 그 문장이 없음을 확인했다** — 노트북 전체에 `validation rows` 0건이고
  `best.pt` 행이 올라간다는 사실(0.68769 > 0.68437)과 모순되는 문장이 없다

**4. 학습/held-out 구분 — 결함 없음.** 다섯 자리에서 명시된다:
셀 21 산문(`one the model trained on, one it never saw`), 셀 22 주석, 셀 22 출력 라벨
(`training image` / `held-out image`), **셀 23 네 패널 제목**, 셀 24 산문.
경로로도 확인했다 — `009` 는 `images/train/`, `049` 는 `images/val/` 이며 yaml 의 두 split 과 일치한다.
**그림 자체가 구분을 지고 있어** 마크다운을 건너뛰는 학생도 오해할 수 없다.

**5. `train()`/`val()` 이 마지막 표현식이 아님 — 기계 확인 완료.**
지시받은 스크립트로 전 35셀 검사 결과 `execute_result` **0건**. 셀 19·29 는 `print` 로 닫힌다.

**6. `instance_colors` — 결함 없음.** 셀 16 에서 한 번 정의, 셀 23·27 이 재사용.
`CLAUDE.md` "한 번 정의한 데이터는 노트북 전체에서 같은 이름으로 재사용" 에 정확히 부합한다.
helper function 이 아니라 **리스트 리터럴 + 인덱싱**이므로 "helper function / model factory 금지" 에
걸리지 않는다. 인덱스 안전성은 **세 사용처 전부 `% len(instance_colors)`** 로 보장된다 (셀 16·23·27).
라운드 4 미검사 항목이던 "`fill_colors` 8색 고정 → 9개 이상에서 `IndexError`" 가 이로써 해소됐다.

---

## BLOCK — **[라운드 5 이력. 라운드 6 에서 해소 확인됨]**

> **A-1 은 닫혔다.** `patience=0` 으로 셀 19·29 가 `120 epochs completed` 를 인쇄하고
> `EarlyStopping` 문구가 사라졌으며, `finetune_epochs` 변수가 인자와 라벨 8곳을 모두 공급한다.
> 아래는 **결함이 어떻게 발견되고 왜 BLOCK 이었는지의 기록**이며, 현재 노트북의 상태가 아니다.
> 최신 판정은 이 문서 최상단에 있다.

### A-1. 노트북이 8곳에서 `120 epochs` 라고 말하는데 실제 학습은 101 epoch 에서 멈췄고, 그 사실이 바로 위 셀에 인쇄돼 있다
- 위치: 셀 17 (markdown ×2), 셀 20 (code 주석 + print), 셀 21 (markdown ×2), 셀 22 (code print),
  셀 23 (code 패널 제목 ×2), 셀 29 (code print) — **detection·segmentation 양쪽**
- 원문 (노트북의 주장):
  ```
  셀 17 : | **weights evaluated** | ... | **+ 120 epochs on coco8 (SGD, lr0=0.01)** |
  셀 17 : | **checkpoint**        | ... | **best of 120, chosen on `images/val`**   |
  셀 20 : # train() leaves the best checkpoint loaded, so the weights after all 120 epochs are opened separately
  셀 20 : print(f'B after 120 epochs   {last_epoch_metrics.box.map50:9.5f}  ...')
  셀 21 : `last.pt` is the weights after all 120 epochs, and `best.pt` is whichever epoch scored highest ...
  셀 21 : The next two cells run the pretrained weights and those same 120-epoch weights on two photographs
  셀 22 : print('  after 120 epochs : ' + ', '.join(finetuned_names))
  셀 23 : 'After 120 epochs - training image' / 'After 120 epochs - held-out image'
  셀 29 : print(f'after 120 epochs  {segmentation_last_metrics.box.map50:10.5f}  ')
  ```
- **반증 — 셀 19 자신의 출력** (셀 29 도 동일):
  ```
  engine\trainer: ... epochs=120, ... patience=100, ... optimizer=SGD, lr0=0.01 ...
  EarlyStopping: Training stopped early as no improvement observed in last 100 epochs.
                 Best results observed at epoch 1, best model saved as best.pt.
  To update EarlyStopping(patience=100) pass a new patience value, i.e. `patience=300`
                 or use `patience=0` to disable EarlyStopping.
  101 epochs completed in 0.008 hours.
  ```
  `patience=100` 은 **노트북이 넘기지 않은 ultralytics 기본값**이다. best 가 epoch 1 이라
  `1 + 100 = 101` 에서 종료됐다. 따라서 `last.pt` 는 **101 epoch 가중치**이고
  `best.pt` 는 **epoch 1 가중치**다. 셀 29 도 `101 epochs completed` 로 동일하다.
- 위반 규약: `CLAUDE.md` "Notebook 편집 — 전체 맥락 우선" (마크다운·라벨이 코드 출력과 어긋남),
  "비교 실험 통제 변수" (*"변경된 항목은 마크다운 표나 한 문장으로 명시한다"* — B 팔의 실제 학습량이
  표와 다르다), "Result Comparison" (*"값을 직접 보고 인지하게 한다"* — 고정 문자열이 설정을 대신 주장)
- 왜 문제인가:
  1. **학생 화면에서 직접 충돌한다.** 셀 19 로그의 마지막 줄이 `101 epochs completed` 이고
     바로 다음 셀 출력의 한 줄이 `B after 120 epochs` 다. **스크롤 한 번에 둘이 같이 보인다.**
     라운드 4 B-3 에서 우리가 잡았던 구조 — *"주의 깊은 학생일수록 노트북이 틀렸다고 결론짓는다"* —
     와 같고, 이번에는 **노트북이 실제로 틀렸다.**
  2. **`best.pt` 가 epoch 1 이라는 사실이 서술을 뒤집는다.** 셀 17 의 `best of 120`,
     셀 21 의 `whichever epoch scored highest`, 셀 32 의 *"The higher fine-tuned score came from a
     checkpoint picked on the split being reported"* 는 **"120개 후보 중 val 에서 고른 것"** 으로 읽힌다.
     실제로는 **학습이 망가뜨리기 전인 epoch 1** 이며, 그래서 `best.pt` 점수가 baseline 과
     사실상 같다 (0.68769 vs 0.68437). 선택 편향 이야기 자체는 성립하지만 **진짜 이야기는
     "고를 것이 없어서 맨 앞이 뽑혔다" 이고, 그쪽이 더 단순하고 더 정직하다.**
  3. **재현되지 않는다.** 종료 epoch 이 best epoch 에 의존하므로 다른 환경에서 best 가 epoch 3 이면
     103 에서 멈춘다. `120` 은 **어떤 환경에서도 참이 되지 않는 숫자**다.
  4. 대상이 대학원생이고 이들이 자기 연구에서 그대로 흉내 낼 절차다. `epochs=` 를 적어 놓고
     **실제 학습량을 로그로 확인하지 않는 것**은 논문에서 그대로 반박당하는 형태다.
- 수정 방향: **`verifier2` 와 단일 지시로 합의했다 (그쪽 V-11).** 두 안 중 **리더가 택일**한다.
  - **(a) 라벨을 epoch 비의존 표현으로 바꾼다** — 재실행 불필요, 숫자·그림 그대로:
    `B last epoch` / `the weights at the last epoch` / `After fine-tuning - training image` /
    `best epoch, chosen on images/val`.
    셀 17 의 `+ 120 epochs on coco8` **한 곳만은** 호출 인자를 말하므로
    `` fine-tuned with `epochs=120` (SGD, lr0=0.01) `` 처럼 **인자임이 드러나게** 쓰면 참으로 남는다.
    이 안이 (i) 101≠120 충돌, (ii) 아래 [구 B-1] Exercise 2 충돌, (iii) 재현 불가를 **한꺼번에** 닫는다.
  - **(b) `patience=0` 을 셀 19·29 에 넘겨 실제로 120 epoch 을 돌린다** — 라벨이 전부 참이 되지만
    **전 숫자·셀 22 목록·셀 23 네 패널이 재측정·재렌더링된다.** 착수 전 리더 결정이 필요하다.
  - **분량**: (a) 는 셀 17 을 현재 116/120 안에서 처리할 수 있다. **EarlyStopping 설명은
    산문 여유가 있는 셀 21 (61/80) 에 넣고, 셀 17 에는 넣지 마라.**

> ### 리더 결정 — **(b) `patience=0` 으로 확정** (라운드 5 종료 시점)
>
> **"라벨을 실제에 맞추는" (a) 가 아니라 "실제를 라벨에 맞추는" (b) 를 골랐다.** 리더 근거 셋:
> 1. 이 노트북이 가르치는 것은 *"우리가 정한 만큼 학습시킨다"* 이다. 조기 종료가 조용히 끼어들어
>    **요청과 실행이 갈리는 것 자체가 학생이 자기 실험에서 겪을 함정**이며, 여기서 켜 둘 이유가 없다
> 2. 숫자를 빼면 **Exercise 2 가 무엇을 바꾸는지 학생이 확인할 길이 사라진다**
> 3. **실측으로 결론이 바뀌지 않음을 확인했다** — `patience=0` 120 epoch 완주 시
>    `last.pt` **0.36998** / `best.pt` 0.68769 로 조기종료본(0.36912 / 0.68769)과 사실상 같고,
>    학습 이미지의 orange 검출과 held-out 의 `zebra` 붕괴도 그대로다. **라벨만 참이 된다**
>
> **[구 B-1 이 같은 수정으로 함께 닫힌다.]** `epochs` 를 변수로 한 번만 정의하고 라벨이 f-string 으로
> 재사용한다 — `CLAUDE.md` "비교 실험 통제 변수" 의 *"위에서 한 번만 정의하고 모든 방법이
> 재사용한다"* 를 학습 인자에 적용한 것이다. 내 초판 수정 방향 1안과 같은 형태다.
>
> **V-12(`best.pt` 가 epoch 1)는 반영하지 않는다 — 리더 결정.** 사실이지만 **epoch 1~11 지표가
> 전부 같아 노트북 출력만으로는 epoch 1 을 특정할 수 없다.** 적으면 출력이 뒷받침하지 못하는
> 주장이 되고, 그것은 이 회차 내내 걷어낸 종류다. **판단이 옳다** — 내 A-1 "왜 문제인가 2" 의
> 논거도 같은 이유로 노트북에 인쇄될 수 없으며, **감사 근거로만 유효하고 학생 배포물의 문장이
> 될 수 없다.** 최종 보고에 미반영으로 명시된다.
>
> #### 재검증 시 볼 것 (범위 확정)
> | # | 확인 | 방법 |
> |---|---|---|
> | 1 | 셀 19·29 로그에 **`120 epochs completed`**, **`EarlyStopping` 문구 소멸** | 로그 전문 grep |
> | 2 | `epochs` 변수가 **인자와 라벨 양쪽**에 실제로 쓰였는가 (8곳 전부) | source grep — 잔존 리터럴 `120` 0건 |
> | 3 | 수정이 **다른 단정문을 깨뜨리지 않았는가** | 셀 21 `about half`(0.36998/0.68437 = **0.541**, 유지), 셀 24 orange·zebra, 셀 32 `halved` 재대조 |
> | 4 | 셀 29 **segmentation 수치 변동** | 어떤 마크다운도 seg 숫자를 단정하지 않으므로 파급 없음을 확인만 한다 |
> | 5 | `markdown_budget.py` / `audit_notebook.py` | 재실행 |
>
> **저자에게 미리 경고할 것 둘 (내가 재검증에서 잡을 항목이므로 앞당겨 적는다):**
> - **셀 17 에 `patience` 를 적지 마라.** 현재 **116/120** 이라 두 단어면 하드 캡을 넘긴다.
>   `patience=0` 은 비교 변수가 아니라 *"요청한 만큼 돌게 하는"* 설정이므로 통제표에 행이 필요 없다.
> - **`Closing dataloader mosaic` 로그 줄이 새로 나타난다.** `close_mosaic=10` 기본값이 살아 있고
>   이번에는 120 epoch 을 완주하므로 epoch 111 에서 찍힌다 (101 에서 멈추던 때는 아예 안 나왔다).
>   **어떤 단정문도 이것에 걸려 있지 않으며, 나는 이것을 결함으로 올리지 않는다** —
>   라운드 4 에서 내가 이 로그 줄을 인과로 잘못 지목했다가 리더의 통제 실험에 반증당한 자리다.

> **[구 B-1 — A-1 에 흡수됨]** 초판에서 나는 이 자리를 FIX 로 올리며 *"셀 33 Exercise 2 가
> `epochs` 를 30 으로 바꾸라고 지시하므로 학생이 그때 라벨을 거짓으로 만든다"* 고 적었다.
> **지적한 자리는 맞았고 A-1 (a) 가 함께 닫는다. 틀린 것은 등급의 근거다** —
> 나는 "배포 상태에서는 참" 을 전제했는데 배포 상태에서 이미 거짓이었다.
> Exercise 2 는 결함의 **원인이 아니라 두 번째 증상**이었다.

---

## FIX

**없다.** (초판의 B-1 은 위 A-1 에 흡수됐다.)

---

## NOTE

### C-1. 셀 17 통제표는 B 를 한 팔로 그리는데 셀 20 출력은 세 행이다
표의 `checkpoint` 행 B 열이 `best of 120, chosen on images/val` 하나만 적는데, 실제로는
`last.pt`(무선택)과 `best.pt`(선택됨) **두 체크포인트가 보고된다.**
바로 아래 산문 `Only the last-epoch row escapes that selection.` 이 이를 보완하고 셀 21 이 다시
설명하므로 **거짓은 아니고 공개 의무도 충족된다** — `CLAUDE.md` "비교 실험 통제 변수" 는
"변경된 항목을 표 **또는 한 문장**으로 명시" 를 요구하며 후자를 지켰다.
다만 표만 읽는 학생에게는 세 번째 행이 예고 없이 나타난다.
고친다면 B 열을 `last epoch, and best of 120 chosen on images/val` 로 늘리는 것인데
**셀 17 이 116/120 이라 하드 캡을 넘긴다.** 표의 `dataset` 행을 함께 빼야 하며
(`evaluation split` 이 사실상 같은 것을 말한다), 그만한 가치가 있는지는 리더 판단이다.
**손대지 않는 쪽을 권한다.**

### C-2. 셀 32 두 번째 불릿의 `halved the score` 가 어느 체크포인트인지 말하지 않는다
`Fine-tuning on four images fitted those four and halved the score on the images it had not seen.`
`last.pt` 기준으로는 두 열 모두 참이지만(0.539 / 0.568), `best.pt` 는 **오히려 올라간다**(0.68769 > 0.68437).
바로 다음 불릿이 *"The higher fine-tuned score came from a checkpoint picked on the split being
reported, so the last-epoch number is the honest one"* 로 **그 예외를 명시적으로 처리**하므로
두 불릿을 함께 읽으면 정확하다. 순서상 한정이 뒤에 온다는 것이 유일한 약점이다.
**Summary 가 산문 61/60(소프트 초과)이라 단어를 더할 자리가 없다. 수정하지 않는 쪽을 권한다.**

### C-3. (라운드 4 C-4 이월, 미해소) 셀 3 의 `device` 가 계산·출력되고 어디에도 쓰이지 않는다
`device = 'cuda' if torch.cuda.is_available() else 'cpu'` 를 출력하지만 `train()`/`val()`/`predict()`
어느 호출에도 넘기지 않는다 (전 셀 검색 결과 셀 3 에만 등장). 학생은 이 변수가 배치를 결정한다고 오해한다.
`device=device` 를 실제로 넘기거나 출력만 남기고 변수를 없애는 편이 정직하다.

### C-4. (라운드 4 C-5 부분 해소) `val()` 이 여전히 `seed=42, amp=False` 를 받는다
셀 18·20·29 의 `val()` 호출이 순수 평가에 영향 없는 학습 전용 인자를 넘긴다.
**C-5 의 본질(통제표가 이 둘을 통제 항목으로 나열)은 이번에 해소됐다** — 표에서 두 행이 빠졌다.
남은 것은 인자 자체뿐이며 에러도 오해도 일으키지 않는다. 정리하면 각 호출이 한 줄 짧아진다.

### C-5. (라운드 4 C-1 이월, 미해소) fuse 함정이 코드 주석에만 있고 마크다운에 없다
셀 19 `# training starts from another fresh object, never from one that has already run inference`,
셀 29 `# again a fresh object, so that the inference above cannot affect this training`.
`predict()` 를 돌린 객체로 학습하면 **에러 없이 mAP 만 0 이 되는** 이 함정은 이 노트북에서 가장
실용적인 교훈인데 학생이 읽는 마크다운에는 한 줄도 없다. 셀 17 또는 28 에 한 문장을 제안한다.
(주석 자체는 "무엇을 하는가" 범위 안이므로 위반으로 올리지 않는다. 셀 17 은 분량이 없으니 셀 28 이 낫다.)

---

## 검사한 체크포인트 (라운드 5)

| 체크포인트 | 방법 | 결과 |
|-----------|------|------|
| emoji | `audit_notebook.py` + non-ASCII 독립 스캔 | 통과. `—`(U+2014) 1건은 셀 0 제목 house style |
| cross-ref | `audit_notebook.py` + 프로필 패턴 독립 실행 | 통과. `Practice 12` 자기 참조뿐 |
| meta-comment | `audit_notebook.py` + 전 주석 육안 | 통과 (판단 근거는 아래 오탐 표) |
| cryptic-var | `audit_notebook.py` + 신규 변수 19종 육안 | 통과. 약어·shadowing 0건 |
| assert-compare | `audit_notebook.py` + 육안 | 통과. `assert`/`allclose`/`array_equal` 0건 |
| gridspec | `audit_notebook.py` + 독립 grep | 통과. 셀 23 은 `plt.subplots(2, 2)` |
| epsilon-trick / loss-weight-trick | `audit_notebook.py` | 통과 (해당 없음) |
| helper-factory | `audit_notebook.py` + `def` 전수 | 통과. **함수 정의 0개** |
| docstring | `audit_notebook.py` + `"""`/`'''` grep | 통과 |
| hangul | `audit_notebook.py` + 독립 스캔 | 통과. 한글 0자 |
| test-leak | 육안 (split 추적) | **셀 17 이 자진 공개.** `coco8` 무 test split, `images/val` 이중 사용을 표+산문으로 명시 |
| lr-symbol / theta-symbol / np-matrix / chained-fit | `audit_notebook.py` + 육안 | 통과. **`alpha=` 오탐도 미발화** (셀 27 인자 순서 변경으로 자연 해소) |
| markdown 분량 | `markdown_budget.py` | **0 over cap**, soft 6 |
| **Step 4 삭제 연쇄** | 삭제 식별자 10종 + `IoU` 전수 검색, Exercises·Summary 참조 추적 | **통과. 잔재 0건** |
| **Step 번호 연속성** | 헤더 정규식 전수 | **통과. 0~7 연속, 옛 번호 0건** |
| **비교 실험 통제** | 셀 17 표 6행 ↔ 셀 18·19·20 인자 한 줄씩 대조 + 셀 22 predict 인자 대조 | **통과. 6행 전부 일치** |
| **결과 해설 정직성** | 셀 21·24·32 ↔ 셀 20·22 출력 숫자 대조, 열·행 양방향 | **통과. 한 행만 보고 쓴 문장 없음** |
| **검증되지 않은 인과** | 인과 표현 8종 + `mosaic`/`warmup` 전수 | **통과. 0건** |
| **학습/held-out 구분** | 5개 명시 자리 + 파일 경로 ↔ yaml split 대조 | **통과** |
| **`execute_result` 잔존** | 지시받은 JSON 스크립트 | **통과. 0건** |
| `train()` 이 새 `YOLO()` 인스턴스인가 | 셀 10·14·18·19·20·22·26·29 객체 출처 추적 | 통과. 셀 19·29 둘 다 `.train()` 직전 줄에서 신규 생성 |
| 마크다운이 코드보다 앞서 나감 | 전 마크다운 주장 ↔ 대응 셀 출력 | **통과.** 라운드 4 B-1(`masks.data`)은 **출력이 서술을 증명하는 상태로 해소** — 셀 26 이 `sample.jpg` 로 돌아가 `(7, 448, 640)` vs `orig_shape (1380, 2070)` |
| 학생 눈높이 | 전 마크다운 육안 | 통과. Python 문법 설명 0건 |
| 한 셀에 한 가지 | 변경 셀 16종 육안 | 통과 (셀 20·22·29 는 아래 오탐 표) |
| 루트 오염 | `ls -A` | 통과. `.pt`/`runs/` 없음, **`_workspace/` 이동 확인** |
| 그림 렌더링 | — | **리더가 육안 확인·통과 판정. 지시대로 재판정하지 않았다.** 그리는 코드만 검사 (차트 텍스트 전부 영어, `gridspec` 0, helper 0) |

### 오탐으로 판정해 걸러낸 것 (하네스 갱신용)

| 후보 | 판정 | 근거 |
|---|---|---|
| 셀 23 이 `panels` 리스트를 `for` 로 순회해 네 패널을 그린다 ("`for` 문으로 옵션을 순회하지 않는다") | **오탐** | 해당 규약은 `CLAUDE.md` "Model Definition Style" 소속으로 **모델 정의**에 한정된다. 여기서 순회하는 것은 (축, 이미지, 결과, 제목) 이며 모델이 아니다. 함수도 아니다 (`def` 0개). 대안은 9줄 × 4 복사이며 가독성이 낮다 |
| 셀 16 주석 `# ... reused by every figure below` ("메타 코멘트" 의심) | **오탐** | 금지 예시(`# same pattern as Section 3`)와 형태가 닮았으나, 이것은 설계 근거·diff 이력이 아니라 **변수의 사용 범위**를 알려 준다. 정의가 왜 지역 변수가 아닌지를 읽는 사람이 바로 알게 하는 정보이며 "이 줄이 무엇을 하는가" 안에 든다 |
| `mAP` 이 정의 없이 9회 등장 (라운드 4 A-3 의 재발 여부) | **결함 아님 — 사용자 결정** | 강의 슬라이드가 다루기로 확정됐다. **틀린 주장이 없음을 별도로 확인**했고(셀 32 의 잘못된 정의문은 삭제됨), 그것이 감사 대상이다 |
| 셀 20·22·29 가 한 셀에서 두세 가지를 한다 | **오탐** | 셋 다 **하나의 산출물**(3행 표 / 2이미지 목록 / 2행 표)을 만드는 데 필요한 최소 단위다. 쪼개면 `val()` 결과 변수만 셀 경계를 넘나든다 |
| 셀 17 `The second row is not controlled` 이 1행(독립변수)도 통제 안 됨을 말하지 않는다 | **오탐 — 등급 낮춤** | 문구가 느슨하나 이어지는 두 문장이 의미를 복원한다. 분량 여유 4단어라 손대면 캡을 넘긴다. 애매하면 NOTE 이하로 두라는 원칙에 따라 지적하지 않는다 |
| 셀 18·20·29 `val(..., seed=42, amp=False)` | **NOTE 로만 (C-4)** | 라운드 4 C-5 의 본질(통제표 오기재)은 해소됐다. 남은 인자는 무해하다 |
| 셀 22 가 셀 14 의 `detection_result` 를 재사용 (stale 의심) | **오탐** | `detection_model` 은 학습에 쓰이지 않는 별개 객체다 (`finetune_model` 이 학습). `conf=0.25` 도 동일해 **비교가 통제된다** |
| 루트의 `.gitignore` / `.omc` ("루트에 그 외 파일을 두지 않는다") | **저자 대상 아님** | 도구 파일이며 노트북 저작물이 아니다. 레이아웃 표의 갱신 대상 — 아래 하네스 갱신 6번 |
| Step 6 에 segmentation baseline arm 이 없음 (라운드 4 B-3 의 1항) | **재지적하지 않음 — 리더 결정** | 저자 보고 §3.7: 리더가 "Step 6 은 숫자 2줄 유지" 로 확정. B-3 의 2항(로그↔출력 모순)은 셀 28 의 `both the last epoch and the chosen checkpoint are shown` 으로 해소됐다 |

---

## 미검사 항목 (라운드 5)

- **셀 31 Roboflow 다운로드 경로 — 실행되지 않았다.** `ROBOFLOW_API_KEY` 부재로 가드 분기만 탔다.
  `'yolov8'` 문자열, `.download()` 반환값의 `.location`, `rglob` 출력 형태는 **감사에서 검증되지 않았다.**
  키를 가진 환경에서 한 번 돌리는 것 외에 해소 방법이 없다 (라운드 4 부터 이월).
- **셀 6 의 다운로드 분기 미실행.** `data/coco8*` 가 이미 존재해 최초 실행 학생 경로는 미검증이다.
- **셀 2 의 `except ImportError` 분기 미실행.** 패키지가 이미 설치돼 있었다. Colab 최초 경로 미검증.
- **그림 렌더링을 내가 판정하지 않았다.** 리더가 `build/figures/` 의 PNG 4장(셀 12·16·23·27)을
  직접 열어 통과 판정했고, 지시에 따라 재판정하지 않았다. 내가 검사한 것은 **그리는 코드의 규약 준수**뿐이다
  (차트 텍스트 영어 · `gridspec` 미사용 · helper 미사용 · 인덱스 안전성).
  저자가 보고한 **셀 23 아랫줄 왼쪽 `person` 라벨 잘림**은 육안 항목이므로 내 판정 대상이 아니다.
- **강의자료 대조를 하지 않았다.** 이 회차에 대응하는 강의 PDF 가 없고, 수식·기호 정합성은
  `consistency-verifier`(`verifier2`) 레인이다. mAP 정의 부재의 교육적 타당성은 사용자 결정이므로
  판정하지 않았다.
- **`plots=False` 로 학습 곡선 산출물이 없다.** 101 epoch 구간의 손실 거동을 그림으로 확인하지 못했다.
  이 리포트는 학습 **동역학**(왜 성능이 떨어지는가)에 어떤 주장도 걸지 않는다 — 라운드 4 에서
  내가 두 번 틀린 지점이다. A-1 이 쓰는 것은 동역학이 아니라 **로그에 평문으로 인쇄된 사실**
  (`101 epochs completed`, `patience=100`, `Best results observed at epoch 1`)뿐이다.
- **초판에서 셀 19·29 의 학습 로그를 읽지 않았다 — 미검사가 아니라 누락이었다.**
  `verifier2` 의 반증으로 뒤늦게 검사해 A-1 을 올렸다. 같은 셀의 로그에 **아직 내가 대조하지 않은
  다른 인쇄 사실이 있을 수 있다** — 이번에 확인한 것은 `epochs`·`patience`·`optimizer`·`lr0`·
  `close_mosaic`·`seed`·`imgsz`·`batch` 여덟 인자와 종료 epoch 뿐이다. 하네스 갱신 9번의
  스크립트가 없는 한 이 대조는 매번 사람 손에 달려 있다.
- **저자가 보고한 나머지 val 3장 프로브**(036·042·061)를 재실행하지 않았다. 노트북에 실리지 않는
  숫자이므로 학생 배포물에 영향이 없다.
- **Exercise 2 를 실제로 수행해 보지 않았다.** B-1 은 셀 19·20·22·23 의 소스 대조로 판정했다.

---

## 검증자 조율 (`verifier2`)

| 항목 | 감사자 | 검증자 | 합의 |
|---|---|---|---|
| 변수명 전반 (신규 19종 포함) | 지적 0건 | 지적 0건 | **양쪽 0건 확정.** 저자에게 변수명 지시를 보내지 않는다 |
| `instance_colors` 명명 | 규약 부합 | 개선으로 평가 | 지적하지 않음 |
| `lr-symbol`/`theta-symbol` 미발화 | 정상 | 정상 (강의자료 부재로 기호 권위가 ultralytics API 뿐) | 재지적 없음 |
| **`120 epochs` ↔ 실제 101** | 초판 **FIX(B-1)** — Exercise 2 발현으로만 판단 | **치명 V-11 — 배포 상태에서 이미 거짓** | **검증자가 옳다.** 내가 반증을 직접 확인해 **BLOCK(A-1) 으로 승격**하고 판정을 PASS → **FAIL** 로 정정했다 |
| `best.pt` 가 epoch 1 이라는 사실 미기재 | A-1 "왜 문제인가 2" 로 흡수 | **경미 V-12** | **검증자 소관.** 내가 중복 발신하지 않는다. 셀 21 에서 처리하며 **셀 17 에는 넣지 않는다** (116/120) |

**저자 발신 창구**: 저자에게는 **검증자 리포트의 V-11 하나로만** 나간다 (내 A-1 이 그 안에 명시적으로
포함돼 있음을 검증자가 확인했다). **나는 별도 지시를 보내지 않는다.**
수정 착수 전에 **리더가 (a)/(b) 를 택일**해야 한다 — (b) 를 고르면 전 숫자와 그림 4장이 재측정된다.

**저자 발신**: 리더 지시대로 저자에게 직접 발신하지 않는다. 리더가 병합해 단일 지시로 낸다.

---

## 하네스 갱신 제출 (라운드 5)

1. **[즉시 반영 — 오탐 억제, 라운드 4 제안 1번 재확인]** `lr-symbol` 의 matplotlib `alpha=` 오탐이
   이번 라운드에는 **발화하지 않았다** — 셀 27 의 인자 순서가 바뀌어 `alpha=` 가 줄머리를 벗어났다.
   프로필의 `known_false_positive` 기재가 회피법까지 적어 둔 것이 실제로 작동한 사례다. **유지할 것.**
   다만 패턴을 좁히자는 제안(`^\s*(alpha|eta|step_size|learning_rate)\s*=(?!.*[,)])`)은 여전히 미반영이며,
   줄바꿈 형태가 바뀌면 재발한다.
2. **[체크포인트 신설 — 2회 반복 확인, 승인 대상]**
   **"Exercise 가 학생에게 바꾸라고 지시하는 값이, 출력 라벨·`print` 문자열·그림 제목에 하드코딩돼
   있지 않은지 확인한다."** 라운드 4 의 B-2(Exercise 3 → `best_iou` 판정 문자열)와 이번 B-1
   (Exercise 2 → `120 epochs` 라벨 4곳)이 **같은 결함의 2회 반복**이다. 정규식으로 잡히지 않지만
   기계화는 가능하다 — Exercises 셀에서 백틱 식별자와 숫자를 뽑아 코드 셀의 문자열 리터럴과 대조.
   최소한 `notebook-convention-audit/SKILL.md` 수동 체크리스트에 한 줄.
3. **[규약 보강 — 라운드 4 제안 3번, 여전히 미반영]** `CLAUDE.md` "비교 실험 통제 변수" 의 통제 항목
   목록에 **"모델 선택 절차(early stopping / best checkpoint 를 어느 split 에서 고르는가)"** 를 추가할 것.
   이번 통제표는 저자가 `checkpoint` 행을 자발적으로 넣어 통과했지만, **규약 문서에는 아직 없어서
   다음 회차 저자는 같은 것을 다시 발명해야 한다.** 라운드 4 에서 BLOCK 1건의 원인이었던 항목이다.
4. **[체크포인트 신설 — 라운드 4 제안 2번의 결말 기록]** "학습 후 지표를 `best.pt` 에서 읽으면서
   그 사실을 서술하지 않음" 은 이번 개정에서 **저자가 `last`/`best` 두 행을 모두 출력하고 셀 21 에
   설명을 붙여 자발적으로 해소**했다. 이 형태(두 행 + 한 문단)가 다음 회차의 기본값이 되도록
   `code-patterns.md` 에 체크포인트 보고 절을 신설할 것을 제안한다 — 저자 보고 §9 의 같은 제출과 병합.
5. **[트리거 3 — 2회 반복, 사용자 판단 필요]** `CLAUDE.md` "Notebook Generator Scripts" 절이 여전히
   `_gen_p{NN}.py` 명명과 **"검증이 끝나면 삭제한다"** 를 말하는데, 실제는 `gen/p{NN}.py` 보존이고
   **이번 개정 3건이 전부 그 보존 덕분에 가능했다.** 저자가 라운드 4·5 연속 제출했고 미반영이다.
   규약과 관행이 어긋난 상태이므로 어느 쪽을 고칠지 사용자 판단이 필요하다.
6. **[트리거 3 — 신규]** `CLAUDE.md` "디렉터리 레이아웃" 표와 프로필 `notebooks.layout` 이 루트에
   `Practice*.ipynb` 와 `CLAUDE.md` **만** 허용하는데, 실제 루트에 `.gitignore` 와 `.omc/` 가 있다.
   둘 다 정당한 도구 파일이다. 표에 "루트의 dotfile·도구 디렉터리는 예외" 한 행을 넣지 않으면
   **매 라운드 감사자가 같은 판단을 반복하고 언젠가 저자에게 잘못 지적한다.**
   (`_workspace/` 는 `md/p12_workspace/` 로 이동해 해소됐다 — 이번 이동이 옳은 처리다.)
7. ~~**[트리거 1 — 내 절차의 개선 기록]**~~ **철회한다.** 초판에서 나는 *"이번 라운드에는 학습
   동역학에 어떤 주장도 걸지 않고 판정을 재현 가능한 대조 셋으로만 구성했다"* 를 개선 사례로 적었다.
   **그 절차가 바로 이번 BLOCK 을 놓친 원인이다.** 8번으로 대체한다.

8. **[트리거 1 — 이번 라운드에서 가장 값진 항목. 즉시 반영 대상]**
   **"학습·평가 API 를 호출하는 셀은 최종 `print` 블록이 아니라 로그 전문을 읽는다.
   최소한 `epochs completed`·`EarlyStopping`·`optimizer:`·`patience` 네 줄은 반드시 확인하고,
   노트북이 인쇄한 하이퍼파라미터 문구와 한 줄씩 대조한다."**
   / `notebook-convention-audit/SKILL.md` 수동 체크리스트 /
   근거: **나는 PASS 를 냈고 틀렸다.** 셀 18·20·22·26·29 의 마지막 `print` 블록만 뽑아 숫자를
   대조했고 셀 19·29 의 학습 로그 본문은 열지 않았다. 그 로그 안에 `101 epochs completed` 가
   **평문으로** 있었고, 노트북은 여덟 곳에서 `120` 이라고 말하고 있었다.

   **왜 놓쳤는지가 핵심이다.** 라운드 4 에서 나는 학습 로그의 지표 변화를 두 번 잘못 해석했고
   (한 번은 없는 결함을 만들고, 한 번은 검증되지 않은 인과를 노트북에 넣자고 제안했다),
   그 반성으로 *"학습 동역학에 주장을 걸지 않는다"* 를 세웠다. **그것을 "로그를 읽지 않는다" 로
   잘못 실행했다.** 로그에는 두 종류가 있다 —
   - **해석이 필요한 것** (epoch 12 에서 mAP 이 왜 떨어지는가): 여기서는 등급을 올리지 말고
     통제 실행 없이 원인을 적지 않는다. 라운드 4 의 교훈은 여기까지만 유효하다
   - **해석이 필요 없는 사실** (`101 epochs completed`, `Best results observed at epoch 1`,
     `patience=100`): **이것은 그냥 읽으면 되는 숫자이고, 마크다운과 대조하는 것이 감사의 본업이다**

   **과잉 교정이 새 사각지대를 만들었다.** 지난 실패를 피하려고 만든 규칙이 다음 실패의 원인이 되는
   형태이며, 이것이 이번 라운드에 내가 남길 가장 중요한 기록이다. `SKILL.md` 에 위 두 종류의
   구분을 명시하지 않으면 다음 감사자도 같은 곳에 선다.

9. **[트리거 4 — 정규식 체크포인트 신설 후보, 기계화 가능]**
   **"노트북 텍스트가 주장하는 하이퍼파라미터 수치 ↔ 셀 출력 로그의 실측치 대조."**
   마크다운·`print`·차트 제목에서 `\d+\s*epochs?` 를 뽑고, 같은 노트북의 stream 출력에서
   `(\d+) epochs completed` 를 뽑아 다르면 후보로 올린다. **이번 결함은 이 정규식 두 줄로 잡힌다.**
   `epochs` 뿐 아니라 `lr`·`batch`·`imgsz` 로 확장 가능하며, 라이브러리가 인자를 무시하거나
   덮어쓰는 경우(라운드 4 의 `optimizer='auto'` → `AdamW(lr=0.000119)` 건과 **같은 계열**)를
   구조적으로 잡는다. `audit_notebook.py` 는 현재 **source 만 보고 outputs 를 보지 않는데,
   이 회차의 결함 2건이 연속으로 outputs 안에 있었다.** 스크립트 확장 1순위.

10. **[트리거 3 — 저자 보고의 자기 판정을 믿으면 안 되는 사례]**
    저자 보고 §3.6 은 셀 17·21·24·32 의 주장을 출력과 대조한 표를 실었고 **전부 "참"** 으로 적었다.
    그 표에 **`120 epochs` 행이 없다** — 저자도 자기가 쓴 숫자를 대조 대상으로 삼지 않았다.
    `practice-notebook-authoring` 의 자체 점검 절에 **"노트북이 인쇄한 하이퍼파라미터는 전부
    로그 실측치와 대조한다"** 를 추가할 것. 저자·감사자·검증자 셋 중 **검증자만** 잡았다.

---
---

# [보존] 이전 라운드 이력

# Convention Audit — Practice12_Object_Detection_and_Segmentation.ipynb (라운드 4, 최종)

## 판정: PASS
- **BLOCK 2건 전부 해소** (A-1, A-3)
- **FIX 3건 전부 해소** (B-1, B-2, B-3)
- **NOTE 4건 전부 미반영** (C-1, C-2, C-4, C-5) — 전부 선택 사항이며 판정에 영향 없음.
  아래 "NOTE 미반영 현황" 에 개별 상태를 적었다
- 초안 NOTE 1건(C-3)은 **내 오탐으로 판정해 철회**했다 (저자 미조치가 아니라 지적 자체를 거둔 것)

라운드 4 최종본을 **노트북 실물로 재검증했다** (리더 보고를 그대로 받지 않고 독립 확인):

| 확인 항목 | 결과 |
|---|---|
| `audit_notebook.py` (16종) | 후보 없음 |
| `markdown_budget.py` | 0 over cap (셀 24 = 76/80 산문, 셀 17 = 118/120) |
| 전 35셀 실행 | 에러 0, `execution_count` 1–20 연속 = 통짜 재실행 |
| 셀 24 ↔ 셀 23 출력 | 네 숫자 일치. 두 지표 방향 모두 참 (0.63026 < 0.68437, **0.41646 > 0.39986**) |
| 인과 표현 | `mosaic`/`warmup`/`overfit`/`due to`/`caused`/`therefore`/`thus`/`hence` **0건**. `because` 1건은 셀 10 의 검증된 인과(오탐 확정) |
| 한글·이모지 | 0. source 의 non-ASCII 는 셀 0 제목의 `—`(U+2014) 하나뿐 |
| 루트 청결 | `.pt`/`runs/` 없음 |

**미검증으로 남는 항목** (실행으로 해소되지 않음 — 최종 보고에 반드시 포함할 것):

- **Roboflow `.download()` 왕복** — `ROBOFLOW_API_KEY` 부재로 가드 분기만 실행됐다.
  `model_format='yolov8'`, `.location` 반환, `rglob` 출력은 **키를 가진 환경에서만 검증된다**
- **셀 6 최초 다운로드 분기** — `data/coco8*` 가 이미 존재해 미실행. 최초 실행 학생 경로 미검증
- **셀 2 `except ImportError` 분기** — 패키지가 이미 설치돼 있어 미실행. Colab 최초 실행 경로 미검증
- **시각화 4셀(12·16·19·27)의 렌더링** — 코드와 출력 메타데이터로만 판정. 그림 육안 확인 안 함

> **라운드 이력.** 이 문서는 라운드 1 초안에서 네 번 개정됐고, **개정의 다수가 내 판정을 낮추는
> 방향이었다.** 판정 이력과 내가 틀린 지점은 본문에 그대로 남겼다 — A-1 각주(내 오판 2회와 각각의
> 반증 경로), C-3(내 오탐 철회), 하네스 갱신 6번(오판의 유형화). 다음 회차 감사자가 같은 로그를
> 보고 같은 경로를 밟지 않도록 지운 것이 없다.

---

## NOTE 미반영 현황 (전부 선택 사항, 판정 무관)

| 항목 | 상태 | 확인 |
|---|---|---|
| C-1. fuse 함정이 코드 주석에만 있고 마크다운에 없다 | **미반영** | 마크다운 15셀 전수 검색 — fuse/fresh-object 함정 언급 0건. 셀 22·29 주석은 그대로 |
| C-2. 셀 10 이 세 가지를 한 셀에서 한다 | **미반영** | 여전히 27줄 단일 셀, `open(...)` 2회 |
| C-4. 셀 3 의 `device` 가 계산·출력되고 안 쓰인다 | **미반영** | `device` 는 셀 3 에서만 등장하고 어떤 `train()`/`val()`/`predict()` 에도 전달되지 않는다 |
| C-5. 셀 21 이 `val()` 에 학습 전용 인자를 넘기고 통제표가 통제 항목으로 적는다 | **미반영** | 셀 20 표에 `seed`/`amp` 행 유지 |

---

---

## BLOCK

### A-1. 셀 20 통제표의 B 행이 사실과 다르다 — 두 팔의 체크포인트 선택 절차가 다르다
- 위치: 셀 20 (markdown), 셀 21·22·23 (code)
- 원문 (셀 20): `The comparison below changes one thing only.`
- 원문 (셀 20 표 B 행): `| **weights before evaluation** | **COCO pretrained, untouched** | **COCO pretrained + 20 epochs on coco8** |`
- **반증 — 셀 22 자신의 출력**:
  ```
  epoch  1–11 :  mAP50 0.688   mAP50-95 0.422    (소수 3자리까지 불변)
  epoch 12–20 :  mAP50 0.630   mAP50-95 0.416    (close_mosaic=10 지점에서 바뀜)
  Validating ...\detect_finetune\weights\best.pt
  all  4  17   0.677  0.667  0.688  0.423          <- 보고된 값의 출처
  ```
  즉 B 가 평가받는 대상은 "20 epoch 학습한 가중치" 가 아니라 **21개 체크포인트(초기 + 20 epoch) 중
  val mAP 가 가장 높은 것**이다.
- 위반 규약: `CLAUDE.md` "비교 실험 통제 변수" — *"격리하려는 변수 하나만 다르고 나머지는 반드시
  동일해야 한다"*, *"변경된 항목은 마크다운 표나 한 문장으로 명시한다"*
- 왜 문제인가: **A 는 체크포인트 1개, B 는 21개 중 최댓값**이고, 그 최댓값을 고른 기준이
  **보고하는 바로 그 split(`images/val`)** 이다. A 에는 그런 선택 기회가 없다.
  선택 절차가 두 팔에서 다르므로 `changes one thing only` 은 성립하지 않으며, 통제표에는 이 행이 없다.
  대상이 대학원생이고 이들이 자기 연구에서 그대로 흉내 낼 절차다 — 평가 split 위에서 체크포인트를
  고르고 그 split 의 점수를 성능으로 보고하는 것은 논문에서 그대로 반박당하는 형태다.
- 수정 방향 (표 한 행 + 해설 한 줄이면 끝난다):
  1. 셀 20 표의 B 행을 `**COCO pretrained + 20 epochs on coco8, best checkpoint by val mAP**` 로 고치고,
     `changes one thing only` 을 두 팔의 선택 절차가 다르다는 한 문장으로 대체한다 —
     *"One thing differs by design, and one more differs by necessity: the fine-tuned arm reports its
     best checkpoint, chosen on the same `images/val` we score on."*
  2. 셀 24 에 한 줄 추가 — *"Validation mAP drops from epoch 12, where the mosaic augmentation closes;
     the number above is the best checkpoint, not the last one."*

> **각주 — 초안의 과잉 판정을 철회한다.** 초안에서 나는 여기에 "셀 24 의 결론은 부호가 틀렸다
> (20 epoch 종료 시점 0.630 < baseline 0.684 이므로 fine-tuning 이 모델을 나쁘게 만들었다)" 를
> 함께 올렸다. **검증자 반론을 검증해 철회한다.** 이유 셋이며 셋 다 내가 직접 확인했다.
> - `.train()` 이 산출하는 것은 `best.pt` 이고 실무에서 배포하는 것도 그것이다. 셀 23 이 비교하는
>   대상은 `last.pt` 가 아니라 `best.pt` 이므로, 셀 24 의 "gain" 은 **정확히 계산된 값**이다.
> - 지표가 epoch 1–11 동안 소수 3자리까지 고정된 이유는 **학습이 사실상 일어나지 않아서**다 —
>   4장 / `batch=4` → `nb=1`, 즉 **전 학습이 optimizer step 20회**이고 적용 lr 은 `AdamW(lr=0.000119)`
>   (셀 22·29 로그 첫머리에서 확인).
> - val 4장·17 instances 의 mAP 는 **성긴 계단함수**다. 11 epoch 동안 소수 3자리까지 *완전히* 고정이다가
>   한 번에 한 단 떨어지는 모양은 "서서히 열화" 가 아니라 "임계를 한 번 넘음" 이다.
>   값이 연속으로 흐르지 않는 것 자체가 근거다.
> - epoch 12 하락의 **원인은 확인되지 않았다.** 나는 한때 `close_mosaic=10` 을 원인으로 지목했으나
>   **리더의 통제 실험이 이를 반증했다** — `close_mosaic=0` 으로 두어 `Closing dataloader mosaic` 이
>   로그에 아예 없는 조건에서도 하락은 **똑같이 epoch 12 에서** 일어난다 (mAP50 0.688 → 0.630).
>   로그 두 줄이 가까운 자리에 보인 것은 **상관이지 인과가 아니었다.**
> - 따라서 "fine-tuning 이 모델을 열화시켰다" 도, "증강 스케줄 때문이다" 도 **둘 다 근거가 없다.**
>   셀 24 의 결론(*"An eight-image dataset shows that the pipeline runs, not that fine-tuning helps"*)은
>   오히려 이 데이터가 뒷받침한다.
>
> **이 각주에서 노트북에 인쇄해도 되는 것은 관측뿐이다** — 4장 / `batch=4` → optimizer step 20회,
> `lr0=0.000119`, 20 epoch 동안 val 지표가 **두 값밖에 갖지 않음**. 결론은 인과 단정이 아니라
> *"이 규모에서는 어느 방향으로도 판단할 수 없다"* 로 닫는다 (리더 확정).
>
> **내가 두 번 틀렸다는 것을 기록해 둔다.** 처음에는 하락을 학습 효과로 읽었고(검증자가 반박),
> 다음에는 그 원인을 `close_mosaic` 으로 지목했다(리더가 통제 실험으로 반박).
> 두 번째가 더 나빴다 — **나는 그 설명 문장을 노트북에 넣으라고 제안했다.** 그대로 들어갔다면
> 검증되지 않은 인과가 학생에게 인쇄됐을 것이고, 그것은 이번 라운드 내내 우리가 잡아 온 결함
> (마크다운의 단정문을 출력이 뒷받침하지 못함)과 **정확히 같은 종류**다.
> 감사자가 그 결함을 잡는 쪽이 아니라 만드는 쪽에 섰다.
>
> 초안의 수정 지시를 저자가 그대로 따랐다면 **정직한 결론을 거짓으로 고쳐 쓰게 됐을 것이다.**
> 거짓인 것은 셀 24 가 아니라 셀 20 통제표의 B 행이며, 위 (1)(2)가 그것을 고친다.
> 감사자가 만들어낸 위반을 저자가 반영하는 것이 가장 나쁜 실패라 여기 남긴다.

### A-3. mAP 을 6회 보고하면서 한 번도 정의하지 않았고, 유일한 설명은 틀렸다
- 위치: 셀 17 (markdown, 정의 누락), 셀 32 (markdown, 부정확한 서술)
- 원문 (셀 17 — IoU·Precision·Recall 은 수식으로 주고 곧바로 표로 넘어간다):
  ```
  | Metric | IoU thresholds it uses |
  | mAP@50    | 0.50 only |
  | mAP@75    | 0.75 only |
  | mAP@50-95 | the average over 0.50, 0.55, ... 0.95 |
  ```
- 원문 (셀 32): `IoU decides whether a detection counts, and mAP averages precision over a range of IoU thresholds.`
- 위반 규약: 명세 §2.3 이 **도입할 수식으로 명시**한
  $AP = \int_0^1 p(r)\,dr$, $mAP = \frac{1}{C}\sum_{c=1}^{C} AP_c$ 가 노트북에 없다.
  `CLAUDE.md` "Target Audience" (*"수식의 각 항이 코드의 어느 줄인지를 드러낸다"*),
  "설명 분량 — Summary 에 새 내용을 넣지 않는다"
- 왜 문제인가:
  - `mAP` 은 이 노트북의 중심 지표다 — 셀 21·23(×2)·29(×2)·32 에서 보고된다. 그런데 **`A` 와 `m` 이
    무엇의 약자인지조차 나오지 않는다.** 셀 17 은 IoU 와 Precision/Recall 은 수식으로 정의해 놓고
    정작 mAP 은 "어떤 IoU 임계값을 쓰는가" 만 표로 준다. 임계값은 mAP 의 정의가 아니라 변형의 축이다.
  - 셀 32 의 서술은 **셀 17 의 표와 모순된다.** 표는 mAP@50 이 임계값 "0.50 only" 라고 하는데,
    Summary 는 mAP 을 "averages precision over a range of IoU thresholds" 로 정의한다.
    그 정의대로면 mAP@50 은 mAP 이 아니게 된다. 그리고 정작 진짜 두 평균 — **recall 에 대한 적분(AP)**
    과 **클래스에 대한 평균(m)** — 은 둘 다 빠졌다.
  - 클래스 평균은 셀 29 출력에 `person / dog / horse / elephant / umbrella / potted plant` 행으로
    **화면에 이미 나와 있다.** 학생은 그 행들이 `all` 행과 무슨 관계인지 알 방법이 없다.
- 수정 방향: 셀 17 에 명세 §2.3 의 두 수식을 넣는다.
  ```
  $$AP = \int_0^1 p(r)\,dr, \qquad mAP = \frac{1}{C}\sum_{c=1}^{C} AP_c$$
  ```
  한 줄 덧붙일 것 — *"AP is the area under one class's precision-recall curve; the m averages it over classes."*
  그 뒤 셀 32 의 mAP 문장을 그에 맞게 고친다.
  - **분량 주의**: 셀 17 은 현재 102 단어(하드 120)다. 자리를 만들려면 **`mAP@75` 행을 지운다** —
    노트북 어디에서도 `.box.map75` 를 계산하지 않는다. 수정 후 `markdown_budget.py` 를 반드시 재실행할 것.
- **공식 문서 근거** (검증자가 이번 라운드에 직접 확인해 제공):
  `docs.ultralytics.com/guides/yolo-performance-metrics/` — mAP 는 *"computes the **area under the
  precision-recall curve**"*, *"extends across **multiple object classes**"*. **IoU 임계값 범위 평균은
  mAP50-95 에만 해당한다.** 즉 셀 32 의 현재 문장은 공식 문서 정의와도 어긋난다.

---

## FIX

### B-1. 셀 25 의 `masks.data` 서술을 셀 26 의 출력이 반박한다 — 저자의 자체 수정이 미완이다
- 위치: 셀 25 (markdown), 셀 26 (code 출력)
- 원문 (셀 25 표): `| masks.data | binary grid sized by the model input, not by the image |`
- 원문 (셀 26 출력):
  ```
  masks.data   : (5, 480, 640)
  orig_shape   : (480, 640)
  ```
- 위반 규약: `CLAUDE.md` "Notebook 편집 — 전체 맥락 우선" (마크다운이 코드 출력과 어긋남), 명세 §3.5
- 왜 문제인가: 표가 `not by the image` 라고 단언하는데 바로 아래 출력은 두 값이 **정확히 같다**.
  일반 명제로는 참(격자는 모델 입력이 정한다)이지만 `not by the image` 라는 표현은 "이미지와 다르다"로
  읽히고, 학생 화면의 숫자가 그것을 반박한다.
  저자 보고 §4-4 는 이 항목을 고쳤다고 했으나 **산문 한 문장만 고치고 표의 부정 표현은 그대로 남았다.**
  근본 원인은 따로 있다 — 명세 셀 26 은 `data/sample.jpg`(2070×1380, 실재 확인함)를 쓰도록
  리더가 §6 #3 에서 판정했는데, 저자가 480×640 coco8 이미지로 바꾸면서 두 shape 가 우연히 일치하게 됐다.
- 수정 방향: **(a) 로 확정** (검증자·리더 합의).
  - (a) 셀 26·27 의 source 를 `DATA / 'sample.jpg'` 로 되돌린다. 검증자가 설치본
    `ultralytics 8.4.120` 소스로 기전까지 확인했다 — `SegmentationPredictor.construct_result` 가
    기본 `retina_masks=False` 에서 `ops.process_mask(..., img.shape[2:], upsample=True)` 를 쓰므로
    masks 는 **letterbox 된 모델 입력** 해상도다. 480×640 은 `imgsz=640` letterbox 결과가 480×640
    (480 이 32 배수라 패딩 0)이라 우연히 원본과 같아진 것이고, sample.jpg(2070×1380)면
    640/2070 = 0.3092 → 1380 × 0.3092 = 426.7 → stride 32 패딩 → **(N, 448, 640)** 으로
    명세 §3.5 의 리더 실측과 정확히 일치한다.
  - 따라서 (a) 는 "되돌리면 서술이 참이 된다" 가 아니라 **"출력이 서술을 증명한다"** 이다.
    이 편이 (b)(문구만 완화)보다 교육적으로 우월하다.
  - **동반 수정 필요**: 셀 27 `fill_colors` 는 8색 고정이라 인스턴스 수가 바뀌면 `IndexError` 가 난다.
- **저자 지시 발신자**: 이 항목은 **검증자가 이미 저자에게 (a)안으로 보냈다** (`fill_colors` 동반 수정 포함).
  중복 발신 방지를 위해 나는 보내지 않았다.

### B-3. Step 7 에 baseline 이 없어 보고된 숫자를 해석할 수 없고, 로그와 출력이 모순돼 보인다
> 초안에서 BLOCK(A-2)으로 올렸다가 **FIX 로 강등**했다. A-1 각주의 반론이 여기에도 적용된다 —
> `best.pt` 보고는 표준 관행이고 "fine-tuning 이 열화시켰다" 는 내 초안 서술은 과잉이었다.
> 다만 아래 두 가지는 그 반론과 무관하게 남는다.

- 위치: 셀 28 (markdown), 셀 29 (code)
- 원문 (셀 29 출력): `box mAP@50 : 0.76875` / `mask mAP@50 : 0.73537`
- 원문 (셀 29 로그, 위 출력 직전): epoch 20 시점 `Box mAP50 0.565` / `Mask mAP50 0.555`
- 위반 규약: `CLAUDE.md` "설명 분량 — 코드 뒤 해설은 '이 출력이 무엇을 말하는가' 만",
  "Notebook 편집 — 전체 맥락 우선"
- 왜 문제인가: 두 가지다.
  1. **baseline arm 이 없다.** Step 5 는 pretrained/fine-tuned 두 줄을 나란히 주는데 Step 7 은
     학습 후 숫자만 준다. 0.769 가 좋은지 나쁜지 판단할 기준이 학생에게 없고, 절 제목이
     "Segmentation Fine-tuning" 이라 숫자는 자연히 성과로 읽힌다. Step 5·7 의 구성이 비대칭이다.
  2. **화면상 로그와 출력이 모순돼 보인다.** 마지막 epoch 이 `0.555` 로 찍힌 직후
     `mask mAP@50 : 0.73537` 이 나온다. `best.pt` 의 존재를 노트북이 한 번도 설명하지 않으므로
     **주의 깊은 학생일수록 노트북이 틀렸다고 결론짓는다.** (실제로 초안 단계의 나도 그렇게 읽었다.)
- 수정 방향: A-1 (2)의 한 줄 해설을 Step 7 에도 넣는다. 가능하면 Step 5 와 형태를 맞춰
  `segmentation_baseline_model` 로 학습 전 `val()` 을 한 셀 추가하면 1·2 가 동시에 해소된다.

### B-2. 셀 18 이 검사하지 않은 판정을 출력 문자열에 박아 두었다
- 위치: 셀 18 (code, 마지막 줄)
- 원문: `print(f'best match : IoU={best_iou:.3f}  -> a true positive at threshold 0.50')`
- 위반 규약: `CLAUDE.md` "Notebook 편집 — 전체 맥락 우선", "Result Comparison"
  (*"값을 직접 보고 일치를 인지하게 한다"* — 판정을 코드가 아닌 문자열이 대신하고 있다)
- 왜 문제인가: `-> a true positive at threshold 0.50` 은 **`best_iou` 와 무관한 고정 문자열**이다.
  현재 값 0.939 에서는 참이지만, 임계값과 비교하는 코드가 어디에도 없다.
  그리고 이 노트북의 **셀 33 Exercise 3 이 학생에게 정확히 그 상태를 만들라고 지시한다** —
  *"Run the IoU loop again with `ground_truth_box` built from the second label line."*
  두 번째 라벨로 바꾸면 best IoU 가 0.50 아래로 내려갈 수 있고, 그때 노트북은 **거짓을 출력한다.**
  Exercise 1(`conf` 0.25 → 0.5)도 검출 집합을 줄여 같은 상황을 만들 수 있다.
  Step 4 의 교육 목표가 "임계값이 판정을 가른다" 인데, 정작 그 판정을 코드가 하지 않는다.
- 수정 방향: 임계값을 이름 있는 변수로 올리고 비교를 코드로 시킨다. 셀 17 의 표와도 이어진다.
  ```python
  iou_threshold = 0.50
  ...
  verdict = 'true positive' if best_iou >= iou_threshold else 'false positive'
  print(f'best match : IoU={best_iou:.3f}  -> {verdict} at threshold {iou_threshold:.2f}')
  ```

---

## NOTE

### C-1. fuse 함정이 코드 주석에만 있고 마크다운에 없다
셀 22 `# training starts from another fresh object, never from one that has already run inference`,
셀 29 `# again a fresh object, so that the inference above cannot affect this training`.
명세 §3.2 가 "예외 없이 mAP 만 0 이 된다"고 기록한 이 함정은 **이 노트북에서 가장 실용적인 교훈**인데,
학생이 읽는 마크다운에는 한 줄도 없다. `CLAUDE.md` "Notebook Comments" 는 근거·대안 설명을
마크다운으로 옮기라고 한다. 셀 20 또는 셀 28 에 한 문장 추가를 제안한다 —
*"Each training call starts from a freshly constructed `YOLO(...)`. An object that has already run
`predict()` is fused, and training a fused model silently produces zero mAP."*
(주석 자체는 "무엇을 하는가"의 범위 안에 있다고 보아 위반으로 올리지 않았다.)

### C-2. 셀 10 이 세 가지를 한 셀에서 한다
클래스 이름 확보 + `data.yaml` 2개 생성 + 앞 8줄 출력. `CLAUDE.md` "길어지지 않게 하는 규칙 — 한 셀에
한 가지". yaml 쓰기 블록 7줄이 그대로 두 번 반복돼 셀이 26줄이다. 이름 확보/출력과 yaml 생성을
두 셀로 나누면 읽기 쉬워진다. (해당 규칙의 소속 절이 마크다운 분량이므로 코드 셀에는 NOTE 로 둔다.
반복 자체는 "helper function 금지" 규약상 오히려 옳은 형태이므로 함수로 묶지는 말 것.)

### ~~C-3. 같은 클래스 사전을 두 이름으로 참조한다~~ — **철회 (내 오탐)**
초안에서 셀 18 의 `class_names[ground_truth_class]` 와 `detection_result.names[class_index]` 병용을
"한 번 정의한 데이터는 같은 이름으로 재사용한다" 위반 후보로 올렸다. **검증자와 조율하며 다시 보니
오탐이다.** 노트북은 혼용하는 것이 아니라 **일관된 규칙을 지키고 있다** — ground truth(라벨 파일에서
온 것)에는 `class_names`(우리가 쓴 yaml 의 어휘), 예측에는 `detection_result.names`(모델의 어휘)를
쓴다. 셀 12(라벨)→`class_names`, 셀 15·16(예측)→`result.names`, 셀 18(양쪽)→각각, 셀 26(예측)→
`result.names` 로 예외 없이 일관된다. 두 어휘가 원리적으로 다를 수 있으므로 오히려 **정확한 구분**이다.
저자에게 보내지 않는다.

### C-4. 셀 3 의 `device` 가 계산·출력되고 어디에도 쓰이지 않는다
`device = 'cuda' if torch.cuda.is_available() else 'cpu'` 를 출력하지만 `train()`/`val()`/`predict()`
어느 호출에도 넘기지 않는다 (ultralytics 가 자체 선택한다). 학생은 이 변수가 배치를 결정한다고 오해한다.
`device=device` 를 실제로 넘기거나, 출력만 남기고 변수를 없애는 편이 정직하다.

### C-5. 셀 21 이 `val()` 에 학습 전용 인자를 넘기고 통제표가 그것을 통제 항목으로 적는다
`baseline_model.val(..., seed=42, amp=False, ...)` — `seed` 와 `amp` 는 순수 평가 호출에 영향이 없다.
셀 20 표가 이 둘을 A/B 통제 항목으로 나열하면 "맞춰야 할 요인" 으로 읽힌다. 에러는 나지 않고
A-1 의 진짜 통제 문제를 가리는 효과가 있으므로, A-1 수정 시 함께 정리할 것을 권한다.

---

## 검사한 체크포인트

| 체크포인트 | 방법 | 결과 |
|-----------|------|------|
| emoji | `audit_notebook.py` + 전 셀 non-ASCII 독립 스캔 | 통과. source 의 non-ASCII 는 셀 0 의 `—`(U+2014) 하나뿐 — 제목 house style |
| cross-ref | `audit_notebook.py` + 육안 | 통과. `Practice 12` 자기 참조만, `Ch`/`Lec`/타 `Practice` 없음 |
| meta-comment | `audit_notebook.py` + 전 주석 육안 | 통과 (판단 근거는 C-1) |
| cryptic-var | `audit_notebook.py` + 전 변수 육안 | 통과. 약어 없음, 명세 §5 표와 대조 완료 |
| assert-compare | `audit_notebook.py` + 육안 | 통과. `assert`/`allclose` 0건, 셀 23 은 `print` 나란히 출력 |
| gridspec | `audit_notebook.py` | 통과. `plt.subplots` 만 사용 |
| epsilon-trick | `audit_notebook.py` | 통과 (해당 없음 — 표준화 블록 없음) |
| loss-weight-trick | `audit_notebook.py` | 통과 (해당 없음) |
| helper-factory | `audit_notebook.py` + 육안 | 통과. 함수 정의 0개 |
| docstring | `audit_notebook.py` | 통과. `#` 주석만 |
| hangul | `audit_notebook.py` + 독립 스캔 | 통과. 한글 0자 |
| test-leak | 육안 (분할 구조 추적) | **A-1 로 보고.** test split 자체가 없고 `images/val` 이 체크포인트 선택과 최종 보고에 이중 사용됨 |
| lr-symbol / theta-symbol / np-matrix / chained-fit | `audit_notebook.py` + 육안 | 통과. 명세 §1 대로 발화할 자리가 없음. `.train()`/`.predict()` 체이닝은 정규식이 못 잡으므로 셀 14·21·22·26·29·31 을 육안 확인 — 전부 두 줄 분리됨 |
| markdown 분량 | `markdown_budget.py` | 하드 초과 0, 소프트 초과 6 (셀 4·5·17·20·30·33). 통과 |
| **비교 실험 통제** | 셀 20 통제표 8행을 셀 21·22·23 인자와 한 줄씩 대조 + 셀 22 학습 로그 판독 | **BLOCK (A-1).** 표의 `imgsz`/`batch`/`seed`/`amp`/`model size`/`data.yaml`/`split` 은 코드와 **전부 일치**했다. 표에 **없는** 차이(두 팔의 체크포인트 선택 절차)가 결함이다 |
| **`train()` 이 새 `YOLO()` 인스턴스인가** | 셀 10·14·21·22·26·29 의 모델 객체 출처를 코드로 추적 | **통과.** 셀 22 `finetune_model`, 셀 29 `segmentation_finetune_model` 모두 `.train()` 직전 줄에서 새로 생성. 추론용 `detection_model`(셀 14)·`segmentation_model`(셀 26)과 이름·객체 모두 분리. 지표(0.688/0.769 ≫ 0.0006/0.0000)로도 교차 확인 |
| 마크다운이 코드보다 앞서 나감 | 전 마크다운 주장을 대응 셀 출력과 대조 | **FIX (B-1)** 1건. 저자가 고쳤다고 보고한 `masks.data` 건이 미완 |
| 결과 해설의 정직성 | 셀 24·28·32 를 셀 22·23·29 출력과 대조 | **셀 24 는 통과.** 과장이 없고 결론이 데이터의 뒷받침을 받는다 (초안의 "부호가 틀렸다" 는 내 과잉 판정 — A-1 각주에서 철회). Step 7 은 해설 부재로 **B-3** |
| 학생 눈높이 (기계공학 대학원생) | 전 마크다운 육안 | Python 문법 설명 **0건** (통과). IoU 는 정의 후 사용(통과). **mAP 은 정의 없이 사용 → A-3** |
| 한 셀에 한 가지 | 전 35셀 육안 | NOTE (C-2). 그 외 셀 구성은 타당 |
| 루트 오염 | `ls -A` 실측 | 통과. `.pt`/`runs/` 없음. `build/`·`data/` 는 레이아웃 표가 허용 |

### 오탐으로 판정해 걸러낸 것 (리더 하네스 갱신용)

| 후보 | 판정 | 근거 |
|---|---|---|
| 셀 18 이 셀 12 의 `center_x`/`center_y`/`box_width`/`box_height`/`tokens` 를 덮어씀 | **오탐** | `CLAUDE.md` "in-place 덮어쓰기 주의" 는 데이터가 조용히 바뀌는 경우를 겨냥한다. 이들은 매번 라벨에서 새로 파싱하는 스크래치 변수이며 뒤에서 원본으로 참조되지 않는다 |
| 셀 21·22 앞에 seed 재설정이 없음 ("Reproducibility — 비교 대상 생성 직전마다 재설정") | **오탐** | 규약의 취지는 랜덤 초기화 차이 혼입 방지다. 두 모델 모두 동일 `.pt` 파일에서 로드되고 학습 난수는 `seed=42` 인자가 통제한다 |
| 셀 10 의 yaml 쓰기 블록 7줄 중복 | **오탐** | `CLAUDE.md` 는 helper function/factory 를 금지하고 명시적 반복을 선호한다. 함수로 묶는 것이 오히려 위반 |
| 셀 2 의 `subprocess` + `pip install` ("ML 이 아닌 구문" 의심) | **오탐** | 명세 §4 셀 02 가 Colab 대응 환경 가드로 지정했다 |
| 셀 8 수식은 $x_2,y_2$ 를 주는데 셀 12 코드는 width/height 를 씀 | **오탐** | `Rectangle` API 가 (좌상단, 폭, 높이)를 요구한다. 수학적으로 동일 |
| 셀 29 가 train+val+print 3가지 (한 셀에 한 가지) | **오탐** | 명세 §4 셀 29 가 "seg fine-tune + val" 로 지정했다. 다만 B-3 수정 시 셀 분리가 자연히 따라온다 |
| 셀 30 이 명세 §4.3 의 폴더 배치 대조표와 다름 | **오탐 — 리더 판정으로 확정** | 리더 회신: *"그 정정 지시는 내가 라운드 중에 냈고, 현재 문구가 내가 의도한 것이다."* 내 보류 판단("미검증 사실을 단정하지 않아 더 정직하다")이 맞았다. 결함 아님 |
| **셀 24 "The gain is small" 의 부호가 틀렸다 (내 초안 BLOCK)** | **오탐 — 내가 만들어낸 위반** | 검증자 반론을 검증해 철회. `.train()` 의 산출물은 `best.pt` 이므로 셀 24 의 "gain" 은 정확한 값이다. epoch 12 의 지표 변화는 **학습 효과가 아니라 `close_mosaic=10` 증강 스케줄 아티팩트**이며, epoch 1–11 이 소수 3자리까지 고정된 것은 4장/`batch=4`(iteration 1회) + `lr0=0.000119` 로 **학습이 사실상 일어나지 않았기** 때문이다. 상세는 A-1 각주 |
| **`_workspace/` 가 과목 루트에 있음** | **저자 대상 아님 — 리더 소관 확정** | 리더 회신: 하네스 산출물이며 Phase 6 에서 처리. 저자에게 지적하지 않은 것이 옳다 |

---

## 미검사 항목

- **셀 31 Roboflow 다운로드 경로 — 실행되지 않았다.** `ROBOFLOW_API_KEY` 부재로 가드 분기만 탔다.
  `model_format='yolov8'` 문자열, `.download()` 반환값의 `.location` 속성, `rglob` 출력 형태는
  **감사에서 검증되지 않았다.** 명세 §6 이 "지역 검증으로는 절대 걸리지 않는 실패" 로 분류한 잔여
  위험이며, 키를 가진 환경에서 한 번 돌리는 것 외에 해소 방법이 없다.
- **`data/coco8*` 가 이미 존재하는 상태에서 실행됐다.** 셀 6 의 다운로드·압축해제 분기
  (`if not ...exists()`)는 이번 실행에서 타지 않았다. 최초 실행 학생의 경로는 미검증이다.
- **셀 2 의 `except ImportError` 분기 미실행.** 두 패키지가 이미 설치돼 있었다. Colab 최초 실행
  경로는 미검증이다.
- **강의자료 대조를 하지 않았다.** 명세 §1 대로 이 회차에 대응하는 강의 PDF 가 없다. 따라서
  A-3 의 수식 판정 근거는 **강의 원문이 아니라 명세 §2.3 과 COCO 표준 정의**다. 향후 슬라이드가
  제작되면 mAP 서술 깊이를 재검증해야 한다.
- **시각화 셀(12·16·19·27)의 렌더링 결과를 이미지로 열어 보지 않았다.** 코드와 출력 메타데이터로만
  판정했다. 박스·폴리곤이 물체에 실제로 얹혔는지는 육안 확인이 필요하다 (셀 18 의 IoU 0.939 가
  간접 근거는 된다).
- **`plots=False` 로 학습 곡선 산출물이 없어** 열화 시점(epoch 12, mosaic 종료)의 손실 거동을
  그림으로 확인하지 못했다. per-epoch 지표 로그로만 판정했다.
- **셀 27 `fill_colors` 는 8색 고정**이라 인스턴스가 9개 이상인 이미지에서 `IndexError` 가 난다.
  현재 이미지(5개)에서는 발생하지 않아 **실행으로 검증되지 않은 경로**다. B-1(a) 로 `sample.jpg`
  로 바꿀 경우 인스턴스 수가 달라지므로 재확인이 필요하다.

---

## 검증자 조율 — **완료 (합의됨)**

`consistency-verifier` 와 겹치는 영역을 직접 조율했다. 저자에게는 **단일 지시**로 나간다.

| 항목 | 감사자 | 검증자 | 합의 |
|---|---|---|---|
| 변수명 전반 | 지적 0건 | 지적 0건 | **양쪽 0건.** 저자에게 변수명 지시를 보내지 않는다 |
| 명세 §5 에 없는 이름 4종 (`pretrained_model`, `best_iou`, `best_index`, `fill_colors`) | 정당한 추가 | 정당한 추가 | 지적하지 않음 |
| `iou` (셀 18) | 약어 아님 — 수식 $\mathrm{IoU}$ 와 1:1 | 풀어쓰기 요구 금지 | 지적하지 않음 |
| `annotated_image` 부재 (명세 §5 에 있으나 미사용) | 검증자 지적으로 인지 | `plot()` 미사용 결정의 결과, 정당한 이탈 | **이름을 억지로 만들라고 지시하지 않음** |
| `class_names` vs `result.names` 병용 | 초안 NOTE → **철회** | 미제기 | 내 오탐. C-3 참조 |
| `lr-symbol` / `theta-symbol` 미발화 | 정상 (명세 §1) | 정상 | 재지적 없음 |
| 셀 25 `masks.data` ↔ 셀 26 출력 모순 | **B-1 (FIX)** | 치명 (a) | **(a)안 확정.** 검증자가 letterbox 산술로 `sample.jpg` → `(N, 448, 640)` 를 예측해 명세 §3.5 리더 실측과 일치시켰다. **검증자가 저자에게 발신 완료** |
| mAP 정의 오류 | **A-3 (BLOCK)** | 치명 (b) — 셀 32 Summary | **A-3 이 검증자 건(V-2)을 포함한다.** 검증자가 V-2 를 철회하고 내 문구를 단일 지시로 채택. 공식 문서 인용은 검증자 제공 |
| **`best.pt` 체크포인트 선택** | 초안 A-1 (BLOCK) | **독립 발견 — V-9 (치명)** | **양쪽이 독립적으로 같은 결함에 도달했다.** 검증자가 추가로 내 "부호가 틀렸다" 프레이밍을 반박했고 **내가 검증 후 채택**했다. A-1 은 검증자 제안 (i)(ii) 형태로 재작성 |

**저자 지시 창구**: 리더 지시에 따라 **앞으로 저자에게 직접 발신하지 않는다.** 세 쪽 지적을 리더가
병합해 단일 지시로 낸다. (이번 라운드의 저자 발신은 B-1 을 검증자가 보낸 것 1건이며, 감사자 발신은 없다.)

## 하네스 갱신 제출

1. **[즉시 반영 — 오탐 억제]** 저자 보고 §5 의 `lr-symbol` 오탐(matplotlib `alpha=` 가 줄머리에 오면
   발화)은 이번 라운드에도 재발할 수 있는 형태다. 프로필 `lr-symbol.pattern` 을 대입 좌변이
   식별자 단독인 경우로 좁히거나(`^\s*(alpha|eta|step_size|learning_rate)\s*=(?!.*[,)])`),
   `lr-symbol.why` 에 회피법을 적어 둘 것. 현재 `noise: true` 라 후보로만 뜨지만, 매 라운드 사람이
   같은 판단을 반복한다.
2. **[체크포인트 신설 후보 — 승인 대상]** **"학습 후 지표를 `best.pt` 에서 읽으면서 그 사실을
   서술하지 않음"** 을 감사 항목으로 신설할 것을 제안한다. 정규식으로는 잡히지 않지만
   `.train()` 뒤에 `.val()` 이 오는 노트북에서 **구조적으로 반복될 결함**이며, 이번 회차에서
   BLOCK 2건(A-1·A-2)의 단일 원인이었다. `notebook-convention-audit/SKILL.md` 의 수동 체크리스트에
   "학습 API 가 체크포인트를 자동 선택하는가, 그 선택이 평가 split 위에서 이뤄지는가" 한 줄 추가.
3. **[규약 보강 후보]** `CLAUDE.md` "비교 실험 통제 변수" 의 통제 항목 목록
   (데이터셋·전처리·초기 가중치·학습률·반복 횟수·batch·optimizer·seed)에 **"모델 선택 절차
   (early stopping / best checkpoint 를 어느 split 에서 고르는가)"** 를 추가할 것을 제안한다.
   현재 목록만 대조하면 이번 결함이 통과한다 — 실제로 저자의 통제표는 목록의 8항목을 전부 지켰다.
4. **[명세 단계 보강]** 명세 §4.2 의 통제표가 **`val()` 호출 인자만** 대조 대상으로 삼게 돼 있어
   학습 절차 자체가 만드는 비대칭을 담지 못했다. 명세 템플릿의 통제표에 "선택 절차" 행을 기본 포함시킬 것.
5. **[미검증 경로 기록]** 이 노트북은 `data/` 가 이미 존재하고 패키지가 이미 설치된 상태에서만
   실행 검증됐다. 외부 리소스를 받는 회차는 **깨끗한 상태에서 한 번 더 실행**하는 절차를
   `practice-notebook-authoring` 에 명시할 것.
6. **[감사자 자신의 실패 유형 — 이번 라운드에서 가장 값진 항목]**
   나는 같은 로그를 두고 **연속으로 두 번 틀렸고, 두 번 다 남이 반증했다.**
   - 1차: epoch 12 의 mAP 하락을 **학습 효과**로 읽고 BLOCK 을 만들었다 → 검증자가 반박
     (`.train()` 산출물은 `best.pt`, epoch 1–11 고정은 학습이 사실상 없었다는 뜻)
   - 2차: 그 하락의 원인을 **`close_mosaic`** 으로 지목하고 **그 설명을 노트북에 넣자고 제안**했다
     → 리더가 통제 실험으로 반박 (`close_mosaic=0` 에서도 하락은 똑같이 epoch 12)
   - 검증자도 같은 함정에 한 발 들어갔다 — `warmup` 이 전 구간을 덮었다는 설명을 쓰려다
     소스에서 `round(min(3.0, 19) * 1)` = **3 iteration** 임을 확인하고 철회했다 (기억 속 다른 버전의
     하한 100 iteration 과 혼동). **넘겨받는 쪽도 면역이 없다.**

   → `notebook-convention-audit/SKILL.md` 에 아래를 추가 제안한다 (리더 강화 지시 반영):

   > **학습 로그의 지표 변화를 결함 근거로 쓰기 전에 (a) epoch 당 iteration 수, (b) 실제 적용된
   > learning rate, (c) epoch 의존 설정(`close_mosaic`·`warmup`·scheduler)이 그 지점에서 바뀌는지
   > — 셋을 먼저 확인한다. 셋은 전부 로그 첫머리에 인쇄되므로 확인 비용이 0이다.
   > **그러나 셋을 확인해도 부족하다.** 로그에서 두 사건이 같은 지점에 보이는 것은 **상관이지
   > 인과가 아니다.** 인과를 주장하려면 **그 변수만 바꾼 실행**이 필요하다 — 이번 사례에서
   > 그 비용은 10초였다. 통제 실행 없이는 원인을 적지 않는다.
   > 동역학 해석이 필요한 순간 감사자는 등급을 올리지 말고 검증자에게 넘기는 것을 기본값으로 하되,
   > **넘겨받는 쪽에는 "기억이 아니라 설치본 소스로 확인" 의무가 따라붙는다.** 버전마다 기본값이
   > 다르고, 그럴듯한 설명일수록 검증 없이 통과하기 쉽다.

   **가장 중요한 교훈은 등급이 아니라 방향이다.** 1차 오판은 없는 결함을 만든 것이고,
   2차 오판은 **검증되지 않은 인과를 학생 배포물에 인쇄하자고 제안한 것**이다.
   후자는 이번 라운드 내내 우리가 잡아 온 결함(마크다운 단정문을 출력이 뒷받침하지 못함)과
   같은 종류이며, 감사자가 그것을 만드는 쪽에 선 사례로 남긴다.
