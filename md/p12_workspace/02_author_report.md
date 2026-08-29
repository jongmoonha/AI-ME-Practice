# Author Report — Practice12_Object_Detection_and_Segmentation.ipynb

- generator: `D:/Main/00_Research/00_Python/00_Lecture/01_AI-ME_Graduate/gen/p12.py`
- 라운드: **7 (최종)** — 라운드 5 사용자 개정 3건 + 그림 가독성, 라운드 6 V-11·B-1,
  **라운드 7 = 감사 PASS / 정합성 PASS 이후 잔여 2건 (치명 0)**
- 실행 검증: **성공** — 전 35 셀 실행, 에러 셀 0, 미실행 코드셀 0, `execution_count` 1–20 연속
- 총 셀 수: code 20개 / markdown 15개 (합 35 — 개정 전과 같다)
- 실행 시간: **90.4초** (라운드 6 은 171.8초. 같은 120 epoch 이며 실행 편차다)
- 명세 대비 미구현: 아래 §6
- 이번 라운드 반영한 지적: **감사 B-1 / 검증 V-13 · V-14 (치명 0).**
  누적으로는 사용자 개정 3건, 리더 강화 2건, 라운드 6 의 V-11(치명)·B-1.
  부수 해소: 감사 NOTE **C-5**(라운드 5), Summary 소프트 캡(라운드 7).
  **미반영 확정: 감사 C-1 · C-2, 검증 V-12** — 사유는 §4.3 말미

> **명세서(`01_spec_Practice12.md`)는 아직 개정 전 상태다.** 이 라운드에서 뒤집힌 것이
> §3.6(하이퍼파라미터), §3.7(실측 수치), §4(셀 구성·번호), §5(변수표)로 넓다.
> 리더가 `spec` 에게 재동기화를 지시했다. **아래 §1 대응표가 그 입력이다.**

---

## 1. 셀 번호 / Step 번호 재배치 — 개정 전 → 개정 후 (0-based)

**Step 번호 재배치는 리더 승인 완료** ("Step 3 다음이 5 로 뛰면 학생은 셀이 지워졌다고 읽는다").

| 개정 전 셀 | 개정 후 셀 | Step 번호 | 내용 |
|---|---|---|---|
| 0–16 | 0–16 | Step 0–3 유지 | 변동 없음 (제목 ~ Step 3 박스 그리기) |
| **17, 18, 19** | **삭제** | **Step 4 소멸** | "How Detection Is Scored" md / IoU 손계산 루프 / IoU 그림 |
| 20 | **17** | Step 5 → **Step 4** | Detection Fine-tuning 통제표 — **재작성** |
| 21 | 18 | 〃 | baseline `val()` |
| 22 | 19 | 〃 | detection `train()` — 하이퍼파라미터 변경 + 말미 `print` |
| 23 | 20 | 〃 | 3-arm `val()` + 3줄 출력 |
| 24 | 21 | 〃 | 결과 해설 — **전면 재작성** |
| — | **22** | 〃 | **신규** — 두 이미지 × (GT / pretrained / after 120 epochs) 텍스트 |
| — | **23** | 〃 | **신규** — `plt.subplots(2, 2)` 네 패널 |
| — | **24** | 〃 | **신규** — 그림 해설 |
| 25 | 25 | Step 6 → **Step 5** | Instance Segmentation |
| 26, 27 | 26, 27 | 〃 | seg 추론 / box vs mask 그림 |
| 28 | 28 | Step 7 → **Step 6** | Segmentation Fine-tuning |
| 29 | 29 | 〃 | seg fine-tune + 2-arm val — 하이퍼파라미터 변경 |
| 30 | 30 | Step 8 → **Step 7** | Building Your Own Dataset |
| 31–34 | 31–34 | — | Roboflow / Summary / Exercises / 빈 셀 |

**Step 헤더는 `Step 0` ~ `Step 7` 연속이다** (기계 확인).
명세 §4 의 셀 행 번호는 17 이후 전부 밀렸다.

---

## 2. 개정 1 — Step 4 삭제와 연쇄 정리

삭제: IoU/precision/recall/AP/mAP 수식 마크다운, IoU 손계산 루프, IoU 시각화 (3셀).

**연쇄 정리 전수 확인 (노트북 source 전체 문자열 검색, 전부 0건):**

```
ground_truth_box 0   predicted_box 0   best_iou 0   best_index 0   iou_threshold 0
verdict 0   ground_truth_area 0   class_matches 0   ground_truth_line 0   ground_truth_class 0
IoU 0   iou 0   Precision 0   Recall 0   intersection 0   union 0   'AP =' 0   'mAP =' 0
```

- **Summary**: `A detection counts only when the class matches and the IoU clears the threshold.` 삭제
- **Exercises 1**: `does the best IoU match survive?` 절만 제거 (앞 절은 유효하므로 유지)
- **Exercises 3**: IoU 루프 문항 → segmentation polygon 그리기 문항으로 교체
- **Exercises 2**: `20 to 60` → `120 to 30`, 질문도 "두 fine-tuned 행의 간격" 으로 갱신
- **mAP 정의·설명 0건.** 남은 `precision`/`recall` 문자열은 셀 18 의 `pretrained precision :` print
  라벨 두 개뿐이며 정의문이 아니다

`detection_label_path`·`sample_image`·`detected_boxes`·`image_width`/`image_height` 는
셀 9·12·15·16 에서 계속 쓰이므로 남겼다.

---

## 3. 개정 2 — fine-tuning 효과의 시각 확인 (리더 강화 지시 반영)

### 3.1 강화 전/후

초안은 **1×2 (학습 이미지에서만 전/후)** 였다. 사용자 지적 — *"train 한 이미지로 test 하는 것
아니냐"* — 을 받아 리더가 **2×2 네 패널**로 강화했고, 그대로 반영했다.
**초안대로 갔으면 memorization 이 개선으로 읽혔다.** 지적이 정확하다.

### 3.2 셀 23 — `plt.subplots(2, 2)` 네 패널

| | 왼쪽 (pretrained) | 오른쪽 (`last.pt`) |
|---|---|---|
| 윗줄 | `Pretrained - training image` | `After 120 epochs - training image` |
| 아랫줄 | `Pretrained - held-out image` | `After 120 epochs - held-out image` |

- 이미지: 윗줄 `images/train/000000000009.jpg`, 아랫줄 `images/val/000000000049.jpg`
- 박스는 Step 3 방식 그대로 (`Rectangle` + `ax.text`). `plot()` 미사용, `gridspec` 미사용
- 그리기 루프가 `range(len(panel_result.boxes))` 라 **예측 0개 패널에서도 그림이 깨지지 않는다**
- 네 패널 사양을 `panels` 리스트에 한 줄씩 명시하고 하나의 `for` 로 그린다.
  같은 9줄을 네 번 복사하는 것보다 읽기 쉽고, 규약이 금지하는 helper function/factory 가 아니다

### 3.3 셀 22 실측 출력 — 리더 표를 재현했다

> 아래 수치는 **라운드 6 (`patience=0`) 재실행 기준**이다. 라운드 5 의 조기종료 값과 다르다 (§4.2).

```
training image 000000000009.jpg
  ground truth     : bowl, bowl, broccoli, bowl, orange, orange, orange, orange
  pretrained       : bowl, broccoli, bowl, bowl, broccoli, fork, bowl
  after 120 epochs : bowl, broccoli, bowl, bowl, orange, orange, orange, orange, orange, orange

held-out image 000000000049.jpg
  ground truth     : horse, horse, person, person, person, potted plant, person, person, person
  pretrained       : horse, horse, person, person, person, person, person, person
  after 120 epochs : zebra
```

**학습 이미지에서는 orange 0개 → 6개, 학습에 안 쓴 이미지에서는 검출 8개 → `zebra` 1개.**
나머지 val 3장도 별도 프로브로 확인했고 리더 표와 일치한다 (036: umbrella·person → zebra,
042: dog → bowl×3, 061: elephant×2 → 검출 0개). 노트북에는 049 한 장만 싣는다.

### 3.4 val 지표 — 리더 프로브와 자릿수까지 일치

| arm | mAP@50 | mAP@50-95 | 리더 프로브 |
|---|---|---|---|
| A pretrained | 0.68437 | 0.39986 | **일치** |
| B after 120 epochs (`last.pt`) | **0.36998** | 0.23863 | **일치** |
| B best checkpoint (`best.pt`) | 0.68769 | 0.41834 | **일치** |

segmentation 셀 29 (개정 전보다 붕괴 폭이 훨씬 크다):

| arm | box mAP@50 | box mAP@50-95 | mask mAP@50 | mask mAP@50-95 |
|---|---|---|---|---|
| after 120 epochs (`last.pt`) | 0.32036 | 0.22290 | 0.25711 | 0.16023 |
| best checkpoint (`best.pt`) | 0.76877 | 0.48625 | 0.73535 | 0.40272 |

### 3.5 통제표 (셀 17) — 리더 지시대로 확정

```
| **weights evaluated** | **COCO pretrained, untouched** | **+ 120 epochs on coco8 (SGD, lr0=0.01)** |
| **checkpoint**        | **no selection**               | **best of 120, chosen on `images/val`**   |
| dataset               | same                           | same                                       |
| evaluation split      | `images/val`                   | `images/val`                               |
| model size            | `yolo11n`                      | `yolo11n`                                  |
| `imgsz`               | 320                            | 320                                        |

The second row is not controlled.
`coco8` has no test split, so the same four photographs choose B's checkpoint and then score it.
Only the last-epoch row escapes that selection.
```

- **`seed`·`amp` 행 삭제** — 순수 평가 호출에 영향이 없는데 통제 항목으로 적혀 있었다.
  감사자 **NOTE C-5** 가 지적한 바로 그 항목이며 이번에 해소됐다
- **`batch` 행 삭제 / `model size` 행 유지** — 리더 정정 지시. `batch` 는 평가 결과를 바꾸지 않고,
  `model size` 는 두 arm 이 다르면 비교 자체가 무의미해지므로 빠지면 안 되는 행이다
- **`epochs`/`optimizer`/`lr0` 는 통제 항목 행으로 넣지 않고 B 서술 안에 담았다** — A 는 학습하지
  않으므로 A 열에 쓸 값이 없다 (리더 지시)
- 분량 **116/120** (하드 여유 4단어). 이 셀은 다음 라운드에 문장을 더할 여유가 거의 없다

### 3.6 서술 — 출력 대조 전수

| 셀 | 주장 | 대조 대상 | 결과 |
|---|---|---|---|
| 17 | `coco8` has no test split | `data/coco8/images/` 에 `train`·`val` 뿐 (실측) | 참 |
| 17 | model size / imgsz / split 이 A·B 동일 | 셀 18·19·20 인자 | 참 |
| 21 | Only `last.pt` is scored without any selection | 셀 17 표 2행 + `train()` 동작 | 참 |
| 21 | it keeps about half of the pretrained mAP@50 | 0.36998 / 0.68437 = **0.541** | 참 |
| 24 | The training image gains the oranges the pretrained weights were missing | 셀 22 (pretrained orange **0** → 6) | 참 |
| 24 | The held-out image loses almost everything those same pretrained weights found | 셀 22 (8건 → `zebra` 1건) | 참 |
| 24 | That is overfitting, and the `last.pt` row above is its number | 셀 20 + 셀 23 네 패널 | 참 |
| 32 | fitted those four and **lowered** the score on images it had not seen | 위 두 줄. `halved` 는 라운드 7 에서 제거 (§4.3) | 참 |

**금지 항목 확인:** `overfitting` 은 리더가 허용한 단어이며 네 패널과 두 숫자가 직접 보여준다.
zebra 가 왜 나오는지는 **설명하지 않았다.** 학습률·epoch 수를 원인으로 지목하는 문장 없음.
`mosaic`/`warmup`/`due to`/`caused`/`therefore` 노트북 전체 0건.

> **초안에서 한 번 틀렸고 실행 전에 잡았다.** 셀 24 초안 3번째 문장이
> `The validation rows above are where generalization is measured, and they went down.` 였다.
> **`best.pt` 행은 오히려 올라간다** (0.68769 > 0.68437, 0.41834 > 0.39986). 라운드 4 에서 지적받은
> "여러 행·열을 내는 셀의 해설을 한 행만 보고 쓴다" 와 **같은 결함**이라, `last.pt` 한 행에
> 한정되도록 고친 뒤 재생성·재실행했다.

### 3.7 segmentation 시각 비교는 넣지 않았다 (리더 지시)

Step 6 은 하이퍼파라미터만 맞추고 `last`/`best` 2줄 숫자를 유지했다.

---

## 4. 개정 3 — `execute_result` 와 파일 크기

셀 19 의 `.train()` 뒤에 `print(f'checkpoints written to {...}')` 를 붙여 마지막 표현식이 아니게 했다.

**기계 확인 (사용자가 지정한 스크립트):**

```
execute_result 출력이 남은 셀 : 0
error 출력이 남은 셀          : 0
미실행 코드셀                 : 0
execution_count               : [1..20] 연속
```

**파일 크기: 3,538,639 → 4,157,329 bytes (+0.62MB).** 줄지 않았고, 늘었다.
**`execute_result` 는 그 3.4MB 의 원인이 아니었다** — 59,415자는 전체의 1.7% 였다. 실측:

| 구성 | 크기 |
|---|---|
| PNG base64 (그림 4장: 셀 12·16·23·27) | **약 3,780,000 자 (92%)** |
| stream text (학습 로그 2회 + print) | 약 125,000 자 |
| 나머지 (소스·메타데이터) | ~210,000 자 |

가독성 수정(§4.1) 후 4,140,233 → 4,108,222 로 32KB 줄었고, 라운드 6 에서 조기종료를 끄면서
학습 로그가 101 → 120 epoch 으로 길어져 4,212,939 가 됐다가, 라운드 7 에서 **4,157,329 bytes** 다.
**리더 판정대로 `dpi` 는 건드리지 않았다.**

증가분의 출처는 둘이다 — 새 그림이 **2×2 (`figsize=(13, 9)`)** 라 삭제한 IoU 그림보다 크고,
학습 로그가 20 → 120 epoch 으로 6배가 됐다. 더 줄이려면 `figsize`/`dpi` 를 손대야 하며
이번 개정 범위 밖으로 두었다.

### 4.1 그림 가독성 수정 — 리더가 PNG 를 직접 보고 지시

리더가 네 패널을 눈으로 확인해 교육 메시지는 전달된다고 판정하고, **가독성 결함 2건**을 지적했다.
**나는 그때까지 그림을 열어 보지 않았다** — 코드와 출력 메타데이터로만 판정했고, 그래서 못 잡았다.

| 결함 | 수정 |
|---|---|
| 박스가 전부 같은 빨강이라 라벨↔박스를 짝지을 수 없다 | `instance_colors` 8색을 **셀 16 에서 한 번만 정의**하고 셀 23·27 이 재사용. 인덱스에 `% len(...)` |
| 라벨이 이미지 위쪽에서 잘리고, 축 밖 여백에 찍히고, 서로 겹쳐 안 읽힌다 | `clip_on=True` + `va='top'` + 박스 안쪽(`y1 + 2`) 배치 + 박스와 **같은 색 불투명 배경**에 흰 글씨 |

- `fill_colors`(셀 27 지역 변수)를 없애고 `instance_colors` 로 통일했다 —
  `CLAUDE.md` "한 번 정의한 데이터는 노트북 전체에서 같은 이름으로 재사용한다"
- 이름을 리더 제안 `box_colors` 가 아니라 **`instance_colors`** 로 했다. 셀 27 이 같은 리스트로
  **마스크 폴리곤**도 칠하므로 `box_` 가 그 자리에서 거짓이 된다. 되돌리라면 한 단어다
- helper function 을 만들지 않았다. 리스트 하나 + 인덱싱이다

**셀 23 에서 confidence 숫자를 뺐다** (리더가 내 판단에 맡긴 항목). 4패널이라 밀도가 높고
이 그림의 논점이 "무엇을 찾았나" 이기 때문이다. **셀 16 은 단일 큰 패널이라 confidence 를 유지**했다.

#### 셀 12 도 함께 고쳤다 — 범위 판단, 리더 확인 요청

리더가 명시한 것은 셀 16·23 이지만, PNG 를 뽑아 보니 **셀 12(ground truth 라벨)에 같은 결함이
있었다** — 상단의 `bowl` 과 `orange` 라벨이 겹쳐 **읽을 수 없었다.**
범위 제한 문장이 *"그림 가독성만 고쳐라. 텍스트·수치·구조·하이퍼파라미터는 손대지 마라"* 로
**변경의 종류를 한정한 것**이지 셀을 한정한 것이 아니라고 읽었고, 학생 배포물에 읽을 수 없는
라벨을 남기는 것이 더 나쁘다고 판단했다.

- **색 구성은 바꾸지 않았다** — ground truth 는 lime 하나, polygon 은 cyan 하나 그대로다.
  여기서 색은 인스턴스 식별이 아니라 "이것은 라벨 파일이다" 라는 의미이므로 다색이 오히려 틀린다
- 라벨만 `clip_on=True` + `va='top'` + lime 배경 + **검은 글씨** (lime 이 밝아 흰 글씨는 안 읽힌다)
- 되돌리라면 셀 12 의 `ax.text` 한 줄이다

#### 육안 재확인 (이번에는 내가 직접 봤다)

네 그림을 PNG 로 추출해 `build/figures/` 에 두고 전부 열어 봤다.

| 그림 | 확인 |
|---|---|
| 셀 12 | GT 라벨 8개 전부 읽힌다. `bowl`/`orange` 겹침 해소 |
| 셀 16 | 검출 7개가 서로 다른 색. `bowl 0.25`/`bowl 0.64` 잘림 해소, confidence 유지 |
| 셀 23 | **윗줄 오른쪽에 `orange` 4개가 각각 다른 색으로 뚜렷하다.** 아랫줄 오른쪽은 `zebra` 박스 하나. 잘린 라벨 없음 |
| 셀 27 | 색 리스트 이름만 바뀌고 그림은 동일 (마스크 폴리곤 정상) |

**남은 미세 결함 1건 (수정하지 않았다):** 셀 23 아랫줄 **왼쪽** 끝의 `person` 라벨이 이미지
오른쪽 경계에서 잘려 `per` 까지만 보인다. 그 검출의 박스가 이미지 가장자리에 걸쳐 있어 라벨을
놓을 폭이 없다. `clip_on` 이 **의도대로 동작한 결과**이며(이전에는 축 밖 여백에 찍혔다),
경계 근처에서 라벨을 오른쪽 정렬로 뒤집으려면 조건 분기가 필요해 코드가 복잡해진다.
**단어 일부가 잘리는 것이 여백 침범보다 낫다**고 판단해 그대로 두었다.

---

## 4.2 라운드 6 — V-11 (치명) 과 감사 B-1 (FIX)

### V-11 — `120 epochs` 가 거짓이었다. 실제로는 101 epoch 만 돌았다

`EarlyStopping` 기본값 `patience=100` 이 detection·segmentation 양쪽에서 발동했다.
지표가 epoch 1 이후 개선되지 않아 1+100 = **101 에서 멈췄고**, 같은 셀 로그에
`101 epochs completed` 가 찍혀 있는데 라벨은 `after 120 epochs` 라고 단정했다.

**이 회차 내내 잡아 온 결함(마크다운/라벨의 단정을 같은 셀 출력이 반박)이 학습 인자에서 재발한 것이다.**
앞선 사례들과 다른 점은 **내가 쓴 문장이 아니라 라이브러리 기본값이 거짓을 만들었다**는 것이다.
그래서 문장 검토로는 잡히지 않았고, 나는 로그의 `120 epochs completed` 를 확인한 적이 없었다.

- **해법: `train()` 에 `patience=0` 추가** (detection·segmentation 양쪽).
  ultralytics 는 `patience` 가 falsy 면 `inf` 로 두므로 0 이 곧 비활성화다
- 코드 주석은 동작만 적었다 — `# patience=0 disables early stopping, so all of the requested epochs run`

**재실행 검증 (리더가 요구한 항목):**

```
cell 19: 120 epochs completed in 0.019 hours.
cell 29: 120 epochs completed in 0.023 hours.

EarlyStopping 문구       : 0건
'stopping training early': 0건
'101 epochs'             : 0건
```

### 감사 B-1 — `120` 하드코딩과 Exercise 2 의 충돌

셀 20·22·23(×2)·29 의 출력 라벨에 `120` 이 문자열로 박혀 있었는데 Exercise 2 는 학생에게
그 값을 30 으로 바꾸라고 지시한다. **학생이 문제를 풀면 30 epoch 을 돌리고도 "after 120 epochs"
가 찍힌다.**

**해법: `finetune_epochs` 를 셀 19 에서 한 번만 정의하고 학습 인자와 모든 라벨이 재사용.**
`CLAUDE.md` "비교 실험 통제 변수" 의 `w_init`/`rho`/`n_iter` 규칙을 학습 인자에 적용한 것이다.

| 셀 | 재사용 형태 |
|---|---|
| 19 | `finetune_epochs = 120` (유일한 정의) / `epochs=finetune_epochs` |
| 20 | `last_epoch_label = f'B after {finetune_epochs} epochs'` |
| 22 | `finetuned_label = f'after {finetune_epochs} epochs'` |
| 23 | 패널 제목 2개 f-string |
| 29 | `epochs=finetune_epochs` + `segmentation_last_label` |

- detection·segmentation 이 **같은 값을 쓰므로 정의는 하나뿐**이다
- 라벨 폭이 값 길이에 따라 흔들리지 않도록 `:<21s`·`:<16s`·`:<18s` 로 고정했다.
  학생이 30 으로 바꿔도 표 정렬이 유지된다
- 노트북 source 에 남은 리터럴 `120` 은 **마크다운 4곳(셀 17·21)과 Exercise 2 지시문, 그리고
  정의 한 줄뿐**이다. 마크다운은 리더가 허용한 항목이며 `patience=0` 으로 이제 참이다

### 수치 재보고 — 리더 실측과 대조

| | 리더 실측 (patience=0) | **내 재실행** | 일치 |
|---|---|---|---|
| A pretrained mAP@50 / @50-95 | 0.68437 / 0.39986 | **0.68437 / 0.39986** | 일치 |
| `last.pt` mAP@50 / @50-95 | 0.36998 / 0.23863 | **0.36998 / 0.23863** | 일치 |
| `best.pt` mAP@50 / @50-95 | 0.68769 / 0.41834 | **0.68769 / 0.41834** | 일치 |
| 학습 이미지 검출 | bowl 3, broccoli 1, orange 6 | **bowl 3, broccoli 1, orange 6** | 일치 |
| held-out 검출 | `zebra` 1 | **`zebra` 1** | 일치 |

segmentation 은 101 → 120 epoch 로 늘면서 `last.pt` 만 미세하게 움직였다
(box mAP@50 `0.32123 → 0.32036`, mask `0.25789 → 0.25711`). `best.pt` 는 `0.76877 / 0.73535` 로 불변.

**서술 재검산:** 셀 21 "keeps about half of the pretrained mAP@50" → 0.36998 / 0.68437 = **0.541** 로 여전히 참.
셀 24 세 문장과 Summary 2번째 불릿도 전부 참을 유지한다 (오렌지 6개, held-out `zebra` 1개).
**그림은 구조·색·라벨 모두 그대로이고 제목의 `120` 만 이제 사실이다** (PNG 재추출해 확인함).

### 반영하지 않은 것 — V-12

**`best.pt` 가 epoch 1 의 가중치라는 사실은 노트북에 넣지 않았다** (리더 지시).
노트북 출력만으로는 epoch 1 을 특정할 수 없다 — epoch 1~11 의 지표가 전부 같기 때문이다.
적으면 출력이 뒷받침하지 못하는 주장이 되고, 그것이 이 회차 내내 걷어낸 결함 그 자체다.

---

## 4.3 라운드 7 — 최종 2건 (감사 PASS / 정합성 PASS 이후, 치명 0)

**마크다운 2셀만 바뀌었다. 코드·구조·그림·수치는 손대지 않았다.**

### 수정 1 — Exercise 2 가 없는 이름을 가리켰다 (감사 B-1 / 검증 V-14 부수)

라운드 6 에서 `120` 을 `finetune_epochs` 로 파라미터화하면서, **`train()` 호출부에 `120` 리터럴이
사라졌다.** 그런데 Exercise 2 는 여전히 *"Change `epochs` in the detection fine-tuning call
from 120 to 30"* 이라고 지시하고 있었다. 지시를 문자 그대로 따르는 학생은 호출부를 `epochs=30` 으로
고치고 `finetune_epochs` 는 120 으로 남긴다 → **셀 20·22·23 라벨이 30 epoch 가중치에
`after 120 epochs` 를 붙인다.**

**내가 방금 닫은 결함이 학생 손에서 되살아나는 경로였다.** 수정이 만든 결함이라는 점에서
라운드 3·4 와 같은 계열이며, 이번에는 **노트북이 아니라 학생의 편집 결과에서** 발생한다는 점이 다르다.

```
2. Change `finetune_epochs` from 120 to 30.
   Both training calls read that variable, so the segmentation run shortens with it.
   How far apart are the two fine-tuned rows now?
```

- `finetune_epochs` 는 코드에 정확히 한 곳(셀 19)에만 있으므로 학생이 찾을 수 있다
- **두 번째 줄이 리더 지시로 추가됐다** — 이 값은 segmentation 학습(셀 29)도 함께 짧게 만든다.
  두 `train()` 이 같은 변수를 읽기 때문이고, 이것이 셀 28 의 *"The training call is the one used
  for detection"* 을 코드로 보증하는 구조다. 모르면 segmentation 결과가 왜 같이 변했는지 헷갈린다

### 수정 2 — Summary 의 `halved` 가 열을 한정하지 않았다 (검증 V-13)

| 열 | A | B (`last.pt`) | 남은 비율 |
|---|---|---|---|
| mAP@50 | 0.68437 | 0.36998 | **54.1%** |
| mAP@50-95 | 0.39986 | 0.23863 | **59.7%** |

첫 열은 절반에 가깝지만 둘째 열은 40% 손실이다. **한 열만 보고 쓴 문장이며, 이 회차에서 네 번째로
나온 같은 계열이다** (라운드 4 열 단위, 라운드 5 초안 행 단위, 라운드 6 V-11 인자↔로그, 그리고 이것).

`halved` → `lowered` 로 바꿔 **검증되지 않은 크기 주장을 없앴다.** 정확한 비율을 지어내지 않았다 —
크기는 바로 위 셀 20 의 표가 이미 보여준다.

> Fine-tuning on four images fitted those four and **lowered** the score on images it had not seen.

**부수 효과:** Summary 산문이 61 → **58 단어**로 소프트 캡(60) 아래로 내려갔다.
Exercises 는 한 줄이 늘어 66/80 이 됐고 하드 캡 안이다.

### 수치 불변 확인 (마크다운만 바뀌었으므로 필수)

| | 라운드 6 | **라운드 7** |
|---|---|---|
| A pretrained | 0.68437 / 0.39986 | **0.68437 / 0.39986** |
| `last.pt` | 0.36998 / 0.23863 | **0.36998 / 0.23863** |
| `best.pt` | 0.68769 / 0.41834 | **0.68769 / 0.41834** |
| segmentation `last.pt` | 0.32036 / 0.22290 / 0.25711 / 0.16023 | **동일** |
| `120 epochs completed` | 2건 | **2건** |
| `EarlyStopping` | 0건 | **0건** |

### 반영하지 않은 것 (리더 확정, 최종 보고용)

| 항목 | 사유 |
|---|---|
| 감사 **C-1** — 통제표 B열이 3행 출력을 2팔로 그림 | 셀 17 이 116/120 이라 여유가 없다. 산문이 보완하므로 거짓이 아니다 |
| 감사 **C-2** — `halved`(현 `lowered`)의 체크포인트 미지정 | Summary 도 여유가 없다. 본문이 `last.pt`/`best.pt` 를 구분해 설명한다 |
| 검증 **V-12** — `best.pt` 가 epoch 1 의 가중치 | **노트북 출력만으로 epoch 1 을 특정할 수 없다** (epoch 1~11 지표 동일). 적으면 출력이 뒷받침 못 하는 주장이 되고, 그것이 이 회차 내내 걷어낸 결함 그 자체다 |
| 셀 17·21 마크다운의 `120` 리터럴 | **마크다운 표는 배포 상태를 서술**하는 것이고 학생 편집을 따라갈 의무가 없다. 코드 라벨만 파라미터화되면 충분하다 (리더 판정) |

---

## 5. 이번 라운드에 바뀐 셀 (감사자·검증자 재검증 범위)

**라운드 7 에서 바뀐 셀은 33(Exercises)·32(Summary) 두 개뿐이며 전부 마크다운이다.**
코드·구조·그림·수치는 손대지 않았고, 세 숫자가 불변임을 재실행으로 확인했다 (§4.3).

| 셀 | 라운드 7 변경 |
|---|---|
| **32** | Summary 2번째 불릿 `halved` → `lowered` (V-13) |
| **33** | Exercise 2 가 `finetune_epochs` 를 가리키게 + segmentation 동반 변화 한 줄 (B-1/V-14) |

**라운드 6 에서 바뀐 셀은 19, 20, 22, 23, 29 다섯 개** (전부 코드).

| 셀 | 라운드 6 변경 |
|---|---|
| **19** | `finetune_epochs = 120` 정의 + `epochs=finetune_epochs` + **`patience=0`** + 주석 1줄 |
| **20** | 라벨을 `last_epoch_label` f-string 으로, `:<21s` 고정폭 |
| **22** | 라벨을 `finetuned_label` f-string 으로, `:<16s` 고정폭 |
| **23** | 패널 제목 2개를 f-string 으로 |
| **29** | `epochs=finetune_epochs` + **`patience=0`** + `segmentation_last_label` f-string, `:<18s` |

아래는 라운드 5 까지 누적된 변경 이력이다.

| 셀 | 무엇을 |
|---|---|
| **12** | 라벨 렌더링만 (`clip_on`·`va='top'`·lime 배경). 색 구성·박스·수치 무변경 |
| **16** | `instance_colors` 정의 + 인스턴스별 색 + 라벨 렌더링. confidence 유지 |
| **17** | Step 5 → Step 4. 통제표 전면 재작성 (행 구성·B 서술·test split 서술) |
| **19** | `epochs=120, optimizer='SGD', lr0=0.01` + 말미 `print` |
| **20** | 출력 라벨 `B after 20 epochs` → `B after 120 epochs` |
| **21** | 전면 재작성 |
| **22** | 신규 |
| **23** | 신규 |
| **24** | 신규 |
| **25, 28, 30** | Step 번호만 6→5, 7→6, 8→7 |
| **27** | `fill_colors` → `instance_colors` 이름만. 그림 동일 |
| **29** | `epochs=120, optimizer='SGD', lr0=0.01` + 출력 라벨 |
| **32** | Summary 2번째 불릿 교체 |
| **33** | Exercises 1·2·3 |

셀 0–11, 13–15, 18, 26, 31, 34 는 **손대지 않았다.**

---

## 6. 명세 대비 이탈 — 전수

| # | 이탈 | 사유 | 상태 |
|---|---|---|---|
| 1 | 명세 §4 셀 17–19 (Step 4 전체) 삭제 | **사용자 결정** — 강의 슬라이드가 다룬다 | 명세 개정 필요 |
| 2 | 명세 §2.1–2.3 의 IoU/precision/recall/AP/mAP 수식이 노트북에 없음 | 위와 같음. §2 는 이제 대응 셀이 없다 | 명세 개정 필요 |
| 3 | 명세 §3.6 `epochs=20` → **120**, `optimizer='SGD'`, `lr0=0.01` 추가 | **리더 실측 확정.** `optimizer=auto` 는 `lr0` 를 무시하고 `AdamW(lr=0.000119)` 를 골라 가중치가 움직이지 않는다 | 명세 개정 필요 |
| 4 | 명세 §3.7 실측 표 전부 무효 | 위와 같음 | 명세 개정 필요 |
| 5 | 명세 §3.7 의 "확정 4문장" 폐기 | `one iteration per epoch at lr=0.000119`, `mAP50 holds two values` 가 새 설정에서 **전부 거짓** | 명세 개정 필요 |
| 6 | 명세 §4.2 통제표 9행 → **6행** | `seed`·`amp`(C-5), `batch`(리더 지시) 삭제. `model size` 유지 | 명세 개정 필요 |
| 7 | 명세 §5 변수표에 없는 이름 8종: `heldout_image_path`, `heldout_label_path`, `heldout_image`, `pretrained_heldout_result`, `last_epoch_result`, `last_epoch_heldout_result`, `image_rows`, `panels` (+ 루프 지역 `ground_truth_names`/`pretrained_names`/`finetuned_names`/`panel_*`) | 신규 셀 22·23 이 요구. 전부 약어 아님 | 보고함 |
| 8 | Step 번호 4→7 재배치, 명세 §4 셀 번호 전부 밀림 | **리더 승인 완료.** §1 대응표 | 명세 개정 필요 |
| 9 | 시각 비교에 `best.pt` 아닌 `last.pt` 사용 | **리더 지시** ("`best.pt` 는 val 에서 골라진 것이라 논점이 흐려진다") | 확정 |

---

## 7. 자체 점검 결과 (판정은 감사자·검증자 몫)

```
markdown_budget.py  -> 0 over cap, 6 above soft cap
audit_notebook.py   -> 후보 없음 (16종 체크포인트)
execute_result      -> 0건 / error 0건 / 미실행 0건
루트 오염           -> .pt / runs/ 없음 (실행 후 확인)
non-ASCII (source)  -> ['—'] 하나. 셀 0 제목의 em dash
Step 헤더           -> Step 0 ~ Step 7 연속
```

soft 초과 6건: 셀 4·5·17·30 은 표 마크업이 밀어올린 Step 헤더 셀, 셀 21·32 는 산문 61/80.
**하드 캡에 가장 붙은 것은 셀 17 (116/120, 여유 4단어)** 이다.

`build/runs/` 에 이 노트북이 만들지 않은 `p8_ft/` 가 있다. 다른 회차 작업의 산출물로 보이며
이 노트북과 무관하다 (`build/` 안이라 레이아웃 위반도 아니다).

---

## 8. 알려진 제약 / 리더 확인 요청

1. **셀 17 여유 4단어** (116/120). 다음 라운드에 이 셀에 문장을 더하려면 표에서 행을 빼야 한다.
   `dataset` 행이 다음 후보다 (`evaluation split` 이 사실상 같은 것을 말한다).
2. **파일 크기 3.95MB.** `execute_result` 는 원인이 아니었고 92% 가 그림이다 (§4).
   줄이려면 `figsize`/`dpi` 조정이 필요하며 범위 밖으로 두었다.
3. **명세서가 개정 전 상태다.** §2·§3.6·§3.7·§4·§4.2·§5 가 실물과 어긋난다. `spec` 동기화 필요.
4. **Roboflow `.download()` 미검증** (잔존). `ROBOFLOW_API_KEY` 없음.
5. **셀 2·6 의 미실행 분기 잔존.** 패키지·데이터가 이미 있어 최초 실행 경로는 이번에도 안 돌았다.
6. ~~시각화 4셀의 렌더링 미확인~~ — **해소.** 리더가 먼저 보고 가독성 2건을 지적했고, 수정 후
   **내가 PNG 4장을 직접 열어 재확인**했다 (§4.1). `build/figures/` 에 추출본이 있다.
   남은 미세 결함 1건(셀 23 아랫줄 왼쪽 `person` 라벨 잘림)은 §4.1 에 근거와 함께 적었다.
7. `gen/p12.py` 헤더에 하드 규칙 **6번(`optimizer='SGD'` 명시)**, **7번(`.train()`/`.val()` 을 셀
   마지막 표현식으로 두지 말 것)** 을 추가했고, Step 재번호·2×2 패널 요구사항·`last.pt` 사용 이유를
   "되돌리기 전에 이유를 볼 것" 항목에 적었다.

---

## 9. 하네스 갱신 제출

**[트리거 1 — 3라운드 연속 재발]** "여러 행·열을 출력하는 셀의 해설문은 **모든 행과 열에 대해 참인지**
확인하고, 한 행만 보고 쓴 문장에는 한정어를 붙인다" / `CLAUDE.md` 새 절 /
근거: 라운드 4 에서 열 단위로 틀렸고(0.41646 > 0.39986), **이번 라운드 초안에서 행 단위로 또 틀렸다**
(`the validation rows ... went down` 인데 `best.pt` 행은 올라간다). 라운드 4 의 갱신 제출이 이미
같은 것을 지적했는데도 재발했다 — **제출만 되고 규약 문서에 반영되지 않았다는 신호다.**

**[트리거 1 — 이번 라운드에서 가장 값진 항목]** "학습 전/후를 그림으로 보일 때 **학습 데이터만
보여주면 안 된다.** held-out 샘플을 같은 그림에 넣는다" / `code-patterns.md` 새 절 /
근거: 내 초안은 1×2 (학습 이미지만) 였고 리더 초기 지시도 "학습 이미지임을 명시하라" 였다.
**둘 다 부족했고 사용자가 잡았다** — *"train 한 이미지로 test 하는 것 아니냐"*.
학습 이미지만 보이면 마크다운이 아무리 정직해도 **그림이 개선을 주장한다.**
문장으로 막을 수 없고 **패널 구성으로만 막힌다** — 라운드 1 의 `masks.data` 건(문장을 두 번 고쳐도
안 됐고 이미지를 바꿔서 풀렸다)과 **정확히 같은 계열**이다. 규약은 "명시하라" 가 아니라
"held-out 을 같은 그림에 넣으라" 로 서 있어야 한다.

**[트리거 4 — generator script 함정, 신규]** "학습·평가 API 호출(`.train()`, `.val()`, `.fit()`)을
**셀의 마지막 표현식으로 두지 않는다.** Jupyter 가 반환 객체를 렌더링하고, 지표 객체는 곡선 배열을
통째로 들고 있어 한 셀에 수만 자가 박힌다. 셀 끝을 `print` 로 닫고, 재생성 후 `execute_result`
잔존을 기계로 확인한다" / `generator-script-mechanics.md` / 근거: 셀 19, 59,415자.
정규식 감사로 잡히지 않고 **저작자가 노트북 JSON 을 열어 보지 않으면 발견되지 않는다.**

**[트리거 4 — 신규]** "라이브러리가 `optimizer='auto'` 류 기본값을 가질 때 **`lr0`/`lr` 을 주는 것만으로는
적용되지 않는다.** 학습이 실제로 일어났는지는 인자가 아니라 **로그에 인쇄된 optimizer 행**으로 확인한다" /
`code-patterns.md` / 근거: 개정 전 학습이 `lr0` 를 무시당해 `AdamW(lr=0.000119)` 로 돌았고,
그 결과 "가중치가 거의 안 움직였다" 는 해설을 **세 라운드에 걸쳐 다듬었다.**
원인은 서술이 아니라 설정이었고, 설정을 고치자 해설 문제가 통째로 사라졌다.

**[트리거 2 — 다음 회차가 재사용할 판단 기준]** "학습 전/후 그림에는 **숫자를 낸 것과 같은
체크포인트**를 쓴다. `best.pt` 로 그리고 `last.pt` 로 보고하면 그림과 표가 다른 모델을 말한다" /
`code-patterns.md` 의 체크포인트 선택 절에 하위 규칙으로 / 근거: 셀 22·23, 리더 확정.

**[트리거 2]** "파일 크기 문제는 원인을 추정하지 말고 **출력 종류별로 실측한다**" /
`practice-notebook-authoring` 검증 절차 / 근거: `execute_result` 59,415자가 3.4MB 의 원인으로
지목됐으나 기여는 1.7% 였고 92% 는 PNG 였다. 제거는 옳았지만(렌더링 노이즈) 크기 문제는 그대로 남는다.

**[트리거 1 — 라운드 6, 이 회차 결함 계열의 마지막 형태]**
"**학습 인자로 준 값이 실제로 그렇게 돌았는지 로그로 확인한다.** `epochs=120` 을 넘긴 것과
120 epoch 이 돈 것은 다르다 — early stopping·`close_mosaic`·scheduler 가 조용히 개입한다.
`N epochs completed` 를 눈으로 확인하기 전에는 라벨에 N 을 쓰지 않는다" /
`code-patterns.md` 학습 API 절 / 근거: V-11. `EarlyStopping(patience=100)` 이 발동해 101 에서
멈췄는데 라벨 8곳이 `120 epochs` 라고 단정했고, **반박 증거가 같은 셀 로그에 인쇄돼 있었다.**
이 회차가 다섯 라운드 내내 잡아 온 결함(단정문을 같은 셀 출력이 반박)과 같은 계열이되,
**거짓을 만든 것이 내 문장이 아니라 라이브러리 기본값**이라는 점이 다르다. 그래서 문장 검토로는
잡히지 않았다 — 검토 대상이 "내가 쓴 주장" 이었지 "인자와 로그의 일치" 가 아니었기 때문이다.

**[트리거 1 — 규약이 있는데 적용 범위가 좁았다]**
"`CLAUDE.md` "비교 실험 통제 변수" 의 *'`w_init`·`rho`·`n_iter` 를 한 번만 정의하고 재사용한다'* 를
**출력 라벨과 학습 API 인자까지** 명시적으로 확장한다" / `CLAUDE.md` 해당 절 /
근거: 감사 B-1. 규약은 이미 있었고 나는 그것을 알고 있었지만 **선형모델의 `n_iter` 에만 적용되는
것으로 읽었다.** 그 결과 Exercise 2 가 학생에게 바꾸라고 지시하는 값이 라벨에 하드코딩돼,
문제를 푼 학생 화면에 거짓이 찍히는 상태였다. **규약의 적용 범위가 예시로만 적혀 있으면
예시 밖에서는 지켜지지 않는다.**

**[트리거 1 — 저작 절차의 구멍, 이번 라운드에서 가장 값진 항목]**
"**그림을 내는 셀은 PNG 를 추출해 눈으로 보기 전까지 검증되지 않은 것으로 간주한다.**
`nbconvert` 통과·`output_type == 'display_data'` 존재는 '그림이 생성됐다' 는 뜻이지
'읽을 수 있다' 는 뜻이 아니다" / `practice-notebook-authoring` 의 실행 검증 절차 /
근거: 나는 라운드 4·5 내내 시각화 셀을 "코드와 출력 메타데이터로만 판정" 하고 **미검증 항목으로
보고만 했다.** 리더가 PNG 를 열자 라벨 잘림·경계 밖 렌더링·색 중복이 즉시 나왔고,
**셀 12 는 라벨이 겹쳐 아예 읽을 수 없는 상태로 5라운드를 통과했다.**
추출 비용은 10줄짜리 스크립트와 수 초다. 미검증으로 보고하는 것과 검증하는 것의 비용 차이가
이렇게 작은데도 다섯 라운드 동안 하지 않았다 — **"사람이 봐야 한다" 고 적는 것으로 내 몫을
다했다고 여긴 것이 잘못이다.** 저자가 볼 수 있는 것을 리뷰어에게 넘기면 안 된다.

**[트리거 4 — matplotlib 라벨 렌더링 기본형]**
"이미지 위에 박스와 라벨을 그릴 때 `ax.text` 에 **`clip_on=True`**(축 밖 렌더링 방지),
**`va='top'` + 박스 안쪽 배치**(이미지 상단 잘림 방지), **불투명 `bbox`**(겹침 시 가독성)
셋을 기본으로 둔다. 인스턴스가 여럿이면 **박스 색과 라벨 배경색을 같게** 해 짝짓기를 색으로 시킨다" /
`code-patterns.md` §5 시각화 / 근거: 이번 라운드 셀 12·16·23. 셋 다 기본값으로는 발생하는 문제이며
detection/segmentation 을 다루는 모든 회차에서 반복된다.

**[트리거 3 — 프로필·규약과 실제의 불일치, 라운드 4 에서 이미 제출됐고 미반영]**
`CLAUDE.md` "Notebook Generator Scripts" 절이 여전히 `_gen_p{NN}.py` 와 "검증 후 삭제" 를 말한다.
실제는 `gen/p12.py` 보존이며 **이번 개정 3건이 전부 그 보존 덕분에 가능했다** / 사용자 승인 사항.
