# Practice 12 개정 계획 — 진단과 재설계

> 대상: `Practice12_Object_Detection_and_Segmentation.ipynb` (현재 35셀, 라운드 7 확정본).
> 이 문서는 **재설계 근거**다. 확정되면 `01_spec_Practice12.md` 를 이 설계로 다시 쓴다.
>
> 이 문서의 모든 수치는 **실측**이다 (`ultralytics 8.4.120`, torch 2.6.0+cu124, TITAN RTX).
> 측정 스크립트와 로그는 개정 작업이 끝날 때까지 스크래치에 보관한다.

---

## 1. 진단 — 지금 노트북이 실패하는 지점

### 1.1 치명 — 노트북의 중심 실험이 "fine-tuning 은 손해다" 를 가르친다

Step 4·6 은 **COCO 사전학습 가중치**를 **coco8 4장**으로 120 epoch 학습시킨다.
그 4장은 COCO 학습셋의 일부라 사전학습 모델이 이미 맞히는 사진이다.

| 현재 노트북이 내는 결과 | 값 |
|---|---|
| A 사전학습 그대로 | mAP@50 0.684 |
| B `last.pt` (120 epoch, 선택 없음) | 사전학습의 **약 절반** |
| B `best.pt` (평가할 그 4장으로 고른 체크포인트) | A 보다 조금 높음 |

문제는 숫자가 아니라 **학생이 가져가는 결론**이다.

- 정직하게 읽으면 "fine-tuning 은 성능을 반으로 깎았다" 다. 노트북도 셀 24 에서 그렇게 적는다
  ("That is overfitting").
- 정직하지 않게 읽으면 `best.pt` 행이 개선처럼 보이는데, 그 행은 **평가할 4장으로 고른 체크포인트**다.
  노트북이 셀 17 통제표에 "The second row is not controlled" 라고 스스로 자백해야 하는 상태다.
- `coco8` 에는 test 분할이 없으므로 이 자백은 노트북 안에서 해소할 수 없다.

**사용자 지적이 정확하다.** 사전학습 모델이 이미 잘하는 클래스를 4장으로 이기려는 실험이라
설계상 이길 수 없다. 20 epoch 로 줄여도(리더 실측 0.68437 → 0.68790) 개선폭이 소수점 셋째 자리라
그림으로도 표로도 보이지 않는다. 120 epoch 는 그 효과를 짜내려다 오히려 과적합만 키운 흔적이다.

**연쇄 피해.** 35셀 중 **10셀 이상**(17·18·19·20·21·22·23·24·28·29·32)이 이 실패한 실험을
수행하고 변호하는 데 쓰인다. 감사 보고서의 미해결 항목 C-1·C-2·V-12 도 전부 이 한 뿌리에서 나왔다.

### 1.2 구조 — 같은 이야기를 두 번 한다

| 갈래 | 셀 | 하는 일 |
|---|---|---|
| detection | 13~24 | coco8 로드 → 사전학습 추론 → fine-tune → 지표 → 그림 |
| segmentation | 25~29 | coco8-seg 로드 → 사전학습 추론 → fine-tune → 지표 |

두 갈래가 **데이터셋 2개, `data.yaml` 2개(생성 코드 27줄), 학습 2회, 지표 표 2개**를 만든다.
그런데 **segmentation 모델은 box 를 함께 낸다** — `val()` 이 box 계열과 mask 계열 지표를 한 번에
돌려준다(셀 29 가 이미 그렇게 출력한다). 즉 detection 갈래는 원리적으로 중복이다.

세그 쪽 표에는 **비교 대상(사전학습 행)이 아예 없다.** 학생은 0.76 이라는 숫자를 무엇과 비교할지 모른다.

### 1.3 실행 결함 — 학생 환경에서 노트북이 멈춘다

셀 26 이 `DATA / 'sample.jpg'` 를 읽는데 **노트북 어디에도 이 파일을 만드는 코드가 없다.**
로컬 `data/sample.jpg` 가 이미 있어서 실행 검증을 통과했을 뿐, 학생이 Colab 에서 받으면
`FileNotFoundError` 로 Step 5 에서 멈춘다. (감사·정합성 보고서 양쪽이 이 파일을 "실재 확인" 하고
지나갔다 — 확인한 것은 *리더 PC 에 있다* 는 사실이지 *노트북이 만든다* 는 사실이 아니었다.)

### 1.4 죽은 코드가 가장 중요한 실무 단계를 차지한다

Step 7(Roboflow, 셀 31)은 `ROBOFLOW_API_KEY` 가 없으면 안내문만 찍고 끝난다.
키가 있어도 `workspace('your-workspace')`, `project('your-project')` 가 placeholder 라 실행되지 않는다.
**"직접 데이터셋을 만든다" 는 이 회차에서 가장 실무적인 단계인데, 그 자리를 한 번도 실행되지 않는
셀이 차지하고 있다.** (작업 기록도 이 왕복을 "미검증" 으로 남겼다.)

### 1.5 polygon — 예측을 잘못된 형태로 보여준다

셀 27 이 `masks.xy` 를 `matplotlib.patches.Polygon` 으로 그린다. 학생 눈에는
**"모델이 다각형을 출력한다"** 로 보인다. 실제 계산은 그렇지 않다.

| 무엇 | 실체 | 노트북의 현재 취급 |
|---|---|---|
| 라벨의 polygon | 저장 포맷. 가변 길이 — detection 라벨과 갈리는 지점 | 표시함 (**옳다**) |
| `masks.data` | 모델이 실제로 내는 per-pixel mask | shape 만 `print`, **그림으로 한 번도 안 보임** |
| `masks.xy` | 그 mask 에서 뽑은 contour | **예측의 대표 표현으로 그림** (오해 유발) |

실측으로 확인한 관계 (`data/sample.jpg` 2070x1380, `yolo11n-seg.pt`):

| `retina_masks` | `masks.data` | `orig_shape` | `masks.xy[0]` |
|---|---|---|---|
| `False` (기본) | `(7, 448, 640)` | `(1380, 2070)` | `(354, 2)` |
| `True` | `(7, 1380, 2070)` | `(1380, 2070)` | `(1088, 2)` |

같은 물체의 contour 꼭짓점이 354개에서 1088개로 바뀐다 — **polygon 은 모델의 출력이 아니라
mask 해상도에 따라 달라지는 파생물**이라는 증거다.

**그리고 mask 라야 되는 일을 노트북이 하나도 하지 않는다.** box 와 mask 를 나란히 그려 놓고
"무엇이 다른가" 를 숫자로 묻지 않는다. 픽셀 수(면적), box 대비 채움 비율, 겹침 — mask 의 쓸모가
전부 빠져 있다.

### 1.6 나머지 (감사 미반영분 + 신규)

| 항목 | 근거 |
|---|---|
| 셀 33 Exercise 2 가 `epochs` 를 바꾸라는데 실제 변수명은 `finetune_epochs` | 감사 B-1 (FIX, 미반영) |
| 셀 3 `device` 를 계산·출력하고 어디에도 안 씀 | 감사 C-3 |
| 셀 18·20·29 `val()` 에 학습 전용 인자 `seed`/`amp` | 감사 C-4 |
| 80개 클래스 `names` 를 두 번 써 내려가는 yaml 생성 27줄 | 이 회차 학습목표 대비 지분이 과다 |
| 그림 셀마다 반복되는 라벨 텍스트박스 코드 8줄 | 클래스가 여러 개라 필요했던 코드 |

---

## 2. 재설계 — 하나의 이야기로 바꾼다

### 2.1 전환

> **"사전학습 모델을 이겨라" → "사전학습 모델이 모르는 것을 가르쳐라"**

사전학습 모델은 COCO 80 클래스를 안다. **연구·현장의 대상은 대개 그 안에 없다.**
fine-tuning 의 쓸모는 성능을 몇 % 올리는 것이 아니라 **없던 클래스를 만들어 내는 것**이다.
이 전환 하나로 §1.1 의 실패가 사라지고, 효과가 표 한 줄과 그림 한 장으로 보인다.

### 2.2 대상 클래스 — `crack` (구조물 균열)

`crack-seg` (ultralytics 공식 자산, `https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip`).
train 3717 / val 200 / test 112, 단일 클래스 `crack`, **segmentation 라벨(polygon)**.

이 데이터셋을 고른 이유는 셋이다.

1. **COCO 에 없다** — `'crack' in model.names` → `False`. 사전학습 모델은 이 클래스를 출력할 수 없다
2. **기계공학 도메인** — 구조물·부품 검사. 대학원생이 자기 연구로 옮길 수 있는 과제다
3. **box vs mask 질문이 그림 한 장으로 결판난다** — 균열은 가늘고 비스듬한 선이라
   **자기 bounding box 의 9~33% 만 채운다**(실측). COCO 사물(cup 0.84, bus 0.65)과 대비된다

### 2.3 실측 — 이 설계가 실제로 무엇을 보여주는가

`crack-seg` 에서 **train 200장 / val 100장** 을 잘라 `yolo11n-seg`, `imgsz=320`, `batch=16`,
`SGD lr0=0.01`, `seed=42`, `epochs=20` 으로 측정했다.

| 조건 | box mAP@50 | mask mAP@50 | 학습 시간 |
|---|---|---|---|
| A. COCO 사전학습 그대로 (학습 없음) | **0.0118** | **0.0025** | — |
| B. A 의 가중치에서 출발해 20 epoch | **0.6079** | **0.3305** | 84.6초 |
| C. 같은 구조, 무작위 초기화에서 20 epoch | **0.0094** | **0.0049** | 51.9초 |

**A → B 가 이 노트북의 답이다.** 0.01 에서 0.61 로, 표를 읽지 않아도 그림에서 보인다.

**B vs C 는 그 이유다.** 데이터·epoch·lr·optimizer·batch·seed·imgsz 가 전부 같고
**다른 것은 초기 가중치 하나뿐**인 완전 통제 비교다. 200장으로 되는 이유가
"YOLO 가 좋아서" 가 아니라 **사전학습된 backbone 을 물려받았기 때문**임을 보인다.
현재 노트북이 실패한 자리에 **제대로 통제된 A/B** 가 들어선다.

사전학습 모델의 실제 반응 (val 이미지 4장, `conf=0.25`):

```
1604...jpg : 0 objects []
1605...jpg : 2 objects ['bird', 'bird']
1610...jpg : 0 objects []
1619...jpg : 0 objects []
```

아무것도 못 찾거나 새를 본다. **그림 한 장으로 동기가 끝난다.**

### 2.4 box vs mask — 숫자로 답한다

fine-tune 한 모델의 예측에서 `mask 픽셀 수 / box 면적` (실측):

| 대상 | 채움 비율 |
|---|---|
| crack (4장, `retina_masks=True`) | 0.091 / 0.101 / 0.327 / 0.255 |
| COCO 사물 (`bus.jpg`) | bus 0.65, person 0.33~0.62, stop sign 0.83 |

균열의 box 는 **90% 가 배경**이다. "균열 면적·길이를 재려면 box 로는 안 되고 mask 여야 한다" 는
결론이 학생의 화면에서 나온다. 이것이 지금 노트북에 통째로 빠져 있는 부분이다.

### 2.5 polygon 을 어디에 둘 것인가 — 사용자 질문에 대한 답

**예측 표현은 mask 로 바꾼다. polygon 은 라벨 포맷 자리에만 남긴다.**

| 자리 | 표현 | 이유 |
|---|---|---|
| 라벨 파일 (Step 3) | **polygon** | 실제 저장 포맷이다. `class x1 y1 ... xn yn` 의 **가변 길이**가 detection 라벨 `class cx cy w h` 의 고정 4개와 갈리는 지점이며, 그것이 두 과제의 차이 그 자체다 |
| 예측 시각화 (Step 5·6) | **mask** (`imshow` + alpha) | 모델이 실제로 내는 것이 per-pixel mask 다. 픽셀을 셀 수 있어 면적·채움 비율이 나온다. 구멍·분리된 조각도 그대로 그려진다 |
| `masks.xy` (Step 7) | **한 번만, 파생물로서** | mask 에서 뽑은 contour 임을 `retina_masks` 로 꼭짓점 수가 354 → 1088 로 바뀌는 실측으로 보인다 |

덤으로 코드가 짧아진다. 단일 클래스라 **인스턴스마다 색·이름 텍스트박스를 그리던 8줄이 통째로 사라지고**,
마스크 오버레이는 `imshow(mask, alpha=0.5)` 한 줄이다.

### 2.6 "라벨도 그냥 픽셀 mask 로 하면 되지 않나" — 사용자 질문에 대한 답

**선택지가 아니다.** ultralytics 로더가 읽는 세그멘테이션 라벨은 텍스트 polygon 한 줄뿐이다.
PNG mask 를 이미 갖고 있을 때의 공식 경로가 `ultralytics.data.converter.convert_segment_masks_to_yolo_seg`
— **마스크를 polygon 으로 바꿔서** 넣는다.

포맷이 그렇게 정해진 이유:

| 이유 | 내용 |
|---|---|
| 증강이 좌표 곱셈으로 끝난다 | mosaic·scale·flip·perspective 를 polygon 은 아핀 변환 한 번으로 처리한다. 픽셀 mask 는 매 배치 warp + 재샘플링이고 경계가 뭉개진다 |
| 해상도 독립 | 정규화 좌표라 `imgsz` 를 바꿔도 같은 라벨이 유효하다 |
| 사람이 만드는 방식 | 라벨링 도구가 클릭으로 꼭짓점을 찍는다 |
| detection 라벨과 같은 계열 | `class + 좌표들` 한 줄. 4개면 box, 2n개면 outline |

**용량은 근거로 쓰지 않는다.** 실측: crack200 의 20장 기준 polygon 텍스트 **72.9 KB** vs
1-bit PNG 마스크 **22.8 KB** — 균열은 꼭짓점이 수백 개라 **오히려 polygon 이 3배 크다.**
"polygon 이 가볍다" 는 통설은 COCO 사물 기준이며 이 데이터셋에서는 반대다. 노트북에 쓰면 거짓이 된다.

**그리고 학습 시점에는 결국 픽셀 mask 가 된다.** 로더가
`ultralytics.data.utils.polygons2masks(imgsz, polygons, color, downsample_ratio)` 로 rasterize 하고,
학습 인자에 찍히는 `mask_ratio=4` 가 그 다운샘플 비율이다. 손실은 픽셀 단위로 계산된다.

```
polygon 텍스트 (저장·증강) → rasterize → 픽셀 mask (손실) → 픽셀 mask (예측) → contour (masks.xy)
```

**→ Step 3 에 셀 하나를 추가한다.** 라벨 polygon 을 그리고, 같은 polygon 을 채워 이진 mask 로 만들고
(`PIL.ImageDraw.polygon`, 5줄), 두 패널을 나란히 놓아 "손실이 비교하는 것은 오른쪽" 을 보인다.
지금 노트북에 없는 다리이며, 이 질문은 학생에게서 그대로 나온다.

---

## 3. 개정 노트북 구성안 (약 28셀, 현재 35셀)

| Step | 셀 | 내용 | 남기는 것 |
|---|---|---|---|
| — | 1 | 제목 + 이 노트북의 주장 (사전학습이 아는 것 / 모르는 것) | |
| 0. Setup | 3 | 설치, import, `BUILD`/`DATA`/`RUNS` 경로, seed | 현재 셀 2·3 (device 삭제) |
| 1. Two outputs | 3 | `bus.jpg` 한 장에 detect 모델과 seg 모델. `boxes.xyxy/conf/cls`, `masks.data` shape 출력 + box \| mask 2패널 그림 | 현재 Step 1·3 을 합쳐 압축 |
| 2. The blind spot | 3 | `crack-seg.zip` 다운로드, `'crack' in model.names` → `False`, 균열 4장에 사전학습 추론 → 0 objects / 'bird' 그림 | **신규 — 동기** |
| 3. Making a dataset | 6 | 폴더 트리 출력, polygon 라벨 한 줄 파싱(꼭짓점 수), 라벨을 이미지에 되그리기, **polygon → 이진 mask rasterize**(§2.6), **polygon 에서 box 유도**(두 포맷의 관계), train 200 / val 100 subset 구성 + `data.yaml` 작성 | 현재 Step 2 + **죽은 Roboflow 대체** |
| 4. Fine-tuning | 4 | 통제표(A/B/C), `train()` 1회, `val()` 3행 출력 | 현재 Step 4·6 을 **하나로 병합** |
| 5. Before / after | 2 | 같은 held-out 4장, 위 = 사전학습, 아래 = fine-tune. mask 오버레이 | 현재 셀 22·23 의 자리 |
| 6. Box vs mask | 2 | `mask 픽셀 / box 면적` 을 균열과 `bus.jpg` 사물에 나란히 | **신규 — mask 의 쓸모** |
| 7. Where the mask comes from | 3 | `masks.data` 해상도 ≠ 원본, `retina_masks=True` 로 원본 해상도, `masks.xy` 꼭짓점 354 → 1088 | 현재 Step 5 의 핵심만 |
| — | 2 | Summary + Exercises | |

### 사라지는 것

- `coco8`, `coco8-seg` 두 데이터셋과 80개 `names` yaml 생성 코드 27줄
- detection 전용 fine-tuning 갈래 전체 (Step 4) — seg 학습 1회가 box·mask 지표를 함께 낸다
- `best.pt` vs `last.pt` 체크포인트 선택 논쟁 — **val 100장에 test 분할 근거가 생기므로 `last.pt` 하나만 본다**
- Roboflow 셀 (실행되지 않는 코드) → markdown 두 줄로 축약: "export 도 같은 폴더 구조와 `data.yaml` 이다"
- `data/sample.jpg` 의존 (다운로드되는 `bus.jpg` 로 교체, 134KB)

### 남는 위험과 확인할 것

| 항목 | 상태 |
|---|---|
| `crack-seg.zip` 96.1MB 다운로드 (10.7초, 압축 해제 6.9초) | 300장만 쓰는데 96MB. 허용 가능하나 학생 안내에 명시 |
| zip 이 **flat 하게 풀린다** (`crack-seg/` 최상위 폴더 없음, `coco8.zip` 과 다름) | 노트북 코드가 이 차이를 흡수해야 함. 실측 확인 완료 |
| **CPU 런타임 학습 시간 미측정** | GPU 84.6초. CPU 는 재측정 필요. 상단에 `subset_size` 를 변수로 노출해 줄일 수 있게 한다 |
| 균열 이미지가 416x416 이라 `masks.data` 해상도 차이가 드러나지 않음 | Step 7 은 `bus.jpg`(810x1080)로 보인다 — 실측 `(6, 640, 480)` vs `orig (1080, 810)` |
| 강의 슬라이드 없음 (이 회차 공통) | 개정 후 `01_spec` 이 다시 유일한 재생성 근거가 된다 |

---

## 4. 작업 순서

1. 이 계획 승인 → `01_spec_Practice12.md` 를 이 설계로 재작성 (셀 단위 명세 + 통제표)
2. `gen/p12.py` 재작성 → `jupyter nbconvert --execute` 전 셀 실행 검증
3. `python .claude/scripts/markdown_budget.py` + `audit_notebook.py` 감사
4. 정합성 검증 (강의자료 없음 → ultralytics 공식 문서 축)
5. 실행 후 과목 루트 오염 확인 (`.pt` / `runs/` 잔여 0건 — 실측 스크립트에서 `[cwd extras] []` 확인함)
