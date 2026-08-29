# Detection / Segmentation 회차 재구성 기록

> 사용자(교수) 결정으로 **한 회차짜리 노트북 하나를 세 개로 쪼갰다.**
> 근거는 `05_revision_plan.md` 의 진단이고, 이 문서는 그 결과와 실측값을 남긴다.
>
> **이 문서가 세 노트북의 현재 상태에 대한 기준이다.** `01_spec_Practice12.md` 는 개정 전
> 35셀 노트북의 명세이므로 **더 이상 유효하지 않다** — 대조에 쓰지 말 것.

## 1. 무엇이 어디로 갔나

| 새 산출물 | 내용 | 원본 |
|---|---|---|
| `Practice12_Object_Detection_and_Segmentation.ipynb` (35셀) | 사전학습 추론만. `result.boxes` / `result.masks` 구조를 열어 보고 이미지에 덧그린다 | 2025_2 과목의 `HW4_answer.ipynb` 를 확장 |
| `Practice13_Fine_Tuning_on_a_New_Class.ipynb` (41셀) | crack 데이터 로드 → 라벨 포맷 → subset 구성 → **detection fine-tuning + 박스 출력** → 통제 비교 → 가중치만 바꿔 segmentation | `HW5.pdf` 의 절차를 키 없이 도는 데이터로 옮김 |
| `HW/HW_PCB_Defect_Detection.ipynb` + `HW/Answer/..._answer.ipynb` (각 16셀) | PCB 결함 6클래스 fine-tuning 과제 | `HW5.pdf` (Roboflow PCB defect) |
| `tools/prepare_pcb_dataset.py` | PKU PCB 원본 693장을 640 으로 축소 + VOC XML → YOLO 변환 + zip | 신규 |

생성기는 `gen/p12.py`, `gen/p13.py`, `gen/hw_pcb.py` 다. `hw_pcb.py` 하나가 배포본과 정답본을
`answer` 플래그로 동시에 만든다 (`CLAUDE.md` "HW 규약").

**번호는 사용자 지시로 붙이지 않았다** — HW 파일명에 숫자가 없다. 나중에 부여한다.

## 2. 삭제된 것과 그 이유

| 삭제 | 이유 |
|---|---|
| coco8 / coco8-seg 로 4장 fine-tuning (구 Step 4·6) | 사전학습이 이미 맞히는 사진이라 효과가 음수였다. `05_revision_plan.md` §1.1 |
| `best.pt` vs `last.pt` 체크포인트 논쟁 | val 100장이 생겨 통제가 성립한다. `last.pt` 하나만 본다 |
| Roboflow 셀 (`your-workspace` placeholder) | 키가 있어도 실행되지 않는 죽은 코드였다. 학생이 직접 subset·yaml 을 만드는 코드로 대체 |
| 80개 클래스 `names` yaml 생성 27줄 | 단일 클래스 데이터셋이라 `names: {0: crack}` 다섯 줄로 줄었다 |
| `data/sample.jpg` 의존 (다운로드 코드 없음) | Practice12 가 unsplash URL 에서 직접 받는다. 학생 환경에서 멈추던 결함 해소 |
| 예측을 `matplotlib.patches.Polygon` 으로 그리기 | 모델이 다각형을 낸다는 오해를 준다. `masks.data` 오버레이로 교체 |

## 3. 실측값 — 노트북이 실제로 출력한 숫자

측정 환경: `ultralytics 8.4.120`, torch 2.6.0+cu124, NVIDIA TITAN RTX, 과목 루트에서 실행.

### Practice12 (학습 없음)

| 항목 | 값 |
|---|---|
| `data/sample.jpg` | 2070x1380, detection 6개 / segmentation 7개 (`conf=0.25`) |
| `masks.data` 기본 | `(7, 448, 640)` — 원본 `(1380, 2070)` 과 다르다 |
| `masks.data` `retina_masks=True` | `(7, 1380, 2070)` |
| `masks.xy[0]` | 기본 354점 → retina 1088점. **outline 은 mask 에서 뽑은 파생물** |
| mask/box 채움 비율 | cup 0.842, remote 0.885, chair 0.382~0.621, dining table 0.385 |

### Practice13 (crack, train 200 / val 100, imgsz=320, batch=16, SGD lr0=0.01, seed=42, 20 epoch)

> **이 절의 수치는 3차 수정으로 대체됐다 — §3-2 를 볼 것.** 평가 분할이 val 100장에서 test 112장으로,
> 학습 데이터가 복사한 subset 200장에서 `fraction=0.05` (186장) 로 바뀌었다.
> 아래 표는 그 변경 전 기록이며, 결론(A ≈ 0, B 크게 상승, C 바닥)은 동일하다.

**구성은 detection → segmentation 의 단계 상승이다** (사용자 요청). 같은 데이터셋·같은 호출에서
가중치 파일 한 단어만 바뀐다. 학습은 3회 — detection fine-tune, detection scratch, segmentation
fine-tune.

`yolo11n.pt` (detection) — Step 4~7 의 통제 비교:

| 조건 | box mAP@50 | box mAP@50-95 | 학습 시간 |
|---|---|---|---|
| A 사전학습 그대로 | 0.00272 | 0.00110 | — |
| B 사전학습에서 fine-tune | **0.61904** | **0.35435** | 39.2초 |
| C 무작위 초기화, 나머지 동일 | 0.00209 | 0.00051 | 37.5초 |

`yolo11n-seg.pt` (segmentation) — Step 8:

| 조건 | box mAP@50 | mask mAP@50 |
|---|---|---|
| 사전학습 그대로 | 0.01183 | 0.00248 |
| fine-tune (84.6초) | **0.60785** | **0.33046** |

A vs B 는 학습 여부만, B vs C 는 초기 가중치만 다르다.
crack 의 mask/box 채움 비율 0.091 / 0.101 / 0.327 / 0.255 — **박스의 70~90% 가 배경**이다.

> **ultralytics 는 polygon 라벨 데이터셋으로 detection 모델도 학습한다** (라벨의 extent 를 박스로
> 삼는다). 실측으로 확인했고, 덕분에 데이터셋 하나가 두 과제를 모두 먹인다.
> Step 2 가 polygon 에서 `class cx cy w h` 를 유도해 보이는 것이 그 근거다.

### HW (PCB, train 561 / val 66 / test 66, imgsz=640, batch=16, SGD lr0=0.01, seed=42, 30 epoch)

| 항목 | 값 |
|---|---|
| 학습 시간 | 18.5초/epoch → 30 epoch 약 9분 (TITAN RTX) |
| 데이터 | 원본 693장 960MB → 640 축소 후 **zip 50.6MB** |
| 분할 | 클래스마다 순서대로 8/10 train, 1/10 val, 1/10 test |
| 클래스 균형 | train 기준 392~409개로 고르다 (Problem 6 의 근거) |
| **test mAP@50** | **0.85871** (mAP@50-95 0.43725) |
| 클래스별 AP@50-95 | missing_hole 0.553 / short 0.457 / mouse_bite 0.412 / open_circuit 0.411 / spurious_copper 0.410 / **spur 0.381 (최저)** |
| 사전학습 모델 (Problem 2) | 같은 PCB 이미지에서 **0개 검출** |

**Problem 5 의 `conf` 는 0.10 이다.** 0.25 에서는 6장 중 3장이 0개로 나와 "학습이 안 된 것" 처럼 보인다
(실측: mouse_bite 0, open_circuit 0, spur 0). 30 epoch 모델의 confidence 가 COCO 모델보다 낮다는 사실
자체를 노트북에 적었고, 0.25 로 되돌려 보는 것을 Problem 6-2 로 물었다. conf=0.10 에서 검출/정답 =
3/3, 3/3, 3/3, 3/4, 1/2, 5/3 이다.

## 3-1. 2차 수정 — 사용자 피드백 4건

| 지적 | 조치 |
|---|---|
| 학습 로그가 배치마다 찍혀 지저분하다 | **`os.environ['YOLO_VERBOSE'] = 'False'`** 를 `import ultralytics` **앞**에 둔다. 실측으로 학습·검증 로그가 완전히 사라졌다 (P13 의 stream 출력이 수백 chunk → **25개**). 대신 `results.csv` 를 읽어 loss·mAP 곡선을 그린다 |
| fine-tuning 전에도 detection·segmentation 을 둘 다 보여야 한다 | P13 Step 1 이 두 모델을 모두 돌린다. 실측 출력: detection `['giraffe','giraffe']`, segmentation `['bird','bird']`, 나머지 3장은 0개 |
| 스텝마다 결과 해설이 없다 | 출력·그림 뒤에 해석 마크다운을 붙였다 (P12 4개, P13 7개, 셀당 3줄 이내) |
| instance vs semantic 구분이 필요한가 | 정의 표를 P12 Step 5 와 P13 Step 9 에 각각 뒀다. **crack 에서는 instance 경계가 라벨의 판단**이라는 점을 P13 에 적었다 (분기하는 균열은 연결된 한 덩어리다) |

`YOLO_VERBOSE` 는 **import 시점에 읽힌다.** 설치 셀에서 `import ultralytics` 보다 먼저 설정해야
효력이 있다 — 나중에 설정하면 조용해지지 않는다.

곡선용 컬럼 (실측): `epoch`, `train/box_loss`, `val/box_loss`, `metrics/mAP50(B)`,
segmentation 은 `train/seg_loss`, `metrics/mAP50(M)` 이 추가된다. `(B)` = box, `(M)` = mask.

## 3-2. 3차 수정 — 학생 제출 답안과의 비교

사용자가 2025_2 HW5 학생 제출본(6셀)을 보여주며 **"이 답안을 보면 엄청 간단한데?"**,
**"checkpoint 를 저장하고 불러서 보는 것보다 훈련 과정을 다 보여주는 게 좋겠다"**,
**"구문이 깔끔한 것보다 직관적인 게 좋다 (기계공학도는 파이썬을 잘 하지 못함)"** 라고 지적했다.
**파이썬 숙련도를 요구하는 구문을 걷어냈다.** P13 50셀 → **41셀**, HW 20셀 → **16셀**.

| 걷어낸 것 | 대신 |
|---|---|
| `YOLO(RUNS/'crack_detect'/'weights'/'last.pt')` 재로드 | `train()` 뒤 **같은 객체로 바로 `val()`**. 학생 답안과 같은 흐름 |
| `best.pt` 가 val 로 선택되는 문제 | **`split='test'` 로 평가**. crack-seg 에 `images/test` 112장 + 라벨이 있다 (실측 확인) |
| subset 폴더 복사 (`shutil` 이중 루프 10줄) | **`train(fraction=0.05)`** 인자 하나 (3717장의 5% = 186장) |
| 손으로 `Rectangle` 그리기, `np.ma.masked_where`, `instance_map` 누적 | **`result.plot()[:, :, ::-1]`** — 학생 답안과 같은 방식 |
| `PIL.ImageDraw` 로 polygon rasterize | `axes.fill(outline_x, outline_y)` — matplotlib 한 줄 |
| HW 문제 6개 | **4개** (코드 3 + 서술 1). 라벨 개수 세기 문제는 삭제하고 사실만 도입부 표에 남겼다 |

**손으로 그리는 코드는 Practice12 에 남아 있다.** 그 회차의 주제가 "결과 구조를 열어 보고 덧그리기"
이기 때문이다. Practice13 의 주제는 학습이므로 그리기는 `plot()` 한 줄로 끝낸다.

> **함정.** `val()` 에서 `project=` 를 빼면 **과목 루트에 `runs/` 가 생긴다.**
> 간결하게 쓰려다 한 번 빠뜨렸고 실행 전에 되돌렸다. `val()` 에도 `project=str(RUNS)` 를 넘긴다.

### 3차 수정 후 실측 (test 분할 112장 기준, fraction=0.05, 20 epoch)

| 조건 | box mAP@50 |
|---|---|
| A 사전학습 그대로 | 0.01057 |
| B fine-tune | **0.49755** |
| C 무작위 초기화 | 0.00388 |

segmentation (같은 설정, `yolo11n-seg.pt`): box 0.46162 / **mask 0.37393**.
mask/box 채움 비율 0.151 ~ 0.429. 사전학습 모델은 test 4장 모두 **0개 검출** (이전 val 4장에서
`giraffe`/`bird` 가 나오던 것과 달라졌으므로 Step 1 해설 문구를 실측에 맞춰 고쳤다).

HW 는 문제 구성만 바뀌고 학습 설정은 그대로다: test mAP@50 **0.85830**, `spur` 0.38068 최저.

## 4. 남은 작업 — 사용자 조치가 필요하다

1. **`build/pcb_defect_640.zip` (50.6MB) 을 공개 URL 에 올려야 한다.**
   HW 노트북은 아래 주소를 받도록 생성돼 있다:
   `https://raw.githubusercontent.com/jongmoonha/AI-ME-Practice/main/data/pcb_defect_640.zip`
   업로드 전까지 이 한 줄만 미검증이다 (검증은 `python gen/hw_pcb.py --local` 로 로컬 zip 을 받아 수행했다).
   주소가 달라지면 `gen/hw_pcb.py` 의 `DATASET_URL` 을 고치고 재생성한다.
2. CPU 전용 환경(Colab CPU 런타임) 학습 시간은 여전히 미측정이다. GPU 기준 시간만 노트북에 적었다.
3. `data/coco8`, `data/coco8-seg`, `data/coco8*.yaml` 은 이제 어느 노트북도 쓰지 않는다. 지워도 된다.

## 5. 검증 상태

| 노트북 | 전 셀 실행 | `markdown_budget.py` | `audit_notebook.py` |
|---|---|---|---|
| Practice12 | 통과 — 14/14 셀, 에러 0, 루트 오염 0 | 0 over cap | 후보 없음 |
| Practice13 | 통과 — 19/19 셀, 에러 0, 루트 오염 0 | 0 over cap | 후보 없음 |
| HW 정답본 | 통과 — 8/8 셀, 에러 0. **단 zip 주소는 로컬 `file://` 로 돌렸다** (§4-1) | 0 over cap | 후보 없음 |
| HW 배포본 | 해당 없음 — 학생 작성 셀이 비어 있다 (`CLAUDE.md` "HW 규약") | 0 over cap | 후보 없음 |

**배포용 정답본에는 출력이 없다.** 검증은 `--local` 로 만든 판으로 했고 그 실행본을
`build/HW_answer_executed_with_local_zip.ipynb` 에 남겼다. zip 을 올린 뒤
`python gen/hw_pcb.py` → 정답본 1회 실행으로 출력을 채우면 된다.

**HW 노트북은 자기 위치 기준으로 `build/` 와 `data/` 를 만든다.** Jupyter 커널의 작업 디렉터리가
노트북 폴더이기 때문이다 (검증 중 `HW/Answer/` 에 116MB 가 생겨 지웠다). 학생은 노트북 하나만 받아
어디서든 돌리므로 이 동작이 맞지만, 과목 저장소 안에서 실행하면 `HW/Answer/data/` 가 생긴다.
`.gitignore` 는 `build/` 만 막으므로 커밋 전에 확인할 것.

또 하나 — URL 두 개는 실측으로 확인했다.
`images.unsplash.com/photo-1760681557777-...&w=2070` → 2070x1380,
`ultralytics.com/images/bus.jpg` → 810x1080.
