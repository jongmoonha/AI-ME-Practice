# 환경 프로브 결과 — Practice 12 (Object Detection and Segmentation)

리더가 노트북 저작 전에 **직접 실행해 실측**한 결과다. 추측이 아니라 측정값이므로 저자는 이 결론을
그대로 따르고, 다르게 하려면 먼저 재측정하라.

측정 환경: `ultralytics 8.4.120`, `torch 2.6.0+cu124`, CUDA NVIDIA TITAN RTX, Python 3.13.5,
과목 루트(`01_AI-ME_Graduate/`)에서 실행.

---

## 1. 확정된 경로 전략 (가장 중요)

과목 규약은 "루트에는 `Practice*.ipynb` 와 `CLAUDE.md` 만 둔다" 이고 "과목 폴더는 독립적으로 완결된다"
이다. ultralytics 는 기본값으로 **둘 다 깬다.** 실측으로 확인한 사항:

| 문제 | 실측 내용 | 확정 해법 |
|------|----------|----------|
| 가중치가 루트에 떨어짐 | `YOLO('yolo11n.pt')` 는 cwd 에 `.pt` 를 받는다 | **`YOLO(str(BUILD / 'yolo11n.pt'))`** — 경로를 주면 그 자리에 받는다. 검증: `build/ ok=True, cwd polluted=False` |
| 데이터셋이 남의 과목 폴더로 감 | 기본 `datasets_dir` 이 `D:\...\01_AI-ME_Graduate\AI-ME-Practice\datasets` 를 가리킨다 (**다른 과목**) | zip 을 직접 받아 `data/` 에 푼다 (아래 3절) |
| `settings.update({'datasets_dir': ...})` | 호출해도 **그 프로세스의 학습에 반영되지 않았다.** 여전히 옛 경로를 스캔했다 | 쓰지 말 것. 자기 `data.yaml` 의 절대 `path` 로 해결한다 |
| `settings.update({'weights_dir': ...})` | 다운로드 위치를 **바꾸지 못했다.** 가중치는 여전히 cwd 로 갔다 | 쓰지 말 것 |
| `project=` 에 상대경로 | `project='build/runs'` 를 주면 `runs/detect/build/runs/...` 로 중첩되어 **루트에 `runs/` 가 생긴다** | **`project` 는 반드시 절대경로**: `project=str(BUILD / 'runs')` |
| `train()` 중 `yolo26n.pt` 가 루트에 생김 | AMP 체크가 별도 모델을 cwd 로 받는다 | **`amp=False`** 를 넘긴다. 지표 영향 없음 (map50 0.6879 → 0.68777) |

`amp=False` 로 학습한 뒤 루트 확인 결과 `root extras: []` — 오염 없음.

## 2. 학습 시간 — 상한 300초에 여유롭게 들어온다

| 작업 | 설정 | 실측 |
|------|------|------|
| detection fine-tune | coco8, epochs=20, imgsz=320, batch=4 | **10.2초** |
| segmentation fine-tune | coco8-seg, epochs=20, imgsz=320, batch=4 | **9.2초** |
| 가중치 다운로드 | `yolo11n.pt` 5.4MB / `yolo11n-seg.pt` 5.9MB | 각 ~1초 |
| 데이터셋 다운로드 | coco8 / coco8-seg zip (각 ~1MB) | 각 1초 미만 |

epochs 를 20 으로 잡아도 충분하다. CPU 전용 환경(Colab 무료 티어의 CPU 런타임)에서는 더 걸리므로
저자는 `imgsz=320` 을 유지하라.

## 3. 데이터셋 준비 — 검증된 형태

`data/` 에 직접 받는다. **패키지에 들어 있는 `coco8.yaml` 을 노트북에서 출력하지 마라** — 헤더에
`🚀` 이모지와 `←`, `└`, `├` 박스문자가 들어 있어 이모지 금지 규약을 즉시 위반한다
(실측: `non-ascii chars: ['←', '─', '└', '├', '🚀']`).

```python
url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip'
# coco8-seg 는 .../v0.0.0/coco8-seg.zip
```

압축을 풀면 `data/coco8/` 아래에 `images/`, `labels/`, `LICENSE`, `README.md` 가 생긴다.
`images/train` 4장, `images/val` 4장.

## 4. `data.yaml` — 직접 쓴다. 단 제약이 있다

**`names` 는 반드시 `0 .. N-1` 연속이어야 한다.** 라벨에 등장하는 클래스만 골라 sparse 하게 쓰면
학습이 시작조차 못 한다. 실측 에러:

```
KeyError: '4-class dataset requires class indices 0-3,
           but you have invalid class indices 0-50 defined in your dataset YAML.'
```

coco8 라벨은 COCO 80클래스 인덱스 체계(45=bowl, 49=orange, 50=broccoli 등)를 쓰므로
**80개 이름을 전부 적어야 한다.** 노트북에 80줄을 손으로 적지 말고 사전학습 모델이 이미 갖고 있는
이름표에서 생성하라 — `YOLO(...).names` 가 `{0: 'person', 1: 'bicycle', ...}` 형태의 dict (len 80) 다.
"라벨 파일의 모든 클래스 인덱스가 `names` 에 있어야 한다" 는 것 자체가 가르칠 내용이다.

검증된 `data.yaml` 형태 (`path` 는 절대경로, POSIX 슬래시):

```
path: D:/.../01_AI-ME_Graduate/data/coco8
train: images/train
val: images/val
names:
  0: person
  1: bicycle
  ...
  79: toothbrush
```

## 5. 라벨 포맷 — 실물 확인

detection `data/coco8/labels/train/000000000009.txt` — `class cx cy w h`, 전부 [0,1] 정규화:

```
45 0.479492 0.688771 0.955609 0.5955
45 0.736516 0.247188 0.498875 0.476417
50 0.637063 0.732938 0.494125 0.510583
```

segmentation `data/coco8-seg/labels/train/000000000009.txt` — `class x1 y1 x2 y2 ...` polygon, 정규화.
첫 줄이 토큰 25개 = 클래스 1 + 좌표 24 = **꼭짓점 12개**:

```
45 0.782016 0.986521 0.937078 0.874167 0.957297 0.782021 ...
```

두 포맷의 차이(고정 4개 숫자 vs 가변 길이 polygon)가 곧 detection 과 instance segmentation 의
출력 차이다. 이 대응을 노트북에서 드러내라.

## 6. 추론 결과 객체 — 실측한 속성명

`data/sample.jpg` (1380x2070) 에 `yolo11n.pt`, `conf=0.25`:

| 속성 | 실측 shape / 값 |
|------|----------------|
| `result.boxes.xyxy` | `(6, 4)` — 픽셀 좌표 |
| `result.boxes.xywhn` | `(6, 4)` — 정규화 좌표 (라벨 포맷과 같은 형식) |
| `result.boxes.conf` | `(6,)` |
| `result.boxes.cls` | `(6,)` — float. `int(...)` 로 변환해 `result.names` 를 조회한다 |
| `result.names` | dict, len 80 |
| `result.orig_shape` | `(1380, 2070)` |
| `result.plot()` | `(1380, 2070, 3)` uint8 — **BGR 이다.** matplotlib 에 넣으려면 뒤집어야 한다 |

검출 예: `chair conf=0.833`, `potted plant conf=0.804`, `cup conf=0.766`.

`yolo11n-seg.pt` 같은 이미지:

| 속성 | 실측 |
|------|------|
| `seg_result.masks.data` | `(7, 448, 640)` — **모델 입력 해상도이지 원본 크기가 아니다** |
| `seg_result.masks.xy[0]` | `(354, 2)` — 원본 픽셀 좌표계의 polygon |
| `seg_result.boxes.xyxy` | `(7, 4)` — segmentation 모델도 박스를 함께 낸다 |

`masks.data` 의 해상도가 원본과 다르다는 점에 주의하라. 원본 위에 겹쳐 그릴 때는 `masks.xy` 를 쓰거나
리사이즈해야 한다.

## 7. 지표 — 통제된 before/after 비교가 성립한다

같은 `data.yaml`, 같은 `imgsz=320` 으로 측정:

| 조건 | mAP@50 | mAP@50-95 |
|------|--------|-----------|
| 사전학습 그대로 (fine-tune 없음) | 0.68437 | 0.39986 |
| coco8 로 20 epoch fine-tune | 0.68790 | 0.41836 |

**개선폭이 작다는 것이 정직한 결과다.** 학습 이미지가 4장이고, 그 4장이 이미 COCO 학습셋의 일부라
사전학습 모델이 이미 잘 맞힌다. 저자는 이 숫자를 "fine-tuning 은 효과가 크다" 로 포장하지 말고,
**"8장짜리 장난감 데이터셋에서는 파이프라인이 도는지를 확인하는 것이 목적"** 이라고 적어라.
비교를 넣는다면 통제표(같은 데이터·같은 imgsz·같은 seed·다른 것은 fine-tune 여부 하나)를 반드시 붙인다.

지표 접근 경로:
- `train_result.box.map50`, `.box.map` (mAP@50-95), `.box.mp` (precision), `.box.mr` (recall)
- segmentation 은 `seg_train.seg.map50` 이 추가된다 (실측: box.map50 0.76873 / seg.map50 0.73529)
- `train_result.results_dict` 에 `'metrics/mAP50(B)'` 같은 키로도 들어 있다

## 8. 저자가 지켜야 할 결론 요약

1. `BUILD = Path('build')`, 가중치는 `YOLO(str(BUILD / 'yolo11n.pt'))`
2. `train()`/`val()` 에 항상 `project=str(BUILD.resolve() / 'runs')` (절대경로) + `amp=False` + `seed=42`
3. 데이터셋은 zip 을 받아 `data/coco8`, `data/coco8-seg` 에 푼다
4. `data.yaml` 은 직접 쓰되 `names` 를 80개 전부, `path` 는 절대경로
5. 패키지 내장 `coco8.yaml` 을 출력하지 않는다 (이모지)
6. `result.plot()` 은 BGR
7. `epochs=20, imgsz=320, batch=4` 면 10초 안에 끝난다
8. 노트북 실행이 끝난 뒤 과목 루트에 `.pt` / `runs/` 가 남으면 실패로 간주한다
