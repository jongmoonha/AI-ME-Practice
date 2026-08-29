# Spec — Practice12_Object_Detection_and_Segmentation.ipynb

> **상태: 라운드 7 확정 노트북과 동기화됨 (양축 PASS, 치명 0).**
> **이 회차는 강의 슬라이드가 없어 이 문서가 유일한 재생성 근거다** — 노트북을 다시 만들거나 개정할 때
> 여기부터 읽는다.
>
> 갱신 시 원칙: **노트북을 명세에 맞추지 않고, 명세를 확정된 노트북에 맞춘다.**
> 뒤집힌 자리마다 "갱신 이력" 으로 **왜 그렇게 됐는지**를 함께 남겼다 — 이유 없는 지정은 다음 회차에
> 재량으로 다시 뒤집히기 때문이다.
>
> **라운드 5~7 개정 요약 (사용자 요청 3건 + 리더 지시 2건):**
>
> | # | 무엇이 | 왜 |
> |---|---|---|
> | 1 | **Step 4 "How Detection Is Scored" 3셀 삭제**, Step 5~8 → **4~7** 로 당김 | 지표는 **강의 슬라이드에서 다루기로 확정**. Step 번호를 비워 두면 학생이 "셀이 지워졌다" 로 읽는다 (리더 승인) |
> | 2 | 하이퍼파라미터 전면 교체 (§3.6) | 이전 설정은 `optimizer=auto` 가 lr 1.2e-4 를 골라 **학습이 사실상 일어나지 않았다** |
> | 3 | fine-tuning 효과를 **샘플 이미지로** 보이는 3셀 신설 (셀 22·23·24) | 지표만으로는 무슨 일이 일어났는지 학생에게 보이지 않는다 |
> | 4 | `train()` 반환값이 셀 마지막 표현식이라 `curves_results` 가 통째로 렌더링되던 버그 수정 | 셀 19 말미에 `print(...)` 를 두어 반환값이 표시되지 않게 함 |
> | 5 | 그림 가독성 — `instance_colors`, 라벨 `clip_on`·불투명 배경 (리더 지시) | 박스가 `y=0` 에 닿으면 라벨이 이미지 밖으로 밀려났다 |

## 0. 메타

| 항목 | 내용 |
|------|------|
| 대상 과목 | `D:/Main/00_Research/00_Python/00_Lecture/01_AI-ME_Graduate` |
| 이 문서의 자리 | `md/p12_workspace/01_spec_Practice12.md` (`CLAUDE.md` 레이아웃 표: 작업 문서는 `md/`) |
| 대응 강의 | **없음** (아래 §1 참조) |
| 작업 유형 | 신규 — **완료.** 노트북 35셀 확정 |
| 엔진 | ultralytics YOLO `8.4.120` (실측), torch `2.6.0+cu124`, CUDA 사용 가능 |
| 데이터셋 | `coco8` (detection, 8장), `coco8-seg` (segmentation, 8장) |
| 라이브러리 | `ultralytics`, `roboflow`, `numpy`, `torch`, `matplotlib`, `PIL`, 표준 `pathlib`/`urllib`/`zipfile`/`os` |
| generator | `gen/p12.py` — **보존한다.** `CLAUDE.md` 말미는 검증 후 삭제하라고 하지만, 레이아웃 표·프로필·실제 저장소(`p01.py`~`p12.py` 전부 존재)는 보존이 관행이다. 이 모순은 리더가 사용자 승인 사항으로 올렸다 |
| 실행 검증 | `jupyter nbconvert --to notebook --execute --inplace "Practice12_Object_Detection_and_Segmentation.ipynb"` (프로필 `execution.command`), `KMP_DUPLICATE_LIB_OK=TRUE` |

---

## 1. 강의자료 부재 — 이 명세의 근거 체계

**이 회차에 대응하는 강의자료가 존재하지 않는다.** 통상의 "강의자료 → 명세 추출" 절차가 성립하지 않으므로
근거 체계를 바꿔 작성했다. 확인한 사실:

| 프로필이 선언한 위치 | 실재 여부 | 확인 결과 |
|---|---|---|
| `lecture_notes/*.pdf` | 1개뿐 | `Ch1-ML 1_Linear Regression.pdf` 하나. detection/segmentation 없음 |
| `md/practice_outline_ref.md` | 있음 | 학부 과목 참조본. `Practice 16 — Advanced CNN` 까지이고 detection 항목 없음. 전체 헤더를 훑어 확인함 |
| `md/lectures_and_formulas.md` | 있음 | detection/segmentation/YOLO/Roboflow 키워드 **0건** (grep 확인) |

따라서 이 명세의 근거는 다음 셋으로 한정한다. 본문의 모든 기술적 주장에 이 셋 중 하나를 출처로 붙였다.

- **(a) 사용자 확정 결정사항** — 변경 불가. 아래 §7 에 통제표로 정리
- **(b) `CLAUDE.md` 규약** — 절 이름으로 인용
- **(c) 실측 / 공식 문서** — `md/p12_workspace/00_env_probe.md` (리더 실측), 본 분석가의 재현 실측, `https://docs.ultralytics.com/`

`notation` 항목이 §1 표 형태로 오지 않는 이유도 여기에 있다. 이 회차에는 강의 원문 notation 이 없어
**기호의 권위가 ultralytics API 이름과 COCO 지표 정의**뿐이다. 임의로 $\rho$ 류 과목 기호를 끌어오지 않았다.

> 이 회차는 과목 기호 규약(`rho`, `w`, `E`, `dJ`)이 **적용될 자리가 없다.** 선형모델 학습 루프를 직접
> 작성하지 않고 ultralytics 가 학습을 수행하기 때문이다. `consistency-verifier` 는 `lr-symbol` /
> `theta-symbol` 체크포인트가 이 노트북에서 발화하지 않는 것을 정상으로 판정해야 한다.

---

## 2. 수식 — **노트북에서는 삭제됨. 명세에는 유지한다**

> **갱신 이력 (라운드 5).** 사용자(교수) 요청으로 **Step 4 "How Detection Is Scored" 3셀이 삭제**됐다.
> IoU·precision/recall·AP·mAP 수식과 IoU 손계산·IoU 그림이 노트북에서 **전부 사라졌고**,
> 이 내용은 **강의 슬라이드에서 다루기로 확정**됐다.
> 저자가 연쇄 정리를 전수 확인했다 (`IoU` `iou` `Precision` `Recall` `intersection` `union` `AP =` `mAP =`
> 모두 노트북 전체 **0건**).
>
> **그런데 이 절을 지우지 않는다.** 두 가지가 이 회차 밖에서도 유효하기 때문이다:
>
> | 유지 항목 | 왜 |
> |---|---|
> | §2.3 의 **금지 서술** | mAP 를 "IoU 임계값 평균" 으로 잘못 쓰는 사고가 **실제로 한 번 발생**했다(V-2). 노트북에 정의가 없는 지금은 재발해도 대조할 곳이 없으므로 가드가 **더** 필요하다 |
> | §2.3 의 **지표 접근 표** | `.box.map50` / `.box.map` / `.seg.map50` / `.seg.map` 은 셀 18·20·29 가 지금도 쓰는 API 이며 이번 라운드에도 재검증됐다 |
>
> **§2.1·§2.2·§2.4 는 노트북에 넣지 않는다.** 향후 슬라이드가 제작되면 이 절이 대조 기준이 된다
> (§6 미검증 1번).

### 아래는 노트북에 넣지 않는다 — 슬라이드 대조용 보존분

강의 원문이 없으므로 **COCO / PASCAL VOC 표준 정의**를 쓴다. 출처를 각 항목에 명시했다.

### 2.1 Intersection over Union (IoU)

$$\mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$

- 출처: `https://docs.ultralytics.com/guides/yolo-performance-metrics/` (IoU 정의), 표준 정의
- 두 축정렬 박스 $A=(x_1^A, y_1^A, x_2^A, y_2^A)$, $B$ 에 대한 코드 대응:

```
inter_x1 = max(ax1, bx1);  inter_y1 = max(ay1, by1)
inter_x2 = min(ax2, bx2);  inter_y2 = min(ay2, by2)
inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
union_area = area_a + area_b - inter_area
iou = inter_area / union_area
```

- **교집합이 없을 때 음수 폭이 나오므로 `max(0, ...)` 이 정의의 일부다.** 생략하면 음수 IoU 가 조용히 나온다

### 2.2 Precision / Recall

$$\text{Precision} = \frac{TP}{TP+FP}, \qquad \text{Recall} = \frac{TP}{TP+FN}$$

- detection 에서 $TP$ 의 판정 기준이 **IoU 임계값**이다. 이것이 분류와 갈리는 지점이며 노트북이 드러내야 할 핵심
- 출처: 표준 정의 + `https://docs.ultralytics.com/guides/yolo-performance-metrics/`

### 2.3 Average Precision / mean Average Precision

$$AP = \int_0^1 p(r)\,dr, \qquad mAP = \frac{1}{C}\sum_{c=1}^{C} AP_c$$

**두 수식은 셀 17 에 반드시 들어간다** (§4 셀 17 행). 표만 옮기고 정의를 빠뜨리면 안 된다 —
그 빈자리를 Summary 가 어림짐작으로 채우는 사고가 실제로 발생했다(검증자 V-2).

**두 평균을 혼동하지 마라. 이것이 이 절의 핵심이다.**

| 무엇의 평균인가 | 어디에 붙는가 |
|---|---|
| **$m$ (mean) = 클래스 평균** | `mAP` 라는 이름 자체. IoU 와 무관하다 |
| IoU 임계값 평균 | `@50-95` 라는 꼬리표에만 붙는다 |

공식 문서 원문 (`https://docs.ultralytics.com/guides/yolo-performance-metrics/`, 본 분석가 WebFetch 확인):

> "AP computes the **area under the precision-recall curve**, providing a single value that
> encapsulates the model's precision and recall performance."
> "mAP extends the concept of AP by calculating the **average AP values across multiple object classes**."
> mAP50: "Mean average precision calculated at an intersection over union (IoU) threshold of 0.50."
> mAP50-95: "The average of the mean average precision calculated at varying IoU thresholds, ranging from 0.50 to 0.95."

> **금지 서술:** "mAP averages precision over a range of IoU thresholds" — `m` 을 IoU 평균으로 잘못
> 설명하는 문장이다. `mAP@50` 은 임계값이 **하나**뿐이므로 이 서술이면 자기모순이 된다.

| 지표 | 의미 | 코드 접근 (실측 확인) |
|------|------|---------------------|
| mAP@50 | **클래스 평균** AP, IoU 임계값 0.5 하나에서 | `metrics.box.map50` |
| mAP@50-95 | 그 값을 IoU 0.50:0.05:0.95 열 개 임계값에 대해 다시 평균 | `metrics.box.map` |
| mAP@75 | IoU 0.75 | `metrics.box.map75` |
| mean precision | 클래스 평균 precision | `metrics.box.mp` |
| mean recall | 클래스 평균 recall | `metrics.box.mr` |

- 출처: `https://docs.ultralytics.com/modes/val/` + **본 분석가 실측**으로 속성 존재 확인
- `metrics.results_dict` 실측 키 (그대로 출력해도 되는 이름):
  `['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'fitness']`
- segmentation 은 `(M)` 접미 키가 추가된다. 실측:
  `['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)', 'fitness']`
  → segmentation 지표는 `metrics.seg.map50`, `metrics.seg.map`

### 2.4 라벨 좌표의 정규화

detection 라벨 한 줄 = `class cx cy w h`, 전부 이미지 크기로 나눈 $[0,1]$ 값.
픽셀 좌표 복원이 곧 수식이며 노트북에서 이 변환을 직접 쓴다:

$$x_1 = (c_x - w/2)\,W, \quad y_1 = (c_y - h/2)\,H, \quad x_2 = (c_x + w/2)\,W, \quad y_2 = (c_y + h/2)\,H$$

- 출처: `https://docs.ultralytics.com/datasets/detect/` — "Normalized xywh format (from 0 to 1)",
  "divide `x_center` and `width` by image width; divide `y_center` and `height` by image height", 클래스는 0-indexed

segmentation 라벨 한 줄 = `class x1 y1 x2 y2 ... xn yn` (polygon, 정규화, 최소 3쌍).

- 출처: `https://docs.ultralytics.com/datasets/segment/` — `<class-index> <x1> <y1> ... <xn> <yn>`,
  "normalized polygon coordinates ... values are in `[0, 1]` relative to image width and height"

---

## 3. 확정된 기술 사실 (실측 근거 포함)

### 3.1 경로 정책 — 루트 오염 4종과 해법

`CLAUDE.md` "디렉터리 레이아웃": 루트에는 `Practice*.ipynb` 와 `CLAUDE.md` 만 둔다.
ultralytics 는 기본값으로 이를 4가지 방식으로 깨뜨린다. **전부 실측 확인됨.**

| # | 오염 | 해법 | 근거 |
|---|------|------|------|
| 1 | `YOLO('yolo11n.pt')` 가 **cwd** 에 `.pt` 다운로드 | `YOLO(str(BUILD / 'yolo11n.pt'))` — 경로를 주면 그 자리에 받는다 | 리더 실측 + 본 분석가 실측 (넘긴 경로에 정확히 떨어짐 — 확정 자리는 `build/yolo11n.pt`, `build/yolo11n-seg.pt`). 소스 확인: `attempt_download_asset` 이 `safe_download(file=file)` 로 넘기고 `f.parent.mkdir(parents=True)` 함 |
| 2 | `project` 상대경로 → `runs/detect/build/runs/...` 중첩, 루트에 `runs/` 생성 | **`project` 는 반드시 절대경로** | `cfg/__init__.py: get_save_dir` 소스 확인 — `if not Path(project).is_absolute(): project = Path(SETTINGS['runs_dir']) / args.task / project`. 절대경로면 `project / name` 그대로 |
| 3 | 학습 중 AMP 체크가 **`yolo26n.pt` 를 루트에 다운로드** | **`amp=False`** | `utils/checks.py: check_amp` 소스에 `YOLO("yolo26n.pt")` (bare name). 본 분석가 실측: 깨끗한 cwd → `amp=True` 학습 후 `['build', 'yolo26n.pt']`. `amp=False` 로는 `ROOT EXTRAS: []` |
| 4 | 기본 `datasets_dir` 이 **다른 과목 폴더**를 가리킴 (`D:\...\01_AI-ME_Graduate\AI-ME-Practice\datasets`) | zip 을 직접 받아 `data/` 에 푼다 | 리더 실측 + 본 분석가가 전역 설정 파일에서 확인 |

**`settings.update()` 를 쓰지 마라.** 리더 실측은 "그 프로세스에 반영되지 않았다" 였고,
본 분석가는 그 **원인까지 확인**했다 — `settings.update()` 는 사용자 홈의 전역 설정 파일
(`%APPDATA%\Ultralytics\settings.json`)을 **영구히 덮어쓰고**, 값은 import 시점에 읽히므로
같은 커널에서는 반영되지 않고 **다음 프로세스부터** 반영된다. 즉 두 가지가 동시에 나쁘다.

> 학생 노트북이 학생 PC 의 전역 ultralytics 설정을 조용히 바꾸게 된다. **금지.**
> (본 분석가가 조사 중 실제로 이 파일을 오염시켰고 원래 값으로 복구했다.)

확정 경로:

| 대상 | 자리 |
|------|------|
| 사전학습 `.pt` | `build/yolo11n.pt`, `build/yolo11n-seg.pt` |
| 데이터셋 | `data/coco8/`, `data/coco8-seg/` (`CLAUDE.md` 레이아웃 표: `data/` = 이 과목이 쓰는 데이터셋) |
| 우리가 쓴 `data.yaml` | `data/coco8.yaml`, `data/coco8-seg.yaml` |
| 학습 산출물 | `build/runs/` (`project=str((BUILD / 'runs').resolve())`) |
| zip 임시파일 | `build/` |

`.gitignore` 에 `build/` 가 이미 있다 (확인함). `data/` 는 무시되지 않으므로 데이터셋이 커밋 대상이
되는지 리더 판단 필요 → §6.

### 3.2 치명적 함정 — `predict()` 이후 같은 객체로 `train()` 하면 지표가 0 이 된다

**본 분석가가 발견하고 통제 실험으로 격리했다. 리더 프로브에 없는 항목이며, 이 노트북의 자연스러운
셀 순서(사전학습 추론 → fine-tuning)가 정확히 이 함정을 밟는다.**

`predict()` 는 추론 속도를 위해 모델을 **fuse** 한다 (Conv+BatchNorm 융합). 실측: `predict()` 직후
`model.model.is_fused()` 가 `True`. 융합된 모델로 학습하면 BatchNorm 이 사라진 상태로 갱신되어
**예외 없이 학습이 끝나고 지표만 0 이 된다.**

통제 실험 (`data`·`imgsz=320`·`epochs=20`·`batch=4`·`seed=42`·`amp=False` 전부 동일, 다른 것은 predict 선행 여부 하나).
**아래 숫자는 이 함정을 격리하려고 따로 돌린 실험의 값이며 §3.7 의 노트북 실측값과는 별개다**
(당시 하이퍼파라미터 기준). 격리 대상이 optimizer·epoch 와 무관하므로 **결론은 그대로 유효하다**:

| 조건 | detection mAP@50 | segmentation mAP@50 |
|------|------------------|---------------------|
| 새 `YOLO()` 객체로 바로 학습 | **0.68777** | **0.8254** |
| `predict()` 먼저 한 객체로 학습 | **0.00063** | **0.0000** |

epochs 를 10 → 30 → 60 으로 늘려도 회복되지 않았다 (0.0000 고정). epochs 문제가 아니다.

> **하드 규칙: `train()` 은 반드시 그 자리에서 새로 만든 `YOLO(...)` 인스턴스로 시작한다.**
> 추론용 객체와 학습용 객체의 **이름을 분리**해 저자가 실수로 재사용하지 못하게 한다.
> 이 명세의 변수명 규칙(§5)이 그것을 강제한다.

참고로 detection clean 값 `0.68777` 은 리더 실측 `0.68790` 과 일치한다 — 경로 전략 전체가 재현되었다.

### 3.3 `data.yaml` — `names` 는 0..N-1 연속이어야 한다

라벨에 등장하는 클래스만 sparse 하게 적으면 학습이 시작조차 못 한다. 리더 실측 에러:

```
KeyError: '4-class dataset requires class indices 0-3,
           but you have invalid class indices 0-50 defined in your dataset YAML.'
```

coco8 라벨은 COCO 80클래스 인덱스(45=bowl, 49=orange, 50=broccoli)를 쓰므로 **80개 전부** 필요하다.
80줄을 손으로 적지 말고 사전학습 모델의 이름표에서 생성한다 — `YOLO(...).names` 는 `{0: 'person', ...}`
dict 이고 `len == 80` (본 분석가 실측 확인, `names[0] == 'person'`).

**`data.yaml` 키:** `path`(루트, 절대경로·POSIX 슬래시), `train`/`val`(`path` 기준 상대), `names`(dict).
출처: `https://docs.ultralytics.com/datasets/detect/`.

### 3.4 패키지 내장 `coco8.yaml` 을 노트북에서 출력하지 마라

헤더에 `🚀` 이모지와 `←`, `└`, `├` 박스문자가 있다 (리더 실측: `['←', '─', '└', '├', '🚀']`).
출력하는 순간 `CLAUDE.md` "No Emojis / Icons" 위반이며 감사 체크포인트 `emoji` 가 발화한다.
**`data.yaml` 은 우리가 직접 쓰고, 우리가 쓴 파일만 출력한다.**

### 3.5 추론 결과 객체 — 실측 속성

| 속성 | 실측 | 비고 |
|------|------|------|
| `result.boxes.xyxy` | `(N, 4)` float32, 픽셀 | 그리기용 |
| `result.boxes.xywhn` | `(N, 4)` 정규화 | **라벨 txt 와 같은 형식** — §2.4 와 연결 |
| `result.boxes.conf` | `(N,)` | |
| `result.boxes.cls` | `(N,)` float | `int(...)` 변환 후 `result.names` 조회 |
| `result.names` | dict, len 80 | |
| `result.orig_img` | `(H, W, 3)` uint8 | |
| `result.orig_shape` | `(H, W)` | |
| `result.plot()` | `(H, W, 3)` uint8 | **BGR**. matplotlib 에 넣으려면 `[:, :, ::-1]` |
| `result.masks.data` | `(N, h, w)` uint8 {0,1} | **모델 입력 해상도이지 원본 크기가 아니다** |
| `result.masks.xy[i]` | `(P, 2)` | **원본 픽셀 좌표계 polygon — 원본 위에 겹쳐 그릴 때 이것을 쓴다** |
| `result.masks.xyn[i]` | `(P, 2)` | 정규화 polygon — 라벨 포맷과 같은 형식 |

- `plot()` 이 BGR 인 것은 소스 docstring 으로 확인: "Annotated image as a NumPy array (BGR)"
- 출처: `https://docs.ultralytics.com/modes/predict/` + 양측 실측

#### 3.5.1 `masks.data` 는 원본 크기가 아니다 — 단, coco8 으로는 그것을 증명할 수 없다

**Step 6 의 이미지 선택을 좌우하는 항목이다. 검증자 V-1 의 근본 원인이므로 여기 못박는다.**

아래 표는 **본 분석가가 `yolo11n-seg.pt`, `conf=0.25` 로 두 이미지를 직접 돌려 측정한 값**이다
(리더 실측과 일치):

| 이미지 | 원본 (W, H) | `orig_shape` (H, W) | `masks.data` (N, h, w) | 같은가 | 본문 서술을 |
|---|---|---|---|---|---|
| coco8 `000000000009.jpg` | 640 x 480 | `(480, 640)` | `(5, 480, 640)` | **True** | **반박한다** |
| `data/sample.jpg` | 2070 x 1380 | `(1380, 2070)` | `(7, 448, 640)` | **False** | **증명한다** |

coco8-seg 8장의 실측 크기: `640x480, 640x426, 640x428, 640x425, 481x640, 640x478, 381x500, 640x488`
— **대부분 가로가 640** 이다. `imgsz=640` letterbox 를 거치면 `masks.data` 의 폭이 640 이 되어
원본과 우연히 일치하는 경우가 생긴다. 즉 coco8 은 "`masks.data` 는 이미지 크기가 아니다" 라는
**문장을 증명할 수 없는 유일한 종류의 이미지**다.

> **규칙: Step 6 은 대형 이미지(`data/sample.jpg`)를 쓴다.** 노트북 본문이 주장하는 것을 그 셀의
> 출력이 반박하면, 학생은 본문을 믿지 않게 된다. 원본 위에 겹쳐 그릴 때 `masks.xy` 를 쓰는 이유도 이것이다.

### 3.6 하이퍼파라미터와 학습 시간 — **전면 교체됨**

> **갱신 이력 (라운드 5~6). 이전 판은 `epochs=20` 에 optimizer 를 지정하지 않았고, 그것이 틀렸다.**
> `optimizer` 를 비워 두면 ultralytics 기본값 `optimizer='auto'` 가 **`lr0` 를 무시하고 자기가 고른다** —
> 실측 로그가 `AdamW(lr=0.000119)` 였고, 그 학습률로는 **학습이 사실상 일어나지 않았다.**
> 이전 판 §3.7 이 "가중치가 거의 움직이지 않았다" 고 적은 것이 바로 그 결과였다.

**확정 설정 — detection·segmentation 이 동일하다:**

```
finetune_epochs = 120        # 셀 19 에서 한 번만 정의
optimizer='SGD', lr0=0.01, imgsz=320, batch=4,
seed=42, amp=False, workers=0, patience=0
```

| 인자 | 왜 이 값인가 |
|---|---|
| `optimizer='SGD'` | **명시하지 않으면 `auto` 가 `lr0` 를 무시한다.** 실측 확인: 지정 후 로그가 `SGD(lr=0.01, momentum=0.937)` 로 바뀌었다 |
| `lr0=0.01` | `auto` 가 고르던 `1.2e-4` 로는 학습이 일어나지 않는다 |
| `patience=0` | **아래 별도 항목** |
| `amp=False` | AMP 체크가 `yolo26n.pt` 를 루트에 받는다 (§3.1 #3) |
| `workers=0` | Windows 노트북 커널에서 dataloader worker 스폰이 불안정하다 |
| `imgsz=320`, `batch=4` | 시간 상한(아래) |

#### `patience=0` 은 필수다 — 없으면 라벨이 조용히 거짓이 된다

ultralytics 기본값은 `patience=100` 이고, 이 값이면 `EarlyStopping` 이 발동해 **실제로는 101 epoch 만
돈다.** 그러면 노트북이 8곳에서 단정하는 `120 epochs` 가 **같은 셀의 로그(`101 epochs completed`)와
모순**된다. 라운드 5 에서 실제로 발생해 감사·정합성 양축에서 치명/BLOCK 으로 잡혔다.

- **기전 (검증자가 소스로 확인):** `utils/torch_utils.py` 의 `patience or float("inf")` —
  전용 분기가 아니라 **falsy 치환**이다. 따라서 `0` 이 "무한" 을 뜻한다
- **`best.pt` 선택은 별개 경로라 영향받지 않는다.** 기전이 예측한 대로 `best.pt` 행이 한 자리도
  움직이지 않은 것이 관측으로 확인됐다

> **`patience=0` 을 "기본값이니 생략해도 된다" 로 보고 지우지 마라.** 지우면 에러가 나지 않고
> **라벨만 거짓이 된다.** 로그를 끝까지 읽지 않으면 드러나지 않는 종류다.

#### `finetune_epochs` 단일 정의 — 학습 인자와 출력 라벨을 함께 묶는다

`finetune_epochs = 120` 을 **셀 19 에서 한 번만** 정의하고 **8곳이 그것을 읽는다** —
`train()` 인자(셀 19·29)와 출력 라벨·그림 제목(셀 20·22·23) 전부.

`CLAUDE.md` "비교 실험 통제 변수" 의 *"공통 조건을 위에서 한 번만 정의하고 모두가 재사용한다"* 를
**학습 인자를 넘어 출력 라벨까지 확장 적용**한 것이다. detection 과 segmentation 이 같은 값을 쓰므로
정의는 하나뿐이다.

> **"중복 같다" 며 리터럴 `120` 으로 되돌리지 마라.** 학생이 epoch 수를 바꿔 실험할 때
> **숫자는 바뀌고 라벨은 그대로 남아** 그림과 표가 조용히 거짓말을 한다.

#### 학습 시간 (라운드 7 실측, GPU: NVIDIA TITAN RTX)

| 작업 | 실측 |
|------|------|
| detection fine-tune (120 epoch 전량) | **0.010 h ≈ 36초** |
| segmentation fine-tune (120 epoch 전량) | **0.011 h ≈ 40초** |
| `val()` 1회 | 1초 미만 |
| **노트북 전체 35셀** | **90.4초** |

프로필 `cell_time_warning_sec: 300` 대비 **GPU 에서는 여유가 있다.**
다만 `epochs` 가 20 → 120 으로 **6배**가 됐으므로 **CPU 런타임에서는 재확인이 필요하다** (§6 미검증 4번).
`imgsz=320` 을 올리지 마라.

### 3.7 이 회차의 결론 — **"외웠다"** 이지 "좋다/나쁘다" 가 아니다

> **갱신 이력 (라운드 5~7). 이전 판의 결론 "개선폭이 작다" 는 폐기됐다.**
> 그 결론은 §3.6 대로 **학습이 사실상 일어나지 않은 설정**에서 나온 것이라, 관찰이 아니라 설정의 부작용이었다.
> `optimizer='SGD', lr0=0.01, epochs=120` 으로 바꾸자 **실제로 학습이 일어났고 결론이 완전히 달라졌다.**

**라운드 7 실측 — 셀 20 출력:**

| arm | 가중치 | mAP@50 | mAP@50-95 |
|---|---|---|---|
| A pretrained | COCO 사전학습, 손대지 않음 | 0.68437 | 0.39986 |
| B after 120 epochs | `last.pt` — 선택 없음 | **0.36998** | **0.23863** |
| B best checkpoint | `best.pt` — val 에서 고른 것 | 0.68769 | 0.41834 |

**`last.pt` 가 사전학습의 절반 수준으로 떨어진다.** 이전 판(0.63026)보다 훨씬 크게 무너졌고,
이것이 학습이 실제로 일어났다는 증거다.

segmentation 도 같다 — 셀 29 출력:

| arm | box mAP@50 | box mAP@50-95 | mask mAP@50 | mask mAP@50-95 |
|---|---|---|---|---|
| after 120 epochs (`last.pt`) | **0.32036** | **0.22290** | **0.25711** | **0.16023** |
| best checkpoint (`best.pt`) | 0.76877 | 0.48625 | 0.73535 | 0.40272 |

#### 지표만으로는 무슨 일이 일어났는지 안 보인다 — 샘플 이미지가 답한다

사용자 요청으로 신설된 셀 22 의 실측 출력이다. **이것이 이 회차의 핵심 증거다:**

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

| 이미지 | 무슨 일이 | 
|---|---|
| **학습한 이미지** | 사전학습이 **하나도 못 찾던 `orange` 를 6개** 찾아낸다 |
| **처음 보는 이미지** | 사전학습이 찾던 것을 **전부 잃고 `zebra` 하나**만 남는다 |

**이것이 overfitting 이고, `last.pt` 행이 그 숫자다.** 4장은 외울 수만 있다.

#### 서술 규칙 — 확정

**확정 해설, 셀 24 전문 (4문장). 기준이며 임의로 줄이지 마라:**

> The training image gains the oranges the pretrained weights were missing.
> The held-out image loses almost everything those same pretrained weights found.
> That is overfitting, and the `last.pt` row above is its number.
>
> Eight photographs are what this figure is about, not what fine-tuning does in general.

> **금지: "fine-tuning 이 좋다" 도, "fine-tuning 이 나쁘다" 도 쓰지 마라.**
> 이것은 **8장짜리 데이터셋의 이야기이지 일반론이 아니다.** 마지막 문장이 그 선을 긋는 자리이므로
> 분량을 이유로 지우지 마라. 이전 판의 "파이프라인이 도는지 확인이 목적" 도 이제는 쓰지 않는다 —
> 학습이 실제로 일어났으므로 그 서술은 더 이상 사실이 아니다.

`train()` 이 `best.pt` 를 로드한 채 끝나므로 `last.pt` 는 **별도 인스턴스로 열어야** 한다 (셀 20·29).

### 3.8 Roboflow `model_format` 은 검증되지 않는다 — `'yolov8'` 로 확정

**리더 판정. 본 분석가가 설치본 SDK 소스로 재확인했다** (`roboflow 1.4.0`, `core/version.py`):

```python
friendly_formats = {"yolov5": "yolov5pytorch", "yolov7": "yolov7pytorch"}
return friendly_formats.get(format, format)
```

`__get_format_identifier` 는 이 두 개만 매핑하고 **나머지는 검증 없이 그대로 API 로 넘긴다.**
패키지가 아는 식별자에 `yolov11` 은 없다.

> 위험의 성질이 §3.2 와 같다 — **잘못된 문자열이 우리 검증(API 키 없음 → 가드로 건너뜀)은 통과하고,
> 키를 가진 학생 환경에서만 터진다.** 지역 검증으로는 절대 걸리지 않는 실패다.
> `model_format='yolov8'` 로 확정한다. YOLO 라벨 포맷은 v8/v11 이 동일하므로 export 결과에 문제가 없다.

### 3.9 학습 API 가 평가 split 에서 체크포인트를 고르는 것은 **통제되지 않은 차이**다

**§3.2 fuse 함정과 같은 급의 항목이다. 로컬 검증을 전부 통과하면서 결론만 뒤집기 때문이다.**
(검증자 BLOCK 1 의 근본 원인. 초판 §4.2 가 "격리 변수는 fine-tuning 여부 하나뿐" 이라고 단언한 것이
**정확히 반대**였다.)

ultralytics `train()` 은 매 epoch `val` split 으로 평가해 **가장 좋았던 epoch 을 `best.pt` 로 저장하고,
학습이 끝나면 그 가중치를 로드한 채 반환한다.** 그런데 이 노트북이 보고하는 지표도 **같은 `images/val`**
에서 나온다. 따라서 A vs B 비교에는 격리하려던 변수 말고 **두 번째 차이**가 섞인다:

| | A pretrained | B fine-tuned |
|---|---|---|
| 격리하려던 변수 | 학습 안 함 | coco8 로 20 epoch |
| **의도치 않은 두 번째 차이** | **선택 없음** | **20개 중 val 에서 최고를 고름** |

- `best.pt` 점수는 **성능 향상이 아니라 선택 결과(selection result)** 다. 보고 대상 split 에서 고른
  값이므로 낙관적으로 편향된다 — `CLAUDE.md` "데이터 분할과 스케일링" 이 경계하는 누수와 같은 성질이다
- **완전한 통제는 이 API 로 불가능하다.** `train()` 이 자동으로 하는 일이라 끄는 인자가 없다
- 따라서 이 과목의 대응은 **숨기지 말고 드러내는 것**이다:
  1. `last.pt` 를 **따로 열어** 세 번째 arm 으로 함께 보고한다 (§3.7)
  2. 통제표에 `checkpoint evaluated` 행을 넣고 **"The second row is not controlled."** 를 명시한다 (§4.2)
  3. 해설에서 `best.pt` 행을 "selection result, not a performance gain" 으로 부른다 (셀 24)

> **다음 개정자에게:** `last_epoch_model` / `segmentation_last_model` 셀을 "중복" 으로 보고 지우지 마라.
> 그 줄이 사라지면 통제표의 두 번째 행이 근거를 잃고, 노트북은 다시 "fine-tuning 이 올렸다" 로 읽힌다.
> 지우고 싶어지는 동기는 "성능 향상 오독" 이 아니라 **"똑같은 `val` 을 두 번 부르네" 라는 중복 제거
> 충동**이다. 그 충동이 오면 이 문단으로 돌아와라.

#### `best.pt` 를 "학습 성과" 로 서술하지 마라 (검증 V-12)

라운드 7 설정에서 `best.pt` 행(0.68769)이 사전학습(0.68437)과 거의 같은 것은 우연이 아니다.
**학습이 출발점을 한 번도 넘지 못했고, `best.pt` 는 초기 epoch 동률 구간의 tie-break** 이다.
`patience=0` 이 `best.pt` 선택 경로에 영향을 주지 않는다는 것도 기전과 관측 양쪽으로 확인됐다 (§3.6).

- **`best` 행을 "120 epoch 학습의 결과" 로 부르면 거짓**이 된다. 실제로는 학습 거의 직후의 가중치다
- **단, 이 내용을 노트북에 쓰지 않는다.** epoch 별 로그로만 특정할 수 있는데 그 근거 줄이 이번
  수정으로 출력에서 사라져 **노트북 출력만으로는 독자가 확인할 수 없다.** 확인 불가능한 주장을
  본문에 넣지 않는 것이 이 과목의 규율이라 **리더가 미반영으로 확정**했다 (§6)
- 명세에는 남긴다 — 다음 개정자가 `best` 행에 성과 서술을 붙이려 할 때 막는 자리다

---

## 4. 섹션 구성 — 셀 단위 명세

표기: `MD` = 마크다운 셀, `CODE` = 코드 셀.
분량 상한(`CLAUDE.md` "설명 분량"): 첫 셀 150단어 / Summary 100단어 / 그 외 120단어(하드),
표·헤더·수식 제외 **산문 80단어 하드**, 한 문단 3문장, **한 줄에 한 문장**.

house style: 각 Step 의 첫 MD 셀은 `---` 로 시작한 뒤 `## Step N. Title` (P09/P10/P11 공통 확인).

| # | 유형 | 목적 | 담을 내용 | 주요 변수명 | 예상 출력 |
|---|------|------|-----------|------------|----------|
| 00 | MD | 제목 + 개요 | `# Practice 12 — Object Detection and Segmentation`. 세 과제의 **출력 형태** 표(아래 §4.1). 데이터셋 한 줄. 150단어 이내 | — | — |
| 01 | MD | `---`+`## Step 0. Environment and Setup` | 이 셀은 헤더만. 산문 2문장 이내 | — | — |
| 02 | CODE | **Colab 대응 환경 체크** | `try: import ultralytics, roboflow` / `except ImportError:` → `subprocess` 로 `pip install ultralytics roboflow`. 설치 후 재import | — | `ultralytics 8.4.120` 류 버전 출력 |
| 03 | CODE | 임포트·시드·경로 | `numpy`/`torch`/`matplotlib`/`PIL`/`pathlib`/`urllib`/`zipfile`, `from ultralytics import YOLO`. `np.random.seed(42)`, `torch.manual_seed(42)`. `BUILD`, `DATA`, `RUNS` 정의 + `mkdir`. device 출력 | `BUILD`, `DATA`, `RUNS` | `device: cuda`, 경로 3줄 |
| 04 | MD | `## Step 1. Three Tasks, Three Output Shapes` | classification vs detection vs instance segmentation 을 **출력의 형태**로 구분하는 표(§4.1). 산문 3문장 이내 | — | — |
| 05 | MD | `## Step 2. The YOLO Dataset Format` | 폴더 레이아웃 + `data.yaml` 키 표. 산문 3문장 | — | — |
| 06 | CODE | 데이터셋 2종 다운로드·해제 | `coco8.zip`, `coco8-seg.zip` 를 `build/` 로 받아 `data/` 에 해제. 이미 있으면 건너뜀 | `detection_data_dir`, `segmentation_data_dir` | `coco8: 4 train / 4 val images` 류 |
| 07 | CODE | **폴더 트리 실물 출력** | `images/train`, `images/val`, `labels/train`, `labels/val` 를 순회하며 파일 수와 파일명 몇 개 출력. **ASCII 만 사용** (박스문자 금지) | — | 디렉터리별 파일 목록 |
| 08 | MD | 라벨 포맷 설명 | detection `class cx cy w h` 정규화, segmentation polygon. §2.4 수식. 표로 | — | — |
| 09 | CODE | **라벨 txt 실물 출력** | detection 라벨 1개 전체 출력 + segmentation 같은 이미지의 라벨 첫 줄 토큰 수 출력(클래스 1 + 좌표 2n) | `detection_label_path`, `segmentation_label_path` | `45 0.479492 0.688771 ...` / `tokens=25 -> 12 polygon vertices` |
| 10 | CODE | `data.yaml` 생성 | `pretrained_model.names` 로 80개 이름 생성(§3.3), `path`(절대·POSIX)/`train`/`val`/`names` 기록. **두 개 생성** (det/seg). **yaml 경로에 `.resolve()` 를 붙인다** — 작업 디렉터리가 어디든 파일을 찾게 하기 위해서이고, `path:` 값도 `.resolve().as_posix()` 다 (§3.1 경로 정책) | `pretrained_model`, `class_names`, `detection_yaml_path`, `segmentation_yaml_path` | `80 names, 0 = person, 45 = bowl, 79 = toothbrush` + 우리가 쓴 yaml 앞 12줄 |
| 11 | MD | 정규화 좌표가 가리키는 것 | "라벨을 이미지 위에 직접 그려 확인한다" 1~2문장 | — | — |
| 12 | CODE | **라벨을 이미지 위에 그림** | §2.4 식으로 픽셀 복원 후 `matplotlib.patches.Rectangle`. 왼쪽 detection 박스, 오른쪽 segmentation polygon(`plt.Polygon`). `plt.subplots(1, 2)` | `image_width`, `image_height`, `class_index`, `center_x`, `center_y`, `box_width`, `box_height` | 1×2 그림, 박스/폴리곤이 물체에 얹힘 |
| 13 | MD | `## Step 3. Pretrained Detection Inference` | COCO 사전학습 `yolo11n.pt`. 산문 3문장 | — | — |
| 14 | CODE | 모델 로드 + 추론 | `detection_model = YOLO(str(BUILD / 'yolo11n.pt'))` 다음 줄에서 `.predict(...)`. `source` 는 **로컬 coco8 이미지** (URL 금지 — §6 참조), `conf=0.25`, `verbose=False` | `detection_model`, `detection_result` | 클래스·conf 목록 |
| 15 | CODE | `Results` 를 풀어서 출력 | `boxes.xyxy`/`xywhn`/`conf`/`cls` shape 출력 후 검출별 `name conf xyxy` 한 줄씩 `print` | `detected_boxes` | shape 4줄 + 검출 N줄 |
| 16 | CODE | 박스를 **직접 그림** + **`instance_colors` 정의** | `plot()` 쓰지 않고 `sample_image` 위에 `Rectangle` + `ax.text`. 왼쪽 원본, 오른쪽 검출 결과. **`instance_colors` 를 여기서 한 번 정의**해 셀 23·27 이 재사용한다. 라벨은 **`clip_on=True` + `va='top'` + 박스와 같은 색 불투명 배경**(§4.4). **이 셀은 confidence 를 표시한다** | `instance_colors`, `instance_color`, `sample_image` | 1×2 그림 |
| **— Step 4 "How Detection Is Scored" 3셀은 삭제됨 (라운드 5). §2 갱신 이력 참조 —** | | | | | |
| 17 | MD | `## Step 4. Detection Fine-tuning` | 산문 1문장 + 통제표(§4.2) **6행** + 뒤에 **3문장**: `The second row is not controlled.` / `coco8` 에 test split 이 없어 같은 4장이 B 의 체크포인트를 고르고 채점한다 / 마지막 epoch 행만 그 선택을 벗어난다 | — | — |
| 18 | CODE | **baseline `val()`** | 학습 전 사전학습 그대로의 지표. **새 인스턴스** `baseline_model` | `baseline_model`, `baseline_metrics` | mAP@50 `0.68437` / mAP@50-95 `0.39986` / precision / recall |
| 19 | CODE | **`finetune_epochs` 정의 + detection fine-tune** | **맨 위에서 `finetune_epochs = 120` 을 정의**(§3.6). **새 인스턴스** `finetune_model` → 다음 줄 `.train(...)`. 인자는 §3.6 확정값(`optimizer='SGD'`, `lr0=0.01`, `patience=0` 포함). **말미에 `print(...)` 를 두어** `train()` 반환값이 셀 마지막 표현식이 되지 않게 한다 — 없으면 `curves_results` 가 통째로 렌더링된다 | `finetune_epochs`, `finetune_model` | 학습 로그 + `checkpoints written to ...` |
| 20 | CODE | **3-arm `val()` + 나란히 출력** | `last.pt` 를 **별도 인스턴스로 열어** 평가하고(§3.9), `finetune_model` 로 `best.pt` 를 평가. baseline 과 함께 **`print` 로 3줄** (`assert`/`allclose` 금지). 라벨은 `last_epoch_label = f'B after {finetune_epochs} epochs'` | `last_epoch_model`, `last_epoch_metrics`, `finetune_metrics`, `last_epoch_label` | **3줄**: `0.68437` / `0.36998` / `0.68769` (§3.7) |
| 21 | MD | `last.pt` vs `best.pt` 설명 + 다음 셀 예고 | `last.pt` 는 선택 없이 채점되며 **사전학습 mAP@50 의 약 절반**을 유지한다는 것. 이어지는 두 셀이 **학습한 사진 1장과 처음 보는 사진 1장**에 두 가중치를 돌린다는 예고 | — | — |
| 22 | CODE | **신규 — 샘플 이미지 검출 결과를 텍스트로** | held-out 이미지(`val/000000000049.jpg`)를 열고, 두 이미지 × (ground truth / pretrained / after N epochs) 를 `print`. **`last.pt` 로 추론한다** — `best.pt` 는 val 에서 골라진 것이라 이 비교에 넣으면 논점이 흐려진다 | `heldout_image_path`, `heldout_label_path`, `heldout_image`, `pretrained_heldout_result`, `last_epoch_result`, `last_epoch_heldout_result`, `image_rows`, `finetuned_label` | §3.7 의 6줄 블록 |
| 23 | CODE | **신규 — 4패널 그림** | `plt.subplots(2, 2)`. 윗줄 training 이미지, 아랫줄 held-out. 왼쪽 pretrained / 오른쪽 `after N epochs`. `instance_colors` 재사용, 라벨 `clip_on=True`. **이 셀은 confidence 를 뺀다** — 4패널은 밀도가 높다(§4.4) | `panels`, `panel_axis`, `panel_image`, `panel_result`, `panel_title` | 2×2 그림. 오른쪽 아래 패널은 박스 1개(`zebra`) |
| 24 | MD | **신규 — 그림 해설** | §3.7 의 **확정 4문장 전문**. training 이미지는 `orange` 를 얻고, held-out 은 사전학습이 찾던 것을 잃으며, **그것이 overfitting 이고 `last.pt` 행이 그 숫자**라는 것. 마지막 문장이 "8장의 이야기이지 일반론이 아니다" 로 선을 긋는다 | — | — |
| 25 | MD | `## Step 5. Instance Segmentation` | box 와 mask 의 차이. 산문 3문장 | — | — |
| 26 | CODE | seg 추론 | `segmentation_model = YOLO(str(BUILD / 'yolo11n-seg.pt'))` → `.predict(...)`. **source 는 반드시 `data/sample.jpg`** (2070x1380). **이유(§3.5.1) — coco8 이미지로 바꾸지 마라:** 이 셀은 `masks.data` 가 원본 크기가 아님을 출력으로 증명해야 한다. coco8 은 대부분 가로 640 이라 `imgsz=640` 에서 `masks.data` 가 `orig_shape` 와 **우연히 같은 숫자**가 되어 본문을 반박한다. `masks.data`/`masks.xy`/`orig_shape` 를 나란히 출력 | `segmentation_model`, `segmentation_result` | shape 및 인스턴스 수 |
| 27 | CODE | **box vs mask 나란히** | 왼쪽 detection 박스, 오른쪽 `masks.xy` polygon 채색. `plt.subplots(1, 2)`. **`masks.data` 대신 `masks.xy` 사용** (§3.5). `instance_colors` 재사용 | `mask_polygon`, `mask_color` | 1×2 그림 |
| 28 | MD | `## Step 6. Segmentation Fine-tuning` | 산문 **3문장**. 학습 호출은 detection 과 같고 **바뀌는 것은 출력**(박스 계열 + 마스크 계열 두 벌)이라는 점. **"같은 체크포인트 선택이 여기서도 일어나므로 last 와 best 를 모두 보인다"** 를 명시 (§3.9) | — | — |
| 29 | CODE | seg fine-tune + **2-arm val** | **새 인스턴스** `segmentation_finetune_model` (§3.2). `epochs=finetune_epochs` 와 `patience=0` 을 **detection 과 동일하게** 넘긴다(§3.6). `name='segment_finetune'`. 이어서 `last.pt` 를 **별도 인스턴스로 열어** 평가 | `segmentation_finetune_model`, `segmentation_last_model`, `segmentation_last_metrics`, `segmentation_metrics`, `segmentation_last_label` | **4열 2행 표**: `after 120 epochs` `0.32036/0.22290/0.25711/0.16023` / `best checkpoint` `0.76877/0.48625/0.73535/0.40272` |
| 30 | MD | `## Step 7. Building Your Own Dataset` | Roboflow YOLO export 와 Step 2 의 **대응과 차이**를 표로 (§4.3). "포맷은 같고 폴더 배치는 다르다, `data.yaml` 이 그 차이를 흡수한다" 가 교육 포인트 | — | — |
| 31 | CODE | **Roboflow (API 키 가드)** | `api_key = os.environ.get('ROBOFLOW_API_KEY')` → 없으면 안내 `print` 후 건너뜀. 있으면 `Roboflow(api_key=...)` → `.workspace()` → `.project()` → `.version(n)` → `.download('yolov8', location=str(DATA / 'roboflow'))`. **`'yolov8'` 확정 — §3.8 참조.** **`nbconvert` 가 통과해야 하므로 예외를 던지지 않는다** | `roboflow_api_key` | 키 없으면 안내 문구 |
| 32 | MD | `## Summary` | 100단어 이내, 산문 소프트캡 유의. **새 내용 금지.** 불릿 3개: `data.yaml` 의 `names` 가 `0..N-1` / **4장에 맞추고 못 본 이미지에서 점수가 내려갔다** / 높은 쪽은 보고 대상 split 에서 고른 체크포인트라 **마지막 epoch 값이 정직한 숫자**. **지표를 여기서 정의하지 마라** (V-2 발생 지점이었고, 이제 노트북에 정의 자체가 없다) | — | — |
| 33 | MD | `## Exercises` | 3문항. 다른 회차 참조 금지. **IoU 문항은 Step 4 삭제와 함께 제거됐다** — 3번은 segmentation polygon 문항, 2번은 `finetune_epochs` 를 `120 → 30` 으로 바꿔 두 fine-tuned 행의 간격을 보는 문항 | — | — |
| 34 | CODE | `# Write your code here` | 1줄 | — | — |

총 **35 셀** (MD 15 / CODE 20). Step 4 3셀이 빠지고 셀 22·23·24 가 신설되어 **개정 전과 총수가 같다.**
Step 헤더는 `Step 0` ~ `Step 7` **연속이다** — 번호를 비워 두면 학생이 "셀이 지워졌다" 로 읽는다(리더 승인).

### 4.4 그림 규약 — 라운드 5 리더 지시

| 항목 | 확정 |
|---|---|
| 색 | **`instance_colors` 를 셀 16 에서 한 번 정의**하고 셀 23·27 이 재사용. 인스턴스마다 다른 색이라 겹친 박스를 구분할 수 있다. `index % len(...)` 으로 개수 초과를 막는다 |
| 라벨 위치 | `va='top'` + `x1+2, y1+2` — 박스 **안쪽 위**에 둔다. 바깥에 두면 `y=0` 에 닿은 박스에서 이미지 밖으로 밀린다 |
| 라벨 잘림 | **`clip_on=True`** 필수 |
| 라벨 배경 | 박스와 **같은 색 불투명** `bbox=dict(facecolor=instance_color, edgecolor='none', pad=1)` — 사진 위에서 글자가 읽히게 |
| confidence | **셀 16 은 표시하고 셀 23 은 뺀다.** 4패널은 밀도가 높고, 그 그림이 묻는 것은 "무엇을 찾았나" 이지 "얼마나 확신하나" 가 아니다 |

### 4.1 셀 04 의 표 (확정 내용)

| Task | Output per object | Shape |
|:---|:---|:---|
| Classification | one label for the whole image | 1 class index |
| Object detection | a box per object | 4 numbers + class + confidence |
| Instance segmentation | a per-pixel outline per object | polygon of variable length + class + confidence |

- 근거: 사용자 지침 "출력의 형태로 설명" + §2.4 라벨 포맷 차이(고정 4개 vs 가변 polygon)

### 4.2 셀 20 의 통제표 — 필수

`CLAUDE.md` "비교 실험 통제 변수".

> **갱신 이력 — 판정이 뒤집힌 자리다.** 초판은 "격리 변수는 fine-tuning 여부 **하나뿐**" 이라고 단언했다.
> **그것이 틀렸다.** §3.9 대로 `train()` 이 평가 split 에서 `best.pt` 를 고르므로 **두 번째 차이가 있다.**
> `CLAUDE.md` 는 "의도된 다중 차이라면 단정할 수 없다고 솔직히 적으라" 고 한다 — 그 경로를 따른다.

**격리하려는 변수는 fine-tuning 여부이고, 통제되지 않은 두 번째 차이가 하나 있다 (2행).**
최종 노트북 셀 17 의 표 그대로 — **6행이며 순서까지 이것을 따른다:**

| Item | A: pretrained | B: fine-tuned |
|:---|:---|:---|
| **weights evaluated** | **COCO pretrained, untouched** | **+ 120 epochs on coco8 (SGD, lr0=0.01)** |
| **checkpoint** | **no selection** | **best of 120, chosen on `images/val`** |
| dataset | same | same |
| evaluation split | `images/val` | `images/val` |
| model size | `yolo11n` | `yolo11n` |
| `imgsz` | 320 | 320 |

표 바로 뒤에 **세 문장**을 둔다. 첫 문장이 빠지면 표가 "전부 통제됨" 으로 읽혀 초판의 오류가 되돌아온다:

> The second row is not controlled.
> `coco8` has no test split, so the same four photographs choose B's checkpoint and then score it.
> Only the last-epoch row escapes that selection.

- **감사 C-1 은 미반영으로 확정됐다** — "B 열이 `best of 120` 만 적어 셀 20 의 3행 출력을 2팔로 그린다"
  는 지적인데, **뒤따르는 산문이 보완하므로 거짓이 아니고** 셀 17 이 마크다운 예산 116/120 이라 여유가 없다

- **A 와 B 는 각각 별개의 `YOLO()` 인스턴스여야 한다** (§3.2). B 를 A 의 객체로 학습시키면 지표가 0 이 된다
- 2행이 통제 불가라는 사실 때문에 **셀 23 이 `last.pt` arm 을 함께 보고한다** (§3.7). 표와 그 셀은 한 쌍이다

### 4.3 셀 30 의 표 — Roboflow export 대응 (서술 정확성 지시, 리더)

**"Roboflow export 가 Step 2 구조와 똑같다" 고 쓰지 마라. 사실이 아니다.**
포맷은 같지만 **디렉터리 중첩이 다르다.** 이 차이를 덮지 말고 교육 포인트로 쓴다.

| | This notebook (coco8) | Roboflow YOLO export |
|:---|:---|:---|
| label file | `class cx cy w h`, normalized | **same** |
| dataset descriptor | `data.yaml` | **same** |
| folder layout | `images/train`, `labels/train` | `train/images`, `train/labels` |

- 확정 교육 포인트: **폴더 배치가 달라도 `data.yaml` 의 `path`/`train`/`val` 이 그 차이를 흡수한다.**
  학습 코드는 바뀌지 않는다 — 이것이 `data.yaml` 이 존재하는 이유다
- 근거: 리더 판정. 본 분석가는 API 키가 없어 **export 결과를 직접 확인하지 못했다** (§6 참조)

---

## 5. 변수명 규칙 — `CLAUDE.md` "Variable Naming"

약어 금지(`res`, `bx`, `im`, `dt`, `mdl`). 아래를 그대로 쓴다.

**아래는 최종 노트북에서 실제로 쓰인 이름과 대조해 확정한 목록이다.**

| 용도 | 확정 변수명 |
|------|-----------|
| 경로 | `BUILD`, `DATA`, `RUNS` |
| 클래스 이름 확보용 모델 | `pretrained_model`, `class_names` |
| 추론용 detection 모델 | `detection_model`, `detection_result`, `detected_boxes` |
| **baseline 평가용 (새 인스턴스)** | `baseline_model`, `baseline_metrics` |
| **detection 학습용 (새 인스턴스)** | `finetune_model`, `finetune_metrics` |
| **detection `last.pt` 평가용 (또 다른 인스턴스)** | `last_epoch_model`, `last_epoch_metrics` |
| 추론용 segmentation 모델 | `segmentation_model`, `segmentation_result` |
| **segmentation 학습용 (새 인스턴스)** | `segmentation_finetune_model`, `segmentation_metrics` |
| **segmentation `last.pt` 평가용 (또 다른 인스턴스)** | `segmentation_last_model`, `segmentation_last_metrics` |
| 데이터 경로 | `detection_data_dir`, `segmentation_data_dir` |
| yaml 경로 | `detection_yaml_path`, `segmentation_yaml_path` |
| 라벨 파싱 | `detection_label_path`, `segmentation_label_path`, `tokens`, `polygon_tokens` |
| **학습 설정 (단일 정의)** | **`finetune_epochs`** — §3.6, 8곳이 읽는다 |
| **출력 라벨 (파생)** | `last_epoch_label`, `finetuned_label`, `segmentation_last_label` |
| **샘플 이미지 비교 (셀 22·23)** | `heldout_image_path`, `heldout_label_path`, `heldout_image`, `pretrained_heldout_result`, `last_epoch_result`, `last_epoch_heldout_result`, `image_rows`, `panels`, `panel_axis`, `panel_image`, `panel_result`, `panel_title` |
| 이미지 | `sample_image`, `segmentation_image`, `image_width`, `image_height` |
| 시각화 색 | **`instance_colors`** (셀 16 정의, 23·27 재사용), `instance_color`, `mask_color`, `mask_polygon` |
| Roboflow | `roboflow_api_key` |

**Step 4 삭제로 사라진 이름 (노트북 전체 0건, 저자 전수 확인):**
`ground_truth_box`, `ground_truth_class`, `ground_truth_area`, `ground_truth_line`, `predicted_box`,
`class_matches`, `intersection_area`, `union_area`, `iou`, `best_iou`, `best_index`, `iou_threshold`, `verdict`.
**`fill_colors` 는 `instance_colors` 로 이름이 바뀌었다** — 셀 16·23·27 이 공유하는 것이 드러나는 이름이다.

**모델 인스턴스가 여러 개로 나뉘어 있는 것이 §3.2 와 §3.9 에 대한 구조적 방어다.**
- 추론용과 학습용을 합치면 **§3.2** 로 지표가 0 이 된다
- `*_last_model` 을 지우면 **§3.9** 의 통제되지 않은 차이가 다시 감춰진다

`annotated_image` 는 **쓰지 않는다.** 초판 명세에 있었으나 최종 노트북은 중간 변수 없이 `sample_image`
위에 직접 그린다. 검증자가 **정당한 이탈**로 판정했으므로 명세를 노트북에 맞춘다.

---

## 6. 확인 필요 항목 — **미검증 4건 / 의도적 미반영 4건**

아래 7건은 전부 리더가 판정했다. **결론을 여기 기록해 다음 회차가 같은 항목을 다시 올리지 않게 한다**
(`harness-feedback-loop`: 판단 결과가 기록되지 않으면 다음 회차에 같은 항목이 다시 `[근거 없음]` 으로 올라온다).

| # | 항목 | 판정 | 반영 위치 |
|---|------|------|----------|
| 1 | 강의 슬라이드 부재 | **그대로 진행.** 사용자에게 별도 보고하고, **이 명세가 향후 슬라이드 제작 시 대조 기준**이 된다는 점을 함께 전달 | §1 유지 |
| 2 | Summary 셀 | **넣는다.** `CLAUDE.md` "설명 분량" 표가 Summary 를 항목으로 명시하고 상한 100단어까지 정해 두었다. **P11 에 없는 것은 그 회차의 선택이지 규약 변경이 아니다** | 셀 32 유지 |
| 3 | 추론 이미지 | **로컬 coco8 확정** (본 분석가 판단 승인). 단 **Step 6(box vs mask)에 한해 `data/sample.jpg` 허용** — 2070x1380 이라 마스크 형태가 훨씬 잘 읽히고 이미 과목 자산이다 | 셀 26 수정 |
| 4 | `audit_notebook.py` 부재 | **본 분석가의 오인이었다.** 과목 `.claude/scripts/` 가 아니라 **`00_Lecture/.claude/skills/notebook-convention-audit/scripts/audit_notebook.py`** 에 있다 (실재 확인함). 리더가 P11 에 실제로 돌려 정상 동작 확인. **감사자는 기계 검사를 돌릴 수 있다** | 아래 주 |
| 5 | Roboflow `model_format` | **`'yolov8'` 로 확정.** `'yolov11'` 은 패키지가 아는 식별자가 아니다 | §3.8 신설, 셀 31 수정 |
| 6 | `data/` 커밋 여부 | **리더가 처리한다.** 분석가 조치 없음 | — |
| 7 | `chained-fit` 정규식 공백 | **유효한 지적. Phase 5 하네스 갱신 후보로 등록됨** | 아래 주 |

**#4 는 내 조사 범위 오류다.** 과목 폴더의 `.claude/scripts/` 만 보고 "없다" 고 단정했는데, 감사 스크립트는
**과목이 아니라 `00_Lecture/` 상위의 스킬 디렉터리**에 있다. 하네스 도구를 찾을 때는 과목 로컬만이 아니라
스킬 디렉터리까지 봐야 한다 — 다음 회차 분석가가 같은 오인을 하지 않도록 여기 남긴다.

**남은 진짜 미검증 항목 넷 (양축 PASS 이후에도 유지된다):**

1. **강의 슬라이드 부재 — 최대 잔여 위험.** 이 회차에 대응하는 강의자료가 없다(§1).
   **라운드 5 에서 Step 4(지표)가 노트북에서 빠져 슬라이드로 넘어갔으므로 의존도가 더 커졌다** —
   이제 IoU·precision/recall·AP·mAP 를 학생이 배우는 곳은 **오직 슬라이드뿐이고, 그 슬라이드는 아직 없다.**
   제작되면 **§2 가 대조 기준**이 된다. 특히 §2.3 의 금지 서술(mAP 의 `m` 은 클래스 평균)을 확인할 것
2. **Roboflow `.download()` 왕복.** 키 없는 환경에서는 가드 경로만 실행된다.
   `model_format='yolov8'` 은 SDK 소스로 확정했으나(§3.8), `.download()` 의 실제 반환과
   디렉터리 배치(§4.3)는 **본 분석가도 검증자도 확인하지 못했다.** 키를 가진 환경에서 한 번
   돌려 보는 것 외에 닫을 방법이 없다
3. **셀 2·6 의 미실행 분기.** 셀 2 의 `except ImportError` → `pip install` 경로와 셀 6 의 "이미 있으면
   건너뜀" 반대 분기는 이 환경에서 실행되지 않았다. Colab 첫 실행에서만 도는 경로이므로
   **로컬 `nbconvert` 통과가 그 분기의 정상 동작을 보증하지 않는다**
4. **(신규) CPU 환경에서의 학습 시간.** `epochs` 가 20 → **120 으로 6배**가 됐다.
   GPU 실측은 셀당 36~40초, 노트북 전체 90.4초로 프로필 `cell_time_warning_sec: 300` 대비 여유가 있으나,
   **CPU 런타임(Colab 무료 티어)에서는 측정하지 못했다.** 이 회차에서 상한을 넘길 위험이 가장 큰 자리다

### 6.2 의도적으로 반영하지 않은 항목 (되돌리지 마라)

다음은 **누락이 아니라 판정**이다. 다음 개정자가 "빠졌네" 하고 넣지 않도록 사유와 함께 남긴다.

| 항목 | 사유 |
|---|---|
| **감사 C-1** — 셀 17 통제표 B 열이 `best of 120` 만 적어 셀 20 의 3행 출력을 2팔로 그린다 | 뒤따르는 **산문이 보완하므로 거짓이 아니다.** 셀 17 이 마크다운 예산 **116/120** 이라 여유가 없다 |
| **감사 C-2** — 셀 32 `halved the score` 가 어느 체크포인트인지 말하지 않는다 | **다음 불릿이 예외를 처리한다.** Summary 산문이 **61/60** 이라 여유가 없다 |
| **검증 V-12** — `best.pt` 는 초기 epoch 동률의 tie-break 이지 학습 성과가 아니다 | **노트북 출력만으로 특정할 수 없다.** 근거였던 epoch 별 로그 줄이 이번 수정으로 사라졌다. 확인 불가능한 주장을 본문에 넣지 않는 것이 이 과목의 규율이다. **명세 §3.9 에는 남겼다** |
| **감사 C-5** — fuse 함정이 코드 주석에만 있고 마크다운에 없다 | 선택 사항. 코드 주석(셀 19·29)이 이미 이유를 말한다 |

### 6.1 이전 §6 의 원문 (판정 전 기록 — **행동 근거로 쓰지 마라**)

> 이 절은 판정 이력 보존용이다. **위 표가 최신이고 우선한다.**
> 아래 두 항목은 **사실이 틀린 것으로 판명**되어 취소선으로 표시했다.

- **[근거 없음 — 확인 필요] 강의 슬라이드 부재.** 이 회차에 대응하는 강의자료가 `lecture_notes/` 에
  존재하지 않는다. 이 명세의 §2 수식과 §4 섹션 구성은 **강의 원문이 아니라 COCO 표준 정의와 사용자
  결정사항에서 나왔다.** 향후 슬라이드가 제작되면 이 명세와 대조해 notation·범위를 재검증해야 한다.
  특히 IoU/mAP 를 강의가 어느 깊이까지 다루는지가 확정되면 Step 4 의 분량이 바뀔 수 있다.
- **[확인 필요] Summary 셀 유무.** P09·P10 은 `## Summary` + `## Exercises` 를 갖지만 **직전 회차 P11 은
  Summary 가 없다**(현재 15셀, Exercises 로 끝남). 사용자 지침은 Summary 를 포함하라고 했으므로 넣었다.
  P11 이 최신 형태라면 리더가 제거를 지시할 수 있다.
- **[확인 필요] 추론 이미지의 출처.** P11 은 Unsplash URL 을 썼다. 그러나 **`predict(source=<URL>)` 는
  이미지를 cwd 에 다운로드한다** (본 분석가 실측: `bus.jpg` 가 cwd 에 생성됨) → 루트 오염.
  본 명세는 **로컬 coco8 이미지**를 source 로 쓰도록 확정했다. 라벨과 같은 이미지를 쓰므로 Step 2 의
  ground truth 와 Step 3 의 예측, Step 4 의 IoU 가 하나의 이미지로 연결되는 이점도 있다.
  리더가 외부 사진을 원하면 `data/` 로 먼저 받는 셀이 추가로 필요하다.
- **[확인 필요] `data/` 의 커밋 여부.** `.gitignore` 는 `build/` 만 무시한다. coco8/coco8-seg 은 각 1MB
  미만이라 커밋해도 부담은 없으나, 노트북이 자동 생성하는 산출물이므로 `.gitignore` 에
  `data/coco8/`, `data/coco8-seg/` 를 추가할지 리더 판단이 필요하다.
- ~~**[확인 필요] Roboflow 셀의 실행 검증 범위.** ... `model_format` 문자열로 `'yolov11'` 이 유효한지는
  **미확인**이다.~~ → **판정됨: `'yolov8'` 로 확정.** `'yolov11'` 은 패키지가 아는 식별자가 아니다. §3.8 참조
- ~~**[확인 필요] `.claude/scripts/` 에 `audit_notebook.py` 가 없다.**~~
  → **취소. 내 오인이었다.** 스크립트는 `00_Lecture/.claude/skills/notebook-convention-audit/scripts/`
  에 실재하며 정상 동작한다. 감사자는 기계 검사를 돌릴 수 있다
- **[불일치 — 해소됨] `settings.update()` 효과.** 리더 프로브는 "반영되지 않음", 본 분석가 프로브는
  "반영됨" 으로 관측이 갈렸다. 원인을 확인해 해소했다 — 전역 설정 파일을 덮어쓰고 import 시점에 읽히므로
  **같은 커널에서는 무효, 다음 프로세스부터 유효**하다. 두 관측 모두 사실이며 결론은 리더와 동일하게
  **사용 금지**다 (§3.1).
- **[주의] 프로필 `checkpoints.enabled` 의 `lr-symbol`·`theta-symbol`·`np-matrix`·`chained-fit` 은
  이 회차에서 발화할 자리가 없다.** 다만 `chained-fit` 패턴 `\)\.fit(...)\(` 는 `.train(`/`.predict(` 를
  잡지 않으므로, **모델 생성과 `.train()` 체이닝 금지**는 기계 검사가 아니라 사람이 봐야 한다.
  저자는 `CLAUDE.md` "Model Definition Style" 대로 반드시 두 줄로 분리하라.
