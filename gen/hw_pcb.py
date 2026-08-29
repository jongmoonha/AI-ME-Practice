# Generator script for "HW/HW_PCB_Defect_Detection.ipynb" (배포본) 과
#                      "HW/Answer/HW_PCB_Defect_Detection_answer.ipynb" (정답본).
# 원본은 2025_2_AI-ME (Graduate)/1_HW/HW5.pdf — Roboflow 의 PCB defect 데이터로 YOLO 를 학습시키는 과제.
# 원본은 학생이 Roboflow 무료 키를 발급받아 download code 를 붙여넣게 했다. 이 판은 같은 데이터셋의
# 원본(PKU PCB Defect Dataset)을 640 으로 축소하고 YOLO 포맷으로 바꾼 zip 을 직접 받는다 —
# 키 없이 돌고, 정답본을 실행 검증할 수 있다. 가공 스크립트는 tools/prepare_pcb_dataset.py 다.
#
# 2차 수정 — 학생 제출 답안(2025_2 HW5, 6셀)을 보고 사용자가 "훨씬 간단하다" 고 지적했다. 그에 맞춰:
#   - 문제 수를 6개 -> 4개(코드 3 + 서술 1). 라벨 개수 세기 문제는 삭제하고 사실만 도입부 표에 남겼다
#   - 예측 그림을 손으로 Rectangle 그리던 코드에서 result.plot() 으로 교체 (학생 답안과 같은 방식)
#   - 학습 로그는 YOLO_VERBOSE=False 로 끄고 results.csv 곡선으로 대신한다
#
# 실행: python gen/hw_pcb.py            (배포용 URL 사용)
#       python gen/hw_pcb.py --local    (검증용, build/pcb_defect_640.zip 를 file:// 로 읽는다)
import sys
from pathlib import Path

import nbformat as nbf

DATASET_URL = 'https://raw.githubusercontent.com/jongmoonha/AI-ME-Practice/main/data/pcb_defect_640.zip'
if '--local' in sys.argv:
    DATASET_URL = (Path('build') / 'pcb_defect_640.zip').resolve().as_uri()


def build(answer):
    nb = nbf.v4.new_notebook()
    cells = []

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(source.strip("\n")))

    def code(source):
        cells.append(nbf.v4.new_code_cell(source.strip("\n")))

    def problem(solution, stub):
        # 정답본은 완성 코드를, 배포본은 변수명과 골격만 남긴 TODO 를 받는다
        code(solution if answer else stub)

    # ------------------------------------------------------------------ title
    md(r"""
# HW — PCB Defect Detection

A pretrained detector knows 80 everyday classes.
None of them is a soldering defect, so the weights have to be taught what one looks like.

| Item | Value |
|:---|:---|
| images | 561 train / 66 val / 66 test, with roughly the same number of each defect |
| classes | `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper` |
| source | photographs of printed circuit boards, resized so the long side is 640 |

Defects are small, often 10 to 20 pixels across, which is what makes this harder than everyday objects.
""")

    # ----------------------------------------------------------------- given
    md(r"""
---
## Given: Setup and Data

Run the next three cells as they are.
They install `ultralytics`, download the dataset and write the `data.yaml` that training reads.
""")

    code(r"""
import os
import subprocess
import sys

# ultralytics prints a progress bar for every batch; this keeps the training output to a few lines
os.environ['YOLO_VERBOSE'] = 'False'

try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    import ultralytics

print(f'ultralytics {ultralytics.__version__}')
""")

    code(r"""
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

np.random.seed(42)
torch.manual_seed(42)

BUILD = Path('build')
DATA = Path('data')
RUNS = (BUILD / 'runs').resolve()
BUILD.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device : {device}')
""")

    code(r"""
pcb_dir = DATA / 'pcb_defect'

if not pcb_dir.exists():
    urllib.request.urlretrieve('""" + DATASET_URL + r"""', BUILD / 'pcb_defect_640.zip')
    with zipfile.ZipFile(BUILD / 'pcb_defect_640.zip') as archive:
        archive.extractall(pcb_dir)

class_names = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

pcb_yaml_path = (DATA / 'pcb.yaml').resolve()
with open(pcb_yaml_path, 'w', encoding='utf-8') as yaml_file:
    yaml_file.write(f'path: {pcb_dir.resolve().as_posix()}\n')
    yaml_file.write('train: images/train\n')
    yaml_file.write('val: images/val\n')
    yaml_file.write('test: images/test\n')
    yaml_file.write('names:\n')
    for class_index in range(len(class_names)):
        yaml_file.write(f'  {class_index}: {class_names[class_index]}\n')

for split in ['train', 'val', 'test']:
    print(f'{split:5s} : {len(list((pcb_dir / "images" / split).glob("*.jpg")))} images')
""")

    # ------------------------------------------------------------- problem 1
    md(r"""
---
## Problem 1. Look Before Training

Run the pretrained model on one test image and draw what it reports.
`plot()` returns the annotated image in BGR order, so the channels are reversed for matplotlib.

| Answer variable | Type | Meaning |
|:---|:---|:---|
| `pretrained_model` | `YOLO` | the COCO weights, loaded from `build/yolo11n.pt` |
| `pretrained_result` | `Results` | its prediction on `sample_image_path`, at `conf=0.25` |
""")

    problem(r"""
sample_image_path = pcb_dir / 'images' / 'test' / '01_missing_hole_20.jpg'

pretrained_model = YOLO(str(BUILD / 'yolo11n.pt'))
pretrained_result = pretrained_model.predict(sample_image_path, conf=0.25, verbose=False)[0]

print(f'objects found : {len(pretrained_result.boxes)}')

plt.figure(figsize=(8, 5))
plt.imshow(pretrained_result.plot()[:, :, ::-1])
plt.title('Pretrained COCO weights')
plt.axis('off')
plt.show()
""", r"""
sample_image_path = pcb_dir / 'images' / 'test' / '01_missing_hole_20.jpg'

# TODO: load build/yolo11n.pt and predict on sample_image_path with conf=0.25
pretrained_model =
pretrained_result =

print(f'objects found : {len(pretrained_result.boxes)}')

plt.figure(figsize=(8, 5))
plt.imshow(pretrained_result.plot()[:, :, ::-1])
plt.title('Pretrained COCO weights')
plt.axis('off')
plt.show()
""")

    # ------------------------------------------------------------- problem 2
    md(r"""
---
## Problem 2. Fine-tune

Train from the COCO weights on the PCB images.
Use exactly these settings, so that everyone's numbers are comparable.
On a GPU the run takes about nine minutes.

| Argument | Value |
|:---|:---|
| `data` | `str(pcb_yaml_path)` |
| `epochs` | 30 |
| `imgsz`, `batch` | 640, 16 |
| `optimizer`, `lr0` | `'SGD'`, `0.01` |
| `seed`, `patience`, `workers` | 42, 0, 0 |
| `amp` | `False` |
| `project`, `name` | `str(RUNS)`, `'pcb_finetune'` |

| Answer variable | Type | Meaning |
|:---|:---|:---|
| `finetune_model` | `YOLO` | the model that ran the training |
""")

    problem(r"""
finetune_model = YOLO(str(BUILD / 'yolo11n.pt'))
finetune_model.train(data=str(pcb_yaml_path), epochs=30, imgsz=640, batch=16,
                     optimizer='SGD', lr0=0.01, seed=42, amp=False, workers=0, patience=0,
                     project=str(RUNS), name='pcb_finetune', exist_ok=True)

print('training finished')
""", r"""
# TODO: load build/yolo11n.pt, then train it with the settings in the table above
finetune_model =
finetune_model.train()

print('training finished')
""")

    code(r"""
training_history = pd.read_csv(RUNS / 'pcb_finetune' / 'results.csv')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(training_history['epoch'], training_history['train/box_loss'], 'b-',
             linewidth=2, label='train')
axes[0].plot(training_history['epoch'], training_history['val/box_loss'], 'r--',
             linewidth=2, label='val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('box loss')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(training_history['epoch'], training_history['metrics/mAP50(B)'], 'g-', linewidth=2)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('box mAP@50')
axes[1].set_title('Score during training'); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()
""")

    # ------------------------------------------------------------- problem 3
    md(r"""
---
## Problem 3. Score the Test Split

`val()` reads the split named in its `split` argument.
The test images took no part in training or in checkpoint selection.

| Answer variable | Type | Meaning |
|:---|:---|:---|
| `test_metrics` | metrics object | result of `val(...)` on `split='test'`, at `imgsz=640` |
""")

    problem(r"""
test_metrics = finetune_model.val(data=str(pcb_yaml_path), split='test', imgsz=640,
                                  project=str(RUNS), name='pcb_test', exist_ok=True)

print(f'test mAP@50    : {test_metrics.box.map50:.5f}')
print(f'test mAP@50-95 : {test_metrics.box.map:.5f}')
print()
for class_index in range(len(class_names)):
    print(f'{class_names[class_index]:<18s} AP@50-95 = {test_metrics.box.maps[class_index]:.5f}')
""", r"""
# TODO: evaluate finetune_model on split='test' at imgsz=640,
#       with project=str(RUNS), name='pcb_test', exist_ok=True
test_metrics =

print(f'test mAP@50    : {test_metrics.box.map50:.5f}')
print(f'test mAP@50-95 : {test_metrics.box.map:.5f}')
print()
for class_index in range(len(class_names)):
    print(f'{class_names[class_index]:<18s} AP@50-95 = {test_metrics.box.maps[class_index]:.5f}')
""")

    # ------------------------------------------------------------- problem 4
    md(r"""
---
## Problem 4. Show the Predictions

Predict on the six test images below and draw each result.
A model trained for 30 epochs reports lower confidences than one trained on all of COCO, so the
threshold here is 0.10 rather than the usual 0.25.

| Answer variable | Type | Meaning |
|:---|:---|:---|
| `test_results` | list of 6 `Results` | prediction for each path in `test_image_paths`, at `conf=0.10` |
""")

    problem(r"""
test_image_names = ['01_missing_hole_20.jpg', '01_mouse_bite_20.jpg', '01_open_circuit_20.jpg',
                    '01_short_20.jpg', '01_spur_20.jpg', '01_spurious_copper_20.jpg']
test_image_paths = [pcb_dir / 'images' / 'test' / name for name in test_image_names]

test_results = []
for image_path in test_image_paths:
    test_results.append(finetune_model.predict(image_path, conf=0.10, verbose=False)[0])

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

for panel_axis, image_path, result in zip(axes.ravel(), test_image_paths, test_results):
    panel_axis.imshow(result.plot()[:, :, ::-1])
    panel_axis.set_title(f'{image_path.name}   {len(result.boxes)} found', fontsize=9)
    panel_axis.axis('off')

plt.tight_layout(); plt.show()
""", r"""
test_image_names = ['01_missing_hole_20.jpg', '01_mouse_bite_20.jpg', '01_open_circuit_20.jpg',
                    '01_short_20.jpg', '01_spur_20.jpg', '01_spurious_copper_20.jpg']
test_image_paths = [pcb_dir / 'images' / 'test' / name for name in test_image_names]

# TODO: predict on each path with conf=0.10 and append the first result to the list
test_results = []
for image_path in test_image_paths:
    test_results.append()

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

for panel_axis, image_path, result in zip(axes.ravel(), test_image_paths, test_results):
    panel_axis.imshow(result.plot()[:, :, ::-1])
    panel_axis.set_title(f'{image_path.name}   {len(result.boxes)} found', fontsize=9)
    panel_axis.axis('off')

plt.tight_layout(); plt.show()
""")

    # --------------------------------------------------------------- closing
    md(r"""
---
## Problem 5. Read Your Own Numbers

Answer in the markdown cell below, two or three sentences each.

1. Which class has the lowest AP?
   Every class has roughly the same number of training images, so the answer is not "fewer examples".
   Open that class in `images/test` and say what makes it hard.
2. Rerun Problem 4 with `conf=0.25` and compare the counts.
   Boxes disappear, and yet `test_metrics` is unchanged.
   Why not?
""")

    md(r"""
*Write your answer here.*
""" if not answer else r"""
`spur` scores lowest while every class has between 392 and 409 training objects.
A spur is a copper protrusion a few pixels wide, and at this size it looks much like spurious copper,
so the errors are confusions between classes rather than misses.

Raising `conf` only filters the boxes that are drawn.
mAP ranks all predictions by confidence instead of cutting at one value.
""")

    nb['cells'] = cells
    nb['metadata'] = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    }
    return nb


Path('HW/Answer').mkdir(parents=True, exist_ok=True)

for answer, out_path in [(False, Path('HW') / 'HW_PCB_Defect_Detection.ipynb'),
                         (True, Path('HW') / 'Answer' / 'HW_PCB_Defect_Detection_answer.ipynb')]:
    notebook = build(answer)
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    print(f'{out_path}  {len(notebook["cells"])} cells')

print(f'dataset url: {DATASET_URL}')
