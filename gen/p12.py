# Generator script for "Practice12_Fine_Tuning_on_a_New_Class.ipynb"
# 원본은 2025_2_AI-ME (Graduate)/1_HW/HW5 (PCB 결함 fine-tuning 과제) 의 실습판이다.
# HW5 는 Roboflow 키가 있어야 돌아가므로, 실습에서는 키 없이 받는 crack-seg 로 같은 절차를 밟는다.
#
# 회차의 주장: fine-tuning 은 사전학습 모델을 이기는 것이 아니라 **모르는 클래스를 가르치는 것**이다.
# 구성은 detection -> segmentation 의 단계 상승이다. 가중치 파일 한 단어만 바꾸면 출력이 늘어난다.
#
# 3차 수정 — 사용자 피드백 (학생 제출 답안이 6셀로 훨씬 간단하다는 지적):
#   1. checkpoint 저장/재로드(YOLO(runs/.../last.pt))를 없앴다. 학생 답안처럼 train() 뒤 같은 객체로
#      바로 val() 한다. best.pt 가 val 로 선택되는 문제는 **test 분할로 평가**해 해소한다
#      (crack-seg 에 images/test 112장 + 라벨이 있다. 실측 확인)
#   2. subset 폴더 복사(shutil 이중 루프)를 없앴다. train(fraction=0.05) 인자 하나로 대체
#   3. 손으로 Rectangle/마스크를 그리던 코드를 result.plot() 으로 대체.
#      "결과를 그림에 덧그리는" 코드는 Practice11 의 주제이고, 이 회차의 주제는 학습이다
#   4. np.ma.masked_where / instance_map / PIL ImageDraw 처럼 파이썬 숙련도를 요구하는 구문을 제거.
#      대상 독자는 기계공학도이며 파이썬 자체는 학습 목표가 아니다
#   5. 학습 경과는 results.csv 곡선으로 계속 보여준다 (YOLO_VERBOSE=False 로 배치 진행바는 끈다)
#
# ultralytics 는 polygon 라벨 데이터셋으로 detection 모델도 학습한다 (라벨의 extent 를 박스로 쓴다).
# 경로 정책은 md/p12_workspace/00_env_probe.md 를 따른다. crack-seg.zip 은 flat 하게 풀린다.
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip("\n")))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip("\n")))


# ---------------------------------------------------------------------- title
md(r"""
# Practice 12 — Fine-tuning on a New Class

A pretrained detector knows 80 classes.
The object you care about is usually not one of them.

| Question | Answer this notebook gives |
|:---|:---|
| What is fine-tuning for? | teaching a class the weights have never seen |
| How much data does it take? | a few hundred images, because the backbone is inherited |
| What changes for masks instead of boxes? | the weights file, and nothing else in the call |

The new class is `crack`, from photographs of concrete and asphalt surfaces.
""")

# --------------------------------------------------------------------- step 0
md(r"""
---
## Step 0. Setup

`ultralytics` provides the model, the training loop and the metrics.
Setting `YOLO_VERBOSE` before the import turns off the progress bar it prints for every batch.
""")

code(r"""
import os
import subprocess
import sys

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
from PIL import Image
from ultralytics import YOLO

np.random.seed(42)
torch.manual_seed(42)

BUILD = Path('build')
DATA = Path('data')
# ultralytics needs an absolute path here, otherwise the run folder lands next to the notebook
RUNS = (BUILD / 'runs').resolve()
BUILD.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device : {device}')
""")

code(r"""
crack_dir = DATA / 'crack_full'

if not crack_dir.exists():
    urllib.request.urlretrieve(
        'https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip',
        BUILD / 'crack-seg.zip')
    # this archive has no top level folder inside, so the destination folder is named here
    with zipfile.ZipFile(BUILD / 'crack-seg.zip') as archive:
        archive.extractall(crack_dir)

for split in ['train', 'val', 'test']:
    images = sorted((crack_dir / 'images' / split).glob('*.jpg'))
    print(f'{split:5s} : {len(images)} images')
""")

# --------------------------------------------------------------------- step 1
md(r"""
---
## Step 1. What the Pretrained Models Do Here

Both models below are the COCO weights, untouched.
`result.plot()` draws whatever the model found, in BGR order, so the channels are reversed for
matplotlib.
""")

code(r"""
pretrained_detection_model = YOLO(str(BUILD / 'yolo11n.pt'))
pretrained_segmentation_model = YOLO(str(BUILD / 'yolo11n-seg.pt'))

print(f"'crack' among the 80 class names : {'crack' in pretrained_detection_model.names.values()}")
print()

crack_images = sorted((crack_dir / 'images' / 'test').glob('*.jpg'))[:4]
for image_path in crack_images:
    detection_result = pretrained_detection_model.predict(image_path, conf=0.25, verbose=False)[0]
    segmentation_result = pretrained_segmentation_model.predict(image_path, conf=0.25, verbose=False)[0]
    print(f'{image_path.name[:12]} : detection {len(detection_result.boxes)} objects, '
          f'segmentation {len(segmentation_result.boxes)} objects')
""")

code(r"""
fig, axes = plt.subplots(2, 4, figsize=(15, 8))

for column, image_path in enumerate(crack_images):
    detection_result = pretrained_detection_model.predict(image_path, conf=0.25, verbose=False)[0]
    segmentation_result = pretrained_segmentation_model.predict(image_path, conf=0.25, verbose=False)[0]

    axes[0, column].imshow(detection_result.plot()[:, :, ::-1])
    axes[0, column].set_title('detection')
    axes[0, column].axis('off')

    axes[1, column].imshow(segmentation_result.plot()[:, :, ::-1])
    axes[1, column].set_title('segmentation')
    axes[1, column].axis('off')

plt.tight_layout(); plt.show()
""")

md(r"""
Nothing is drawn on any of the four panels.
The weights are not broken.
Their output layer has 80 slots fixed when they were trained, and none of those slots means `crack`.
""")

# --------------------------------------------------------------------- step 2
md(r"""
---
## Step 2. The Label Format

A YOLO dataset is two mirrored folder trees.
Every image has a label file of the same name under `labels/`.

A segmentation label line is `class x1 y1 x2 y2 ... xn yn`: an outline whose coordinates are divided by
the image size.
Its length varies with the shape, which is what separates it from a detection line of `class cx cy w h`.
""")

code(r"""
image_path = crack_images[0]
label_path = crack_dir / 'labels' / 'test' / (image_path.stem + '.txt')
label_line = label_path.read_text().strip().split('\n')[0]

numbers = label_line.split()
print(f'class          : {numbers[0]}')
print(f'coordinates    : {len(numbers) - 1}')
print(f'outline points : {(len(numbers) - 1) // 2}')
print()
print('first 8 numbers')
print(' '.join(numbers[:9]))
""")

md(r"""
Normalized numbers are hard to judge as text.
Drawing them back onto the image is the fastest check that a label file is right.
""")

code(r"""
crack_image = np.array(Image.open(image_path).convert('RGB'))
image_height = crack_image.shape[0]
image_width = crack_image.shape[1]

# the outline, back in pixels
outline = np.array(numbers[1:], dtype=float).reshape(-1, 2)
outline_x = outline[:, 0] * image_width
outline_y = outline[:, 1] * image_height

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].imshow(crack_image)
axes[0].set_title('Image')

axes[1].imshow(crack_image)
axes[1].plot(outline_x, outline_y, color='cyan', linewidth=2)
axes[1].set_title('Label: an outline of points')

# training fills that outline in, and the loss compares this against the predicted mask
axes[2].imshow(np.zeros_like(crack_image))
axes[2].fill(outline_x, outline_y, color='white')
axes[2].set_title('Filled in: what the loss compares')

for panel_axis in axes:
    panel_axis.axis('off')

plt.tight_layout(); plt.show()
""")

md(r"""
The file stores points and the loader fills them in.
A detection label is the same outline reduced to its extent, which is why one dataset trains both kinds
of model.
""")

code(r"""
center_x = (outline_x.min() + outline_x.max()) / 2 / image_width
center_y = (outline_y.min() + outline_y.max()) / 2 / image_height
box_width = (outline_x.max() - outline_x.min()) / image_width
box_height = (outline_y.max() - outline_y.min()) / image_height

print(f'segmentation label : class 0 with {len(outline)} points')
print(f'detection label    : class 0 {center_x:.4f} {center_y:.4f} {box_width:.4f} {box_height:.4f}')
""")

# --------------------------------------------------------------------- step 3
md(r"""
---
## Step 3. Pointing YOLO at the Data

`data.yaml` says where the folders are and what the class indices mean.
`test` is held out from training and from checkpoint selection, so it is the split reported below.
""")

code(r"""
crack_yaml_path = (DATA / 'crack.yaml').resolve()

with open(crack_yaml_path, 'w', encoding='utf-8') as yaml_file:
    yaml_file.write(f'path: {crack_dir.resolve().as_posix()}\n')
    yaml_file.write('train: images/train\n')
    yaml_file.write('val: images/val\n')
    yaml_file.write('test: images/test\n')
    yaml_file.write('names:\n')
    yaml_file.write('  0: crack\n')

print(crack_yaml_path.read_text().strip())
""")

code(r"""
pretrained_model = YOLO(str(BUILD / 'yolo11n.pt'))
pretrained_metrics = pretrained_model.val(data=str(crack_yaml_path), split='test', imgsz=320,
                                          project=str(RUNS), name='crack_pretrained_test', exist_ok=True)

print(f'pretrained box mAP@50 : {pretrained_metrics.box.map50:.5f}')
""")

md(r"""
The score is not exactly zero only because a stray COCO box occasionally overlaps a crack.
Read it as the floor that training has to beat.
""")

# --------------------------------------------------------------------- step 4
md(r"""
---
## Step 4. Fine-tuning a Detector

`fraction=0.05` trains on 5 percent of the images, which is about 185 of them and takes under a minute
on a GPU.
`patience=0` runs every epoch, and `amp=False` stops ultralytics from downloading a second model to
check mixed precision.

`val()` is called on the same object right after training, the way it would be in a script.
""")

code(r"""
epochs = 20

detection_model = YOLO(str(BUILD / 'yolo11n.pt'))
detection_model.train(data=str(crack_yaml_path), epochs=epochs, fraction=0.05,
                      imgsz=320, batch=16, optimizer='SGD', lr0=0.01,
                      seed=42, amp=False, workers=0, patience=0,
                      project=str(RUNS), name='crack_detect', exist_ok=True)

detection_metrics = detection_model.val(data=str(crack_yaml_path), split='test', imgsz=320,
                                        project=str(RUNS), name='crack_detect_test', exist_ok=True)
print(f'fine-tuned box mAP@50 : {detection_metrics.box.map50:.5f}')
""")

md(r"""
Every epoch is written to `results.csv` inside the run folder.
Reading that file is how the training becomes a picture.
""")

code(r"""
detection_history = pd.read_csv(RUNS / 'crack_detect' / 'results.csv')
print(detection_history[['epoch', 'train/box_loss', 'val/box_loss', 'metrics/mAP50(B)']].head())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(detection_history['epoch'], detection_history['train/box_loss'], 'b-',
             linewidth=2, label='train')
axes[0].plot(detection_history['epoch'], detection_history['val/box_loss'], 'r--',
             linewidth=2, label='val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('box loss')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(detection_history['epoch'], detection_history['metrics/mAP50(B)'], 'g-', linewidth=2)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('box mAP@50')
axes[1].set_title('Score during training'); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()
""")

md(r"""
The score climbs in the first few epochs and then flattens, which is where 20 epochs comes from.
The validation loss stops falling before the training loss does, the usual sign that more epochs would
fit these images rather than cracks in general.
""")

code(r"""
fig, axes = plt.subplots(1, 4, figsize=(15, 4))

for panel_axis, image_path in zip(axes, crack_images):
    result = detection_model.predict(image_path, conf=0.25, verbose=False)[0]
    panel_axis.imshow(result.plot()[:, :, ::-1])
    panel_axis.set_title(f'{len(result.boxes)} cracks')
    panel_axis.axis('off')

plt.tight_layout(); plt.show()
""")

md(r"""
These are test images, which no training step and no checkpoint choice ever read.
A box that covers most of the frame is not a mistake: the crack runs across the frame.
""")

# --------------------------------------------------------------------- step 5
md(r"""
---
## Step 5. Why the Pretrained Weights Matter

The run below changes one thing: the weights start from random numbers instead of the COCO ones.

| Item | A: pretrained | B: fine-tuned | C: from scratch |
|:---|:---|:---|:---|
| **initial weights** | **COCO** | **COCO** | **random** |
| **training** | **none** | **20 epochs** | **20 epochs** |
| data, `fraction`, `imgsz`, `batch`, `lr0`, `seed` | same | same | same |

A against B isolates the training; B against C isolates the starting weights.
""")

code(r"""
# building the model from its yaml means no pretrained weights are loaded
scratch_model = YOLO('yolo11n.yaml')
scratch_model.train(data=str(crack_yaml_path), epochs=epochs, fraction=0.05,
                    imgsz=320, batch=16, optimizer='SGD', lr0=0.01,
                    seed=42, amp=False, workers=0, patience=0,
                    project=str(RUNS), name='crack_scratch', exist_ok=True)

scratch_metrics = scratch_model.val(data=str(crack_yaml_path), split='test', imgsz=320,
                                    project=str(RUNS), name='crack_scratch_test', exist_ok=True)

print('                  box mAP@50')
print(f'A pretrained     {pretrained_metrics.box.map50:11.5f}')
print(f'B fine-tuned     {detection_metrics.box.map50:11.5f}')
print(f'C from scratch   {scratch_metrics.box.map50:11.5f}')
""")

code(r"""
scratch_history = pd.read_csv(RUNS / 'crack_scratch' / 'results.csv')

plt.figure(figsize=(7, 4))
plt.plot(detection_history['epoch'], detection_history['metrics/mAP50(B)'], 'b-',
         linewidth=2, label='B: from COCO weights')
plt.plot(scratch_history['epoch'], scratch_history['metrics/mAP50(B)'], 'r--',
         linewidth=2, label='C: from random weights')
plt.xlabel('Epoch'); plt.ylabel('box mAP@50')
plt.title('Same data, same epochs, different starting point')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

md(r"""
C never leaves the bottom of the plot.
It had the same images, the same epochs and the same learning rate, but no backbone that already knew
edges and textures.
A few hundred images are enough to name a new class, not to learn vision from nothing.
""")

# --------------------------------------------------------------------- step 6
md(r"""
---
## Step 6. The Same Training, With Masks

`yolo11n-seg.pt` replaces `yolo11n.pt` and nothing else in the call changes.
The dataset is the one already on disk, since its labels were outlines all along.

Validation now reports two families of numbers: one for boxes, one for masks.
""")

code(r"""
segmentation_model = YOLO(str(BUILD / 'yolo11n-seg.pt'))
segmentation_model.train(data=str(crack_yaml_path), epochs=epochs, fraction=0.05,
                         imgsz=320, batch=16, optimizer='SGD', lr0=0.01,
                         seed=42, amp=False, workers=0, patience=0,
                         project=str(RUNS), name='crack_segment', exist_ok=True)

segmentation_metrics = segmentation_model.val(data=str(crack_yaml_path), split='test', imgsz=320,
                                              project=str(RUNS), name='crack_segment_test',
                                              exist_ok=True)

print(f'box  mAP@50 : {segmentation_metrics.box.map50:.5f}')
print(f'mask mAP@50 : {segmentation_metrics.seg.map50:.5f}')
""")

code(r"""
fig, axes = plt.subplots(1, 4, figsize=(15, 4))

for panel_axis, image_path in zip(axes, crack_images):
    result = segmentation_model.predict(image_path, conf=0.25, verbose=False)[0]
    panel_axis.imshow(result.plot()[:, :, ::-1])
    panel_axis.set_title(f'{len(result.boxes)} cracks')
    panel_axis.axis('off')

plt.tight_layout(); plt.show()
""")

md(r"""
The box score lands close to the detector's, from the same data and the same schedule.
The mask score sits below it, because an outline has to be right pixel by pixel while a box only has to
be right at its four edges.
""")

md(r"""
This model does **instance** segmentation, one mask per object, and the count is part of its answer.

| Task | One output covers |
|:---|:---|
| Semantic segmentation | every pixel of a class, as a single region |
| Instance segmentation | every pixel of one object, one mask per object |

For a crack the two are close, because a branching crack is one connected shape and the split into
instances is a decision the labels made rather than a physical fact.
""")

# --------------------------------------------------------------------- step 7
md(r"""
---
## Step 7. Why a Mask and Not a Box

A crack runs diagonally across its own bounding box.
`retina_masks=True` returns each mask at the size of the original image, so its pixels can be counted
against the area of the box.
""")

code(r"""
print('image          mask pixels   box pixels   fill')
for image_path in crack_images:
    result = segmentation_model.predict(image_path, conf=0.25, retina_masks=True, verbose=False)[0]
    for index in range(len(result.boxes)):
        x1, y1, x2, y2 = result.boxes.xyxy[index]
        box_pixels = float((x2 - x1) * (y2 - y1))
        mask_pixels = float(result.masks.data[index].sum())
        print(f'{image_path.name[:12]} {mask_pixels:12.0f} {box_pixels:12.0f}   '
              f'{mask_pixels / box_pixels:.3f}')
""")

md(r"""
A fraction of each box is crack and the rest is surface.
Any quantity that scales with the damage, such as area or length, has to be read from the mask.
""")

# ------------------------------------------------------------------- summary
md(r"""
---
## Summary

- Fine-tuning taught the weights a class they could not name before, and the score on the held-out test
  split moved from the floor to a usable number.
- The run from random weights, given the same data and the same epochs, stayed at the floor.
- Swapping the weights file turned the same call into a segmentation run, and the crack fills a small
  part of its own box.
""")

md(r"""
---
## Exercises

1. Change `fraction` to 0.02 and rerun Step 4.
   How much of the score survives on a fifth of the images?
2. Change `epochs` to 5.
   All three training calls read that variable, so they shorten together.
3. Predict with `conf=0.05` on the four test images.
   Which extra cracks appear?
""")

code(r"""
# Write your code here
""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}

with open('Practice12_Fine_Tuning_on_a_New_Class.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'{len(cells)} cells written')
