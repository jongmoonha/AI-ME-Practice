# Generator script for "Practice11_Object_Detection_and_Segmentation.ipynb"
# 원본은 2025_2_AI-ME (Graduate)/1_HW/HW4_answer.ipynb (사전학습 YOLO11 로 detection/segmentation 추론).
# HW4 는 res.plot() 한 장으로 끝나므로, 이 회차는 그 자리를 세 가지로 늘렸다:
#   - result.boxes / result.masks 의 속성과 shape 를 직접 열어 본다
#   - 내장 plot() 과 손으로 그린 그림을 나란히 놓는다 (같은 그림이 나오는 것을 눈으로 확인)
#   - box 와 mask 의 차이를 mask 픽셀 / box 면적으로 숫자화한다
# 학습(fine-tuning)은 전부 Practice12 으로 옮겼다 — 이 노트북은 아무것도 학습하지 않는다.
#
# 이 회차에 대응하는 강의 슬라이드는 없다. 근거는 md/p12_workspace/ 의 명세와 실측이다.
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip("\n")))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip("\n")))


# ---------------------------------------------------------------------- title
md(r"""
# Practice 11 — Object Detection and Segmentation

Two tasks that answer *where*, not only *what*.

| Task | Output per object | Numbers per object |
|:---|:---|:---|
| Classification | one label for the whole image | 1 class index |
| Object detection | one axis-aligned box | 4 coordinates + class + confidence |
| Instance segmentation | one per-pixel mask | a mask the size of the image + class + confidence |

Nothing is trained here.
The weights arrive pretrained on the 80 COCO classes, and this notebook reads what they produce.
""")

# --------------------------------------------------------------------- step 0
md(r"""
---
## Step 0. Setup

`ultralytics` provides the YOLO models and their inference API.
Passing a path to `YOLO(...)` downloads the weights to that path instead of the working directory.
""")

code(r"""
import subprocess
import sys

try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    import ultralytics

print(f'ultralytics {ultralytics.__version__}')
""")

code(r"""
import urllib.request
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO

np.random.seed(42)
torch.manual_seed(42)

# every file this notebook writes goes under one of these two folders
BUILD = Path('build')
DATA = Path('data')
BUILD.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device : {device}')
""")

code(r"""
image_path = DATA / 'sample.jpg'
if not image_path.exists():
    urllib.request.urlretrieve(
        'https://images.unsplash.com/photo-1760681557777-bfac1f18cf90'
        '?ixlib=rb-4.1.0&auto=format&fit=crop&q=80&w=2070', image_path)

second_image_path = DATA / 'bus.jpg'
if not second_image_path.exists():
    urllib.request.urlretrieve('https://ultralytics.com/images/bus.jpg', second_image_path)

image = np.array(Image.open(image_path).convert('RGB'))
print(f'image shape : {image.shape}   height, width, channels')

plt.figure(figsize=(7, 5))
plt.imshow(image)
plt.axis('off')
plt.show()
""")

# --------------------------------------------------------------------- step 1
md(r"""
---
## Step 1. Running the Detector

`yolo11n` is the smallest YOLO11 model and already knows the 80 COCO classes.
`conf=0.25` discards detections the model is less than 25 percent confident about.

One call returns a list with one `Results` object per input image.
""")

code(r"""
detection_model = YOLO(str(BUILD / 'yolo11n.pt'))
detection_results = detection_model.predict(source=str(image_path), conf=0.25, verbose=False)

print(f'results returned : {len(detection_results)}   one per input image')

detection_result = detection_results[0]
print(f'objects found    : {len(detection_result.boxes)}')
print(f'orig_shape       : {detection_result.orig_shape}')
""")

# --------------------------------------------------------------------- step 2
md(r"""
---
## Step 2. What Is Inside `result.boxes`

Every attribute below has one row per detected object.

| Attribute | Shape | Meaning |
|:---|:---|:---|
| `boxes.xyxy` | (N, 4) | corners in pixels, $(x_1, y_1, x_2, y_2)$ |
| `boxes.xywhn` | (N, 4) | centre and size, divided by image size |
| `boxes.conf` | (N,) | confidence |
| `boxes.cls` | (N,) | class index, a float that indexes `result.names` |
""")

code(r"""
print(f'boxes.xyxy  : {tuple(detection_result.boxes.xyxy.shape)}')
print(f'boxes.xywhn : {tuple(detection_result.boxes.xywhn.shape)}')
print(f'boxes.conf  : {tuple(detection_result.boxes.conf.shape)}')
print(f'boxes.cls   : {tuple(detection_result.boxes.cls.shape)}')
print(f'names       : {len(detection_result.names)} class names')
print()

for index in range(len(detection_result.boxes)):
    class_index = int(detection_result.boxes.cls[index])
    confidence = float(detection_result.boxes.conf[index])
    x1, y1, x2, y2 = detection_result.boxes.xyxy[index].cpu().numpy()
    print(f'{detection_result.names[class_index]:<14s} conf={confidence:.3f}  '
          f'x1={x1:7.1f} y1={y1:7.1f} x2={x2:7.1f} y2={y2:7.1f}')
""")

md(r"""
The two coordinate systems hold the same box.

$$x_1 = (c_x - w/2)\,W, \quad y_1 = (c_y - h/2)\,H, \quad
x_2 = (c_x + w/2)\,W, \quad y_2 = (c_y + h/2)\,H$$

Normalized coordinates survive a resize of the image, pixel coordinates do not.
""")

code(r"""
image_height, image_width = detection_result.orig_shape

center_x, center_y, box_width, box_height = detection_result.boxes.xywhn[0].cpu().numpy()
x1_from_normalized = (center_x - box_width / 2) * image_width
y1_from_normalized = (center_y - box_height / 2) * image_height
x2_from_normalized = (center_x + box_width / 2) * image_width
y2_from_normalized = (center_y + box_height / 2) * image_height

print('first box, two ways of writing the same rectangle')
print(f'xyxy               : {detection_result.boxes.xyxy[0].cpu().numpy()}')
print(f'xywhn -> pixels    : [{x1_from_normalized:.4f} {y1_from_normalized:.4f} '
      f'{x2_from_normalized:.4f} {y2_from_normalized:.4f}]')
""")

# --------------------------------------------------------------------- step 3
md(r"""
The two lines agree to the last printed digit.
Anything that resizes the image changes the first line and leaves the second one valid.
""")

md(r"""
---
## Step 3. Drawing the Boxes

`result.plot()` returns an annotated copy of the image in **BGR** order, so the channels are reversed
before matplotlib sees it.

Drawing the same boxes by hand is what makes the numbers above concrete.
""")

code(r"""
annotated_image = detection_result.plot()[:, :, ::-1]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].imshow(annotated_image)
axes[0].set_title('result.plot()')
axes[0].axis('off')

axes[1].imshow(image)
for index in range(len(detection_result.boxes)):
    class_index = int(detection_result.boxes.cls[index])
    confidence = float(detection_result.boxes.conf[index])
    x1, y1, x2, y2 = detection_result.boxes.xyxy[index].cpu().numpy()
    instance_color = plt.cm.tab10(index % 10)
    axes[1].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                fill=False, edgecolor=instance_color, linewidth=2))
    # the label sits inside the top edge, so a box touching y=0 keeps its text on the image
    axes[1].text(x1 + 4, y1 + 4, f'{detection_result.names[class_index]} {confidence:.2f}',
                 color='white', fontsize=9, va='top', ha='left', clip_on=True,
                 bbox=dict(facecolor=instance_color, edgecolor='none', pad=1))
axes[1].set_title('drawn from boxes.xyxy')
axes[1].axis('off')

plt.tight_layout(); plt.show()
""")

# --------------------------------------------------------------------- step 4
md(r"""
The two panels carry the same rectangles, drawn by two different pieces of code.
`plot()` is convenient; the loop on the right is what the coordinates mean.
""")

md(r"""
---
## Step 4. The Confidence Threshold

`conf` is the only thing that changes between the two calls below.
Everything the model computed stays the same; the threshold decides what survives.
""")

code(r"""
low_threshold_result = detection_model.predict(source=str(image_path), conf=0.25, verbose=False)[0]
high_threshold_result = detection_model.predict(source=str(image_path), conf=0.50, verbose=False)[0]

low_threshold_names = []
for index in range(len(low_threshold_result.boxes)):
    low_threshold_names.append(low_threshold_result.names[int(low_threshold_result.boxes.cls[index])])

high_threshold_names = []
for index in range(len(high_threshold_result.boxes)):
    high_threshold_names.append(high_threshold_result.names[int(high_threshold_result.boxes.cls[index])])

print(f'conf=0.25 : {len(low_threshold_names):2d} objects  ' + ', '.join(low_threshold_names))
print(f'conf=0.50 : {len(high_threshold_names):2d} objects  ' + ', '.join(high_threshold_names))
""")

# --------------------------------------------------------------------- step 5
md(r"""
The list shortens from the bottom.
`conf` does not change what the model computed, only which of its answers are shown.
""")

md(r"""
---
## Step 5. The Segmentation Model

A segmentation model returns everything the detector returns, plus one mask per object.

| Attribute | What it holds |
|:---|:---|
| `masks.data` | a binary mask per object, as a grid of 0 and 1 |
| `masks.xy` | the outline of that mask, in original image pixels |
""")

code(r"""
segmentation_model = YOLO(str(BUILD / 'yolo11n-seg.pt'))
segmentation_result = segmentation_model.predict(source=str(image_path), conf=0.25, verbose=False)[0]

print(f'objects found : {len(segmentation_result.boxes)}')
print(f'boxes.xyxy    : {tuple(segmentation_result.boxes.xyxy.shape)}')
print(f'masks.data    : {tuple(segmentation_result.masks.data.shape)}')
print(f'orig_shape    : {segmentation_result.orig_shape}')
print()

for index in range(len(segmentation_result.boxes)):
    class_index = int(segmentation_result.boxes.cls[index])
    confidence = float(segmentation_result.boxes.conf[index])
    outline = segmentation_result.masks.xy[index]
    print(f'{segmentation_result.names[class_index]:<14s} conf={confidence:.3f}  '
          f'masks.xy[{index}] = {outline.shape} outline points')
""")

md(r"""
The mask grid above is **not** the size of the image.
It comes back at the resolution the model ran at, which is what `predict` resized the image to.

`retina_masks=True` asks for the masks at the original image size instead.
""")

code(r"""
retina_result = segmentation_model.predict(source=str(image_path), conf=0.25,
                                           retina_masks=True, verbose=False)[0]

print(f'orig_shape             : {segmentation_result.orig_shape}')
print(f'masks.data  default    : {tuple(segmentation_result.masks.data.shape)}')
print(f'masks.data  retina     : {tuple(retina_result.masks.data.shape)}')
print(f'masks.xy[0] default    : {segmentation_result.masks.xy[0].shape}')
print(f'masks.xy[0] retina     : {retina_result.masks.xy[0].shape}')
""")

md(r"""
The outline changes length with the mask resolution.
That is the clearest sign of what the model actually predicts: a grid of pixels, from which an outline
is traced afterwards.
""")

md(r"""
One mask comes back per object, not one per class.
Two chairs in the list above carry two separate masks, and that count is part of the answer.

| Task | One output covers |
|:---|:---|
| Semantic segmentation | every pixel of a class, as a single region |
| Instance segmentation | every pixel of one object, one mask per object |

The models in this notebook do the second.
""")

# --------------------------------------------------------------------- step 6
md(r"""
---
## Step 6. Drawing the Masks

Each mask is a grid of 0 and 1 at the size of the image, so it can be laid straight on top of it.
The third panel below is one such grid on its own, and `result.plot()` stacks all seven of them.
""")

code(r"""
mask_index = 2
mask_class = retina_result.names[int(retina_result.boxes.cls[mask_index])]
one_mask = retina_result.masks.data[mask_index].cpu().numpy()

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

axes[0].imshow(image)
axes[0].set_title('Input image')
axes[0].axis('off')

axes[1].imshow(image)
for index in range(len(retina_result.boxes)):
    x1, y1, x2, y2 = retina_result.boxes.xyxy[index].cpu().numpy()
    axes[1].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                edgecolor=plt.cm.tab10(index % 10), linewidth=2))
axes[1].set_title('Boxes: 4 numbers per object')
axes[1].axis('off')

axes[2].imshow(one_mask, cmap='gray')
axes[2].set_title(f'One mask ({mask_class}): 1 inside, 0 outside')
axes[2].axis('off')

axes[3].imshow(retina_result.plot()[:, :, ::-1])
axes[3].set_title('All seven, by result.plot()')
axes[3].axis('off')

plt.tight_layout(); plt.show()
""")

# --------------------------------------------------------------------- step 7
md(r"""
The white shape in the third panel is one object, and every pixel outside it is 0.
A box says where an object is; a mask says which pixels are the object.
Overlapping chairs still get separate colours, because each mask belongs to one instance.
""")

md(r"""
---
## Step 7. What the Mask Buys You

A mask can be counted; a box cannot.
Dividing the mask area by the box area says how much of the box the object actually fills.
""")

code(r"""
print('object          mask pixels   box pixels   fill')
for index in range(len(retina_result.boxes)):
    class_index = int(retina_result.boxes.cls[index])
    x1, y1, x2, y2 = retina_result.boxes.xyxy[index].cpu().numpy()
    box_pixels = (x2 - x1) * (y2 - y1)
    mask_pixels = float(retina_result.masks.data[index].cpu().numpy().sum())
    print(f'{retina_result.names[class_index]:<14s} {mask_pixels:11.0f} {box_pixels:12.0f}   '
          f'{mask_pixels / box_pixels:.3f}')
""")

# --------------------------------------------------------------------- step 8
md(r"""
Compact objects fill most of their box, and wide or hollow ones do not.
A number this simple already separates the cases where a box is a fair summary from the cases where it
is mostly background.
""")

md(r"""
---
## Step 8. The Same Two Calls on Another Image

Nothing about the code changes when the picture does.
""")

code(r"""
second_image = np.array(Image.open(second_image_path).convert('RGB'))
second_detection = detection_model.predict(source=str(second_image_path), conf=0.25, verbose=False)[0]
second_segmentation = segmentation_model.predict(source=str(second_image_path), conf=0.25,
                                                 retina_masks=True, verbose=False)[0]

fig, axes = plt.subplots(1, 2, figsize=(11, 6))

axes[0].imshow(second_image)
for index in range(len(second_detection.boxes)):
    class_index = int(second_detection.boxes.cls[index])
    x1, y1, x2, y2 = second_detection.boxes.xyxy[index].cpu().numpy()
    instance_color = plt.cm.tab10(index % 10)
    axes[0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                fill=False, edgecolor=instance_color, linewidth=2))
    axes[0].text(x1 + 4, y1 + 4, second_detection.names[class_index],
                 color='white', fontsize=9, va='top', ha='left', clip_on=True,
                 bbox=dict(facecolor=instance_color, edgecolor='none', pad=1))
axes[0].set_title(f'Detection: {len(second_detection.boxes)} objects')
axes[0].axis('off')

axes[1].imshow(second_segmentation.plot()[:, :, ::-1])
axes[1].set_title(f'Segmentation: {len(second_segmentation.boxes)} masks')
axes[1].axis('off')

plt.tight_layout(); plt.show()
""")

# ------------------------------------------------------------------- summary
md(r"""
---
## Summary

- A detection result holds one row per object in `boxes.xyxy`, `boxes.conf` and `boxes.cls`, and
  `boxes.xywhn` is the same rectangle divided by the image size.
- A segmentation result adds `masks.data`, whose resolution follows the model input unless
  `retina_masks=True` asks for the original size.
- Mask pixels can be counted, and their share of the box says how much of it the object fills.
""")

md(r"""
---
## Exercises

1. Raise `conf` to 0.7 and rerun the detection listing.
   Which objects disappear first, and what were their confidences?
2. Print `boxes.xywhn` for every object and check that no value falls outside 0 to 1.
3. Draw only the object with the largest mask area, using the fill numbers from Step 7.
""")

code(r"""
# Write your code here
""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}

with open('Practice11_Object_Detection_and_Segmentation.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'{len(cells)} cells written')
