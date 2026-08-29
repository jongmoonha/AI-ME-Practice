# PKU PCB defect dataset (693장, 원본 3034x1586, 960MB) 을 학생 배포용 YOLO 포맷으로 가공한다.
#
# 출처: https://github.com/Ironbrotherstyle/PCB-DATASET  (images/{Class}/*.jpg, Annotations/{Class}/*.xml)
#       원 데이터는 Peking University Robotics Institute 의 PCB Defect Dataset 이다.
# 결과: build/pcb_defect_640/  +  build/pcb_defect_640.zip
#       긴 변 640 으로 축소한 JPEG + YOLO detection 라벨 + data.yaml.
#       zip 을 공개 URL(교수 GitHub) 에 올리고 HW 노트북이 그 URL 을 받는다.
#
# 실행: python tools/prepare_pcb_dataset.py   (과목 루트에서)

import io
import json
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

REPO = 'Ironbrotherstyle/PCB-DATASET'
BRANCH = 'master'
CLASS_DIRS = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
CLASS_NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
LONG_SIDE = 640
JPEG_QUALITY = 88
WORKERS = 12

OUT = Path('build') / 'pcb_defect_640'
HEADERS = {'User-Agent': 'Mozilla/5.0'}


def fetch(url, timeout=120):
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def list_dir(path):
    url = f'https://api.github.com/repos/{REPO}/contents/{path}?ref={BRANCH}'
    return json.loads(fetch(url, timeout=60))


def parse_voc(xml_bytes, name_to_index):
    # VOC 어노테이션에서 (class_index, cx, cy, w, h) 를 정규화 좌표로 뽑는다
    root = ET.fromstring(xml_bytes)
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    rows = []
    for obj in root.findall('object'):
        raw_name = obj.find('name').text.strip().lower().replace(' ', '_')
        if raw_name not in name_to_index:
            raise ValueError(f'unknown class {raw_name}')
        box = obj.find('bndbox')
        x1 = float(box.find('xmin').text)
        y1 = float(box.find('ymin').text)
        x2 = float(box.find('xmax').text)
        y2 = float(box.find('ymax').text)
        rows.append((name_to_index[raw_name],
                     (x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
                     (x2 - x1) / width, (y2 - y1) / height))
    return rows


def process_one(class_dir, stem, split, name_to_index):
    image_url = f'https://raw.githubusercontent.com/{REPO}/{BRANCH}/images/{class_dir}/{stem}.jpg'
    xml_url = f'https://raw.githubusercontent.com/{REPO}/{BRANCH}/Annotations/{class_dir}/{stem}.xml'

    rows = parse_voc(fetch(xml_url), name_to_index)

    image = Image.open(io.BytesIO(fetch(image_url))).convert('RGB')
    scale = LONG_SIDE / max(image.size)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    resized.save(OUT / 'images' / split / f'{stem}.jpg', quality=JPEG_QUALITY)

    # 정규화 좌표라 리사이즈해도 라벨은 그대로다
    lines = [f'{index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}' for index, cx, cy, w, h in rows]
    (OUT / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return stem, len(rows)


def main():
    name_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    for split in ['train', 'val', 'test']:
        (OUT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUT / 'labels' / split).mkdir(parents=True, exist_ok=True)

    jobs = []
    for class_dir in CLASS_DIRS:
        entries = sorted(entry['name'] for entry in list_dir(f'images/{class_dir}')
                         if entry['name'].endswith('.jpg'))
        print(f'{class_dir:18s} {len(entries)} images', flush=True)
        # 클래스마다 같은 비율로 쪼갠다. 파일명이 01_missing_hole_01.jpg 순이라 순서가 결정적이다
        for position, name in enumerate(entries):
            stem = name[:-4]
            if position % 10 == 8:
                split = 'val'
            elif position % 10 == 9:
                split = 'test'
            else:
                split = 'train'
            jobs.append((class_dir, stem, split))

    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process_one, class_dir, stem, split, name_to_index): stem
                   for class_dir, stem, split in jobs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failed.append((futures[future], repr(exc)))
            done += 1
            if done % 50 == 0:
                print(f'  {done}/{len(jobs)}', flush=True)

    print('failed:', failed, flush=True)

    yaml_lines = ['path: .', 'train: images/train', 'val: images/val', 'test: images/test', 'names:']
    for index, name in enumerate(CLASS_NAMES):
        yaml_lines.append(f'  {index}: {name}')
    (OUT / 'data.yaml').write_text('\n'.join(yaml_lines) + '\n', encoding='utf-8')

    zip_path = OUT.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(OUT.rglob('*')):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(OUT).as_posix())

    for split in ['train', 'val', 'test']:
        count = len(list((OUT / 'images' / split).glob('*.jpg')))
        print(f'{split:5s} {count} images', flush=True)
    print(f'zip {zip_path} {zip_path.stat().st_size/1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
