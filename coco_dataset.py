import time
import json
import shutil

import random
from tqdm import tqdm
from pathlib import Path

import argparse

import cv2
import numpy as np
from skimage import io
from PIL import Image, ImageDraw

from mrzscanner import MRZScanner
from passporteye.mrz.image import MRZPipeline

from hydra import initialize_config_dir, compose

from src.detectors import DocsaidLab, Passporteye

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


ROOT_PATH = Path(__file__).resolve().parent
CFG_PATH = ROOT_PATH / 'conf'

def split_dataset(data_path, train_ratio=0.8, seed=42):
    """
    Create train/validation splits for passport images dataset
    """
    random.seed(seed)

    train_paths = []
    val_paths = []

    # Get all country folders
    country_paths = sorted([p for p in data_path.iterdir() if p.is_dir()])

    for country_path in country_paths:
        # Get all image paths for this country
        image_paths = sorted([str(p) for p in country_path.iterdir() if p.is_file()])

        # Shuffle paths
        random.shuffle(image_paths)

        # Calculate split point
        split_idx = int(len(image_paths) * train_ratio)

        # Add to train/val lists
        train_paths += image_paths[:split_idx]
        val_paths += image_paths[split_idx:]

    return sorted(train_paths), sorted(val_paths)


def create_coco_json(paths, bboxes):
    result = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 0, 'name': 'mrz'}]
    }

    for i, image_path in enumerate(paths, 1):
        # Read image
        img = Image.open(image_path)
        width, height = img.size
        result['images'].append({
            'id': i,
            'file_name': image_path,
            'width': width,
            'height': height
        })

        x1, y1, x2, y2 = bboxes[image_path]
        bwidth = x2 - x1
        bheight = y2 - y1
        result['annotations'].append({
            'id': i,
            'image_id': i,
            'category_id': 0,
            'bbox': [x1, y1, bwidth, bheight],
            'area': bwidth * bheight,
            'iscrowd': 0
        })

    return result

def create_coco_dataset(bboxes, image_data_path, coco_data_path, train_ratio=0.8, seed=42):
    train_paths, val_paths = split_dataset(image_data_path, train_ratio=train_ratio, seed=seed)
    coco_data_path.mkdir(parents=True, exist_ok=True)
    annotations_path = coco_data_path / 'annotations'
    annotations_path.mkdir(parents=True, exist_ok=True)

    train_coco = create_coco_json(train_paths, bboxes)
    with open(str(annotations_path / 'train.json'), 'w') as f:
        json.dump(train_coco, f, indent=4)

    val_coco = create_coco_json(val_paths, bboxes)
    with open(str(annotations_path / 'val.json'), 'w') as f:
        json.dump(val_coco, f, indent=4)

    train_images_path = coco_data_path / 'images' / 'train'
    train_images_path.mkdir(parents=True, exist_ok=True)
    for img_path in train_paths:
        shutil.copy2(img_path, train_images_path / Path(img_path).name)

    val_images_path = coco_data_path / 'images' / 'val'
    val_images_path.mkdir(parents=True, exist_ok=True)
    for img_path in val_paths:
        shutil.copy2(img_path, val_images_path / Path(img_path).name)

def main():
    with initialize_config_dir(version_base=None, config_dir=str(CFG_PATH)):
        cfg = compose(config_name='conf.yaml')
        cfg_docsaid = compose(config_name='conf.yaml', overrides=[f'detector=docsaidlab'])
        cfg_passporteye = compose(config_name='conf.yaml', overrides=[f'detector=passporteye'])

    data_path = cfg.data_path
    results_path = ROOT_PATH / cfg.results_path

    # Load validation results
    with open(results_path / 'validation_passporteye.json', 'r') as f:
        passporteye_results = json.load(f)

    with open(results_path / 'validation_docsaidlab.json', 'r') as f:
        docsaidlab_results = json.load(f)



    # Get all incorrect images for each model
    passporteye_incorrect = set()
    docsaidlab_incorrect = set()


    for country, results in passporteye_results.items():
        passporteye_incorrect.update(results['incorrect_mages'])

    for country, results in docsaidlab_results.items():
        docsaidlab_incorrect.update(results['incorrect_mages'])


    # Find common incorrect images
    common_incorrect = passporteye_incorrect.intersection(docsaidlab_incorrect)
    for c in common_incorrect:
        print(f'Both models failed on {c} Label bbox manually')

    detector_docsaid = DocsaidLab(cfg_docsaid.detector)
    detector_passporteye = Passporteye(cfg_passporteye.detector)

    bboxes = {}
    for country_folder in sorted(data_path.iterdir()):
        if not country_folder.is_dir():
            continue

        print(f'Processing country: {country_folder.name}')
        image_paths = sorted([p for p in country_folder.iterdir() if p.is_file()])
        for image_path in tqdm(image_paths):
            image_path_str = str(image_path)
            if image_path_str in docsaidlab_incorrect:
                _, bbox = detector_passporteye(image_path)
            else:
                _, bbox = detector_docsaid(image_path)
            bboxes[image_path_str] = bbox

    coco_data_path = ROOT_PATH / 'dataset_coco'
    create_coco_dataset(bboxes, data_path, coco_data_path)

if __name__=='__main__':
    main()


