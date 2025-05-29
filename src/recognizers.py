import os
import time
import json
from pathlib import Path


import cv2
from skimage import io
from tqdm import tqdm

import torch
from passporteye.mrz.image import MRZPipeline

from mrzscanner import MRZScanner

ROOT_PATH = Path(__file__).resolve().parent.parent

class Recognizer:
    """Recognizer Interface"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = self.load_model()

    def load_model(self):
        raise NotImplementedError("Subclasses should implement this method")


class DocsaidLab(Recognizer):
    """
    Pretrained MRZ recognition model from DocsaidLab. https://github.com/DocsaidLab/MRZScanner
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def load_model(self):
        """
        Could use only recognition model from Docsaid:
        MRZScanner(model_type=ModelType.recognition, recognition_cfg='20250221')

        But it performed worse than two-stage for recognition on the same data. 
        Could not find the reason, inference time is roughly the same.
        """

        return MRZScanner()

    def __call__(self, image):
        """
        Detect MRZ in the given image using Docsaid.
        """
        results = self.model(image, do_center_crop=False, do_postprocess=False)
        return results['mrz_texts']
    

class PassportEye(Recognizer):
    """
    Pretrained PassportEye MRZ recognition model. https://passporteye.readthedocs.io/en/latest/python_usage.html.
    Not separated in two stages, do both detection and recognition in one step.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def load_model(self):
        return MRZPipeline

    def __call__(self, image_path):
        """
        Detect MRZ in the given image using PassportEye.
        """
        result = self.model(image_path)
        if not result['text']:
            return None
        return result.data['text'].split('\n')