import os
import sys
import time
import json
from pathlib import Path

import cv2
from tqdm import tqdm
from skimage import io

import torch

from mrzscanner import MRZScanner, ModelType

from src.detectors import Detector, YOLOX
from src.recognizers import Recognizer, DocsaidLab, PassportEye

ROOT_PATH = Path(__file__).resolve().parent.parent


class TwoStagePipeline:
    """
    Two-stage pipeline for document processing. 
    Detect region of interest and recognize text inside it.
    """

    def __init__(self, cfg, country_codes, postprocess=False):
        self.cfg = cfg
        self.country_codes = country_codes
        self.postprocess = postprocess

    def extract_country(self, texts):
        return texts[0][2:5]


    def hamming_closest(self, code, code_etalons):
        """Find the closest predefined code using Hamming distance."""

        best, best_dist = None, float("inf")
        for code in code_etalons:
            p = (code + "<<<")[:3]
            dist = sum(pc != cc for pc, cc in zip(p.upper(), code))
            if dist < best_dist:
                best_dist, best = dist, code
        return best


    def process(self, image_path):
        img, bbox = self.detector(image_path)
        img_crop = self.detector.mrz_crop(img, bbox)
        texts = self.recognizer(img_crop)
        country_code = self.extract_country(texts)
        if self.postprocess:
            country_code= self.hamming_closest(country_code, self.country_codes)
        
        return country_code
    

class YOLOXDocsaidLabPipeline(TwoStagePipeline):
    """Pipeline that uses YOLOX for detection and DocsaidLab for recognition."""

    def __init__(self, cfg, country_codes, postprocess=False):
        super().__init__(cfg, country_codes, postprocess=postprocess)
        self.detector = YOLOX(cfg.detector)
        self.recognizer = DocsaidLab(cfg.recognizer)

    def process(self, image_path):
        img, bbox = self.detector(image_path)
        img_crop = self.detector.mrz_crop(img, bbox)
        texts = self.recognizer(img_crop)
        country_code = self.extract_country(texts)
        if self.postprocess:
            country_code= self.hamming_closest(country_code, self.country_codes)
        
        return country_code
    

class DocsaidLabPipeline(TwoStagePipeline):
    """
    Pretrained MRZ recognition model from DocsaidLab. https://github.com/DocsaidLab/MRZScanner
    """

    def __init__(self, cfg, country_codes, postprocess=False):
        super().__init__(cfg, country_codes, postprocess=postprocess)
        # recongnizer actually performs two steps: detection and recognition
        self.recognizer = DocsaidLab(cfg.recognizer)

    def process(self, image_path):
        img = io.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        texts = self.recognizer(img)
        country_code = self.extract_country(texts)
        if self.postprocess:
            country_code= self.hamming_closest(country_code, self.country_codes)
        
        return country_code
    
class PassportEyePipeline(TwoStagePipeline):
    """
    Pipeline that uses PassportEye for detection and recognition.
    """

    def __init__(self, cfg, country_codes, postprocess=False):
        super().__init__(cfg, country_codes, postprocess=postprocess)
        self.recognizer = PassportEye(cfg.recognizer)  # Placeholder for PassportEye recognizer

    def process(self, image_path):
        texts = self.recognizer(image_path)
        country_code = self.extract_country(texts)
        if self.postprocess:
            country_code= self.hamming_closest(country_code, self.country_codes)
        
        return country_code