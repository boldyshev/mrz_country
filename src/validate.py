import os
import time
import json
from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / 'dataset'
import sys
sys.path.append(str(ROOT_PATH))

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
logging.getLogger('ppocr').setLevel(logging.CRITICAL)
logging.getLogger('paddle').setLevel(logging.CRITICAL)

import cv2
from skimage import io
from tqdm import tqdm
import pytesseract

import torch
from passporteye.mrz.image import MRZPipeline

from mrzscanner import MRZScanner, ModelType


_ = torch.hub.load(
    "Megvii-BaseDetection/YOLOX", 
    "yolox_s", 
    pretrained=False, 
    trust_repo=True
)

hub_dir    = torch.hub.get_dir()
repo_name  = "Megvii-BaseDetection_YOLOX_main"
repo_path  = os.path.join(hub_dir, repo_name)

# 3) Add it to Python’s import path
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
from yolox.exp import get_exp

from yolox.utils import postprocess
from yolox.data.data_augment import preproc

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

CONF_THR    = 0.3
NMS_THR     = 0.45
TILE_SIZE   = (640, 640)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PTH = "/home/boris/syn/dataspark/yolox-s.pth"
EXP_FILE = "/home/boris/syn/YOLOX/exps/example/custom/mrz_s.py"

class CountryValidator:
    def __init__(self, name, model, data_path):
        self.name = name
        self.model = model
        self.data_path = data_path
        self.results = {}

    def model_results(self):
        return {
            'correct': 0,
            'incorrect': 0,
            'accuracy': 0,
            'avg_time_per_image': 0,
            'incorrect_mages': []
        }
    
    def predict_country(self, image_path):
        raise NotImplementedError("Subclasses must implement predict_country()")

    def validate_country(self, country_true):
        country_path = self.data_path / country_true
        image_paths = sorted([p for p in country_path.iterdir() if p.is_file()])
        country_image_count = len(image_paths)
        results = self.model_results()

        if country_true == 'DEU':
            country_true = 'D<<'
        results = self.model_results()
        predicted_countries = []
        incorrect_image_paths = []
        time_country = 0
        for image_path in tqdm(image_paths, desc=f'{self.name} - {country_true}', unit='image'):
            start_time = time.perf_counter()
            pred_correct = country_true == self.predict_country(str(image_path))
            elapsed = time.perf_counter() - start_time
            time_country += elapsed

            predicted_countries.append(pred_correct)
            if not pred_correct:
                incorrect_image_paths.append(str(image_path))

        correct_count = predicted_countries.count(True)
        incorrect_count = predicted_countries.count(False)
        
        results['correct'] = correct_count
        results['incorrect'] = incorrect_count
        results['accuracy'] = correct_count / country_image_count
        results['avg_time_per_image'] = round(time_country / country_image_count, 2)
        results['incorrect_mages'] = incorrect_image_paths
            
        print(f"{self.name} - {country_true}: Correct: {correct_count}, Incorrect: {incorrect_count}")

        return results
    
    def run(self, country_list, output_path):
        for country in country_list:
            print(f"Processing country: {country}")
            self.results[country] = self.validate_country(country)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        return self.results
    
class PassportEyeCountry(CountryValidator):
    def __init__(self, name, model, data_path):
        super().__init__(name, model, data_path)

    def predict_country(self, image_path):
        result = MRZPipeline(image_path)
        if not result['text']:
            return None
        return result.data['text'][2:5]

class DocsaidLabCountry(CountryValidator):
    def __init__(self, name, model, data_path):
        super().__init__(name, model, data_path)

    def predict_country(self, image_path):
        img = io.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = self.model(img, do_center_crop=False, do_postprocess=False)
        if not result:
            return None
        return result['mrz_texts'][0][2:5]

class SerdarHelliCountry(CountryValidator):
    def __init__(self, name, model, data_path):
        super().__init__(name, model, data_path)

    def predict_country(self, image_path):
        text_results,segmented_image ,detected_face = self.model.predict(
            image_path,
            do_facedetect = False,
            preprocess_config = {
                    "do_preprocess": False,
                    "skewness": False,
                    "delete_shadow": False,
                    "clear_background": False
                }
        )
        for i, text in enumerate(text_results):
            if text[1].startswith('P<'):
                return text[1][2:5]


class YOLODocsaidCountry(CountryValidator):
    def __init__(self, name, yolo_model, ocr_model, data_path):
        super().__init__(name, yolo_model, data_path)
        self.ocr_model = ocr_model
        self.country_codes = [c.name if c.name != 'DEU' else 'D<<' for c in DATA_PATH.iterdir() if c.is_dir()]


    def mrz_bbox(self, image_path):
        img = cv2.imread(image_path)
        orig_img = img.copy()
        img_resized, ratio = preproc(img, TILE_SIZE)
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            outputs = postprocess(outputs, 1, CONF_THR, NMS_THR)

        bboxes = []
        if outputs[0] is not None:
            output = outputs[0].cpu().numpy()
            for det in output:
                x0, y0, x1, y1, score, cls_id, _ = det
                bbox = [x0, y0, x1, y1]
                bbox = [round(coord / ratio) for coord in bbox]
                return orig_img, bbox

    def ocr_mrz(self, image, bbox):
        x0, y0, x1, y1 = bbox
        cropped = image[y0:y1, x0:x1]
        if self.ocr_model:
            result = self.ocr_model(cropped, do_center_crop=False)
            return result['mrz_texts'][0]
        else:
            result = pytesseract.image_to_string(cropped, config="--oem 3 --psm 6")
            result = result.strip().split('\n')
            return result[0]

    def predict_country(self, image_path):
        image, bbox = self.mrz_bbox(image_path)
        if bbox is None:
            return "No MRZ detected"
        ocr_result = self.ocr_mrz(image, bbox)
        
        return self.hamming_closest(ocr_result[2:5])[0] if ocr_result else "No OCR result"
    
    def hamming_closest(self, predicted):
        best, best_dist = None, float("inf")
        for code in self.country_codes:
            # pad or truncate to length 3 if needed
            p = (predicted + "<<<")[:3]
            dist = sum(pc != cc for pc, cc in zip(p.upper(), code))
            if dist < best_dist:
                best_dist, best = dist, code
        return best, best_dist


if __name__ == "__main__":
    # names2models = {
    #     'passporteye': None,
    #     'docsaidlab': MRZScanner(),
    #     'serdarhelli': None,
    # }

    # model = MRZReader(
    #     facedetection_protxt = str(ROOT_PATH / "weights/face_detector/deploy.prototxt"),
    #     facedetection_caffemodel = str(ROOT_PATH / "weights/face_detector/res10_300x300_ssd_iter_140000.caffemodel"),
    #     segmentation_model = str(ROOT_PATH / "weights/mrz_detector/mrz_seg.tflite"),
    #     easy_ocr_params = { "lang_list": ["en"], "gpu": True }
    # )

    exp = get_exp(EXP_FILE, None)
    model = exp.get_model().to(DEVICE)
    model.eval()

    ckpt = torch.load(WEIGHTS_PTH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt)


    # ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log=False)
    ocr = MRZScanner(
        model_type=ModelType.recognition,
        recognition_cfg='20250221'  # Latest recognition model version
        )
    ocr = MRZScanner()
    # ocr = None
    # validator = SerdarHelliCountry('serdarhelli', model, DATA_PATH)
    validator = YOLODocsaidCountry('boldyshev', model, ocr, DATA_PATH)

    output_path = ROOT_PATH / 'results' / f'validation_{validator.name}_hun.json'
    # country_list = sorted([p.name for p in DATA_PATH.iterdir()])
    country_list = ['HUN']
    results = validator.run(country_list, output_path)

    print(f"Validation results saved to {output_path}")