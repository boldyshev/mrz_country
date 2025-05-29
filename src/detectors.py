import os
import sys
from pathlib import Path


import cv2
from skimage import io

import torch

from mrzscanner import MRZScanner
from passporteye.mrz.image import MRZPipeline

ROOT_PATH = Path(__file__).resolve().parent.parent

_ = torch.hub.load(
    "Megvii-BaseDetection/YOLOX",
    "yolox_s",
    pretrained=False,
    trust_repo=True
)

hub_dir = torch.hub.get_dir()
repo_name = "Megvii-BaseDetection_YOLOX_main"
repo_path = os.path.join(hub_dir, repo_name)

if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from yolox.utils import postprocess
from yolox.data.data_augment import preproc
from yolox.exp import get_exp


class Detector:
    """Detector Interface"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = self.load_model()

    def load_model(self):
        raise NotImplementedError("Subclasses should implement this method")

    def mrz_crop(self, image, bbox):
        """
        Crop the MRZ region from the image based on the bounding box.
        """
        x0, y0, x1, y1 = bbox
        return image[y0:y1, x0:x1]


class YOLOX(Detector):
    def __init__(self, cfg):
        super().__init__(cfg)
        _ = torch.hub.load(
            "Megvii-BaseDetection/YOLOX", 
            "yolox_s", 
            pretrained=False, 
            trust_repo=True
        )



        self.postprocess = postprocess
        self.preproc = preproc

    def load_model(self):
        """
        Load the YOLOX model from a given path.
        """

        exp = get_exp(self.cfg.exp_file, None)
        model = exp.get_model().to(self.cfg.device)
        checkpoint = torch.load(self.cfg.weights_path, map_location=self.cfg.device, weights_only=False)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def __call__(self, image_path):
        img = cv2.imread(image_path)
        orig_img = img.copy()
        tile_size = (self.cfg.tile_size, self.cfg.tile_size)
        img_resized, ratio = preproc(img, tile_size)
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).float().to(self.cfg.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            outputs = postprocess(outputs, 1, self.cfg.confidence, self.cfg.non_max_suppression)

        if outputs[0] is None:
            return None
        
        output = outputs[0].cpu().numpy()[0]
        x0, y0, x1, y1, score, cls_id, _ = output
        bbox = [x0, y0, x1, y1]
        bbox = [round(coord / ratio) for coord in bbox]
        return orig_img, bbox

class DocsaidLab(Detector):
    def __init__(self, cfg):
        super().__init__(cfg)

    def load_model(self):
        return MRZScanner()

    def __call__(self, image_path):
        img = io.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        orig_img = img.copy()
        result = self.model(img, do_center_crop=False, do_postprocess=False)

        mrz_polygon = result['mrz_polygon']
        bbox = tuple(mrz_polygon[0].astype(int).tolist() + mrz_polygon[-2].astype(int).tolist())
        return orig_img, bbox

class Passporteye(Detector):
    def __init__(self, cfg):
        super().__init__(cfg)

    def load_model(self):
        return MRZPipeline

    def scale_bbox_coordinates(self, small_bbox, small_img_size, large_img_size):
        """
        Scale bounding box coordinates from small image to large image

        Args:
            small_bbox: tuple/list (x1, y1, x2, y2) - coordinates in small image
            small_img_size: tuple (width, height) of small image
            large_img_size: tuple (width, height) of large image

        Returns:
            tuple (x1, y1, x2, y2) - coordinates in large image
        """
        small_width, small_height = small_img_size
        large_width, large_height = large_img_size

        # Calculate scaling factors
        width_scale = large_width / small_width
        height_scale = large_height / small_height

        # Get small image coordinates
        x1, y1, x2, y2 = small_bbox

        # Scale to large image
        large_x1 = int(round(x1) * width_scale)
        large_y1 = int(round(y1) * height_scale) - 5
        large_x2 = int(round(x2) * width_scale)
        large_y2 = int(round(y2) * height_scale) + 5

        return large_x1, large_y1, large_x2, large_y2

    def __call__(self, image_path):
        p = MRZPipeline(image_path)
        p.data
        orig_img = p['img']
        p['text']
        pbox = p['boxes'][p['box_idx']]

        mrz_polygon = pbox.as_poly()
        bbox_small = mrz_polygon[3].tolist()[::-1] + mrz_polygon[1].tolist()[::-1]
        bbox_small = [round(coord) for coord in bbox_small]
        bbox = self.scale_bbox_coordinates(bbox_small, p['img_small'].shape, orig_img.shape)

        return  orig_img, bbox









