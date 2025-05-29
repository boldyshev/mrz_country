# exps/custom/mrz_tiny.py

import os
import sys

import torch
hub_dir = torch.hub.get_dir()
repo_name = "Megvii-BaseDetection_YOLOX_main"
repo_path = os.path.join(hub_dir, repo_name)

if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # 1) Number of object classes (just MRZ = 1)
        self.num_classes = 1

        # 2) Path to your dataset root
        #    Inside it you should have images/train, images/val, annotations/train.json, annotations/val.json
        self.data_dir = '/home/boris/syn/dataspark/dataset_coco'

        # 3) Input image size for training and eval (height, width)
        #    You can adjust to something like (640,640) or (512,512)
        self.input_size = (640, 640)

        # 4) Batch size per GPU
        self.train_batch_size = 16
        self.eval_batch_size = 16

        # 5) Learning rate—YOLOX uses a warmup + cosine schedule by default
        #    Typically lr = 0.01 * (batch_size / 64)
        self.basic_lr_per_img = 0.01 / 64.0

        # 6) Annotation files (COCO format)
        self.train_ann = os.path.join(self.data_dir, "annotations", "train.json")
        self.val_ann = os.path.join(self.data_dir, "annotations", "val.json")

        # 7) Data loader num_workers
        self.data_num_workers = 4

        # 8) Experiment name (where logs and weights will be saved under YOLOX_outputs/)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.cache = False
    # (You can also override other methods, e.g. to adjust augmentations or lr schedule,
    #  but for a first run, the defaults are fine.)
