import json
import argparse
from pathlib import Path

from tqdm import tqdm

import sys
from hydra import initialize_config_dir, compose

from src.two_stage import YOLOXDocsaidLabPipeline

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

ROOT_PATH = Path(__file__).resolve().parent
CFG_PATH = ROOT_PATH / 'conf'

def main():
    with initialize_config_dir(version_base=None, config_dir=str(CFG_PATH)):
        cfg = compose(config_name='conf.yaml')

    data_path = Path(cfg.data_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--image_dir', required=True, type=str,  help='Path to the directory with input images')
    parser.add_argument('-o', '--out_path', type=str,  help='Path to the output JSON file')
    args = parser.parse_args()

    country_codes = [c.name if c.name != 'DEU' else 'D<<' for c in data_path.iterdir() if c.is_dir()]
    pipeline = YOLOXDocsaidLabPipeline(cfg, country_codes, postprocess=cfg.postprocess)
    image_paths = sorted([str(p) for p in Path(args.image_dir).iterdir() if p.is_file()])

    results = {}
    for image_path in tqdm(image_paths, desc=f'{args.image_dir.split("/")[-1]}', unit='image'):
        country_pred = pipeline.process(image_path)
        results[str(image_path)] = country_pred

    with open(args.out_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__=='__main__':
    main()
