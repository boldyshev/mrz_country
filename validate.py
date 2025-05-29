import time
import json
import argparse
from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parent
CFG_PATH = ROOT_PATH / 'conf'
from tqdm import tqdm

from hydra import initialize_config_dir, compose

from src.two_stage import YOLOXDocsaidLabPipeline, DocsaidLabPipeline, PassportEyePipeline

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class Validator:
    def __init__(self, name, pipeline, data_path):
        self.name = name
        self.pipeline = pipeline
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

    def validate_country(self, country_true):
        country_path = self.data_path / country_true
        image_paths = sorted([p for p in country_path.iterdir() if p.is_file()])
        country_image_count = len(image_paths)

        if country_true == 'DEU':
            country_true = 'D<<'
        results = self.model_results()
        predicted_countries = []
        incorrect_image_paths = []
        time_country = 0
        for image_path in tqdm(image_paths, desc=f'{self.name} - {country_true}', unit='image'):
            start_time = time.perf_counter()
            pred_correct = country_true == self.pipeline.process(str(image_path))
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


def main():
    with initialize_config_dir(version_base=None, config_dir=str(CFG_PATH)):
        cfg = compose(config_name='conf.yaml')

    data_path = cfg.data_path

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_path', required=True, type=str,  help='Path to the output file')
    parser.add_argument('-c', '--country_list', nargs='+',
                        help='List of specific countries to validate. If none, will validate all the countries in the dataset')
    args = parser.parse_args()

    country_codes = [c.name if c.name != 'DEU' else 'D<<' for c in data_path.iterdir() if c.is_dir()]
    pipeline = YOLOXDocsaidLabPipeline(cfg, country_codes, postprocess=cfg.postprocess)
    validator = Validator('yolo_docsaid', pipeline, data_path)

    if args.country_list:
        country_list = args.country_list
    else:
        country_list = sorted([p.name for p in data_path.iterdir()])

    validator.run(country_list, args.out_path)

    print(f"Validation results saved to {args.out_path}")


if __name__=='__main__':
    main()
