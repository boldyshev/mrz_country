# Passport Country Detection using MRZ Detection and OCR

This project implements a machine learning pipeline to identify the country of origin from synthetic passport images. It employs a two-stage approach: (1) detecting the Machine Readable Zone (MRZ) using a fine-tuned YOLOX-S model, and (2) recognizing text within the MRZ using a pre-trained OCR model to determine the country code.

* No mistakes on the dataset after using Hamming distance postrocessing
* Average 0.45 sec inference time on 12/24 core CPU and 3090 GPU


### Quick Start
```bash
git clone https://github.com/boldyshev/mrz_country
cd mrz_country
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir weights
gdown 1iJgdk4Ry7Gm4Rjem3_qaifnDlFFpmr3H -O weights/yolox-s.pth
```

Predict countries for all images in the given directory. Outputs JSON with `image_path: country_code`
```
python3 predict.py --image_dir <DATASET>/HUN --out_path <OUT_PATH>.json
```

Validate the pipepline on the whole dataset. Outputs JSON file
```
python3 predict.py --out_path <OUT_PATH>.json
```


## Methodology

### MRZ Detection
- **Model**: YOLOX-S, a lightweight object detection model, was chosen for its balance of accuracy and speed, meeting the inference time requirement of less than 1 second per image.
- **Rationale**: Direct image classification was avoided as it wouldn't localize the MRZ, which is critical for country code extraction.

### Text Recognition
- **Model**: The pre-trained MRZScanner from DocsaidLab was selected after comparing it with PassportEye and PaddleOCR.
- **Comparison**: DocsaidLab outperformed PaddleOCR, which struggled with '<' characters, and provided better accuracy than PassportEye on the dataset.

### Post-processing
- **Approach**: A Hamming distance-based correction adjusts misrecognized three-letter country codes by finding the closest match from a list of 24 predefined codes (e.g., 'HHN' corrected to 'HUN').

## Performance Requirements
- **Inference Time**: The pipeline, using YOLOX-S and DocsaidLab's OCR, achieves inference times below 0.5 second per image on 12/24 Core CPU and RTX 3090 GPU.
- **Scalability**: The modular design supports adaptation to hundreds of countries and various document types by retraining the detection model on new datasets and updating the country code list.

## Data Preparation
- **Labeling**: MRZ bounding boxes were labeled using predictions from DocsaidLab and PassportEye, refined manually where necessary.
- **Format**: The dataset was converted to COCO format for compatibility with YOLOX training.
- **Enhancement**: Open-source models and Visual Language Models can accelerate labeling for new datasets.

## Model Training
- **Framework**: PyTorch, as specified in the task.
- **Detection Model**: YOLOX-S was fine-tuned for 50 epochs with default hyperparameters from the repository (https://github.com/Megvii-BaseDetection/YOLOX).
- **Performance**: Achieved approximately 90 mAP on the validation set.

## Post-processing
- **Method**: Misrecognized country codes are corrected by calculating the Hamming distance to a predefined list of 24 codes.
- **Example**: If the OCR outputs 'HHN', the pipeline compares it to all codes, identifies 'HUN' (Hungary) as the closest match (distance of 1), and selects it as the result.


## Results and Evaluation
It is hardly correct to use a validation dataset as the recognition model was not trained. 
- **Strengths**: High MRZ detection accuracy (~90 mAP) and effective country code correction.
- **Challenges**: Errors persist with certain country codes (e.g., Hungarian 'HUN') due to character misrecognition.


## Challenges: 
Errors persist with certain country codes (e.g., Hungarian 'HUN') due to character misrecognition.
The pipeline relies on a predefined list of country codes, which may require updates for broader applicability.
The aspects of deploying in production were not addressed at all in this solution. 


