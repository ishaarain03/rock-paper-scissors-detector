# Rock-Paper-Scissors Detector

A YOLO11-based object detection model fine-tuned to identify **Rock**, **Paper**, and **Scissors** hand gestures in images.

## Overview

This project fine-tunes a pretrained [YOLO11](https://github.com/ultralytics/ultralytics) (small variant) model on a custom Rock-Paper-Scissors dataset sourced from [Roboflow](https://roboflow.com/). The model detects hand gestures in images and draws a bounding box around each detected gesture along with its predicted class and confidence score.

## Dataset

- **Source:** Roboflow — [Rock Paper Scissors SXSW dataset](https://universe.roboflow.com/)
- **Size:** 3,129 images
- **Classes:** `Rock`, `Paper`, `Scissors`
- **Format:** YOLOv11 annotation format

## Model

- **Base model:** YOLO11s (pretrained on COCO)
- **Fine-tuning:** 10 epochs, image size 416x416, batch size 8
- **Hardware:** Trained on CPU (Intel Core i5-8365U)

## Results

Validation performance on 604 held-out images:

| Class    | Precision | Recall | mAP50 | mAP50-95 |
|----------|-----------|--------|-------|----------|
| Paper    | 0.810     | 0.829  | 0.874 | 0.635    |
| Rock     | 0.901     | 0.850  | 0.876 | 0.648    |
| Scissors | 0.787     | 0.879  | 0.896 | 0.674    |
| **All**  | **0.833** | **0.852** | **0.882** | **0.652** |

## Project Structure

```
rock-paper-scissors-detector/
├── rock_paper_scissors.ipynb   # Main notebook: dataset download, training, validation, inference
├── best.pt                    # Fine-tuned model weights
├── requirements.txt           # Python dependencies
├── sample_results/            # Example predictions on test images
└── README.md
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ishaarain03/RockPaperScissor.git
   cd RockPaperScissor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Get a free API key from [Roboflow](https://app.roboflow.com/) if you want to re-download the dataset and retrain.

## Usage

### Run inference with the trained model
```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(source="path/to/your/image.jpg", conf=0.25, save=True)
```

### Retrain from scratch
Open `rock_paper_scissor.ipynb` and run the cells in order:
1. Download the dataset from Roboflow (requires your own API key)
2. Train the model
3. Validate the model
4. Run inference on test images

## Notes

- Training was done on CPU, so image size and batch size were kept small (`imgsz=416`, `batch=8`) to avoid memory issues. A GPU will train significantly faster and can use larger settings.
- The `datasets/` and `runs/` folders are excluded from this repository (see `.gitignore`) since they contain large generated files.

## Acknowledgements

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/) for dataset hosting and annotation tools

## Author

**Isha Saleem**
