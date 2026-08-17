AI-Based Smart Waste Management

This project is an AI-based waste detection system developed as part of my internship work at the 5G Use Case Lab, IIT Jammu.

## What it does

The system uses **YOLOv8** and a camera to detect waste objects. The trained model can identify different waste-related objects and display their detections on the camera feed.

## Tools Used

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* Roboflow
* GitHub

## Dataset

The dataset was obtained from **Roboflow** and contains 20 waste-related classes.

**Original source:**
https://universe.roboflow.com/5g-lab/waste-detection-saa1-kdi39/dataset/1

The complete dataset is kept separately on Google Drive instead of being uploaded to GitHub.

**Dataset:**
PASTE YOUR OPEN-ACCESS GOOGLE DRIVE LINK HERE

## Model Training

The YOLOv8n pretrained model was trained for **50 epochs** using the dataset configuration in `data.yaml`.

Main training settings:

```text
Model: YOLOv8n
Epochs: 50
Image size: 640
Batch size: 16
Device: CPU
```

The training code is available in `train.py`.

The trained model and training results are preserved in:

```text
runs/detect/train-2/
```

The best trained model is:

```text
best.pt
```

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the main detection program:

```bash
python main.py
```

For camera testing:

```bash
python camera_test.py
```

## Main Files

* `main.py` — main detection program
* `camera_test.py` — camera testing
* `train.py` — model training
* `data.yaml` — dataset configuration
* `requirements.txt` — required packages
* `yolov8n.pt` — pretrained YOLOv8 model
* `runs/detect/train-2/` — training results

## Note

The dataset is stored separately on Google Drive because of its size. The GitHub repository contains the project code, configuration, training information and relevant results.

