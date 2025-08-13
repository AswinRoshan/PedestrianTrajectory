# PedestrianTrajectory — YOLOv8 & YOLOv10 Pedestrian Trajectory Detection

Trains YOLOv8 and YOLOv10 models on Roboflow-prepared pedestrian datasets, then uses the trained models to detect pedestrians in video and extract their movement trajectories through a common pipeline.

---

## Features

* Train YOLOv8 or YOLOv10 models on Roboflow-exported pedestrian datasets directly from Python scripts.
* Automatically load the latest trained weights for each YOLO version.
* Detect pedestrians and log (x1, y1, x2, y2) bounding box coordinates per frame.
* Extract and save movement trajectories from detections.
* Save processed video and trajectory data to Google Drive or local disk.

---

## Project Structure

```
PedestrianTrajectory/
│
├── Scripts/
│   ├── yolov8_train.py      # Train YOLOv8 model
│   ├── yolov10_train.py     # Train YOLOv10 model
│   ├── trajectory.py        # Inference script for both YOLOv8 and YOLOv10
│
├── requirements.txt         # Python dependencies
├── README.md                # Documentation
├── .gitignore               # Ignore large files and cache
```

---

## Installation

```bash
git clone https://github.com/YourUser/PedestrianTrajectory.git
cd PedestrianTrajectory

python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Training

**YOLOv8**

```bash
python Scripts/yolov8_train.py
```

**YOLOv10**

```bash
python Scripts/yolov10_train.py
```

---

## Running Trajectory Detection

1. Place trained runs in:

   * YOLOv8 → `runs/train/v8Coordinates...`
   * YOLOv10 → `runs/train/V10Coordinates...`

2. Edit `trajectory.py`:

   ```python
   MODEL_VERSION = "v8"  # or "v10"
   ```

3. Run:

   ```bash
   python Scripts/trajectory.py
   ```
