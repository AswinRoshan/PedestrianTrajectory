# scripts/yolov8_train.py
# Train YOLOv8 on a Roboflow YOLO export

from ultralytics import YOLO
import os

# EDIT THESE PATHS/SETTINGS
DATA_YAML = "/content/drive/My Drive/Pedestrian Project/YourExport/data.yaml"  # CHANGE: path to your Roboflow export's data.yaml
MODEL     = "yolov8m.pt"       # CHANGE: e.g., 'yolov8n.pt', 'yolov8m.pt', or a custom .pt
EPOCHS    = 100                # CHANGE: training epochs
IMGSZ     = 640                # CHANGE: training image size
RUN_NAME  = "V8Coordinates"    # CHANGE: run name shown under runs/train/

model = YOLO(MODEL)
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    project="runs/train",
    name=RUN_NAME,
    plots=True
)

best = os.path.join(results.save_dir, "weights", "best.pt")
print("Training complete, best model:", best)

# Optional: copy results back to Drive (uncomment when running in Colab)
# import shutil
# src = f"/content/runs/train/{RUN_NAME}"
# dst = "/content/drive/MyDrive/Pedestrian/V8Coordinates"
# os.makedirs(dst, exist_ok=True)
# shutil.copytree(src, dst, dirs_exist_ok=True)
# print("Folder saved to", dst)
