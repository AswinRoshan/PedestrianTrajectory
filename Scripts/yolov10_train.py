# scripts/yolov10_train.py
# Train YOLOv8 on a Roboflow YOLO export

from ultralytics import YOLO
import os

# EDIT THESE PATHS/SETTINGS
DATA_YAML = "/content/drive/My Drive/Pedestrian Project/YourExport/data.yaml"  # CHANGE: path to your Roboflow export's data.yaml
MODEL     = "yolov10m.pt"       # CHANGE: e.g., 'yolov10n.pt', 'yolov10m.pt', or a custom .pt
EPOCHS    = 100                # CHANGE: training epochs
IMGSZ     = 640                # CHANGE: training image size
RUN_NAME  = "V10Coordinates"    # CHANGE: run name shown under runs/train/

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
# dst = "/content/drive/MyDrive/Pedestrian/V10Coordinates"
# os.makedirs(dst, exist_ok=True)
# shutil.copytree(src, dst, dirs_exist_ok=True)
# print("Folder saved to", dst)
