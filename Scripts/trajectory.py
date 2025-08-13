# scripts/trajectory.py
# Run YOLO inference on a video, extract coordinates, and save output

from google.colab import drive
import os
import shutil
from ultralytics import YOLO

# USER CONFIGURATION
# Mount Google Drive (for Colab users)
drive.mount('/content/drive')

# Select model version: "v8" or "v10"
MODEL_VERSION = "v10"  # Change to "v8" for YOLOv8 runs

# Base directory for training runs
BASE_RUNS_DIR = "/content/drive/MyDrive/Pedestrian Project/runs/train"

# Path to input video
VIDEO_PATH = "/content/drive/MyDrive/Pedestrian Project/Test.mp4"

# Output folder in Drive
OUTPUT_FOLDER = "/content/drive/MyDrive/Pedestrian Project/"

# Minimum confidence threshold for detections
CONF_THRESHOLD = 0.25

def get_latest_run_path(base_dir, version_prefix):
    """Get the latest run folder path for the given YOLO version."""
    runs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith(version_prefix)]
    if not runs:
        raise FileNotFoundError(f"No runs found starting with '{version_prefix}' in {base_dir}")
    latest_run = max(runs, key=os.path.getmtime)
    return os.path.join(latest_run, "weights", "best.pt")

# Auto-detect weights path
if MODEL_VERSION == "v8":
    WEIGHTS_PATH = get_latest_run_path(BASE_RUNS_DIR, "V8Coordinates")
elif MODEL_VERSION == "v10":
    WEIGHTS_PATH = get_latest_run_path(BASE_RUNS_DIR, "V10Coordinates")
else:
    raise ValueError("MODEL_VERSION must be 'v8' or 'v10'.")

print(f"Using weights: {WEIGHTS_PATH}")
print(f"Input video: {VIDEO_PATH}")

# Load YOLO model and move to GPU (CUDA)
model = YOLO(WEIGHTS_PATH).to("cuda")

# Run inference
print("Running inference with YOLO")
results = model.predict(
    source=VIDEO_PATH,
    save=True,
    conf=CONF_THRESHOLD,
    device="cuda"  # Force GPU usage
)

# Extract coordinates for each detection
print("\n Detected coordinates per frame:")
for result in results:
    for det in result.boxes.data:
        x1, y1, x2, y2 = det[0:4]
        conf = det[4]
        cls = int(det[5])
        print(f"Coordinates: {x1}, {y1}, {x2}, {y2}, Conf: {conf:.2f}, Class: {cls}")

print("\n Inference completed, check the 'runs/detect/predict/' folder for output.")

# Save output video to Drive
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
output_video_name = os.path.basename(VIDEO_PATH).replace(".mp4", ".avi")
output_video_path = os.path.join("runs/detect/predict", output_video_name)

if os.path.exists(output_video_path):
    destination_file = os.path.join(OUTPUT_FOLDER, output_video_name)
    if os.path.exists(destination_file):
        os.remove(destination_file)
        print(f" Removed existing file: {destination_file}")
    shutil.move(output_video_path, OUTPUT_FOLDER)
    print(f" Output saved to: {OUTPUT_FOLDER}")
else:
    print(" Output video not found. Check if YOLO saved it correctly.")
