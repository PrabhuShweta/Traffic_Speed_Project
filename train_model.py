import os
from ultralytics import YOLO

def main():
    # 1. Get the absolute path to your data.yaml file
    # YOLO often fails if it is given a relative path, so we force an absolute path here.
    yaml_path = os.path.abspath("dataset/data.yaml")

    print(f"[INFO] Using dataset configuration at: {yaml_path}")

    # 2. Load the base YOLO model (Nano version for high-speed inference)
    print("[INFO] Loading pre-trained YOLOv8 model...")
    model = YOLO("yolov8n.pt") 

    # 3. Start Training
    print("[INFO] Starting training phase...")
    results = model.train(
        data=yaml_path,
        epochs=30,                  # 30 epochs is a good starting point for a small dataset
        imgsz=640,                  # Standard YOLO image resolution
        batch=16,                   # Number of images processed at once (lower to 8 if RAM is low)
        device="cpu",               # Use "cpu" for standard laptops. Change to 0 if you have an NVIDIA GPU.
        project="traffic_model",    # Master folder for saving runs
        name="custom_yolo",         # Subfolder for this specific training session
        plots=True                  # Automatically generate evaluation charts
    )

    print("[INFO] Training Complete!")
    print("[INFO] Your custom weights are saved at: traffic_model/custom_yolo/weights/best.pt")

if __name__ == '__main__':
    main()