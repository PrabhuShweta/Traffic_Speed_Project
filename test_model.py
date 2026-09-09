import os
from ultralytics import YOLO

def main():
    # 1. Path to your trained model weights from yesterday
    model_path = "runs/detect/traffic_model/custom_yolo/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Could not find the model at {model_path}. Please check the path.")
        return

    print(f"[INFO] Loading custom trained model from: {model_path}")
    model = YOLO(model_path)

    # 2. Path to your test video
    video_path = "videos/highway_test.mp4"
    
    print(f"[INFO] Running inference on video: {video_path}")

    # 3. Run object detection
    # conf=0.4: Only show detections the AI is at least 40% confident about
    # show=True: Opens a window to play the video live with boxes
    # save=True: Saves a copy of the processed video
    results = model.predict(
        source=video_path,
        conf=0.4,
        show=True,
        save=True
    )

    print("[INFO] Testing complete!")
    print("[INFO] The processed video with bounding boxes is saved inside the 'runs/detect/' folder.")

if __name__ == '__main__':
    main()