from ultralytics import YOLO

def main():
    # Updated to the exact path you found, using forward slashes
    model = YOLO("runs/detect/traffic_model/custom_yolo/weights/last.pt") 

    print("[INFO] Resuming training from the last saved epoch...")
    model.train(resume=True)

if __name__ == '__main__':
    main()