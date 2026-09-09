import cv2
import yaml
import numpy as np
from ultralytics import YOLO
import supervision as sv
from utils.speed_calculator import ViewTransformer, SpeedTracker

def main():
    # 1. Load configurations
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Initialize AI and Video components
    model = YOLO(config["model_path"])
    video_info = sv.VideoInfo.from_video_path(config["video_source"])
    frames_generator = sv.get_video_frames_generator(config["video_source"])

    # 3. Initialize Tracking and Speed modules
    tracker = sv.ByteTrack()
    transformer = ViewTransformer(
        source=np.array(config["source_polygon"]),
        target_width=config["real_world_width"],
        target_height=config["real_world_height"]
    )
    speed_tracker = SpeedTracker(fps=video_info.fps, transformer=transformer)

    # 4. Initialize UI Annotators 
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    
    # Define the Region of Interest (ROI) polygon on screen
    polygon_zone = sv.PolygonZone(polygon=np.array(config["source_polygon"]))
    zone_annotator = sv.PolygonZoneAnnotator(zone=polygon_zone, color=sv.Color.RED, thickness=2)

    print("[INFO] Starting real-time traffic processing...")
    
    for frame in frames_generator:
        # Detect objects
        result = model(frame, conf=config["confidence_threshold"])[0]
        detections = sv.Detections.from_ultralytics(result)

        # Filter out non-target classes (e.g., pedestrians) and apply tracking
        detections = detections[np.isin(detections.class_id, config["target_classes"])]
        detections = tracker.update_with_detections(detections)

        labels = []
        for i in range(len(detections)):
            xyxy = detections.xyxy[i]
            class_id = detections.class_id[i]
            track_id = detections.tracker_id[i]

            # Clean up messy dataset labels (e.g., "carrotation" -> "Car")
            raw_name = model.names[class_id]
            clean_name = raw_name.replace("rotation", "").replace("crop", "").capitalize()

            # Calculate center-bottom of bounding box for accurate speed tracking
            x1, y1, x2, y2 = xyxy
            anchor_point = ((x1 + x2) / 2, y2)
            
            # Calculate Speed
            speed = speed_tracker.update_and_calculate_speed(track_id, anchor_point)
            
            # Format label and flag violations
            if speed > config["speed_limit_kmh"]:
                labels.append(f"VIOLATION: {clean_name} {speed} km/h")
            else:
                labels.append(f"#{track_id} {clean_name} {speed} km/h")

        # Draw all graphics on the frame
        annotated_frame = frame.copy()
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        cv2.imshow("Real-Time Speed Enforcement", annotated_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()