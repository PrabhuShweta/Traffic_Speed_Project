from collections import defaultdict, deque
import cv2
import numpy as np

class ViewTransformer:
    def __init__(self, source: np.ndarray, target_width: float, target_height: float):
        # Maps the 4 pixel points to a 2D top-down grid in real meters
        target = np.float32([
            [0, 0],
            [target_width, 0],
            [target_width, target_height],
            [0, target_height],
        ])
        self.source = np.float32(source)
        self.m = cv2.getPerspectiveTransform(self.source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2))
        
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed.reshape(-1, 2)


class SpeedTracker:
    def __init__(self, fps: float, transformer: ViewTransformer):
        self.fps = fps
        self.transformer = transformer
        # Stores the history of vehicle coordinates (keeping exactly 1 second of frames)
        self.coordinates = defaultdict(lambda: deque(maxlen=int(fps)))

    def update_and_calculate_speed(self, tracker_id: int, anchor_point: tuple) -> float:
        # Convert the vehicle's pixel location to real-world meters
        transformed_pt = self.transformer.transform_points(np.array([anchor_point]))[0]
        self.coordinates[tracker_id].append(transformed_pt)

        # Wait until we have half a second of tracking data before calculating speed
        min_frames = int(self.fps / 2)
        if len(self.coordinates[tracker_id]) < min_frames:
            return 0.0

        start_pt = self.coordinates[tracker_id][0]
        end_pt = self.coordinates[tracker_id][-1]

        # Calculate Euclidean distance in meters
        distance_meters = np.linalg.norm(end_pt - start_pt)
        time_seconds = len(self.coordinates[tracker_id]) / self.fps

        # Convert m/s to km/h
        speed_kmh = (distance_meters / time_seconds) * 3.6
        return round(float(speed_kmh), 1)