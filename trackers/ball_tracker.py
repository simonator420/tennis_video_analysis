from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
from filterpy.kalman import KalmanFilter
import numpy as np


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.kalman_filter = self.init_kalman_filter()
        self.prev_position = None # Last position to prevent fake detections like shoe, towel, e.g.

    # For smoother detection of the ball when it's not being detected
    def init_kalman_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        # State of position and speed
        kf.F = np.array([[1, 0, 1, 0], # x' = x + dx
                         [0, 1, 0, 1], # y' = y + dy
                         [0, 0, 1, 0], # dx' = dx
                         [0, 0, 0, 1]]) # dy' = dy
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.R *= 10  # measurement noise
        kf.P *= 10  # initial uncertainty
        kf.Q *= 0.01  # process noise
        return kf

        
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Convert all values to float
        df_ball_positions = df_ball_positions.apply(pd.to_numeric, errors='coerce')

        # Interpolation of missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # Converting back to format [{1: [x1, y1, x2, y2]}, ...]
        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

        
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []
        
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections
        
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections
        
    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.2)[0]
        
        ball_dict = {}
        # Stores best detected box for the ball
        best_box = None
        # Stores detection that is closest to the last location of the ball
        min_dist = float('inf')

        # Iterate all detected boxes in the frame
        for box in results.boxes:
            # Get edges, center, width and height
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1

            # Filters out boxes that are too big for the ball
            if w > 50 or h > 50:
                continue

            # Determine the last location of the ball
            if self.prev_position is not None:
                # Calculate the new position of the ball
                dist = np.linalg.norm(np.array([cx, cy]) - np.array(self.prev_position))
                if dist > 150:
                    continue  # Not the ball
            else:
                dist = 0 # In case of first frame
            
            # Saves curent detection as the best detection so far
            if dist < min_dist:
                min_dist = dist
                best_box = [x1, y1, x2, y2]
                self.prev_position = [cx, cy]

        if best_box:
            cx, cy = (best_box[0] + best_box[2]) / 2, (best_box[1] + best_box[3]) / 2
            z = np.array([cx, cy])
            self.kalman_filter.predict()
            self.kalman_filter.update(z)
            pred_x, pred_y = self.kalman_filter.x[:2]
            # můžeš upravit i bounding box kolem pred_x, pred_y
            ball_dict[1] = best_box
        else:
            # If the ball is not detected, make the Kalman prediction
            self.kalman_filter.predict()
            pred_x, pred_y = self.kalman_filter.x[:2]
            size = 10  # Predicted size
            ball_dict[1] = [pred_x - size, pred_y - size, pred_x + size, pred_y + size]

        return ball_dict

    
    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames