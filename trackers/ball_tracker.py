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
        results = self.model.predict(frame,conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict
    
        # results = self.model.predict(frame, conf=0.2)[0]
        
        # ball_dict = {}
        # # Stores best detected box for the ball
        # best_box = None
        # # Stores detection that is closest to the last location of the ball
        # min_dist = float('inf')

        # # Iterate all detected boxes in the frame
        # for box in results.boxes:
        #     # Get edges, center, width and height
        #     x1, y1, x2, y2 = box.xyxy.tolist()[0]
        #     cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        #     w, h = x2 - x1, y2 - y1

        #     # Filters out boxes that are too big for the ball
        #     if w > 50 or h > 50:
        #         continue

        #     # Determine the last location of the ball
        #     if self.prev_position is not None:
        #         # Calculate the new position of the ball
        #         dist = np.linalg.norm(np.array([cx, cy]) - np.array(self.prev_position))
        #         if dist > 150:
        #             continue  # Not the ball
        #     else:
        #         dist = 0 # In case of first frame
            
        #     # Saves curent detection as the best detection so far
        #     if dist < min_dist:
        #         min_dist = dist
        #         best_box = [x1, y1, x2, y2]
        #         self.prev_position = [cx, cy]

        # if best_box:
        #     cx, cy = (best_box[0] + best_box[2]) / 2, (best_box[1] + best_box[3]) / 2
        #     z = np.array([cx, cy])
        #     self.kalman_filter.predict()
        #     self.kalman_filter.update(z)
        #     pred_x, pred_y = self.kalman_filter.x[:2]
        #     # můžeš upravit i bounding box kolem pred_x, pred_y
        #     ball_dict[1] = best_box
        # else:
        #     # If the ball is not detected, make the Kalman prediction
        #     self.kalman_filter.predict()
        #     pred_x, pred_y = self.kalman_filter.x[:2]
        #     size = 10  # Predicted size
        #     ball_dict[1] = [pred_x - size, pred_y - size, pred_x + size, pred_y + size]

        # return ball_dict

    
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
    
    def get_ball_shot_frames(self, ball_positions, court_keypoints, player_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]

        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Convert all values to float
        df_ball_positions = df_ball_positions.apply(pd.to_numeric, errors='coerce')

        # Interpolation of missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        df_ball_positions['delta_y_smooth'] = df_ball_positions['delta_y'].rolling(window=3, min_periods=1).mean()
        
        df_ball_positions['ball_hit'] = 0
        
        min_change_frames = 25

        for i in range(1, len(df_ball_positions) - int(min_change_frames*1.2)):
            negative_change = df_ball_positions['delta_y_smooth'].iloc[i] > 0 and df_ball_positions['delta_y_smooth'].iloc[i+1]
            positive_change = df_ball_positions['delta_y_smooth'].iloc[i] < 0 and df_ball_positions['delta_y_smooth'].iloc[i+1]
            
            if negative_change or positive_change:
                change_count = 0
                for change_frame in range(i+1, i+int(min_change_frames*1.2)):
                    negative_position_change_following_frame = df_ball_positions['delta_y_smooth'].iloc[i] >0 and df_ball_positions['delta_y_smooth'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y_smooth'].iloc[i] <0 and df_ball_positions['delta_y_smooth'].iloc[change_frame] >0
                    
                    if negative_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>min_change_frames-1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1
        
        df_ball_positions['group_id'] = (df_ball_positions['ball_hit'] != df_ball_positions['ball_hit'].shift()).cumsum()
        
        print(df_ball_positions[df_ball_positions['ball_hit']==1])
        print("")
        # Keep only the hit detection from the last frame of the hit group
        last_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].groupby('group_id').tail(1)
        print(last_hits)
        print("")
        df_ball_positions['ball_hit_filtered'] = 0
        df_ball_positions.loc[last_hits.index, 'ball_hit_filtered'] = 1

        # Getting the grid position by keypoints (e.g. Y-average between keypoints 6 and 7)
        # If you have 14 points, their coordinates are in an array like [x0, y0, x1, y1, ..., x13, y13]
        # The net will be approximately between points 6 (index 12-13) and 7 (index 14-15)
        net_y_position = (court_keypoints[13] + court_keypoints[15]) / 1.7  # y6 + y7

        # Determine which side of the court the ball was on (True = bottom half, False = top half)
        df_ball_positions['ball_side'] = df_ball_positions['mid_y'] > net_y_position

        # Find all detected hits
        filtered_hits = df_ball_positions[df_ball_positions['ball_hit_filtered'] == 1]
        print(f"Filtered hits {list(filtered_hits.index)}")
        print("")
        
        # Find false hits - if two in a row are on the same side of the court, we cancel the first one
        false_hits_indices = []

        # for i in range(len(filtered_hits) - 1):
        #     current_idx = filtered_hits.index[i]
        #     next_idx = filtered_hits.index[i + 1]

        #     current_side = df_ball_positions.loc[current_idx, 'ball_side']
        #     next_side = df_ball_positions.loc[next_idx, 'ball_side']

        #     if current_side == next_side:
        #         print(f"FALSEEE HIT!!!: {current_idx}, current_side: {current_side}, next_side: {next_side}")
        #         false_hits_indices.append(current_idx)
        
        print(false_hits_indices)
        print("")

        # Cancel false hits
        df_ball_positions.loc[false_hits_indices, 'ball_hit_filtered'] = 0
        
        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit_filtered'] == 1].index.tolist()
        
        if player_positions is not None:
            max_distance = 20  # Tunable parameter (pixels)

            filtered_hits = df_ball_positions[df_ball_positions['ball_hit_filtered'] == 1]
            final_hit_frames = []

            for idx in filtered_hits.index:

                if idx >= len(player_positions):
                    continue  # Avoid index error

                player_dict = player_positions[idx]
                print(f"player_dict: {player_dict}")
                ball_bbox = df_ball_positions.loc[idx, ['x1', 'y1', 'x2', 'y2']].values
                ball_center = np.array([(ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2])
                print(f"ball_center: {ball_center}")
                
                closest_player_bbox = None
                closest_distance_y = float('inf')
                ball_y = ball_center[1]

                for bbox in player_dict.values():
                    player_y = (bbox[1] + bbox[3]) / 2
                    y_distance = abs(player_y - ball_y)
                    if y_distance < closest_distance_y:
                        closest_distance_y = y_distance
                        closest_player_bbox = bbox

                print(f"idx: {idx}, closest_player_bbox: {closest_player_bbox}")

                if closest_player_bbox is not None:
                    # Extract edges of the player and ball bounding boxes
                    px1, py1, px2, py2 = closest_player_bbox
                    bx1, by1, bx2, by2 = ball_bbox

                    # Calculate horizontal and vertical distances between the boxes
                    dx = max(0, max(px1 - bx2, bx1 - px2))
                    dy = max(0, max(py1 - by2, by1 - py2))

                    # Compute shortest edge-to-edge distance (0 means overlapping)
                    distance = np.hypot(dx, dy)
                    print(f"tohle je distance {distance}")
                    if distance <= max_distance:
                        final_hit_frames.append(idx)
      
                print(" ")
                
            print(f"final_hit_frames: {final_hit_frames}")
            
            for i in range(len(final_hit_frames) - 1):
                current_idx = filtered_hits.index[i]
                next_idx = filtered_hits.index[i + 1]

                current_side = df_ball_positions.loc[current_idx, 'ball_side']
                next_side = df_ball_positions.loc[next_idx, 'ball_side']

                if current_side == next_side:
                    final_hit_frames.remove(current_idx)
            
            print(false_hits_indices)
            print("")

            return final_hit_frames
        
        return frame_nums_with_ball_hits