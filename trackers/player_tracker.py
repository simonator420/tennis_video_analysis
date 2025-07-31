from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def choose_and_filter_players(self, court_keypoints, player_detections, frames):
        id_counts = {}
        first_frame = None
        court_center = (
            sum(court_keypoints[0::2]) / (len(court_keypoints) // 2),
            sum(court_keypoints[1::2]) / (len(court_keypoints) // 2)
        )

        
        if player_detections:
            # Assume you have the original frames in a variable 'frames'
            first_frame = frames[0].copy()
            for track_id, bbox in player_detections[0].items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(first_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(first_frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.imwrite("debug_first_frame_raw_detections.jpg", first_frame)
            
            
        # Analyzing first frame (adjust to more frames for more accurate monitoring)
        for player_dict in player_detections[:1]:
            for track_id, bbox in player_dict.items():
                player_center = get_center_of_bbox(bbox)

                # Measure distance to the center of the court
                distance_to_center = measure_distance(player_center, court_center)

                print(f"Player {track_id}, distance to court center: {distance_to_center:.2f}")

                # Only count players close enough to the center
                id_counts[track_id] = id_counts.get(track_id, 0) + 1


        # Select two most frequent IDs
        chosen_players = sorted(id_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        print(f"Chosen players (raw IDs): {chosen_players}")
        chosen_players = [x[0] for x in chosen_players]

        # Filtering the rest of the frames
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        
        # Assure that player ids are 1 and 2
        for d in filtered_player_detections:
            keys = list(d.keys())
            if len(keys) > 0:
                d[1] = d.pop(keys[0])
            if len(keys) > 1:
                d[2] = d.pop(keys[1])
        
        # print(filtered_player_detections[-1])
        
        # Determine which player is on upper side of court
        # player_1_y = filtered_player_detections[0][1][1] + filtered_player_detections[0][1][3]
        # player_2_y = filtered_player_detections[0][2][1] + filtered_player_detections[0][2][3]
            
        # upper_player = lambda player_1_y, player_2_y: 1 if player_1_y > player_2_y else 2
        # print(f"Tohle je upper player: {upper_player(player_1_y, player_2_y)}")
        
        return filtered_player_detections
        
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []
        
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections
        
    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0] # Persist ensures the player is tracked consistently
        id_name_dict = results.names
        
        player_dict = {}
        # Iterate over all detected objects
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            # Coordinates of the bounding box
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            # Name of the bounding box
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
                
        return player_dict
    
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames