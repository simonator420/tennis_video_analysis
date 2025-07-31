import cv2
import sys
sys.path.append('../')
import constants
from utils import (
    convert_pixel_distance_to_meters,
    convert_meters_to_pixel_distance,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)
import numpy as np

class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 320
        self.drawing_rectangle_height = 665
        self.buffer = 55 # space from the edge to the rectangle
        self.padding_court = 35 # space from the mini court to the rectangle
        
        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()
        
    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                                )
    
    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        # point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points
    
    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            (12,13),
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]
    
    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + 60
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        
    def set_canvas_background_box_position(self,frame):
        frame= frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height
        
    def draw_court(self, frame):
        # Draw the points
        # for i in range(0, len(self.drawing_key_points), 2):
        #     x = int(self.drawing_key_points[i])
        #     y = int(self.drawing_key_points[i+1])
        #     cv2.circle(frame, (x,y),5, (0,0,255),-1)
            
        # Draw the lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 2)
        
        # Draw the net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))      
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))  
        cv2.line(frame, net_start_point, net_end_point, (255, 255, 255), 2)
        
        return frame
    
    def draw_background_rectangle(self,frame,dominant_color):
        shapes = np.zeros_like(frame,np.uint8)
        
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0])), cv2.FILLED)
        # cv2.rectangle(frame, (self.start_x, self.start_y), (self.end_x, self.end_y), (int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0])), cv2.FILLED)
        out = frame.copy()
        alpha=0.07
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        
        return out
        # return frame

    def draw_mini_court(self,frames,background_color):
        output_frames = []

        for frame in frames:
            frame = self.draw_background_rectangle(frame, background_color)
            frame = self.draw_court(frame)
            output_frames.append(frame)
            
        return output_frames
    
    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points
    
    def get_mini_court_coordinates(self,
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   player_height_in_meters
                                   ):
        
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Convert pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )
        
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                                player_height_in_meters,
                                                                                player_height_in_pixels
                                                                          )
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_coourt_keypoint = ( self.drawing_key_points[closest_key_point_index*2],
                                        self.drawing_key_points[closest_key_point_index*2+1]
                                        )
        
        mini_court_player_position = (closest_mini_coourt_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1]+mini_court_y_distance_pixels
                                        )

        return  mini_court_player_position

    
    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, ball_hits, original_court_key_points):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS,
        }

        output_player_boxes = []
        output_ball_boxes = []
        previous_positions = {}
        
        for hit in ball_hits:
            print(f"Ball hit index {hit}, ball box v tom bode {ball_boxes[hit]} player_boxes v tom bode {player_boxes[hit]}")

        for i, player_bbox in enumerate(player_boxes):
        
            ball_box = ball_boxes[i][1]
            ball_position = get_center_of_bbox(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_position, get_center_of_bbox(player_bbox[x])))

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Get closest keypoint in pixels
                # print(f"Player id {player_id}")
                closest_key_point_index = get_closest_keypoint_index(foot_position,original_court_key_points, [0,2,12,13])
                closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                     original_court_key_points[closest_key_point_index*2+1])

                # Get player height in pixels
                frame_index_min = max(0, i-20)
                frame_index_max = min(len(player_boxes), i+50)
                
                # print(f"Player box {player_boxes}")
                
                bboxes_heights_in_pixels = [
                    get_height_of_bbox(player_boxes[i][player_id])
                    for i in range(frame_index_min, frame_index_max)
                    if player_id in player_boxes[i]
                ]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                
                if player_id in previous_positions:
                    prev_pos = previous_positions[player_id]
                    dx = mini_court_player_position[0] - prev_pos[0]
                    dy = mini_court_player_position[1] - prev_pos[1]
                    distance = (dx**2 + dy**2)**0.5
                    if distance > 30:  # práh můžeš upravit dle potřeby
                        print(f"POZOR: hráč {player_id} udělal skok {distance:.2f} px mezi snímky {i-1} a {i}")
                
                previous_positions[player_id] = mini_court_player_position
                output_player_bboxes_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    # Get The closest keypoint in pixels
                    closest_key_point_index = get_closest_keypoint_index(ball_position,original_court_key_points, [0,2,12,13])
                    closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                        original_court_key_points[closest_key_point_index*2+1])
                    
                    mini_court_player_position = self.get_mini_court_coordinates(ball_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                    output_ball_boxes.append({1:mini_court_player_position})
            output_player_boxes.append(output_player_bboxes_dict)
        
        ball_line_segments = []

        for i in range(len(ball_hits) - 1):
            start_frame = ball_hits[i]
            end_frame = ball_hits[i + 1]

            # Getting the ball position from the beggining of the frame to its end
            ball_start = ball_boxes[start_frame][1]
            ball_end = ball_boxes[end_frame][1]
            ball_pos_start = get_center_of_bbox(ball_start)
            ball_pos_end = get_center_of_bbox(ball_end)

            # Zjistíme, který hráč je blíže míčku v daném framu, abychom získali správnou výšku
            player_bbox_start = player_boxes[start_frame]
            closest_player_start = min(player_bbox_start.keys(), key=lambda x: measure_distance(ball_pos_start, get_center_of_bbox(player_bbox_start[x])))
            player_height_pixels_start = get_height_of_bbox(player_bbox_start[closest_player_start])
            closest_kp_start_idx = get_closest_keypoint_index(ball_pos_start, original_court_key_points, [0,2,12,13])
            closest_kp_start = (original_court_key_points[closest_kp_start_idx * 2], original_court_key_points[closest_kp_start_idx * 2 + 1])
            mini_pos_start = self.get_mini_court_coordinates(ball_pos_start, closest_kp_start, closest_kp_start_idx,
                                                            player_height_pixels_start, player_heights[closest_player_start])

            player_bbox_end = player_boxes[end_frame]
            closest_player_end = min(player_bbox_end.keys(), key=lambda x: measure_distance(ball_pos_end, get_center_of_bbox(player_bbox_end[x])))
            player_height_pixels_end = get_height_of_bbox(player_bbox_end[closest_player_end])
            closest_kp_end_idx = get_closest_keypoint_index(ball_pos_end, original_court_key_points, [0,2,12,13])
            closest_kp_end = (original_court_key_points[closest_kp_end_idx * 2], original_court_key_points[closest_kp_end_idx * 2 + 1])
            mini_pos_end = self.get_mini_court_coordinates(ball_pos_end, closest_kp_end, closest_kp_end_idx,
                                                        player_height_pixels_end, player_heights[closest_player_end])

            # Vytvoříme segment: zobrazí se mezi start_frame a end_frame
            ball_line_segments.append({
                "start_frame": start_frame,
                "end_frame": end_frame,
                "line": [{1: mini_pos_start}, {1: mini_pos_end}]
            })
            


        return output_player_boxes, output_ball_boxes, ball_line_segments
    
    def draw_points_on_mini_court(self, frames, positions, radius, color=(0,255,0), thickness=-1):
        for i, frame in enumerate(frames):
            for _, position in positions[i].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), radius, color, thickness)
        return frames
    
    def smooth_positions(self, positions, window=5):
        if len(positions) < 3:
            return positions
        
        # Step 1: Remove outliers
        smoothed = []
        for i, pos in enumerate(positions):
            if 1 not in pos:
                smoothed.append(pos)
                continue
            
            # Get surrounding positions
            surrounding = []
            for j in range(max(0, i-2), min(len(positions), i+3)):
                if j != i and 1 in positions[j]:
                    surrounding.append(np.array(positions[j][1]))
            
            if surrounding:
                current = np.array(pos[1])
                avg_pos = np.mean(surrounding, axis=0)
                distance = np.linalg.norm(current - avg_pos)
                
                # If too far from average, use average instead
                if distance > 50:  # Adjust threshold as needed
                    smoothed.append({1: tuple(avg_pos)})
                else:
                    smoothed.append(pos)
            else:
                smoothed.append(pos)
        
        # Step 2: Moving average
        final_smooth = []
        for i, pos in enumerate(smoothed):
            if 1 not in pos:
                final_smooth.append(pos)
                continue
            
            # Get window of valid positions
            valid_positions = []
            for j in range(max(0, i-window//2), min(len(smoothed), i+window//2+1)):
                if 1 in smoothed[j]:
                    valid_positions.append(np.array(smoothed[j][1]))
            
            if valid_positions:
                avg_pos = np.mean(valid_positions, axis=0)
                final_smooth.append({1: tuple(avg_pos)})
            else:
                final_smooth.append(pos)
        
        return final_smooth

    def draw_ball_trajectory_lines(self, frames, ball_line_segments, color=(0,255,255), thickness=2):
        for segment in ball_line_segments:
            start_f = segment["start_frame"]
            end_f = segment["end_frame"]
            p1 = segment["line"][0][1]
            p2 = segment["line"][1][1]
            for i in range(start_f, min(end_f, len(frames))):
                cv2.line(frames[i], (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
        return frames

    def draw_player_heatmap(self, player_mini_court_detections):
        player_positions = []
        for frame_positions in player_mini_court_detections:
            for pos in frame_positions.values():
                player_positions.append(pos)
        
        heatmap = np.zeros((self.drawing_rectangle_height, self.drawing_rectangle_width), dtype=np.float32)

        for x, y in player_positions:
            x = int(x - self.start_x)
            y = int(y - self.start_y)
            if 0 <= x < heatmap.shape[1] and 0 <= y < heatmap.shape[0]:
                heatmap[y, x] += 1
                
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0) 

        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
        
        mask_zero = (heatmap_norm == 0)
        heatmap_colored[mask_zero] = [255, 255, 255]

        for line in self.lines:
            pt1 = (int(self.drawing_key_points[line[0] * 2] - self.start_x),
                int(self.drawing_key_points[line[0] * 2 + 1] - self.start_y))
            pt2 = (int(self.drawing_key_points[line[1] * 2] - self.start_x),
                int(self.drawing_key_points[line[1] * 2 + 1] - self.start_y))
            cv2.line(heatmap_colored, pt1, pt2, (0, 0, 0), 2)

        net_start = (int(self.drawing_key_points[0] - self.start_x),
                    int(((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2) - self.start_y))
        net_end = (int(self.drawing_key_points[2] - self.start_x),
                int(((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2) - self.start_y))
        cv2.line(heatmap_colored, net_start, net_end, (255, 0, 0), 2)
        
        cv2.imwrite("heatmap_players_white.png", heatmap_colored)
        return heatmap_colored
        