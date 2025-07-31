import cv2
import ffmpeg
from sklearn.cluster import KMeans
import numpy as np


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
        
    if fps > 24:
        reduced_path = "./input_videos/reduced_input_video.mp4"

        ffmpeg.input(video_path).output(reduced_path, r=24).overwrite_output().run()

        cap = cv2.VideoCapture(reduced_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
    elif fps < 23:
        print("Input video must be at least 24fps")
        return
    else:
        cap = cv2.VideoCapture(video_path)
        
    frames = []
    while True:
        ret, frame = cap.read()
        # If the return is False
        if not ret:
            break
        frames.append(frame)
        
    if frames:
        dominant_color = get_dominant_color(frames[0])
    cap.release()

    return frames, fps, dominant_color

def save_video(output_video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    

def get_dominant_color(image, k=1):
    # Převod na RGB, pokud je potřeba
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Zploštění obrazu na pole tvaru (počet_pixelů, 3)
    pixels = image.reshape(-1, 3)

    # Použijeme KMeans k určení dominantních barev
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pixels)

    # Center of the biggest cluster
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    
    named_colors = {
        # 'orange': np.array([181, 111, 74]),
        'orange': np.array([255, 128, 0]),
        'green': np.array([0, 128, 0]),
        'blue':  np.array([176, 196, 222])
    }

    # Compute distances
    min_distance = float('inf')
    closest_name = None
    for name, ref_color in named_colors.items():
        distance = np.linalg.norm(dominant_color - ref_color)
        if distance < min_distance:
            min_distance = distance
            closest_name = name
            
    print(f"tohle je closest {closest_name}")
    
    return tuple(named_colors[closest_name])  # (R, G, B)
