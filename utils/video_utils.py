import cv2
import ffmpeg


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"Number of fps: {fps}")
    
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
    cap.release()
    print(f"Number of fps: {fps}")
    return frames, fps

def save_video(output_video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()