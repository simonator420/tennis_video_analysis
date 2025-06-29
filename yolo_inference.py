from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.track('input_videos/input_video.mov', conf=0.2, save=True)
