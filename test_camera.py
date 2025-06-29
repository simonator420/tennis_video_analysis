import cv2
import torch
import subprocess

# Load YOLOv5 model, v8 might be better for real time
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open your webcam (0 is usually the default camera, 1 is iPhone in this case)
cap = cv2.VideoCapture(1)

import subprocess

try:
    output = subprocess.check_output(
        ['ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', 'dummy'],
        stderr=subprocess.STDOUT
    ).decode()
    print("Available camera devices:\n")
    print(output)
except subprocess.CalledProcessError as e:
    print("Failed to list devices. Output:\n")
    print(e.output.decode())


if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 expects RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Inference
    results = model(img)

    # Render results on the original frame
    annotated_frame = results.render()[0]
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Show frame
    cv2.imshow("YOLOv5 - Webcam", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

