from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8n.pt")  # lightweight YOLOv8

# initializing the DeepSort tracker
tracker = DeepSort(max_age=30)

# Read video
cap = cv2.VideoCapture("D:/Footfall-Counter/footfall.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model(frame)  # returns list of detections

    # Filter only people with confidence > 0.5
    person_boxes = []
    for r in results[0].boxes.data.tolist():  # YOLOv8 returns boxes in this format
        x1, y1, x2, y2, conf, cls = r
        if int(cls) == 0 and conf > 0.4:  # human class = 0 (person = 0 in coco dataset)
            person_boxes.append([int(x1), int(y1), int(x2), int(y2), conf])

    # Draw boxes for visualization
    for box in person_boxes:
        x1, y1, x2, y2, conf = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    

cap.release()
cv2.destroyAllWindows()
