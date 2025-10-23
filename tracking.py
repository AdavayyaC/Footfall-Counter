from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize DeepSort tracker
tracker = DeepSort(max_age=30, n_init=3)

# Read video
cap = cv2.VideoCapture("D:/Footfall-Counter/footfall.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Prepare person detections for DeepSort
    person_detections = []
    for r in results[0].boxes.data.tolist():
        x1, y1, x2, y2, confidence, cls = r
        if int(cls) == 0 and confidence > 0.4:
            # Correct format: [[x1,y1,x2,y2], confidence]
            person_detections.append([[float(x1), float(y1), float(x2), float(y2)], float(confidence)])

    # Update tracker
    tracks = tracker.update_tracks(person_detections, frame=frame)

    # Draw tracked boxes with IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
