#!/usr/bin/env python3
"""
Footfall Counter - Final Implementation
Description:
A practical computer vision tool that counts footfall (entries and exits)
in real time using YOLOv8 object detection and tracking. 
Built for efficiency, clarity, and deployability.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import argparse
import sys

class FootfallCounter:
    """
    Tracks and counts people crossing a defined line in a video stream.

    Args:
        video_path (str): Path to input video file or camera index.
        roi_position (float): Position of counting line (0.0–1.0). Default 0.5 = middle.
        confidence (float): YOLO detection confidence threshold.
    """

    def __init__(self, video_path, roi_position=0.5, confidence=0.5):
        self.video_path = video_path
        self.roi_position = roi_position
        self.confidence = confidence

        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')  # Nano model for good speed/accuracy tradeoff
        print("Model loaded successfully.")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {video_path}")

        # Extract basic video info
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define counting line (Region of Interest)
        self.roi_line_y = int(self.height * roi_position)

        # Initialize counts and trackers
        self.entry_count = 0
        self.exit_count = 0
        self.counted_ids = set()
        self.track_history = defaultdict(lambda: [])

        # Output setup
        self.output_path = 'output_footfall.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

    def process_frame(self, frame, frame_idx):
        """Process one frame for detection, tracking, and counting."""

        results = self.model.track(frame, persist=True, classes=[0], conf=self.confidence, verbose=False)
        cv2.line(frame, (0, self.roi_line_y), (self.width, self.roi_line_y), (0, 255, 255), 3)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                track = self.track_history[track_id]
                track.append((cx, cy))
                if len(track) > 30:
                    track.pop(0)

                crossing = None
                if len(track) >= 2 and track_id not in self.counted_ids:
                    prev_cy, curr_cy = track[-2][1], track[-1][1]

                    if prev_cy < self.roi_line_y and curr_cy >= self.roi_line_y:
                        self.entry_count += 1
                        self.counted_ids.add(track_id)
                        crossing = 'ENTRY'
                        print(f"[+] ENTRY | ID:{track_id} | Total: {self.entry_count}")
                    elif prev_cy > self.roi_line_y and curr_cy <= self.roi_line_y:
                        self.exit_count += 1
                        self.counted_ids.add(track_id)
                        crossing = 'EXIT'
                        print(f"[-] EXIT  | ID:{track_id} | Total: {self.exit_count}")

                color = (0, 255, 0) if crossing == 'ENTRY' else (0, 0, 255) if crossing == 'EXIT' else (255, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)

                if len(track) > 1:
                    pts = np.array(track).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(frame, [pts], False, (220, 220, 220), 2)

                if crossing:
                    cv2.putText(frame, crossing, (cx - 40, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

        # Display summary stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, f"ENTRIES: {self.entry_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"EXITS:   {self.exit_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"TOTAL:   {self.entry_count + self.exit_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        return frame

    def run(self):
        """Main loop for reading frames and running detection."""
        print("\nProcessing started...\n")
        frame_count = 0

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            frame_count += 1

            processed = self.process_frame(frame, frame_count)
            self.out.write(processed)

            if frame_count % 30 == 0:
                percent = (frame_count / self.total_frames) * 100 if self.total_frames else 0
                print(f"Progress: {percent:.1f}%")

            cv2.imshow("Footfall Counter", processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Entries : {self.entry_count}")
        print(f"Exits   : {self.exit_count}")
        print(f"Net Flow: {self.entry_count - self.exit_count}")
        print(f"Saved video: {self.output_path}")
        print("="*60 + "\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="YOLOv8 Footfall Counter - Engineer’s Implementation")
    parser.add_argument('--video', required=True, help="Input video path or webcam index (e.g., 0).")
    parser.add_argument('--roi', type=float, default=0.5, help="Vertical position of counting line (0–1). Default = 0.5")
    parser.add_argument('--confidence', type=float, default=0.5, help="YOLO confidence threshold (default 0.5)")
    args = parser.parse_args()

    counter = FootfallCounter(args.video, args.roi, args.confidence)
    counter.run()


if __name__ == "__main__":
    main()
