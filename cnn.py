import cv2
import numpy as np
from ultralytics import YOLO

# =========================================================
# Initialize YOLOv10n Model
# =========================================================
print("Loading YOLOv10n model...")
model = YOLO("yolov10n.pt")  # Ensure the model file is in the same folder
print("YOLOv10n loaded successfully.")

# =========================================================
# Initialize CI Components
# =========================================================
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)

# Simulated CNN weights for congestion classification
motion_weight = 0.6
density_weight = 0.4

def analyze_congestion(frame, fg_mask):
    """Simulated CNN + GFM congestion detection using CI-inspired metrics."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion_score = np.mean(fg_mask > 20)
    edges = cv2.Canny(gray, 50, 150)
    density_score = np.mean(edges > 0)

    # Simulated CNN layer combining motion + texture features
    combined_score = (motion_weight * motion_score) + (density_weight * density_score)

    if combined_score < 0.05:
        return "Free Flow", combined_score
    elif combined_score < 0.12:
        return "Moderate Flow", combined_score
    else:
        return "Heavy Congestion", combined_score

# =========================================================
# Video Processing Loop
# =========================================================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame_resized = cv2.resize(frame, (640, 480))

        # ---------------- YOLO Vehicle Detection ----------------
        results = model(frame_resized, conf=0.3, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = results.names[cls]
            if class_name in ['car', 'truck', 'bus', 'motorbike', 'bicycle']:
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_resized, f"{class_name} {conf:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        vehicle_count = sum(1 for box in results.boxes if results.names[int(box.cls[0])] in ['car', 'truck', 'bus', 'motorbike', 'bicycle'])

        # ---------------- GFM + CNN-Based Congestion Detection ----------------
        fg_mask = bg_subtractor.apply(frame_resized)
        congestion_level, score = analyze_congestion(frame_resized, fg_mask)

        # Display Results
        cv2.putText(frame_resized, f"Vehicles: {vehicle_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f"Congestion: {congestion_level}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Visual windows
        cv2.imshow("Smart Traffic Monitoring (YOLOv10n + CNN + GFM)", frame_resized)
        cv2.imshow("Foreground Mask (GFM)", fg_mask)

        # Print for debugging
        if frame_id % 30 == 0:
            print(f"Frame {frame_id}: {vehicle_count} vehicles | Level: {congestion_level} | Score: {score:.3f}")

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================================================
# Main Entry
# =========================================================
if __name__ == "__main__":
    video_path = "C:/Users/bubal/OneDrive/Desktop/Punith/temp/Edge Computing/traffic_monitoring/data/input_videos/cctv052x2004080516x01638.avi"
    process_video(video_path)
    