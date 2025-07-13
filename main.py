import cv2
import numpy as np
import json
import time
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# === CONFIGURATION ===
RTSP_URL = 'rtsp://user:password@CAMIP:port'
MODEL_PATH = 'yolov8n.pt'
FRAME_INTERVAL = 30  # seconds (capture a frame every 30 seconds) ajustable

# === MQTT SETTINGS ===
MQTT_BROKER = 'IP'       # Replace with your broker IP or domain
MQTT_PORT = 1883
MQTT_CAMERA_STATUS_TOPIC = 'parking/camera/status'  # Topic to publish camera online/offline

# === LOAD PARKING SPOTS FROM JSON ===
with open('parking_spots.json', 'r') as f:
    PARKING_SPOTS = json.load(f)
    PARKING_SPOTS = [[tuple(point) for point in spot] for spot in PARKING_SPOTS]

# === INIT YOLO MODEL ===
model = YOLO(MODEL_PATH)

# === INIT MQTT CLIENT ===
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
mqtt_client.loop_start()

def is_occupied(spot, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        car_box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        inter_area = cv2.intersectConvexConvex(np.array(spot, dtype=np.float32), car_box.astype(np.float32))[0]
        if inter_area > 100:
            return True
    return False

def publish_camera_status(status, last_status):
    if status != last_status:
        mqtt_client.publish(MQTT_CAMERA_STATUS_TOPIC, status)
        print(f"📡 MQTT → Camera status: {status}")
        return status
    return last_status

def main():
    previous_status = [None] * len(PARKING_SPOTS)
    last_camera_status = None
    previous_frame_time = time.time()
    last_saved_file = None  # Track last saved file after status change
    
    # Track last camera status to avoid repeated publishes
    last_camera_status = publish_camera_status("online", last_camera_status)

    while True:
        current_time = time.time()
        elapsed_time = current_time - previous_frame_time

        # Process frame only if 30 seconds have passed
        if elapsed_time >= FRAME_INTERVAL:
            previous_frame_time = current_time  # Update the previous frame time

            # === Reconnect the RTSP stream before grabbing each frame ===
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)  # Force FFmpeg backend
            if not cap.isOpened():
                print("❌ Failed to open RTSP stream.")
                publish_camera_status("offline", None)
                continue

            # Delay to buffer stream and allow it to stabilize
            time.sleep(2)

            # Grab the next frame from the stream
            ret, frame = cap.read()

            if not ret:
                print("⚠️ Failed to grab a frame from the stream.")
                time.sleep(FRAME_INTERVAL)
                cap.release()
                continue

            # Process the frame with YOLO
            results = model(frame)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                        detections.append(box.xyxy[0].cpu().numpy())

            # Handle parking spot status and MQTT publishing
            for i, spot in enumerate(PARKING_SPOTS):
                occupied = is_occupied(spot, detections)
                color = (0, 0, 255) if occupied else (0, 255, 0)

                # === Status change detection ===
                if previous_status[i] is None:
                    previous_status[i] = occupied
                elif previous_status[i] != occupied:
                    status_text = 'occupied' if occupied else 'free'
                    # 📡 Publish status to individual MQTT topic
                    topic = f"parking/spot/{i+1}/status"
                    payload = json.dumps({
                        "spot": i + 1,
                        "status": status_text,
                        "timestamp": int(time.time())
                    })
                    mqtt_client.publish(topic, payload)
                    print(f"📡 MQTT Published to {topic}: {payload}")
                    previous_status[i] = occupied

                    # Save frame with a unique filename ONLY if the status changes
                    if last_saved_file != f"spot_{i+1}_status_{status_text}.jpg":
                        filename = f"spot_{i+1}_status_{status_text}_{int(time.time())}.jpg"
                        # Draw the parking spot status (Occupied/Free) on the frame
                        cv2.polylines(frame, [np.array(spot)], True, color, 2)
                        cv2.putText(frame, f"Spot {i+1}: {'Occupied' if occupied else 'Free'}",
                                    (spot[0][0], spot[0][1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        # Save the frame with annotated parking spots (status change)
                        cv2.imwrite(filename, frame)
                        print(f"🖼️ Image saved as: {filename}")
                        last_saved_file = filename  # Track the last saved file

                # Always draw the parking spot status on every frame (including overwrites)
                cv2.polylines(frame, [np.array(spot)], True, color, 2)
                cv2.putText(frame, f"Spot {i+1}: {'Occupied' if occupied else 'Free'}",
                            (spot[0][0], spot[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Always overwrite `latest_output.jpg` with the current frame (with spots and status)
            cv2.imwrite("latest_output.jpg", frame)
            print("🖼️ Frame overwritten as: latest_output.jpg")

            # Release the capture and move to the next frame
            cap.release()

        # Sleep for a second before checking again
        time.sleep(1)

    mqtt_client.loop_stop()
    mqtt_client.disconnect()

if __name__ == '__main__':
    main()

