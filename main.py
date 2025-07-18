import cv2
import numpy as np
import json
import time
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from datetime import datetime

# === LOAD CONFIGURATION ===
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

RTSP_URL = CONFIG['rtsp_url']
MODEL_PATH = CONFIG['model_path']
FRAME_INTERVAL = CONFIG['frame_interval']

MQTT_BROKER = CONFIG['mqtt']['broker']
MQTT_PORT = CONFIG['mqtt']['port']
MQTT_TOPIC = CONFIG['mqtt']['topic_status']
MQTT_CAMERA_STATUS_TOPIC = CONFIG['mqtt']['topic_camera_status']

# === LOAD PARKING SPOTS FROM JSON ===
with open('parking_spots.json', 'r') as f:
    PARKING_SPOTS = json.load(f)
    PARKING_SPOTS = [[tuple(point) for point in spot] for spot in PARKING_SPOTS]

# === INIT YOLO MODEL ===
model = YOLO(MODEL_PATH)

# === INIT MQTT CLIENT ===
mqtt_client = mqtt.Client()
mqtt_client.will_set(MQTT_CAMERA_STATUS_TOPIC, payload="offline", qos=1, retain=True)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
mqtt_client.loop_start()

# === OFF-HOURS CHECK ===
def is_off_hours():
    if not CONFIG.get("off_hours", {}).get("enabled", False):
        return False

    start_str = CONFIG["off_hours"]["start"]
    end_str = CONFIG["off_hours"]["end"]
    now = datetime.now().time()

    # Convert from AM/PM format to time objects
    start = datetime.strptime(start_str, "%I:%M %p").time()
    end = datetime.strptime(end_str, "%I:%M %p").time()

    if start < end:
        return start <= now <= end
    else:
        return now >= start or now <= end

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
    pending_status = [None] * len(PARKING_SPOTS)
    last_camera_status = None
    previous_frame_time = time.time()
    last_saved_file = None

    last_camera_status = publish_camera_status("online", last_camera_status)

    try:
        while True:
            # Check if we are in off-hours
            if is_off_hours():
                last_camera_status = publish_camera_status("idle", last_camera_status)
                print("🌙 Off-hours: Skipping parking check.")
                time.sleep(60)
                continue
            else:
                last_camera_status = publish_camera_status("online", last_camera_status)

            current_time = time.time()
            elapsed_time = current_time - previous_frame_time

            if elapsed_time >= FRAME_INTERVAL:
                previous_frame_time = current_time

                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    print("❌ Failed to open RTSP stream.")
                    last_camera_status = publish_camera_status("offline", last_camera_status)
                    continue
                else:
                    last_camera_status = publish_camera_status("online", last_camera_status)

                time.sleep(2)  # Allow stream to stabilize
                ret, frame = cap.read()

                if not ret:
                    print("⚠️ Failed to grab a frame from the stream.")
                    cap.release()
                    time.sleep(FRAME_INTERVAL)
                    continue

                # Run YOLOv8 on the frame
                results = model(frame)
                detections = []
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                            detections.append(box.xyxy[0].cpu().numpy())

                # Process each parking spot
                for i, spot in enumerate(PARKING_SPOTS):
                    occupied = is_occupied(spot, detections)
                    color = (0, 0, 255) if occupied else (0, 255, 0)

                    # === Debounce logic ===
                    if previous_status[i] is None:
                        previous_status[i] = occupied
                    elif pending_status[i] is not None:
                        if pending_status[i] == occupied:
                            status_text = 'occupied' if occupied else 'free'
                            topic = f"parking/spot/{i+1}/status"
                            payload = json.dumps({
                                "spot": i + 1,
                                "status": status_text,
                                "timestamp": int(time.time())
                            })
                            mqtt_client.publish(topic, payload)
                            print(f"📡 MQTT Published to {topic}: {payload}")
                            previous_status[i] = occupied
                            pending_status[i] = None

                            filename = f"spot_{i+1}_status_{status_text}_{int(time.time())}.jpg"
                            cv2.polylines(frame, [np.array(spot)], True, color, 2)
                            cv2.putText(frame, f"Spot {i+1}: {status_text.title()}",
                                        (spot[0][0], spot[0][1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.imwrite(filename, frame)
                            print(f"🖼️ Image saved as: {filename}")
                            last_saved_file = filename
                        else:
                            print(f"❌ Spot {i+1}: change reverted before confirmation.")
                            pending_status[i] = None
                    elif previous_status[i] != occupied:
                        pending_status[i] = occupied
                        print(f"🕒 Spot {i+1}: change to {'Occupied' if occupied else 'Free'} pending confirmation...")

                    cv2.polylines(frame, [np.array(spot)], True, color, 2)
                    cv2.putText(frame, f"Spot {i+1}: {'Occupied' if occupied else 'Free'}",
                                (spot[0][0], spot[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.imwrite("latest_output.jpg", frame)
                print("🖼️ Frame overwritten as: latest_output.jpg")
                cap.release()

            time.sleep(1)

    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("🛑 MQTT client disconnected")

if __name__ == '__main__':
    main()

