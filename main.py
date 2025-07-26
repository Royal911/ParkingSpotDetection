import cv2
import numpy as np
import json
import time
import os
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from datetime import datetime
import threading
import dashboard

# === CONFIG ===
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

RTSP_URL = CONFIG['rtsp_url']
MODEL_PATH = CONFIG['model_path']
FRAME_INTERVAL = CONFIG['frame_interval']
CONF_THRESHOLD = CONFIG.get('conf_threshold', 0.5)
ALLOWED_CLASSES = CONFIG.get('allowed_classes', [2, 3, 5, 7])
INTERSECTION_THRESHOLD = CONFIG.get('intersection_threshold', 0.3)

MQTT_BROKER = CONFIG['mqtt']['broker']
MQTT_PORT = CONFIG['mqtt']['port']
MQTT_TOPIC = CONFIG['mqtt']['topic_status']
MQTT_CAMERA_STATUS_TOPIC = CONFIG['mqtt']['topic_camera_status']
MQTT_CONTROL_TOPIC = CONFIG['mqtt']['topic_control']

# === PARKING SPOTS ===
with open('parking_spots.json', 'r') as f:
    PARKING_SPOTS = json.load(f)
    PARKING_SPOTS = [[tuple(point) for point in spot] for spot in PARKING_SPOTS]

# === MODEL ===
model = YOLO(MODEL_PATH)

# === MQTT ===
mqtt_client = mqtt.Client()
mqtt_client.will_set(MQTT_CAMERA_STATUS_TOPIC, payload="offline", qos=1, retain=True)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
mqtt_client.loop_start()

# === GLOBALS ===
detection_enabled = True
previous_status = [None] * len(PARKING_SPOTS)
latest_frame_path = "latest_output.jpg"
latest_annotated_frame = None

# === SNAPSHOT ===
def save_snapshot(frame, save_dir="snapshots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/snapshot_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"[INFO] Snapshot saved: {filename}")

# === FORCE RECHECK ===
def force_recheck_and_publish(frame, detections):
    for i, spot in enumerate(PARKING_SPOTS):
        occupied = is_occupied(spot, detections)
        status_text = 'occupied' if occupied else 'free'
        topic = f"parking/spot/{i+1}/status"
        payload = json.dumps({
            "spot": i + 1,
            "status": status_text,
            "timestamp": int(time.time())
        })
        mqtt_client.publish(topic, payload)
        print(f"📡 [FORCE] Published: {topic} → {status_text}")
        previous_status[i] = occupied

# === MQTT CALLBACK ===
def on_message(client, userdata, msg):
    global detection_enabled, latest_annotated_frame
    payload = msg.payload.decode().strip().lower()
    if msg.topic == MQTT_CONTROL_TOPIC:
        if payload == "start":
            detection_enabled = True
            print("✅ Detection enabled via MQTT.")
        elif payload == "stop":
            detection_enabled = False
            print("⛔ Detection disabled via MQTT.")
        elif payload == "snapshot":
            if latest_annotated_frame is not None:
                save_snapshot(latest_annotated_frame)
        elif payload == "recheck":
            if latest_annotated_frame is not None:
                results = model(latest_annotated_frame)
                detections = []
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        if cls in ALLOWED_CLASSES and conf >= CONF_THRESHOLD:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append((x1, y1, x2, y2, conf))
                force_recheck_and_publish(latest_annotated_frame, detections)
        else:
            print(f"⚠️ Unknown control command: {payload}")

mqtt_client.on_message = on_message
mqtt_client.subscribe(MQTT_CONTROL_TOPIC)

# === TIME CHECK ===
def is_off_hours():
    if not CONFIG.get("off_hours", {}).get("enabled", False):
        return False
    start_str = CONFIG["off_hours"]["start"]
    end_str = CONFIG["off_hours"]["end"]
    now = datetime.now().time()
    start = datetime.strptime(start_str, "%I:%M %p").time()
    end = datetime.strptime(end_str, "%I:%M %p").time()
    return start <= now <= end if start < end else now >= start or now <= end

# === OCCUPANCY ===
def is_occupied(spot, detections):
    spot_contour = np.array(spot, dtype=np.float32)
    spot_area = cv2.contourArea(spot_contour)
    if spot_area == 0:
        return False
    for det in detections:
        x1, y1, x2, y2, conf = det
        box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        inter_area, _ = cv2.intersectConvexConvex(spot_contour, box)
        if inter_area / spot_area >= INTERSECTION_THRESHOLD:
            return True
    return False

def publish_camera_status(status, last_status):
    if status != last_status:
        mqtt_client.publish(MQTT_CAMERA_STATUS_TOPIC, status)
        print(f"📡 MQTT → Camera status: {status}")
        return status
    return last_status

# === MAIN ===
def main():
    global previous_status, latest_frame_path, latest_annotated_frame
    pending_status = [None] * len(PARKING_SPOTS)
    last_camera_status = publish_camera_status("online", None)
    previous_frame_time = time.time()

    try:
        while True:
            if is_off_hours():
                last_camera_status = publish_camera_status("idle", last_camera_status)
                print("🌙 Off-hours: Skipping detection.")
                time.sleep(60)
                continue

            if not detection_enabled:
                last_camera_status = publish_camera_status("paused", last_camera_status)
                print("⏸️ Detection paused.")
                time.sleep(5)
                continue

            if time.time() - previous_frame_time < FRAME_INTERVAL:
                time.sleep(1)
                continue

            previous_frame_time = time.time()
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print("❌ Failed to open RTSP stream.")
                last_camera_status = publish_camera_status("offline", last_camera_status)
                time.sleep(FRAME_INTERVAL)
                continue

            time.sleep(2)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print("⚠️ Failed to read frame.")
                continue

            results = model(frame)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls in ALLOWED_CLASSES and conf >= CONF_THRESHOLD:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append((x1, y1, x2, y2, conf))

            for i, spot in enumerate(PARKING_SPOTS):
                occupied = is_occupied(spot, detections)
                color = (0, 0, 255) if occupied else (0, 255, 0)

                if previous_status[i] is None:
                    previous_status[i] = occupied
                elif pending_status[i] is not None:
                    if pending_status[i] == occupied:
                        status_text = 'occupied' if occupied else 'free'
                        topic = f"parking/spot/{i+1}/status"
                        payload = json.dumps({"spot": i + 1, "status": status_text, "timestamp": int(time.time())})
                        mqtt_client.publish(topic, payload)
                        print(f"📡 Published: {topic} → {status_text}")
                        previous_status[i] = occupied
                        pending_status[i] = None
                    else:
                        print(f"❌ Spot {i+1} reverted.")
                        pending_status[i] = None
                elif previous_status[i] != occupied:
                    pending_status[i] = occupied
                    print(f"🕒 Spot {i+1} change to {'Occupied' if occupied else 'Free'} pending...")

                cv2.polylines(frame, [np.array(spot)], True, color, 2)
                cv2.putText(frame, f"Spot {i+1}: {'Occupied' if occupied else 'Free'}",
                            (spot[0][0], spot[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imwrite("latest_output.jpg", frame)
            latest_annotated_frame = frame.copy()
            latest_frame_path = "latest_output.jpg"
            print("🖼️ Frame saved as latest_output.jpg")

    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("🛑 MQTT client disconnected")

# === START ===
if __name__ == '__main__':
    dashboard_thread = threading.Thread(
        target=dashboard.run_dashboard,
        args=(previous_status, latest_frame_path)
    )
    dashboard_thread.daemon = True
    dashboard_thread.start()

    main()

