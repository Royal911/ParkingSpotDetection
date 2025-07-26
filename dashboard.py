from flask import Flask, render_template, Response, jsonify, request, send_from_directory, send_file
import threading
import json
import paho.mqtt.client as mqtt
import time
import os
import docker
import cv2  # Required to capture from RTSP

app = Flask(__name__, static_folder='static')

# === CONFIGURATION ===
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def save_config(data):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(data, f, indent=4)

CONFIG = load_config()

MQTT_BROKER = CONFIG['mqtt']['broker']
MQTT_PORT = CONFIG['mqtt']['port']
MQTT_CONTROL_TOPIC = CONFIG['mqtt']['topic_control']
MQTT_CAMERA_STATUS_TOPIC = CONFIG['mqtt']['topic_camera_status']
RTSP_URL = CONFIG.get('rtsp_url')

# === GLOBAL SHARED STATE ===
parking_statuses = []
latest_frame_path = "latest_output.jpg"
camera_status = "offline"

# === MQTT ===
def on_connect(client, userdata, flags, rc):
    print("✅ MQTT connected.")
    client.subscribe(MQTT_CAMERA_STATUS_TOPIC)
    print(f"📡 Subscribed to {MQTT_CAMERA_STATUS_TOPIC}")

def on_message(client, userdata, msg):
    global camera_status
    if msg.topic == MQTT_CAMERA_STATUS_TOPIC:
        camera_status = msg.payload.decode().strip().lower()
        print(f"🔄 Camera status updated via MQTT: {camera_status}")

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
mqtt_client.loop_start()

# === ROUTES ===

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/define')
def define():
    return send_from_directory('static', 'define.html')

@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/status')
def status():
    try:
        timestamp = int(time.time())
        return jsonify({
            "statuses": parking_statuses,
            "camera_status": camera_status,
            "latest_frame_url": f"/latest_frame.jpg?{timestamp}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/latest_frame.jpg')
def latest_frame():
    def generate():
        try:
            with open(latest_frame_path, 'rb') as f:
                data = f.read()
            yield data
        except Exception:
            yield b''
    return Response(generate(), mimetype='image/jpeg')

@app.route('/control', methods=['POST'])
def control():
    command = request.json.get('command', '').lower()
    if command in ['start', 'stop', 'snapshot']:
        mqtt_client.publish(MQTT_CONTROL_TOPIC, command)
        return jsonify({"status": "success", "command": command})
    else:
        return jsonify({"status": "error", "message": "Invalid command"}), 400

@app.route('/config', methods=['GET'])
def get_config():
    try:
        return jsonify(load_config())
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/config', methods=['POST'])
def update_config():
    try:
        new_config = request.json
        save_config(new_config)

        def restart_container():
            time.sleep(1)
            try:
                client = docker.DockerClient(base_url='unix://var/run/docker.sock')
                container = client.containers.get('parking-monitor')
                container.restart()
                print("🔄 Container restarted successfully.")
            except Exception as e:
                print(f"❌ Failed to restart container: {e}")

        threading.Thread(target=restart_container).start()
        return jsonify({"status": "success", "message": "Config saved. Restarting container..."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# === Parking spot definition image capture ===
@app.route('/frame_for_define')
def frame_for_define():
    try:
        cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            print("❌ Failed to open RTSP stream.")
            return send_from_directory('static', 'placeholder.jpg')

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("❌ Failed to read frame from RTSP.")
            return send_from_directory('static', 'placeholder.jpg')

        _, buffer = cv2.imencode('.jpg', frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        print(f"❌ Error capturing RTSP frame: {e}")
        return send_from_directory('static', 'placeholder.jpg')

@app.route('/save_spots', methods=['POST'])
def save_spots():
    spots = request.json.get('spots')
    if not spots or not isinstance(spots, list):
        return jsonify({"status": "error", "message": "Invalid spots data"}), 400

    try:
        for spot in spots:
            if len(spot) != 4 or not all(isinstance(pt, list) and len(pt) == 2 for pt in spot):
                return jsonify({"status": "error", "message": "Each spot must have 4 [x,y] points"}), 400

        with open('parking_spots.json', 'w') as f:
            json.dump(spots, f, indent=2)
        print("✅ Saved parking spots.")
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/latest_output.jpg")
def serve_latest_frame():
    return send_file("latest_output.jpg", mimetype="image/jpeg")
def get_frame():
    path = "latest_output.jpg"
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    else:
        return "Frame not available", 404

# === Integration mode ===
def run_dashboard(parking_status_ref, frame_path_ref):
    global parking_statuses, latest_frame_path
    parking_statuses = parking_status_ref
    latest_frame_path = frame_path_ref
    app.run(host='0.0.0.0', port=5000)

# === Standalone run ===
if __name__ == '__main__':
    parking_statuses = [None, True, False]
    latest_frame_path = "latest_output.jpg"
    app.run(host='0.0.0.0', port=5000)

