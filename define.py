import cv2
import json
import numpy as np
import os
import sys

# Load RTSP URL from config.json
CONFIG_PATH = 'config.json'
SPOTS_OUTPUT_PATH = 'parking_spots.json'

if not os.path.exists(CONFIG_PATH):
    print("Error: config.json not found.")
    sys.exit(1)

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

rtsp_url = config.get('rtsp_url')
if not rtsp_url:
    print("Error: 'rtsp_url' not found in config.json")
    sys.exit(1)

# Capture a frame from the RTSP stream
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("Error: Cannot connect to RTSP stream.")
    sys.exit(1)

ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Failed to capture frame from RTSP.")
    sys.exit(1)

image = frame.copy()
spots = []
current_spot = []

def mouse_callback(event, x, y, flags, param):
    global current_spot, spots
    if event == cv2.EVENT_LBUTTONDOWN:
        current_spot.append((x, y))
        if len(current_spot) == 4:
            spots.append(current_spot.copy())
            print(f"Spot {len(spots)} defined: {current_spot}")
            current_spot = []

# Set up OpenCV window
cv2.namedWindow("Define Spots")
cv2.setMouseCallback("Define Spots", mouse_callback)

print("🖱️  Instructions: Click 4 points to define each parking spot.")
print("▶️  Press 'u' to undo last spot, 'q' to quit and save.")

while True:
    temp_image = image.copy()

    # Draw in-progress spot
    for point in current_spot:
        cv2.circle(temp_image, point, 5, (0, 255, 255), -1)

    # Draw completed spots
    for i, spot in enumerate(spots):
        cv2.polylines(temp_image, [np.array(spot)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(temp_image, f"Spot {i + 1}", spot[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Instructions overlay
    cv2.putText(temp_image, "Click 4 points | 'u': Undo | 'q': Save & Quit", 
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Define Spots", temp_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('u'):
        if spots:
            spots.pop()
            print("⏪ Last spot removed.")

cv2.destroyAllWindows()

# Save spots to JSON
with open(SPOTS_OUTPUT_PATH, "w") as f:
    json.dump(spots, f, indent=2)

print(f"✅ Saved {len(spots)} spot(s) to {SPOTS_OUTPUT_PATH}")
