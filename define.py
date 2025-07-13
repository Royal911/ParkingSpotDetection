import cv2
import json
import numpy as np 

image_path = 'reference.jpg'  # A snapshot from your camera
spots = []
current_spot = []

def mouse_callback(event, x, y, flags, param):
    global current_spot
    if event == cv2.EVENT_LBUTTONDOWN:
        current_spot.append((x, y))
        if len(current_spot) == 4:
            spots.append(current_spot.copy())
            current_spot = []

image = cv2.imread(image_path)
cv2.namedWindow("Define Spots")
cv2.setMouseCallback("Define Spots", mouse_callback)

while True:
    temp_image = image.copy()

    # Draw current spot
    for point in current_spot:
        cv2.circle(temp_image, point, 5, (0, 255, 255), -1)

    # Draw saved spots
    for i, spot in enumerate(spots):
        cv2.polylines(temp_image, [np.array(spot)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(temp_image, f"Spot {i+1}", spot[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Define Spots", temp_image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Save to JSON
with open("parking_spots.json", "w") as f:
    json.dump(spots, f)

print("Saved parking spots to parking_spots.json")
