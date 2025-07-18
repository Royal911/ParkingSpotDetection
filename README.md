<h1>Parking Spot Detection</h1>

<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/b4d10d10-ea9d-4112-bef1-3dd5be62f574" />

<h2>Python Requirements</h2>

	- cv2 module
 
	- numpy module
 
	- paho-mqtt module
 
	- ultralytics module
 <h2>Steps</h2>

   - Run define.py and chose the spots (You need a jpg from the camera pointing to the parking lot ) 
   - Press "Q" when finish and will generate the parking_spots.json
   - Configure the config.json
      - RTSP Link
      - Frame interval ( How often a picture is taken )
      - Enable or Disable the OFF Hours ( when the parking check is disabled )
      - IF OFF Hours is ON:
         - Select the starting time and the stoping time off the OFF Hours
      - Broker IP address and Port

<h2>Future Features</h2>
   - Option to turn on and off the parking check from via MQTT payload 
