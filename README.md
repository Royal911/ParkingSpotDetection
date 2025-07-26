<h1>Parking Spot Detection</h1>

<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/b4d10d10-ea9d-4112-bef1-3dd5be62f574" />
<img width="700" height="200" alt="image" src="https://github.com/user-attachments/assets/46583c5a-92a7-4c41-9e15-e0df8059c800" />

 <h2>Steps</h2>
 
   - Go to your home location ( My documents or where you want  to have the folder created )
   - Run the docker Container in that folder
    	
	docker run -d \
	  --name parking-monitor \
	  -p 5000:5000 \
	  -v "$(pwd)/ParkingSpotDetection:/app" \
	  -v /var/run/docker.sock:/var/run/docker.sock \
	  -v /etc/localtime:/etc/localtime:ro \
	  -v /etc/timezone:/etc/timezone:ro \
	  --restart unless-stopped \
	  --network bridge \
	  royal911/parking-monitor:latest
   - Clone the repository
   - 	 git clone https://github.com/Royal911/ParkingSpotDetection.git
    
       
   - Edit the config.json
      - RTSP Link
      - Frame interval ( How often a picture is taken )
      - Enable or Disable the OFF Hours ( when the parking check is disabled )
      - Allowed Classes [2, 3, 5, 7] | [Car,Motorcycle,Bus,Truck]
      - Intersection Threshold -  Determines how much overlap is required to say an object is inside a spot | You could go a bit higher (0.4) for stricter rules or lower (0.2) for loose inclusion.
      - Conf Threshold  - Very low — will detect almost everything, including weak/confused predictions | But if you get too many false positives, increase to 0.2 or 0.3
      - IF OFF Hours is ON:
         - Select the starting time and the stoping time off the OFF Hours
      - Broker IP address and Port
   - Restart the container
   - Open the webpage : http://localhost:5000/   - replace the locahost with your host ip
    	<img width="1020" height="780" alt="image" src="https://github.com/user-attachments/assets/e408560b-84ea-4271-8c0e-2e23bd7e3b2a" />
   - Go to Define Spots and Define your own spots for detection
