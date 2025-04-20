import cv2
import requests
import xml.etree.ElementTree as ET
import time

# Dynamically retrieve the HLS playlist URL for "Hwy 80 at Northgate"
status_url = "https://cwwp2.dot.ca.gov/data/d3/cctv/cctvStatusD03.xml"
resp = requests.get(status_url)
resp.raise_for_status()
root = ET.fromstring(resp.content)

stream_url = None
for cam in root.findall(".//cctv"):
    # Extract the human-readable location and route
    loc = cam.find("location/locationName").text or ""
    route = cam.find("location/route").text or ""
    # Match route I-80 and location containing "Northgate"
    if "I-80" in route and "Northgate" in loc:
        # streamingVideoURL lives under imageData
        elem = cam.find("imageData/streamingVideoURL")
        if elem is not None and elem.text and "http" in elem.text:
            stream_url = elem.text
        break
if not stream_url:
    raise RuntimeError("Could not find I-80 Northgate camera in District 3 XML")

cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    raise RuntimeError("Cannot open stream")

# Throttle playback to real-time based on stream FPS
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0  # fallback if FPS not available
frame_interval = 1.0 / fps
last_time = time.time()

while True:
    # Flush buffered frames to keep only the most recent
    while cap.grab():
        pass
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or error")
        break

    # Throttle to maintain real-time playback
    now = time.time()
    elapsed = now - last_time
    if elapsed < frame_interval:
        time.sleep(frame_interval - elapsed)
    last_time = time.time()

    # Display (or process) the frame
    cv2.imshow("Northgate Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()