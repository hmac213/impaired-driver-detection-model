import requests
import xml.etree.ElementTree as ET
import cv2

# 1. Download the District 3 CCTV status XML
status_url = "https://cwwp2.dot.ca.gov/data/d3/cctv/cctvStatusD03.xml"
resp = requests.get(status_url)
resp.raise_for_status()

# 2. Parse it
root = ET.fromstring(resp.content)

# 3. Find the I‑80 Northgate camera and print its .m3u8 URL
for cam in root.findall(".//cctv"):
    # Safely extract route and locationName from the <location> element
    route_elem = cam.find("location/route")
    route = route_elem.text if route_elem is not None else ""
    loc_elem = cam.find("location/locationName")
    loc = loc_elem.text if loc_elem is not None else ""
    if "I-80" in route and "Northgate" in loc:
        url_elem = cam.find("imageData/streamingVideoURL")
        if url_elem is not None and url_elem.text.startswith("http"):
            stream_url = url_elem.text
            print(f"Streaming URL: {stream_url}")
            # Open the live HLS stream and display with OpenCV
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                raise RuntimeError("Cannot open video stream")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Stream ended or error")
                    break
                cv2.imshow("Live Highway Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Playback interrupted by user")
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            print("Found camera but no streaming URL element.")
        break
else:
    raise RuntimeError("Could not find I‑80 Northgate camera in XML")