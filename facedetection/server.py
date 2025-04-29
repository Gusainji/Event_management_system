import cv2
import base64
import numpy as np
from flask import Flask
from flask_socketio import SocketIO
from ultralytics import YOLO 

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO("yolov8n.pt")  

@socketio.on("frame")
def handle_frame(image_data):
    try:
        # Convert base64 image to OpenCV format
        image_data = image_data.split(",")[1]
        buffer = base64.b64decode(image_data)
        np_arr = np.frombuffer(buffer, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Run YOLOv8 detection
        results = model(frame)
        
        people_count = 0
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  # Get class ID
                if class_id == 0:  # YOLO class ID for 'person' is 0
                    people_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Encode the processed frame back to base64
        _, buffer = cv2.imencode(".jpg", frame)
        processed_frame = base64.b64encode(buffer).decode("utf-8")

        # Send processed frame and people count to the frontend
        socketio.emit("processedFrame", {
            "frame": f"data:image/jpeg;base64,{processed_frame}",
            "peopleCount": people_count
        })
    except Exception as e:
        print("Error processing frame:", str(e))

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)