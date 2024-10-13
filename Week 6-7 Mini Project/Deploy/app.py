from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder to store uploaded videos

# Load YOLO model
model = YOLO('yolov8n.pt')

# Adjust green signal time based on vehicle count
def adjust_green_signal_time(vehicle_count):
    base_green_time = 1  # Base green time in seconds
    vehicle_multiplier = 0.001  # Increase green time by seconds per vehicle
    green_time = base_green_time + (vehicle_count * vehicle_multiplier)
    return int(green_time)

# Process video to detect vehicles and adjust signal time
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    total_vehicle_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        vehicle_count = 0
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                
                # Count vehicles
                if label in ['car', 'truck', 'bus', 'motorcycle']:
                    vehicle_count += 1
        
        total_vehicle_count += vehicle_count
    
    cap.release()
    # Adjust green signal time based on total detected vehicles
    return adjust_green_signal_time(total_vehicle_count)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has a video file
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)
            
            # Process the uploaded video
            green_signal_time = process_video(video_path)
            
            if green_signal_time is not None:
                return render_template('index.html', green_signal_time=green_signal_time)
            else:
                return "Error processing video."
    
    return render_template('index.html', green_signal_time=None)

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
