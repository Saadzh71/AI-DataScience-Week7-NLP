import cv2
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Load a pre-trained YOLOv8 model

def adjust_green_signal_time(vehicle_count):
    base_green_time = 20  # Base green time in seconds
    vehicle_multiplier = 4  # Green time increases by 4 seconds per vehicle
    
    # Adjust green time based on vehicle count
    green_time = base_green_time + (vehicle_count * vehicle_multiplier)
    
    # Write the adjusted green signal time to a text file
    with open("adjusted_green_time.txt", "w") as file:
        file.write(f"Green signal adjusted to: {green_time} seconds")
    
    print(f"Adjusted Green Signal Time: {green_time} seconds")
    return green_time

video_path = '"C:\Users\nujud\Desktop\Deploy\Traffic IP Camera video.mp4"'
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Apply YOLO model to detect vehicles in the frame
        results = model(frame)
        vehicle_count = 0
        
        # Loop through results and count vehicles
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                
                # Count only vehicles
                if label in ['car', 'truck', 'bus', 'motorcycle']:
                    vehicle_count += 1
        
        # Adjust the green signal time based on the detected vehicle count
        adjust_green_signal_time(vehicle_count)
        
        # Add a delay to simulate real-time processing
        time.sleep(5)  # Adjust this delay as needed for real-time

    cap.release()
    cv2.destroyAllWindows()

# Example: Processing a random video input
video_path = 'input_video.mp4'  # Path to your video file
process_video(video_path)
