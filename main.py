import time
from ultralytics import YOLO
from process_video import process_video

# Start timer
start_time = time.time()

# Load trained YOLO models
vehicle_model = YOLO(r"C:\Users\keesa\Desktop\Traffic violation.v4i.yolov9\YOLOv8\Vehicle Detection\runs_vehicle_detection\detect\train\weights\best.pt")
rider_model = YOLO(r"C:\Users\keesa\Desktop\Traffic violation.v4i.yolov9\YOLOv8\Rider\runs_rider_detection\detect\train\weights\best.pt")
helmet_model = YOLO(r"C:\Users\keesa\Desktop\Traffic violation.v4i.yolov9\YOLOv8\Helmet_detection\runs\detect\train\weights\best.pt")

# Define video paths
video_path = r"C:\Users\keesa\Desktop\Traffic violation.v4i.yolov9\YOLOv8\TVDS\test_2.mp4"
output_path = r"C:\Users\keesa\Desktop\Traffic violation.v4i.yolov9\YOLOv8\TVDS\output_video5.mp4"

# Run processing
process_video(video_path, vehicle_model, rider_model, helmet_model, output_path)

# End timer
end_time = time.time()

# Print execution time
total_time = end_time - start_time
print(f"Total execution time: {total_time:.2f} seconds")
