import cv2
from detect_objects import detect_objects
from tracking import initialize_tracker, update_tracker
from trapezium import create_trapezium
from utils import assign_riders_to_motorcycles
from helmet_detection import detect_helmets
import numpy as np

def process_video(video_path, vehicle_model, rider_model, helmet_model, output_path="output_video.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video!")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    tracker = initialize_tracker()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vehicles and filter out motorcycles
        vehicles = detect_objects(vehicle_model, frame)
        motorcycles = [d for d in vehicles if d['class'] == 3]  # Motorcycle class = 3
        non_motorcycles = [d for d in vehicles if d['class'] != 3]  # Exclude motorcycles

        # Detect riders
        riders = detect_objects(rider_model, frame)

        # Assign riders to motorcycles
        assignments = assign_riders_to_motorcycles(motorcycles, riders)

        trapeziums = []
        helmet_detections = []
        triple_riding_detections = []
        
        for motorcycle_key, rider_list in assignments.items():
            motorcycle = {'x': motorcycle_key[0], 'y': motorcycle_key[1], 'w': motorcycle_key[2], 'h': motorcycle_key[3]}
            trapezium = create_trapezium(motorcycle, rider_list)
            trapeziums.append(trapezium)
            
            # Detect helmets in the trapezium
            helmet_count = len(detect_helmets(frame, helmet_model, trapezium))
            rider_count = len(rider_list)
            
            if rider_count > 2 or helmet_count > 2:
                triple_riding_detections.append(trapezium)
        
        # Convert trapezium to bounding boxes for tracking
        trapezium_detections = [
            {
                'x': (min(p[0] for p in trapezium) + max(p[0] for p in trapezium)) / 2,
                'y': (min(p[1] for p in trapezium) + max(p[1] for p in trapezium)) / 2,
                'w': max(p[0] for p in trapezium) - min(p[0] for p in trapezium),
                'h': max(p[1] for p in trapezium) - min(p[1] for p in trapezium),
                'class': 'Trapezium'
            }
            for trapezium in trapeziums
        ]
        
        all_detections = non_motorcycles + helmet_detections + trapezium_detections
        tracks = update_tracker(tracker, all_detections, frame)
        
        # Draw tracked bounding boxes
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            class_name = track.get_det_class()
            color = (255, 0, 0)  # Default color

            if class_name in ['Helmet', 'No_Helmet']:
                color = (0, 255, 0) if class_name == 'Helmet' else (0, 0, 255)
            
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), color, 2)
            cv2.putText(frame, class_name, (int(ltrb[0]), int(ltrb[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw triple riding bounding boxes
        for trapezium in triple_riding_detections:
            pts = np.array(trapezium, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(frame, "Triple Riding", (int(trapezium[0][0]), int(trapezium[0][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Output video saved to {output_path}")




# import cv2
# from detect_objects import detect_objects
# from tracking import initialize_tracker, update_tracker
# from trapezium import create_trapezium
# from utils import assign_riders_to_motorcycles
# from helmet_detection import detect_helmets
# import numpy as np

# def process_video(video_path, vehicle_model, rider_model, helmet_model, output_path="output_video.mp4"):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video!")
#         return
    
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     tracker = initialize_tracker()
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Detect vehicles and filter out motorcycles
#         vehicles = detect_objects(vehicle_model, frame)
#         motorcycles = [d for d in vehicles if d['class'] == 3]  # Motorcycle class = 3
#         non_motorcycles = [d for d in vehicles if d['class'] != 3]  # Exclude motorcycles

#         # Detect riders
#         riders = detect_objects(rider_model, frame)

#         # Assign riders to motorcycles
#         assignments = assign_riders_to_motorcycles(motorcycles, riders)

#         # Detect helmets
#         # Create trapezium bounding boxes
#         trapeziums = []
#         for motorcycle_key, rider_list in assignments.items():
#             motorcycle = {'x': motorcycle_key[0], 'y': motorcycle_key[1], 'w': motorcycle_key[2], 'h': motorcycle_key[3]}
#             trapezium = create_trapezium(motorcycle, rider_list)
#             trapeziums.append(trapezium)
#         helmet_detections = []
#         for trapezium in trapeziums:
#             helmet_detections.extend(detect_helmets(frame, helmet_model, trapezium))
#         # Combine all detections for tracking
#         trapezium_detections = [
#             {
#                 'x':  (min(p[0] for p in trapezium)+max(p[0] for p in trapezium))/2,  # Leftmost x
#                 'y': (min(p[1] for p in trapezium)+max(p[1] for p in trapezium))/2,  # Topmost y
#                 'w': max(p[0] for p in trapezium) - min(p[0] for p in trapezium),  # Width = xmax - xmin
#                 'h': max(p[1] for p in trapezium) - min(p[1] for p in trapezium),  # Height = ymax - ymin
#                 'class': 'Trapezium'
#             }
#             for trapezium in trapeziums
#         ]
        
#         all_detections = non_motorcycles  + helmet_detections + trapezium_detections
#         # Update tracker with all objects
#         tracks = update_tracker(tracker, all_detections, frame)

#         # Draw tracked bounding boxes
#         for track in tracks:
#             if not track.is_confirmed():
#                 continue
#             ltrb = track.to_ltrb()
#             class_name = track.get_det_class()  # Ensure track stores class names
#             color = (255, 0, 0) #if class_name not in ['Helmet', 'No_helmet'] else ((0, 255, 0) if class_name == 'Helmet' else (0, 0, 255))

#             cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), color, 2)
#             cv2.putText(frame, class_name, (int(ltrb[0]), int(ltrb[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
#         # # Draw trapezium bounding boxes
#         # for trapezium in trapeziums:
#         #     pts = np.array(trapezium, np.int32).reshape((-1, 1, 2))
#         #     cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
#         #     cv2.putText(frame, "Trapezium", (trapezium[0][0], trapezium[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         out.write(frame)

#     cap.release()
#     out.release()
#     print(f"Output video saved to {output_path}")

 