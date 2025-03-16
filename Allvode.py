import cv2
import numpy as np
from detect_objects import detect_objects
from tracking import initialize_tracker, update_tracker
from trapezium import create_trapezium
from utils import assign_riders_to_motorcycles
from helmet_detection import detect_helmets

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
    
    # Initialize DeepSORT trackers
    vehicle_tracker = initialize_tracker()
    helmet_tracker = initialize_tracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect motorcycles and riders
        motorcycles = [d for d in detect_objects(vehicle_model, frame) if d['class'] == 3]
        riders = [d for d in detect_objects(rider_model, frame) if d['class'] == 0]

        # Assign riders to motorcycles
        assignments = assign_riders_to_motorcycles(motorcycles, riders)

        # Track motorcycles using DeepSORT
        motorcycle_tracks = update_tracker(vehicle_tracker, motorcycles, frame)

        # Draw tracked motorcycles and link trapezium bounding boxes
        for track in motorcycle_tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()
            motorcycle_bbox = {'x': ltrb[0], 'y': ltrb[1], 'w': ltrb[2] - ltrb[0], 'h': ltrb[3] - ltrb[1]}

            # Draw bounding box for motorcycle
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (255, 0, 0), 2)
            cv2.putText(frame, f"Motorcycle {track.track_id}", (int(ltrb[0]), int(ltrb[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Get riders assigned to this motorcycle
            if (motorcycle_bbox['x'], motorcycle_bbox['y'], motorcycle_bbox['w'], motorcycle_bbox['h']) in assignments:
                riders_assigned = assignments[(motorcycle_bbox['x'], motorcycle_bbox['y'], motorcycle_bbox['w'], motorcycle_bbox['h'])]
                
                # Create trapezium around motorcycle + riders
                trapezium = create_trapezium(motorcycle_bbox, riders_assigned)
                pts = np.array(trapezium, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                # Detect helmets inside the trapezium region
                helmet_detections = detect_helmets(frame, helmet_model, trapezium)

                # Track helmets using DeepSORT
                helmet_tracks = update_tracker(helmet_tracker, helmet_detections, frame)

                for helmet_track in helmet_tracks:
                    if not helmet_track.is_confirmed():
                        continue
                    
                    ltrb = helmet_track.to_ltrb()
                    label = "Helmet" if helmet_track.det_class == "Helmet" else "No Helmet"
                    color = (0, 255, 0) if label == "Helmet" else (0, 0, 255)

                    cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), color, 2)
                    cv2.putText(frame, label, (int(ltrb[0]), int(ltrb[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write frame to output video
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Output video saved to {output_path}")




# import cv2
# from detect_objects import detect_objects
# from tracking import initialize_tracker, update_tracker
# from trapezium import create_trapezium
# from utils import assign_riders_to_motorcycles
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
        
#         motorcycles = [d for d in detect_objects(vehicle_model, frame) if d['class'] == 3]
#         riders = [d for d in detect_objects(rider_model, frame) if d['class'] == 0]
        
#         assignments = assign_riders_to_motorcycles(motorcycles, riders)
#         detections = detect_objects(vehicle_model, frame)
        
#         tracks = update_tracker(tracker, detections, frame)
        
#         for track in tracks:
#             if not track.is_confirmed():
#                 continue
#             ltrb = track.to_ltrb()
#             cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (255, 0, 0), 2)
        
#         for motorcycle_key, riders in assignments.items():
#             motorcycle = {'x': motorcycle_key[0], 'y': motorcycle_key[1], 'w': motorcycle_key[2], 'h': motorcycle_key[3]}
#             trapezium = create_trapezium(motorcycle, riders)
            
#             pts = np.array(trapezium, np.int32).reshape((-1, 1, 2))
#             cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
#             cv2.putText(frame, "Motorcycle", (int(motorcycle['x']), int(motorcycle['y'])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         out.write(frame)
    
#     cap.release()
#     out.release()
#     print(f"Output video saved to {output_path}")

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

#         # Detect motorcycles and riders
#         motorcycles = [d for d in detect_objects(vehicle_model, frame) if d['class'] == 3]
#         riders = [d for d in detect_objects(rider_model, frame) if d['class'] == 0]

#         # Assign riders to motorcycles
#         assignments = assign_riders_to_motorcycles(motorcycles, riders)

#         # Track objects
#         detections = detect_objects(vehicle_model, frame)
#         tracks = update_tracker(tracker, detections, frame)

#         # Draw bounding boxes for tracked objects
#         for track in tracks:
#             if not track.is_confirmed():
#                 continue
#             ltrb = track.to_ltrb()
#             cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (255, 0, 0), 2)

#         # Process each motorcycle and its riders
#         for motorcycle_key, riders in assignments.items():
#             motorcycle = {'x': motorcycle_key[0], 'y': motorcycle_key[1], 'w': motorcycle_key[2], 'h': motorcycle_key[3]}
#             trapezium = create_trapezium(motorcycle, riders)

#             # Draw the trapezium
#             pts = np.array(trapezium, np.int32).reshape((-1, 1, 2))
#             cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
#             cv2.putText(frame, "Motorcycle", (int(motorcycle['x']), int(motorcycle['y'])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             helmet_result = detect_helmets(frame, helmet_model, trapezium)
#             for det in helmet_result:
#                 x, y, w, h = det['x'], det['y'], det['w'], det['h']
#                 x1, y1 = int(x), int(y)
#                 x2, y2 = int(x + w), int(y + h)
                
#                 label = 'Helmet' if det['class'] == 0 else 'No Helmet'
#                 color = (0, 255, 0) if det['class'] == 0 else (0, 0, 255)

#                 cv2.rectangle(trapezium, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(trapezium, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


#         out.write(frame)

#     cap.release()
#     out.release()
#     print(f"Output video saved to {output_path}")



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

#         # Detect motorcycles and riders
#         motorcycles = [d for d in detect_objects(vehicle_model, frame) if d['class'] == 3]
#         riders = [d for d in detect_objects(rider_model, frame) if d['class'] == 0]

#         # Assign riders to motorcycles
#         assignments = assign_riders_to_motorcycles(motorcycles, riders)

#         # Track objects
#         detections = detect_objects(vehicle_model, frame)
#         tracks = update_tracker(tracker, detections, frame)

#         # Draw bounding boxes for tracked objects
#         for track in tracks:
#             if not track.is_confirmed():
#                 continue
#             ltrb = track.to_ltrb()
#             cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (255, 0, 0), 2)

#         # Process each motorcycle and its riders
#         for motorcycle_key, riders in assignments.items():
#             motorcycle = {'x': motorcycle_key[0], 'y': motorcycle_key[1], 'w': motorcycle_key[2], 'h': motorcycle_key[3]}
#             trapezium = create_trapezium(motorcycle, riders)

#             # Draw the trapezium
#             pts = np.array(trapezium, np.int32).reshape((-1, 1, 2))
#             cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
#             cv2.putText(frame, "Motorcycle", (int(motorcycle['x']), int(motorcycle['y'])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             # Detect helmets **directly** on the full frame but **only** keep detections inside the trapezium
#             helmet_result = detect_helmets(frame, helmet_model, trapezium)

#             # Draw detected helmets
#             for det in helmet_result:
#                 x, y, w, h = det['x'], det['y'], det['w'], det['h']
#                 x1, y1 = int(x), int(y)
#                 x2, y2 = int(x + w), int(y + h)
                
#                 label = 'Helmet' if det['class'] == 0 else 'No Helmet'
#                 color = (0, 255, 0) if det['class'] == 0 else (0, 0, 255)

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         out.write(frame)

#     cap.release()
#     out.release()
#     print(f"Output video saved to {output_path}")
