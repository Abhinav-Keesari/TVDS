import cv2
from detect_objects import detect_objects
import numpy as np

def detect_helmets(frame, helmet_model, trapezium):
    """
    Detects helmets within the given trapezium bounding box and converts coordinates to full-frame.

    :param frame: The current frame from the video.
    :param helmet_model: The YOLO model for helmet detection.
    :param trapezium: List of four (x, y) points defining the trapezium.
    :return: List of detected helmet and no-helmet bounding boxes in full-frame format.
    """
    x_min = min(p[0] for p in trapezium)
    y_min = min(p[1] for p in trapezium)
    x_max = max(p[0] for p in trapezium)
    y_max = max(p[1] for p in trapezium)
    
    # Define center and size for getRectSubPix
    center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    size = (x_max - x_min, y_max - y_min)
    
    # Crop the trapezium region using getRectSubPix
    roi = cv2.getRectSubPix(frame, (int(size[0]), int(size[1])), center)
    
    if roi is None or roi.size == 0:
        return []
    
    # Run YOLO helmet detection on cropped region
    detections = detect_objects(helmet_model, roi)
    
    # Convert coordinates to full-frame reference
    full_frame_detections = []
    for det in detections:
        x_center=det['x']
        y_center=det['y']
        w=det['w']
        h=det['h']
        class_id=det['class']
        # Convert (x_center, y_center) in ROI to full-frame
        full_x_center = x_center + x_min
        full_y_center = y_center + y_min
        
        # Append bounding box in full-frame format
        full_frame_detections.append({
            'x': full_x_center,
            'y': full_y_center,
            'w': w,
            'h': h,
            'class': class_id
        })
    
    return full_frame_detections




# import cv2
# from detect_objects import detect_objects
# import numpy as np

# def detect_helmets(frame, helmet_model, trapezium):
#     """
#     Detects helmets within the given trapezium bounding box.
    
#     :param frame: The current frame from the video.
#     :param helmet_model: The YOLO model for helmet detection.
#     :param trapezium: List of four (x, y) points defining the trapezium.
#     :return: List of detected helmet and no-helmet bounding boxes inside the trapezium.
#     """
#     x_min = min(p[0] for p in trapezium)
#     y_min = min(p[1] for p in trapezium)
#     x_max = max(p[0] for p in trapezium)
#     y_max = max(p[1] for p in trapezium)
    
#     # Define center and size for getRectSubPix
#     center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
#     size = (x_max - x_min, y_max - y_min)
    
#     # Crop the trapezium region using getRectSubPix
#     roi = cv2.getRectSubPix(frame, (int(size[0]), int(size[1])), center)
#     names = ['Helmet', 'No_Helmet']
    
#     if roi is None or roi.size == 0:
#         return []
    
#     detections = detect_objects(helmet_model, roi)
#     return detections


# import cv2
# from detect_objects import detect_objects
# import numpy as np

# def detect_helmets(frame, helmet_model, trapezium):
#     """
#     Detects helmets within the given trapezium bounding box.

#     :param frame: The current frame from the video.
#     :param helmet_model: The YOLO model for helmet detection.
#     :param trapezium: List of four (x, y) points defining the trapezium.
#     :return: List of detected helmet and no-helmet bounding boxes inside the trapezium.
#     """
#     # Convert trapezium to numpy array if not already
#     if not isinstance(trapezium, np.ndarray):
#         trapezium = np.array(trapezium)
#     # Validate the trapezium
#     if trapezium is None or trapezium.shape != (4, 2):
#         return []
#     exit()

#     # Create a mask for the trapezium
#     mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#     cv2.fillPoly(mask, [trapezium], 255)

#     # Apply the mask to the frame
#     masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

#     # Find the bounding box around the trapezium
#     x_min, y_min = np.min(trapezium, axis=0)
#     x_max, y_max = np.max(trapezium, axis=0)

#     # Ensure the bounding box is within the frame size
#     x_min, x_max = max(0, x_min), min(frame.shape[1], x_max)
#     y_min, y_max = max(0, y_min), min(frame.shape[0], y_max)

#     # Crop the region of interest (ROI) within the trapezium
#     roi = masked_frame[y_min:y_max, x_min:x_max]

#     # Return empty if the ROI is invalid
#     if roi.size == 0:
#         return []

#     # Detect helmets within the cropped trapezium region
#     detections = detect_objects(helmet_model, roi)

#     # Adjust bounding box coordinates to the original frame
#     adjusted_detections = []
#     for det in detections:
#         x1, y1, x2, y2 = det['bbox']
#         adjusted_detections.append({
#             'class': det['class'],
#             'bbox': [x1 + x_min, y1 + y_min, x2 + x_min, y2 + y_min]
#         })

#     return adjusted_detections
