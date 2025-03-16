import cv2

def draw_triple_riding_bbox(frame, trapezium_bbox):
    """Draws a bounding box for triple riding detection."""
    x1, y1, x2, y2 = trapezium_bbox  # Extract trapezium bounding box coordinates
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box
    cv2.putText(frame, "Triple Riding", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 0, 255), 2)
