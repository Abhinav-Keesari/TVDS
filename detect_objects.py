import numpy as np
from ultralytics import YOLO

def detect_objects(model, image):
    results = model(image,conf=0.35)
    detections = []
    
    for result in results:
        for box, cls in zip(result.boxes.xywh, result.boxes.cls):  
            x, y, w, h = box.cpu().numpy()
            class_id = int(cls.cpu().numpy())
            detections.append({'x': x, 'y': y, 'w': w, 'h': h, 'class': class_id})
    
    return detections
