from shapely.geometry import Polygon
from trapezium import bbox_to_polygon
def iou(bbox1, bbox2):
    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)
    
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    
    intersection = poly1.intersection(poly2)
    if intersection.is_empty:
        return 0.0
    
    return intersection.area / (poly1.area + poly2.area - intersection.area)

def assign_riders_to_motorcycles(motorcycles, riders):
    assignments = {}
    
    for rider in riders:
        best_iou = 0
        best_motorcycle = None
        
        for motorcycle in motorcycles:
            current_iou = iou(bbox_to_polygon(motorcycle), bbox_to_polygon(rider))
            if current_iou > best_iou:
                best_iou = current_iou
                best_motorcycle = motorcycle
        
        if best_motorcycle:
            key = (best_motorcycle['x'], best_motorcycle['y'], best_motorcycle['w'], best_motorcycle['h'])
            if key not in assignments:
                assignments[key] = []
            assignments[key].append(rider)
    
    return assignments
