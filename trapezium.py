# import numpy as np
# import cv2
# from shapely.geometry import Polygon

# def bbox_to_polygon(bbox):
#     """ Convert bounding box (x, y, w, h) to a polygon representation. """
#     x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
#     return np.array([
#         [x - w / 2, y - h / 2],  # Top-left
#         [x + w / 2, y - h / 2],  # Top-right
#         [x + w / 2, y + h / 2],  # Bottom-right
#         [x - w / 2, y + h / 2]   # Bottom-left
#     ], dtype=np.float32)

# def iou(polygon1, polygon2):
#     """ Calculate the Intersection over Union (IoU) of two polygons. """
#     poly1 = Polygon(polygon1)
#     poly2 = Polygon(polygon2)
    
#     if poly1.is_valid and poly2.is_valid:
#         intersection = poly1.intersection(poly2).area
#         union = poly1.union(poly2).area
#         return intersection / union if union != 0 else 0
#     return 0

# def create_trapezium(motorcycle, riders):
#     """ Generate a trapezium bounding box that efficiently encloses a motorcycle and its riders. """
#     if not riders:
#         return bbox_to_polygon(motorcycle)  # If no riders, return motorcycle bbox as polygon
    
#     # Collect all points from the motorcycle and riders
#     combined_points = [bbox_to_polygon(motorcycle)]
#     for rider in riders:
#         combined_points.append(bbox_to_polygon(rider))
    
#     combined_points = np.vstack(combined_points)  # Stack all points together
#     hull = cv2.convexHull(combined_points)  # Compute convex hull
    
#     return hull.reshape(-1, 2).tolist() if len(hull) > 2 else combined_points.tolist()

# def assign_riders_to_motorcycles(motorcycles, riders, iou_threshold=0.5):
#     """ Assign riders to motorcycles based on IoU threshold. """
#     assignments = {}
    
#     if motorcycles:  # If there is at least one motorcycle
#         motorcycle = motorcycles[0]  # Only take the first motorcycle
#         motorcycle_bbox = {'x': motorcycle['x'], 'y': motorcycle['y'], 'w': motorcycle['w'], 'h': motorcycle['h']}
#         motorcycle_polygon = bbox_to_polygon(motorcycle_bbox)
        
#         assigned_riders = []
        
#         # Check IoU for each rider
#         for rider in riders:
#             rider_bbox = {'x': rider['x'], 'y': rider['y'], 'w': rider['w'], 'h': rider['h']}
#             rider_polygon = bbox_to_polygon(rider_bbox)
            
#             # Calculate IoU
#             iou_score = iou(motorcycle_polygon, rider_polygon)
            
#             # Assign rider to motorcycle if IoU exceeds the threshold
#             if iou_score >= iou_threshold:
#                 assigned_riders.append(rider)
        
#         assignments[(motorcycle_bbox['x'], motorcycle_bbox['y'], motorcycle_bbox['w'], motorcycle_bbox['h'])] = assigned_riders
    
#     return assignments

import numpy as np
import cv2
from shapely.geometry import Polygon

def bbox_to_polygon(bbox):
    """ Convert bounding box (x, y, w, h) to a polygon representation. """
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    return np.array([
        [x - w / 2, y - h / 2],  # Top-left
        [x + w / 2, y - h / 2],  # Top-right
        [x + w / 2, y + h / 2],  # Bottom-right
        [x - w / 2, y + h / 2]   # Bottom-left
    ], dtype=np.float32)

def create_trapezium(motorcycle, riders):
    """ Generate a trapezium bounding box that efficiently encloses a motorcycle and its riders. """
    if not riders:
        return bbox_to_polygon(motorcycle)  # If no riders, return motorcycle bbox as polygon
    
    # Collect all points from the motorcycle and riders
    combined_points = [bbox_to_polygon(motorcycle)]
    for rider in riders:
        combined_points.append(bbox_to_polygon(rider))
    
    combined_points = np.vstack(combined_points)  # Stack all points together
    hull = cv2.convexHull(combined_points)  # Compute convex hull
    
    return hull.reshape(-1, 2).tolist() if len(hull) > 2 else combined_points.tolist()


