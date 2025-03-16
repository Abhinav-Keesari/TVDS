from deep_sort_realtime.deepsort_tracker import DeepSort

def initialize_tracker():
    return DeepSort(max_age=30, n_init=3, nn_budget=100)

def update_tracker(tracker, detections, frame):
    formatted_detections = [
        ([d['x'] - d['w'] / 2, d['y'] - d['h'] / 2, d['w'], d['h']], 0.6, str(d['class']))
        for d in detections if d['class'] != 3
    ]
    return tracker.update_tracks(formatted_detections, frame=frame)
