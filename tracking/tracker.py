#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from collections import deque

class ObjectTracker:
    """Tracks objects across frames using simple IoU-based tracking"""
    
    def __init__(self, max_disappeared=30, min_iou=0.3, max_trajectory_points=30):
        # Maximum number of frames an object can be missing before being removed
        self.max_disappeared = max_disappeared
        
        # Minimum IoU for matching detections to existing tracks
        self.min_iou = min_iou
        
        # Maximum number of trajectory points to store
        self.max_trajectory_points = max_trajectory_points
        
        # Initialize tracking state
        self.reset()
    
    def reset(self):
        """Reset the tracker state"""
        # Dictionary to store tracked objects
        # Key: object ID, Value: object data
        self.objects = {}
        
        # Counter for assigning unique IDs
        self.next_object_id = 0
        
        # Counter for disappeared objects
        self.disappeared = {}
    
    def update(self, detections):
        """Update object tracks with new detections"""
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.objects.keys()):
                self.disappeared[object_id] += 1
                
                # Remove object if it has been missing for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.remove_object(object_id)
            
            # Return the remaining tracked objects
            return list(self.objects.values())
        
        # If no existing objects, register all detections as new objects
        if len(self.objects) == 0:
            for detection in detections:
                self.register_object(detection)
                
            return list(self.objects.values())
        
        # Match detections to existing objects
        object_ids = list(self.objects.keys())
        object_bboxes = [self.objects[object_id]['bbox'] for object_id in object_ids]
        
        # Calculate IoU between each detection and existing objects
        matched_object_ids = []
        unmatched_detections = []
        
        for detection in detections:
            detection_bbox = detection['bbox']
            max_iou = 0
            max_iou_id = None
            
            for i, object_id in enumerate(object_ids):
                # Skip already matched objects
                if object_id in matched_object_ids:
                    continue
                
                # Calculate IoU
                iou = self.calculate_iou(detection_bbox, object_bboxes[i])
                
                # Update if this is the best match so far
                if iou > max_iou and iou >= self.min_iou:
                    max_iou = iou
                    max_iou_id = object_id
            
            # If a match was found, update the object
            if max_iou_id is not None:
                self.update_object(max_iou_id, detection)
                matched_object_ids.append(max_iou_id)
            else:
                unmatched_detections.append(detection)
        
        # Mark unmatched objects as disappeared
        for object_id in object_ids:
            if object_id not in matched_object_ids:
                self.disappeared[object_id] += 1
                
                # Remove object if it has been missing for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.remove_object(object_id)
        
        # Register new objects for unmatched detections
        for detection in unmatched_detections:
            self.register_object(detection)
        
        # Return the current set of tracked objects
        return list(self.objects.values())
    
    def register_object(self, detection):
        """Register a new object"""
        # Create a new object with a unique ID
        object_id = self.next_object_id
        self.next_object_id += 1
        
        # Initialize object data
        self.objects[object_id] = detection.copy()
        self.objects[object_id]['track_id'] = object_id
        
        # Initialize trajectory with the center point of the bounding box
        bbox = detection['bbox']
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        self.objects[object_id]['trajectory'] = deque([(center_x, center_y)], maxlen=self.max_trajectory_points)
        
        # Initialize disappeared counter
        self.disappeared[object_id] = 0
    
    def update_object(self, object_id, detection):
        """Update an existing object with new detection data"""
        # Update object data
        for key, value in detection.items():
            self.objects[object_id][key] = value
        
        # Ensure track_id is preserved
        self.objects[object_id]['track_id'] = object_id
        
        # Update trajectory
        bbox = detection['bbox']
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        self.objects[object_id]['trajectory'].append((center_x, center_y))
        
        # Reset disappeared counter
        self.disappeared[object_id] = 0
    
    def remove_object(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0  # No intersection
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area
        
        return iou