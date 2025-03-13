#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
from pathlib import Path

class YOLODetector:
    """YOLOv8 detector for object detection in traffic scenes"""
    
    def __init__(self, model_path=None, conf_threshold=0.25, classes=None):
        # Set default model path if not provided
        if model_path is None:
            # Use YOLOv8n by default
            self.model = YOLO('yolov8n.pt')
        else:
            self.model = YOLO(model_path)
        
        # Set confidence threshold for detections
        self.conf_threshold = conf_threshold
        
        # Set classes to detect (None means all classes)
        self.classes = classes
        
        # Define traffic-related class mapping (COCO dataset)
        self.class_mapping = {
            0: 'person',      # person
            1: 'bicycle',     # bicycle
            2: 'car',         # car
            3: 'motorcycle',  # motorcycle
            5: 'bus',         # bus
            7: 'truck',       # truck
        }
        
        # Define colors for each class (BGR format for OpenCV)
        self.colors = {
            'person': (0, 128, 255),     # Orange
            'bicycle': (0, 255, 0),      # Green
            'car': (0, 0, 255),          # Red
            'motorcycle': (255, 0, 0),   # Blue
            'bus': (255, 0, 255),        # Magenta
            'truck': (255, 255, 0),      # Cyan
            'unknown': (128, 128, 128)   # Gray
        }
    
    def detect(self, frame):
        """Run detection on a frame and return results"""
        if frame is None:
            return []
        
        # Run YOLOv8 inference
        results = self.model(frame, conf=self.conf_threshold, classes=self.classes)
        
        # Process results
        detections = []
        
        if results and len(results) > 0:
            # Get the first result (only one image was processed)
            result = results[0]
            
            # Extract boxes, confidence scores, and class IDs
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Create detection objects
            for i, box in enumerate(boxes):
                class_id = class_ids[i]
                confidence = confs[i]
                
                # Map class ID to name
                class_name = self.class_mapping.get(class_id, 'unknown')
                
                # Skip if not a traffic-related object
                if class_name == 'unknown':
                    continue
                
                # Create detection object
                detection = {
                    'bbox': box.astype(int).tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'class_name': class_name
                }
                
                detections.append(detection)
        
        return detections
    
    def annotate_frame(self, frame, objects):
        """Draw detection boxes and tracking info on the frame"""
        if frame is None or objects is None:
            return frame
        
        # Make a copy of the frame to avoid modifying the original
        annotated_frame = frame.copy()
        
        # Draw each object
        for obj in objects:
            # Extract information
            bbox = obj.get('bbox', [])
            class_name = obj.get('class_name', 'unknown')
            confidence = obj.get('confidence', 0)
            track_id = obj.get('track_id', None)
            
            if not bbox or len(bbox) != 4:
                continue
            
            # Get color for this class
            color = self.colors.get(class_name, self.colors['unknown'])
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            if track_id is not None:
                label = f"{class_name} #{track_id} {confidence:.2f}"
            else:
                label = f"{class_name} {confidence:.2f}"
            
            # Draw label background
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw trajectory if available
            if 'trajectory' in obj and len(obj['trajectory']) > 1:
                points = np.array(obj['trajectory'], dtype=np.int32)
                cv2.polylines(annotated_frame, [points], False, color, 2)
        
        return annotated_frame
    
    def cleanup(self):
        """Clean up resources"""
        # Release CUDA memory if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()