#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

class VideoDisplayWidget(QWidget):
    """Widget for displaying video streams with status information"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up the layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: #1e1e1e; color: white;")
        self.video_label.setText("No Video")
        
        # Create status label
        self.status_label = QLabel("No Stream Selected")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("background-color: #333333; color: white; padding: 5px;")
        self.status_label.setFont(QFont("Arial", 9))
        
        # Add widgets to layout
        self.layout.addWidget(self.video_label, 1)
        self.layout.addWidget(self.status_label, 0)
        
        # Store the current frame for snapshot functionality
        self.current_frame = None
    
    @pyqtSlot(object)
    def update_frame(self, frame):
        """Update the displayed frame"""
        if frame is None:
            return
            
        # Store the current frame
        self.current_frame = frame.copy()
        
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create QImage from the frame
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Create a pixmap from the QImage and scale it to fit the label
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(), 
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Set the pixmap to the label
        self.video_label.setPixmap(pixmap)
    
    def set_status(self, status_text):
        """Update the status text"""
        self.status_label.setText(status_text)
    
    def clear(self):
        """Clear the video display"""
        self.video_label.clear()
        self.video_label.setText("No Video")
        self.current_frame = None
    
    def save_snapshot(self, filename):
        """Save the current frame as an image file"""
        if self.current_frame is not None:
            # Create snapshots directory if it doesn't exist
            os.makedirs("snapshots", exist_ok=True)
            filepath = os.path.join("snapshots", filename)
            
            # Save the image
            cv2.imwrite(filepath, self.current_frame)
            self.set_status(f"Snapshot saved: {filepath}")
            return True
        else:
            self.set_status("No frame to save")
            return False
    
    def resizeEvent(self, event):
        """Handle resize events to scale the video properly"""
        super().resizeEvent(event)
        
        # If we have a pixmap, rescale it
        if not self.video_label.pixmap() is None:
            pixmap = self.video_label.pixmap()
            scaled_pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(), 
                                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)