#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QComboBox, QPushButton, QGridLayout, 
                             QGroupBox, QSplitter, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

from src.video.video_processor import VideoProcessor
from src.detection.detector import YOLODetector
from src.tracking.tracker import ObjectTracker
from src.ui.video_widget import VideoDisplayWidget
from src.ui.stats_widget import StatsWidget

class MainWindow(QMainWindow):
    """Main window for the Traffic Monitoring System"""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Traffic Monitoring System")
        self.setMinimumSize(1200, 800)
        
        # Initialize components
        self.init_ui()
        
        # Initialize video processors, detectors and trackers
        self.init_processors()
        
        # Start processing timer
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.process_frames)
        self.processing_timer.start(30)  # ~33 fps
        
    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create control panel at the top
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Create splitter for video displays and statistics
        splitter = QSplitter(Qt.Horizontal)
        
        # Video display area (left side)
        self.video_container = QWidget()
        video_layout = QGridLayout(self.video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create video display widgets (2x2 grid)
        self.video_widgets = []
        for i in range(4):
            video_widget = VideoDisplayWidget()
            row, col = divmod(i, 2)
            video_layout.addWidget(video_widget, row, col)
            self.video_widgets.append(video_widget)
        
        # Statistics panel (right side)
        self.stats_widget = StatsWidget()
        
        # Add widgets to splitter
        splitter.addWidget(self.video_container)
        splitter.addWidget(self.stats_widget)
        splitter.setSizes([700, 500])  # Set initial sizes
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Set central widget
        self.setCentralWidget(central_widget)
    
    def create_control_panel(self):
        """Create the control panel with stream selection and controls"""
        control_panel = QGroupBox("Control Panel")
        control_layout = QHBoxLayout(control_panel)
        
        # Stream selection
        stream_group = QGroupBox("Stream Selection")
        stream_layout = QGridLayout(stream_group)
        
        # Create stream selectors for each video widget
        self.stream_selectors = []
        for i in range(4):
            label = QLabel(f"Stream {i+1}:")
            selector = QComboBox()
            # Add sample stream options (will be populated from config)
            selector.addItems(["None", "1625973-hd (25fps)", "4791734-hd (30fps)", 
                              "Webcam", "Pula Traffic Cam", "Zagreb Traffic Cam"])
            selector.setCurrentIndex(0)
            selector.currentIndexChanged.connect(lambda idx, pos=i: self.change_stream(pos, idx))
            
            row, col = divmod(i, 2)
            stream_layout.addWidget(label, row, col*2)
            stream_layout.addWidget(selector, row, col*2+1)
            self.stream_selectors.append(selector)
        
        # Control buttons
        button_group = QGroupBox("Controls")
        button_layout = QHBoxLayout(button_group)
        
        self.start_button = QPushButton("Start All")
        self.start_button.clicked.connect(self.start_all_streams)
        
        self.stop_button = QPushButton("Stop All")
        self.stop_button.clicked.connect(self.stop_all_streams)
        
        self.snapshot_button = QPushButton("Take Snapshot")
        self.snapshot_button.clicked.connect(self.take_snapshot)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.snapshot_button)
        
        # Add groups to control panel
        control_layout.addWidget(stream_group, 2)
        control_layout.addWidget(button_group, 1)
        
        return control_panel
    
    def init_processors(self):
        """Initialize video processors, detectors and trackers"""
        # Create video processors for each stream
        self.video_processors = []
        self.active_streams = [False] * 4
        
        for i in range(4):
            processor = VideoProcessor()
            self.video_processors.append(processor)
        
        # Initialize YOLOv8 detector (shared among streams)
        self.detector = YOLODetector()
        
        # Initialize trackers for each stream
        self.trackers = [ObjectTracker() for _ in range(4)]
    
    @pyqtSlot(int, int)
    def change_stream(self, position, stream_index):
        """Change the stream source for a specific position"""
        # Map stream_index to actual source
        sources = [None, "sample_videos/1625973-hd_1920_1080_25fps.mp4", "sample_videos/4791734-hd_1920_1080_30fps.mp4", 
                  0, "https://cdn-004.whatsupcams.com/hls/hr_pula01.m3u8", "https://cdn-004.whatsupcams.com/hls/hr_zagreb01.m3u8"]
        
        source = sources[stream_index] if stream_index < len(sources) else None
        
        if source is not None:
            # Stop current stream if active
            if self.active_streams[position]:
                self.video_processors[position].stop()
            
            # Start new stream
            success = self.video_processors[position].start(source)
            self.active_streams[position] = success
            
            # Reset tracker for this position
            self.trackers[position].reset()
            
            # Update UI
            if success:
                self.video_widgets[position].set_status(f"Connected: {source}")
            else:
                self.video_widgets[position].set_status(f"Failed to connect: {source}")
        else:
            # Stop stream if active
            if self.active_streams[position]:
                self.video_processors[position].stop()
                self.active_streams[position] = False
            
            # Clear display
            self.video_widgets[position].clear()
            self.video_widgets[position].set_status("No Stream Selected")
    
    def process_frames(self):
        """Process frames from all active streams"""
        detection_results = {}
        
        for i, processor in enumerate(self.video_processors):
            if not self.active_streams[i]:
                continue
                
            # Get frame from video processor
            frame = processor.get_frame()
            if frame is None:
                continue
            
            # Run detection on frame
            detections = self.detector.detect(frame)
            
            # Update tracker with new detections
            tracked_objects = self.trackers[i].update(detections)
            
            # Draw detection results and tracking info on frame
            annotated_frame = self.detector.annotate_frame(frame, tracked_objects)
            
            # Update video display
            self.video_widgets[i].update_frame(annotated_frame)
            
            # Collect statistics
            detection_results[i] = self.count_objects(tracked_objects)
        
        # Update statistics display
        if detection_results:
            self.stats_widget.update_stats(detection_results)
    
    def count_objects(self, tracked_objects):
        """Count objects by class"""
        counts = {}
        for obj in tracked_objects:
            obj_class = obj.get('class_name', 'unknown')
            if obj_class in counts:
                counts[obj_class] += 1
            else:
                counts[obj_class] = 1
        return counts
    
    def start_all_streams(self):
        """Start all selected streams"""
        for i, selector in enumerate(self.stream_selectors):
            if selector.currentIndex() > 0:  # If not 'None'
                self.change_stream(i, selector.currentIndex())
    
    def stop_all_streams(self):
        """Stop all active streams"""
        for i in range(4):
            if self.active_streams[i]:
                self.video_processors[i].stop()
                self.active_streams[i] = False
                self.video_widgets[i].clear()
                self.video_widgets[i].set_status("Stream Stopped")
    
    def take_snapshot(self):
        """Take snapshots of all active streams"""
        for i, widget in enumerate(self.video_widgets):
            if self.active_streams[i]:
                widget.save_snapshot(f"snapshot_stream_{i+1}.jpg")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop all streams
        self.stop_all_streams()
        
        # Stop processing timer
        self.processing_timer.stop()
        
        # Clean up resources
        self.detector.cleanup()
        
        # Accept the close event
        event.accept()