#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QGroupBox, QScrollArea, QFrame, QGridLayout,
                             QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont, QColor
import pyqtgraph as pg

class StatsWidget(QWidget):
    """Widget for displaying detection statistics and tracking information"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up the layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # Create title label
        title_label = QLabel("Traffic Statistics")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        
        # Create scroll area for statistics
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        # Create container widget for scroll area
        self.stats_container = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_container)
        
        # Create statistics groups for each stream
        self.stream_stats = []
        for i in range(4):
            stats_group = self.create_stream_stats_group(i)
            self.stats_layout.addWidget(stats_group)
            self.stream_stats.append(stats_group)
        
        # Add stretch to push everything to the top
        self.stats_layout.addStretch(1)
        
        # Set the container as the scroll area widget
        scroll_area.setWidget(self.stats_container)
        
        # Create graph for historical data
        self.graph_group = self.create_graph_group()
        
        # Add widgets to main layout
        self.layout.addWidget(title_label)
        self.layout.addWidget(scroll_area, 3)  # Give more space to statistics
        self.layout.addWidget(self.graph_group, 2)  # Give less space to graph
        
        # Initialize data structures for statistics
        self.object_counts = {i: {} for i in range(4)}
        self.historical_data = {}
        
        # Set up graph update timer
        self.graph_data = {}
        self.time_axis = list(range(100))  # Last 100 data points
        
        # Set up color map for different object classes
        self.color_map = {
            'car': (255, 0, 0),       # Red
            'truck': (0, 255, 0),     # Green
            'bus': (0, 0, 255),       # Blue
            'motorcycle': (255, 255, 0), # Yellow
            'bicycle': (255, 0, 255),  # Magenta
            'person': (0, 255, 255)    # Cyan
        }
    
    def create_stream_stats_group(self, stream_index):
        """Create a group box for stream statistics"""
        group = QGroupBox(f"Stream {stream_index + 1}")
        layout = QVBoxLayout(group)
        
        # Create grid for object counts
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)  # Make count column stretch
        
        # Add headers
        grid.addWidget(QLabel("Object Type"), 0, 0)
        grid.addWidget(QLabel("Count"), 0, 1)
        
        # Add placeholders for different object types
        self.add_object_type_row(grid, 1, "Cars", "0")
        self.add_object_type_row(grid, 2, "Trucks", "0")
        self.add_object_type_row(grid, 3, "Buses", "0")
        self.add_object_type_row(grid, 4, "Motorcycles", "0")
        self.add_object_type_row(grid, 5, "Bicycles", "0")
        self.add_object_type_row(grid, 6, "Pedestrians", "0")
        
        # Add total row
        grid.addWidget(QLabel("Total"), 7, 0)
        total_label = QLabel("0")
        total_label.setStyleSheet("font-weight: bold;")
        grid.addWidget(total_label, 7, 1)
        
        # Add grid to layout
        layout.addLayout(grid)
        
        # Store labels for updating later
        group.count_labels = {
            'car': grid.itemAtPosition(1, 1).widget(),
            'truck': grid.itemAtPosition(2, 1).widget(),
            'bus': grid.itemAtPosition(3, 1).widget(),
            'motorcycle': grid.itemAtPosition(4, 1).widget(),
            'bicycle': grid.itemAtPosition(5, 1).widget(),
            'person': grid.itemAtPosition(6, 1).widget(),
            'total': total_label
        }
        
        # Add status label
        status_label = QLabel("No data")
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(status_label)
        group.status_label = status_label
        
        return group
    
    def add_object_type_row(self, grid, row, label_text, count_text):
        """Add a row for an object type in the statistics grid"""
        label = QLabel(label_text)
        count = QLabel(count_text)
        count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        grid.addWidget(label, row, 0)
        grid.addWidget(count, row, 1)
    
    def create_graph_group(self):
        """Create a group box for the historical data graph"""
        group = QGroupBox("Traffic Flow History")
        layout = QVBoxLayout(group)
        
        # Create pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')  # White background
        self.plot_widget.setLabel('left', 'Count')
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.showGrid(x=True, y=True)
        
        # Create plot items for each object class
        self.plots = {}
        
        # Add plot widget to layout
        layout.addWidget(self.plot_widget)
        
        return group
    
    @pyqtSlot(dict)
    def update_stats(self, detection_results):
        """Update statistics with new detection results"""
        # Update object counts for each stream
        for stream_id, counts in detection_results.items():
            if stream_id < len(self.stream_stats):
                self.update_stream_stats(stream_id, counts)
        
        # Update historical data
        self.update_historical_data(detection_results)
        
        # Update graph
        self.update_graph()
    
    def update_stream_stats(self, stream_id, counts):
        """Update statistics for a specific stream"""
        group = self.stream_stats[stream_id]
        labels = group.count_labels
        
        # Store the counts
        self.object_counts[stream_id] = counts
        
        # Update count labels
        total = 0
        for obj_class, count in counts.items():
            # Map YOLOv8 class names to our display names
            display_class = self.map_class_name(obj_class)
            if display_class in labels:
                labels[display_class].setText(str(count))
                total += count
        
        # Update classes that weren't detected
        for label_class in labels.keys():
            if label_class != 'total' and label_class not in [self.map_class_name(c) for c in counts.keys()]:
                labels[label_class].setText("0")
        
        # Update total
        labels['total'].setText(str(total))
        
        # Update status
        if total > 0:
            group.status_label.setText(f"Last updated: {self.get_timestamp()}")
        else:
            group.status_label.setText("No objects detected")
    
    def map_class_name(self, yolo_class):
        """Map YOLOv8 class names to our display names"""
        # YOLO class mapping
        mapping = {
            'car': 'car',
            'truck': 'truck',
            'bus': 'bus',
            'motorcycle': 'motorcycle',
            'bicycle': 'bicycle',
            'person': 'person',
            '0': 'person',  # Sometimes YOLO returns numeric class IDs
            '2': 'car',
            '5': 'bus',
            '7': 'truck',
            '3': 'motorcycle',
            '1': 'bicycle',
            # Add more mappings as needed
        }
        return mapping.get(yolo_class.lower(), yolo_class.lower())
    
    def update_historical_data(self, detection_results):
        """Update historical data for graphing"""
        # Aggregate counts across all streams
        aggregated_counts = {}
        for stream_id, counts in detection_results.items():
            for obj_class, count in counts.items():
                display_class = self.map_class_name(obj_class)
                if display_class in aggregated_counts:
                    aggregated_counts[display_class] += count
                else:
                    aggregated_counts[display_class] = count
        
        # Update historical data
        for obj_class, count in aggregated_counts.items():
            if obj_class not in self.graph_data:
                # Initialize with zeros if this is a new class
                self.graph_data[obj_class] = [0] * len(self.time_axis)
            
            # Add new data point and remove oldest
            self.graph_data[obj_class].append(count)
            if len(self.graph_data[obj_class]) > len(self.time_axis):
                self.graph_data[obj_class].pop(0)
    
    def update_graph(self):
        """Update the historical data graph"""
        # Clear previous plots
        self.plot_widget.clear()
        
        # Add new plots for each object class
        for obj_class, data in self.graph_data.items():
            if obj_class in self.color_map:
                color = self.color_map[obj_class]
                pen = pg.mkPen(color=color, width=2)
                self.plot_widget.plot(self.time_axis[-len(data):], data, name=obj_class, pen=pen)
    
    def get_timestamp(self):
        """Get a formatted timestamp for display"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")