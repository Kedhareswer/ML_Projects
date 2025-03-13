#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import base64
import threading
import time
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from queue import Queue
import sys

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components from the original application
from src.video.video_processor import VideoProcessor
from src.detection.detector import YOLODetector
from src.tracking.tracker import ObjectTracker

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'traffic_monitoring_secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
video_processors = []
active_streams = [False] * 4
trackers = []
detector = None
processing_thread = None
processing_active = False

# Stream sources
stream_sources = [
    None,
    "sample_videos/1625973-hd_1920_1080_25fps.mp4",
    "sample_videos/4791734-hd_1920_1080_30fps.mp4",
    0,  # Webcam (only works on server machine)
    "https://cdn-004.whatsupcams.com/hls/hr_pula01.m3u8",
    "https://cdn-004.whatsupcams.com/hls/hr_zagreb01.m3u8"
]

# Stream names for display
stream_names = [
    "None",
    "1625973-hd (25fps)",
    "4791734-hd (30fps)",
    "Webcam",
    "Pula Traffic Cam",
    "Zagreb Traffic Cam"
]

def init_system():
    """Initialize video processors, detector and trackers"""
    global video_processors, trackers, detector
    
    # Initialize video processors
    video_processors = [VideoProcessor() for _ in range(4)]
    
    # Initialize YOLOv8 detector
    detector = YOLODetector()
    
    # Initialize trackers
    trackers = [ObjectTracker() for _ in range(4)]

def process_frames():
    """Process frames from all active streams"""
    global processing_active
    
    while processing_active:
        detection_results = {}
        frame_data = {}
        
        for i, processor in enumerate(video_processors):
            if not active_streams[i]:
                continue
                
            # Get frame from video processor
            frame = processor.get_frame()
            if frame is None:
                continue
            
            # Run detection on frame
            detections = detector.detect(frame)
            
            # Update tracker with new detections
            tracked_objects = trackers[i].update(detections)
            
            # Draw detection results and tracking info on frame
            annotated_frame = detector.annotate_frame(frame, tracked_objects)
            
            # Convert frame to JPEG for streaming
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_data[i] = base64.b64encode(buffer).decode('utf-8')
            
            # Collect statistics
            detection_results[i] = count_objects(tracked_objects)
        
        # Emit frames and statistics via WebSocket
        if frame_data:
            socketio.emit('frame_update', frame_data)
        
        if detection_results:
            socketio.emit('stats_update', detection_results)
            
        # Sleep to control frame rate
        time.sleep(0.03)  # ~33 fps

def count_objects(tracked_objects):
    """Count objects by class"""
    counts = {}
    for obj in tracked_objects:
        obj_class = obj.get('class_name', 'unknown')
        if obj_class in counts:
            counts[obj_class] += 1
        else:
            counts[obj_class] = 1
    return counts

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', stream_names=stream_names)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('stream_options', stream_names)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('change_stream')
def handle_change_stream(data):
    """Handle stream change request"""
    position = data.get('position')
    stream_index = data.get('stream_index')
    
    if position is None or stream_index is None:
        return
    
    position = int(position)
    stream_index = int(stream_index)
    
    if position < 0 or position >= 4:
        return
    
    source = stream_sources[stream_index] if stream_index < len(stream_sources) else None
    
    if source is not None:
        # Stop current stream if active
        if active_streams[position]:
            video_processors[position].stop()
        
        # Start new stream
        success = video_processors[position].start(source)
        active_streams[position] = success
        
        # Reset tracker for this position
        trackers[position].reset()
        
        # Update client
        if success:
            socketio.emit('stream_status', {'position': position, 'status': f"Connected: {source}"})
        else:
            socketio.emit('stream_status', {'position': position, 'status': f"Failed to connect: {source}"})
    else:
        # Stop stream if active
        if active_streams[position]:
            video_processors[position].stop()
            active_streams[position] = False
        
        # Update client
        socketio.emit('stream_status', {'position': position, 'status': "No Stream Selected"})

@socketio.on('start_all')
def handle_start_all():
    """Start all streams with selected sources"""
    for i, selector_value in enumerate(request.get_json()):
        if selector_value > 0:
            handle_change_stream({'position': i, 'stream_index': selector_value})

@socketio.on('stop_all')
def handle_stop_all():
    """Stop all active streams"""
    for i in range(4):
        if active_streams[i]:
            video_processors[i].stop()
            active_streams[i] = False
            socketio.emit('stream_status', {'position': i, 'status': "No Stream Selected"})

@socketio.on('take_snapshot')
def handle_take_snapshot(data):
    """Take a snapshot of the specified stream"""
    position = int(data.get('position', 0))
    
    if position < 0 or position >= 4 or not active_streams[position]:
        return
    
    frame = video_processors[position].get_frame()
    if frame is None:
        return
    
    # Save snapshot
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"snapshot_{position+1}_{timestamp}.jpg"
    filepath = os.path.join("static", "snapshots", filename)
    
    # Ensure directory exists
    os.makedirs(os.path.join("static", "snapshots"), exist_ok=True)
    
    # Save image
    cv2.imwrite(filepath, frame)
    
    # Notify client
    socketio.emit('snapshot_taken', {
        'position': position,
        'filepath': filepath,
        'filename': filename
    })

def start_processing():
    """Start the frame processing thread"""
    global processing_thread, processing_active
    
    if processing_thread is None or not processing_thread.is_alive():
        processing_active = True
        processing_thread = threading.Thread(target=process_frames)
        processing_thread.daemon = True
        processing_thread.start()

def stop_processing():
    """Stop the frame processing thread"""
    global processing_active
    processing_active = False
    if processing_thread:
        processing_thread.join(timeout=1.0)

if __name__ == '__main__':
    # Initialize the system
    init_system()
    
    # Start processing thread
    start_processing()
    
    # Start the Flask app
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    finally:
        # Clean up resources
        stop_processing()
        for processor in video_processors:
            processor.stop()