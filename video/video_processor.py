#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import time
import threading
import numpy as np
from queue import Queue

class VideoProcessor:
    """Handles video capture and processing from various sources"""
    
    def __init__(self):
        self.cap = None
        self.source = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=10)  # Buffer a few frames
        self.last_frame = None
        self.thread = None
        self.lock = threading.Lock()
    
    def start(self, source):
        """Start capturing from the specified source"""
        # Stop any existing capture
        self.stop()
        
        self.source = source
        
        # Try to open the video source
        try:
            # For network streams, set additional parameters to improve reliability
            if isinstance(source, str) and (source.startswith('http://') or source.startswith('https://')):
                self.cap = cv2.VideoCapture(source)
                # Set buffer size to reduce stuttering
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                # Set timeout for network operations
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                # Set read timeout
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            else:
                self.cap = cv2.VideoCapture(source)
                
            if not self.cap.isOpened():
                print(f"Failed to open video source: {source}")
                return False
                
            # Start the capture thread
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop)
            self.thread.daemon = True
            self.thread.start()
            return True
            
        except Exception as e:
            print(f"Error starting video capture: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def stop(self):
        """Stop capturing and release resources"""
        self.is_running = False
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        # Release capture device
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
    
    def _capture_loop(self):
        """Background thread for continuous frame capture"""
        while self.is_running:
            with self.lock:
                if not self.cap or not self.cap.isOpened():
                    break
                    
                ret, frame = self.cap.read()
                
            if not ret:
                # Try to reconnect for network streams
                if isinstance(self.source, str) and (
                    self.source.startswith('rtsp://') or 
                    self.source.startswith('http://') or
                    self.source.startswith('https://')):
                    
                    print(f"Lost connection to {self.source}, attempting to reconnect...")
                    time.sleep(1.0)  # Wait before reconnecting
                    
                    with self.lock:
                        if self.cap:
                            self.cap.release()
                        self.cap = cv2.VideoCapture(self.source)
                        
                    continue
                else:
                    # End of file or disconnected camera
                    break
            
            # If queue is full, remove oldest frame
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            
            # Add new frame to queue
            try:
                self.frame_queue.put(frame, block=False)
            except:
                pass
                
            # Small delay to prevent CPU overuse
            time.sleep(0.01)
    
    def get_frame(self):
        """Get the latest frame from the video source"""
        if not self.is_running:
            return None
            
        try:
            # Get the newest frame from the queue
            frame = self.frame_queue.get_nowait()
            self.last_frame = frame
            return frame
        except:
            # Return the last valid frame if queue is empty
            return self.last_frame
    
    def is_active(self):
        """Check if the video processor is active"""
        return self.is_running and (self.cap is not None) and self.cap.isOpened()