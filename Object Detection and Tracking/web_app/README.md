# Traffic Monitoring System - Web Interface

This is a web-based interface for the Traffic Monitoring System, allowing users to access the traffic monitoring capabilities through a browser.

## Features

- Real-time traffic monitoring with up to 4 simultaneous video streams
- Object detection using YOLOv8 for vehicles, pedestrians, and other traffic participants
- Live statistics and traffic flow visualization
- Snapshot capability for capturing important moments
- Responsive design that works on desktop and mobile devices

## Requirements

All dependencies are listed in the `requirements.txt` file. The main requirements are:

- Python 3.7+
- Flask and Flask-SocketIO for the web server
- OpenCV for video processing
- Ultralytics YOLOv8 for object detection
- PyTorch for the deep learning backend

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Web Interface

To start the web interface, run:

```bash
python app.py
```

This will start the Flask server on http://localhost:5000. Open this URL in your web browser to access the interface.

## Usage

1. Select video sources from the dropdown menus for each stream
2. Click "Start All" to begin processing all selected streams
3. View real-time detection results in the video panels
4. Monitor traffic statistics in the right panel
5. Click on any video feed to select it for taking snapshots
6. Click "Take Snapshot" to capture the current frame from the selected stream

## Video Sources

The system supports various video sources:

- Local video files (place them in the `sample_videos` directory)
- Webcam (only available when accessing from the server machine)
- Network streams (RTSP, HTTP)

## Extending

To add more video sources, modify the `stream_sources` and `stream_names` lists in `app.py`.

## Troubleshooting

- If you encounter issues with video streaming, check that the video sources are accessible
- For performance issues, consider reducing the number of simultaneous streams or using a smaller YOLOv8 model
- Make sure all required Python packages are installed correctly