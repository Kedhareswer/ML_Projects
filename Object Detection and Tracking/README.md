# Traffic Monitoring System

A comprehensive system for real-time traffic monitoring, analysis, and visualization using computer vision and deep learning techniques.

## Overview

This project provides a traffic monitoring solution with both desktop and web interfaces. It uses YOLOv8 for object detection and custom tracking algorithms to monitor traffic flow, count vehicles, and analyze traffic patterns.

## Features

- Real-time traffic monitoring with up to 4 simultaneous video streams
- Object detection using YOLOv8 for vehicles, pedestrians, and other traffic participants
- Object tracking across video frames
- Live statistics and traffic flow visualization
- Snapshot capability for capturing important moments
- Available as both a desktop application (PyQt5) and web interface (Flask)

## System Requirements

- Python 3.7+
- CUDA-compatible GPU recommended for real-time performance
- Webcam or video sources (files, network streams)

## Installation

1. Clone this repository:

```bash
git clone https://github.com/Kedhareswer/ML_Projects.git
cd traffic-monitoring-system
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Desktop Application

### Running the Desktop Application

To start the desktop application, run:

```bash
python main.py
```

This will launch the PyQt5-based user interface.

### Usage

1. Select video sources from the dropdown menus for each stream
2. Click "Start All" to begin processing all selected streams
3. View real-time detection results in the video panels
4. Monitor traffic statistics in the right panel
5. Click "Take Snapshot" to capture the current frame from the selected streams

## Web Interface

The project also includes a web-based interface that provides similar functionality through a browser.

### Running the Web Interface

To start the web interface, navigate to the web_app directory and run:

```bash
cd web_app
pip install -r requirements.txt  # Install web-specific dependencies
python app.py
```

This will start the Flask server on http://localhost:5000. Open this URL in your web browser to access the interface.

For more details about the web interface, see the [Web Interface README](web_app/README.md).

## Video Sources

The system supports various video sources:

- Local video files (place them in the `sample_videos` directory)
- Webcam (only available when accessing from the local machine)
- Network streams (RTSP, HTTP)

## Project Structure

```
├── main.py                 # Main entry point for desktop application
├── requirements.txt        # Project dependencies
├── sample_videos/          # Sample video files for testing
├── src/                    # Source code
│   ├── detection/          # Object detection modules
│   ├── tracking/           # Object tracking algorithms
│   ├── ui/                 # Desktop UI components
│   └── video/              # Video processing utilities
├── web_app/                # Web interface
│   ├── app.py              # Flask application
│   ├── requirements.txt    # Web-specific dependencies
│   ├── static/             # Static assets
│   └── templates/          # HTML templates
└── yolov8n.pt             # Pre-trained YOLOv8 model
```

## Configuration

The system uses default configurations that work well for most scenarios. To customize:

- For the desktop app: Modify parameters in the respective module files
- For the web interface: Edit the configuration in `web_app/app.py`

## Extending

### Adding New Video Sources

- For the desktop app: Modify the sources list in `src/ui/main_window.py`
- For the web interface: Modify the `stream_sources` and `stream_names` lists in `web_app/app.py`

### Using Different Detection Models

The system uses YOLOv8n by default. To use a different model:

1. Download or train your preferred YOLOv8 model variant
2. Update the model path in the detector configuration

## Troubleshooting

- If you encounter issues with video streaming, check that the video sources are accessible
- For performance issues, consider reducing the number of simultaneous streams or using a smaller YOLOv8 model
- Make sure all required Python packages are installed correctly
- Check GPU compatibility and drivers if detection is slow

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) for the desktop interface
- [Flask](https://flask.palletsprojects.com/) for the web interface
