# Gate Cameras

Gate Cameras is a web application that allows you to monitor multiple gates equipped with cameras and perform real-time object detection to count the number of people entering and exiting through each gate.

The application uses Flask, OpenCV, and the MobileNet SSD (Single Shot MultiBox Detector) deep learning model for object detection.

## Features

- Real-time object detection to count the number of people entering and exiting through each gate
- Display of live video feeds from the gate cameras
- Overall counts for each gate and total counts across all gates
- Automatic updates of counts through Server-Sent Events (SSE)
- Responsive web design to adapt to different screen sizes

## Prerequisites

- Python 3.7 or above
- Flask
- OpenCV
- MobileNet SSD pre-trained model
