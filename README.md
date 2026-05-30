# MLFF Integration

## Overview

MLFF Integration is a Multi-Lane Free Flow vehicle identification system that combines computer vision and RFID technology for automatic vehicle verification.

The system integrates:

* Vehicle detection using YOLO
* License plate recognition
* RFID-based vehicle verification
* ROS 2 communication between nodes
* Real-time data exchange using serial communication

## Features

* Real-time vehicle monitoring
* RFID tag reading through ESP32
* ROS 2 publisher and subscriber architecture
* Vehicle identification and verification
* Modular node-based implementation

## Project Structure

```text
integrasi-baru/
├── scripts/
│   ├── kamera_node.py
│   ├── sensor.py
│   ├── subscriber.py
│   └── tes-rfid.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Prerequisites

- Ubuntu 24.04
- ROS 2 Jazzy
- Python 3.12

Install ROS 2 Jazzy before running this project.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Notes

This project uses PaddlePaddle GPU.

Please install the appropriate CUDA version and PaddlePaddle build for your GPU before running the system.


## Running

Start the camera node:

```bash
python3 kamera_node.py
```

Start the RFID node:

```bash
python3 sensor.py
```

Start the subscriber node:

```bash
python3 subscriber.py
```

## Author

pikipik5758

vibecoders
