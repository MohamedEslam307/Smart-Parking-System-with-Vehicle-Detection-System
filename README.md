# Smart Parking System with Vehicle Detection System

This project implements a smart parking system that utilizes a Pi Camera for vehicle detection, OpenCV for image processing and motion detection, and MQTT for remote monitoring with Node-Red. It features real-time tracking of parking slots and calculates fees based on parking duration. Additionally, the system integrates with serial communication to check parking availability.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Overview

This system detects vehicles entering and leaving a parking lot, monitors available parking slots, and calculates parking fees based on the time a vehicle is parked. It uses a Raspberry Pi with a camera module to capture and process video, and it sends real-time data to an MQTT broker for remote access. Serial communication is used to interface with an external system (e.g., Arduino) for additional parking availability checks.

### Key Technologies:
- **Pi Camera**: Captures video for parking lot monitoring.
- **OpenCV**: Detects vehicles and motion, and processes video frames.
- **MQTT Protocol**: Sends real-time data on parking availability to remote systems.
- **Serial Communication**: Interfaces with other hardware (e.g., Arduino) to check parking availability.

## Features

- **Real-Time Vehicle Detection**: Monitors parking slots in real-time using video processing.
- **Motion Detection**: Detects movements to determine parking status.
- **Fee Calculation**: Calculates parking fees based on occupancy.
- **Remote Monitoring**: Sends updates about the parking lot status to a remote MQTT broker.
- **Serial Communication**: Communicates with external devices to update parking availability.

## System Requirements

To run this project, you will need:
- **Raspberry Pi** with a camera module.
- **Python 3.x**.
- **Libraries**:
  - `OpenCV`
  - `NumPy`
  - `PyYAML`
  - `Paho-MQTT`
  - `Picamera2`
  - `PySerial`
  - `ssl`
  - `threading`

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/MohamedEslam307/Smart-Parking-System-with-Vehicle-Detection-System.git
    ```


## Usage

1. **Camera Setup**: Ensure the Pi Camera is connected and configured.
2. **Vehicle Detection**: The system automatically detects vehicles entering and leaving the parking slots and updates the parking availability.
3. **Remote Monitoring**: Data about parking availability and fees are sent to the MQTT broker for remote monitoring and you can view all system garages using user-friendly ui with Node-Red.
4. **Serial Communication**: The system can send parking availability status to external devices via serial communication.

## Configuration

Some key configuration settings are available in the `CarPark.yml` file:
- **Parking space polygons**: Define the areas for parking detection.
- **Video output options**: Enable or disable video recording.
- **Detection parameters**: Configure motion and parking detection thresholds.

