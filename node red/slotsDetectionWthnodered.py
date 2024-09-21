import yaml
import numpy as np
import cv2
import time
from picamera2 import Picamera2
import serial
import threading



####################
# node red code
import paho.mqtt.client as mqtt
import time
import ssl
import json
import pandas as pd

phone_number = "+201121088265" 
api_key = "2336781"

# MQTT broker information
broker_URL = "broker.hivemq.com"
broker_port = 8883

# Create an MQTT client instance
client = mqtt.Client(client_id="sensordata1")

# Set TLS for secure connection
client.tls_set(ca_certs=None, certfile=None, keyfile=None, cert_reqs=ssl.CERT_NONE, tls_version=ssl.PROTOCOL_TLSv1_2)

# Set username and password for MQTT broker
client.username_pw_set("sicteam", "Aa123456")

# Callbacks for connect/disconnect
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully to broker")
    else:
        print(f"Failed to connect with result code {rc}")

def on_disconnect(client, userdata, rc):
    print(f"Disconnected with result code {rc}")

client.on_connect = on_connect
client.on_disconnect = on_disconnect

# Connect to the broker
try:
    client.connect(broker_URL, broker_port)
    client.loop_start()  # Start the loop
    print("Connected to Broker!")
except Exception as e:
    print(f"Failed to connect to broker: {e}")
    exit(1)

# Class to hold the number of empty and full places, location, and fee rate
class GarageStatus:
    def __init__(self, emptyplaces, fullplaces, location, parking_fee_rate):
        self.emptyplaces = int(emptyplaces)  # Ensure conversion to int
        self.fullplaces = int(fullplaces)    # Ensure conversion to int
        self.location = location
        self.parking_fee_rate = parking_fee_rate  # Fee rate per car

    # Method to calculate the total parking fee for all cars in the garage
    def calculate_fee(self):
        return self.fullplaces * self.parking_fee_rate

    # Method to convert object to a dictionary for JSON serialization
    def to_dict(self):
        return {
            "emptyplaces": self.emptyplaces,  # Already converted to int
            "fullplaces": self.fullplaces,    # Already converted to int
            "location": self.location,
            "total_fee": self.calculate_fee(),  # Calculate the total fee
            "parking_fee_rate": self.parking_fee_rate
        }

 ########################

# Initialize serial connection
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)  # Adjust the port and baud rate
# Global variable to track running status of the serial listener thread
running = True

def handle_message(message):
    """Function to handle messages received from Arduino."""
    print(f"Received message from Arduino: {message}")
    
    # Example: Check for specific message from Arduino
    if message == "CHECK_PARKING":
        parking_status = check_parking_availability()  # Function to check parking status
        ser.write(str(parking_status).encode('utf-8'))
        ser.write(b'\n')  # Send newline after the message

def listen_to_serial():
    """Thread function to listen to the serial port for incoming messages."""
    while running:
        if ser.in_waiting > 0:  # Check if there is data in the serial buffer
            message = ser.readline().decode('utf-8').strip()  # Read the message
            handle_message(message)  # Trigger the handler function
        else:
            continue  # No message received, continue listening

# Start the serial listening thread
serial_thread = threading.Thread(target=listen_to_serial)
serial_thread.start()


# Path references
fn_yaml = "CarPark.yml"
fn_out = "CarPark.avi"
global_str = "Last change at: "
change_pos = 0.00
dict = {
    'parking_overlay': True,
    'parking_id_overlay': True,
    'parking_detection': True,
    'motion_detection': True,
    'pedestrian_detection': False,
    'min_area_motion_contour': 500,
    'park_laplacian_th': 2.8,
    'park_sec_to_wait': 1,
    'show_ids': True,
    'classifier_used': True,
    'save_video': False
}

# Initialize the Pi Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(config)
picam2.start()


# Initialize a dictionary to store entry times for each parking slot
parking_entry_times = [None] * len(parking_data)

def calculate_parking_fee(start_time, end_time):
    duration = end_time - start_time
    hours_parked = duration / 3600  # Convert seconds to hours
    fee = hours_parked * 10  # 10 pounds per hour
    return round(fee, 2)

def check_parking_availability():
    # Use your parking detection code here to update parking_status
    # For example, if there's at least one free spot, return 0 (available), otherwise 1 (full)
    free_slots = [i for i, status in enumerate(parking_status) if status]
    if len(free_slots) > 0:
        return 0  # Parking available
    else:
        return 1  # Parking full


# Set up video writer if needed
if dict['save_video']:
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter(fn_out, fourcc, 20, (1280, 720))

# Initialize HOG descriptor for pedestrian detection
if dict['pedestrian_detection']:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize background subtractor for motion detection
if dict['motion_detection']:
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)

# Read YAML data (parking space polygons)
with open(fn_yaml, 'r') as stream:
    parking_data = yaml.safe_load(stream)

parking_contours = []
parking_bounding_rects = []
parking_mask = []

if parking_data:
    for park in parking_data:
        points = np.array(park['points'])
        rect = cv2.boundingRect(points)
        points_shifted = points.copy()
        points_shifted[:, 0] = points[:, 0] - rect[0]
        points_shifted[:, 1] = points[:, 1] - rect[1]
        parking_contours.append(points)
        parking_bounding_rects.append(rect)
        mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1,
                                color=255, thickness=-1, lineType=cv2.LINE_8)
        mask = mask == 255
        parking_mask.append(mask)

kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 19))

if parking_data:
    parking_status = [False] * len(parking_data)
    parking_buffer = [None] * len(parking_data)

def print_parkIDs(park, coor_points, frame_rev):
    moments = cv2.moments(coor_points)
    centroid = (int(moments['m10'] / moments['m00']) - 3, int(moments['m01'] / moments['m00']) + 3)
    cv2.putText(frame_rev, str(park['id']), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

while True:
    # Capture frame-by-frame from the Pi Camera
    frame_initial = picam2.capture_array()
    frame = cv2.resize(frame_initial, (1280, 720))

    # Background Subtraction
    frame_blur = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
    frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    frame_out = frame.copy()

    # Motion detection
    if dict['motion_detection']:
        fgmask = fgbg.apply(frame_blur)
        bw = np.uint8(fgmask == 255) * 255
        bw = cv2.erode(bw, kernel_erode, iterations=1)
        bw = cv2.dilate(bw, kernel_dilate, iterations=1)
        contours, _ = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < dict['min_area_motion_contour']:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (255, 0, 0), 1)
    if ser.in_waiting > 0:
        incoming_msg = ser.readline().decode('utf-8').strip()  # Read the message
        
        # If the message is "CHECK_PARKING", send the parking status
        if incoming_msg == "CHECK_PARKING":
            parking_status = check_parking_availability()
            ser.write(str(parking_status).encode('utf-8'))  # Send response to Arduino
            ser.write(b'\n')  # End message with a newline
##################################################################################################
# Initialize a dictionary to store entry times for each parking slot
parking_entry_times = [None] * len(parking_data)

def calculate_parking_fee(start_time, end_time):
    duration = end_time - start_time
    hours_parked = duration / 3600  # Convert seconds to hours
    fee = hours_parked * 10  # 10 pounds per hour
    return round(fee, 2)

while True:
    # Capture frame-by-frame from the Pi Camera
    frame_initial = picam2.capture_array()
    frame = cv2.resize(frame_initial, (1280, 720))


    # Parking detection
    if dict['parking_detection']:
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            rect = parking_bounding_rects[ind]
            roi_gray = frame_gray[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]

            laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
            points[:, 0] = points[:, 0] - rect[0]
            points[:, 1] = points[:, 1] - rect[1]
            delta = np.mean(np.abs(laplacian * parking_mask[ind]))
            status = delta < dict['park_laplacian_th']

            # Check for car entry (car occupied the slot)
            if status and not parking_status[ind]:
                parking_entry_times[ind] = time.time()  # Record entry time
                print(f"Car entered slot {ind} at {time.ctime(parking_entry_times[ind])}")

            # Check for car exit (car leaves the slot)
            if not status and parking_status[ind]:
                if parking_entry_times[ind] is not None:
                    exit_time = time.time()  # Record exit time
                    print(f"Car left slot {ind} at {time.ctime(exit_time)}")

                    # Calculate the fee
                    fee = calculate_parking_fee(parking_entry_times[ind], exit_time)
                    print(f"Parking fee for slot {ind}: {fee} pounds")

                    parking_entry_times[ind] = None  # Reset the entry time

            if status != parking_status[ind] and parking_buffer[ind] is None:
                parking_buffer[ind] = time.time()
                change_pos = time.time()
            elif status != parking_status[ind] and parking_buffer[ind] is not None:
                if time.time() - parking_buffer[ind] > dict['park_sec_to_wait']:
                    parking_status[ind] = status
                    parking_buffer[ind] = None
            elif status == parking_status[ind] and parking_buffer[ind] is not None:
                parking_buffer[ind] = None

    # Parking overlay
    if dict['parking_overlay']:
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            if parking_status[ind]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.polylines(frame_out, [points], True, color, 2)

            if dict['parking_id_overlay']:
                print_parkIDs(park, points, frame_out)

    # Pedestrian detection
    if dict['pedestrian_detection']:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pedestrians, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Print free parking slots
    free_slots = [i for i, status in enumerate(parking_status) if status]
    free_slots_text = f"Free slots: {len(free_slots)}"
    cv2.putText(frame_out, free_slots_text, (5, frame_out.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow('Parking Detection', frame_out)

    # Save video if the option is enabled
    if dict['save_video']:
        out.write(frame_out)

    # Handle key events
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break



    emptyplaces = len(free_slots)
    # Creating 3 instances for 3 different garages, including location URLs and fee rates
    garage1 = GarageStatus(emptyplaces=emptyplaces, fullplaces= 2- emptyplaces, location="https://maps.app.goo.gl/ADbYHBEvx9AT3J54A", parking_fee_rate=5)  # $5 per car
    garage2 = GarageStatus(emptyplaces=5, fullplaces=9, location="https://maps.app.goo.gl/2aAutYdkyRy2NjAF8", parking_fee_rate=5)  # $7 per car
    garage3 = GarageStatus(emptyplaces=1, fullplaces=3, location="https://maps.app.goo.gl/bibvj2RMA3KZg6DG8", parking_fee_rate=10) # $10 per car

    # Serialize the objects to JSON strings, including phone number and API key
    messages = {
        "Garage1": json.dumps({
            "phoneNumber": phone_number,
            "apiKey": api_key,
            "payload": garage1.to_dict()
        }),
        "Garage2": json.dumps({
            "phoneNumber": phone_number,
            "apiKey": api_key,
            "payload": garage2.to_dict()
        }),
        "Garage3": json.dumps({
            "phoneNumber": phone_number,
            "apiKey": api_key,
            "payload": garage3.to_dict()
        })
    }

    # Publish messages for each garage with QoS 1 and retain the messages
    for topic in messages.keys():
        try:
            client.publish(topic, messages[topic], qos=1, retain=True)
            print(f"Published: {messages[topic]} to topic: {topic}")
        except Exception as e:
            print(f"Failed to publish message to {topic}: {e}")

    # Wait for a bit before disconnecting
    time.sleep(2)

########################
# Cleanup
picam2.stop()
if dict['save_video']:
    out.release()
cv2.destroyAllWindows()

########################

# Stop the loop and disconnect from the broker
client.loop_stop()
client.disconnect()
print("Disconnected from Broker.")

