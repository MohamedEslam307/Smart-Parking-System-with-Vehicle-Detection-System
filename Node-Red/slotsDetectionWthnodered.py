import yaml
import numpy as np
import cv2
import time
from picamera2 import Picamera2
import serial
import threading
import paho.mqtt.client as mqtt
import ssl
import json

phone_number = "+201009668973"
api_key = "9488283"

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
            "emptyplaces": self.emptyplaces,
            "fullplaces": self.fullplaces,
            "location": self.location,
            "total_fee": self.calculate_fee(),
            "parking_fee_rate": self.parking_fee_rate
        }

# Path references
fn_yaml = "CarPark.yml"
fn_out = "CarPark.avi"
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

# Set up video writer if needed
if dict['save_video']:
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter(fn_out, fourcc, 20, (1280, 720))

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

if parking_data:
    parking_status = [False] * len(parking_data)
    parking_buffer = [None] * len(parking_data)

# Serial communication
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
running = True

def check_parking_availability():
    free_slots = [i for i, status in enumerate(parking_status) if status]
    return 0 if len(free_slots) > 0 else 1

def handle_message(message):
    # print(f"Received message from Arduino: {message}")
    if message == "CHECK_PARKING":
        parking_status = check_parking_availability()
        ser.write(str(parking_status).encode('utf-8'))
        ser.write(b'\n')

def listen_to_serial():
    while running:
        if ser.in_waiting > 0:
            message = ser.readline().decode('utf-8').strip()
            handle_message(message)

serial_thread = threading.Thread(target=listen_to_serial)
serial_thread.start()

# Function to send data to the MQTT broker
def mqtt_publish_loop():
    while True:
        free_slots = [i for i, status in enumerate(parking_status) if status]
        emptyplaces = len(free_slots)

        garage_topics = ["Garage1", "Garage2", "Garage3", "Garage4", "Garage5", "Garage6"]

        garage1 = GarageStatus(emptyplaces=emptyplaces, fullplaces= 2- emptyplaces, location="https://maps.app.goo.gl/ADbYHBEvx9AT3J54A", parking_fee_rate=5)  # $5 per car
        garage2 = GarageStatus(emptyplaces=5, fullplaces=9, location="https://maps.app.goo.gl/2aAutYdkyRy2NjAF8", parking_fee_rate=5)  # $7 per car
        garage3 = GarageStatus(emptyplaces=3, fullplaces=3, location="https://maps.app.goo.gl/bibvj2RMA3KZg6DG8", parking_fee_rate=10) # $10 per car
        garage4 = GarageStatus(emptyplaces=7, fullplaces=1, location="https://maps.app.goo.gl/nnjKH9VKw9SQtUWF7", parking_fee_rate=5)  # $5 per car
        garage5 = GarageStatus(emptyplaces=5, fullplaces=5, location="https://maps.app.goo.gl/hEyPc49y3ne1BFYMA", parking_fee_rate=5)  # $7 per car
        garage6 = GarageStatus(emptyplaces=1, fullplaces=3, location="https://maps.app.goo.gl/AudC5NvDBkVyjPve7", parking_fee_rate=10) # $10 per car

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
            }),
            "Garage4": json.dumps({
                "phoneNumber": phone_number,
                "apiKey": api_key,
                "payload": garage4.to_dict()
            }),
            "Garage5": json.dumps({
                "phoneNumber": phone_number,
                "apiKey": api_key,
                "payload": garage5.to_dict()
            }),
            "Garage6": json.dumps({
                "phoneNumber": phone_number,
                "apiKey": api_key,
                "payload": garage6.to_dict()
            })
        }

        for topic in messages.keys():
            try:
                client.publish(topic, messages[topic], qos=1, retain=True)
                print(f"Published: {topic} : {messages[topic]}")
            except Exception as e:
                print(f"Failed to publish message to {topic}: {e}")

        time.sleep(10)

# Start the MQTT publish thread
mqtt_thread = threading.Thread(target=mqtt_publish_loop)
mqtt_thread.start()

while True:
    frame_initial = picam2.capture_array()
    frame = cv2.resize(frame_initial, (1280, 720))

    frame_blur = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
    frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    frame_out = frame.copy()

    if dict['motion_detection']:
        fgmask = fgbg.apply(frame_blur)
        bw = np.uint8(fgmask == 255) * 255
        bw = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 19)), iterations=1)
        contours, _ = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < dict['min_area_motion_contour']:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (255, 0, 0), 1)

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

            if status != parking_status[ind] and parking_buffer[ind] is None:
                parking_buffer[ind] = time.time()
            elif status == parking_status[ind] and parking_buffer[ind] is not None:
                parking_buffer[ind] = None
            elif status != parking_status[ind] and parking_buffer[ind] is not None:
                if time.time() - parking_buffer[ind] > dict['park_sec_to_wait']:
                    parking_status[ind] = status
                    parking_buffer[ind] = None

            color = (0, 255, 0) if parking_status[ind] else (0, 0, 255)
            cv2.drawContours(frame_out, [points + rect[:2]], contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_8)

    if dict['parking_id_overlay']:
        cv2.putText(frame_out, str(park['id']), (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    if dict['parking_overlay']:
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            color = (0, 255, 0) if parking_status[ind] else (0, 0, 255)
            cv2.drawContours(frame_out, [points], contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_8)

    if dict['save_video']:
        out.write(frame_out)

    cv2.imshow('frame', frame_out)
    # Handle key events
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break


picam2.stop()
cv2.destroyAllWindows()
if dict['save_video']:
    out.release()

