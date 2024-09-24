import paho.mqtt.client as mqtt
import time
import ssl
import json
import pandas as pd

# Load the Excel sheet from the specified path
excel_file = r'C:\Users\Admin\Desktop\car_details.xlsx'  # Path to your Excel file
df = pd.read_excel(excel_file)  # Read the Excel file into a pandas DataFrame

# Display the contents of the DataFrame for debugging
print("Data in Excel:")
print(df)

# Ask the user for the car number to search
car_number = input("Enter the car number: ").strip()  # Stripping any leading/trailing whitespace

# Search for the car number in the DataFrame
car_info = df[df['car number'].astype(str).str.strip() == car_number]

# If the car number is found, extract the phone number and API key
if not car_info.empty:
    phone_number = str(car_info.iloc[0]['phonenumber']).strip("'")  # Remove the extra apostrophe
    api_key = str(car_info.iloc[0]['api'])  # Ensure it's a string
    print(f"Phone number: {phone_number}, API key: {api_key}")
else:
    print("Car number not found in the Excel sheet.")
    exit(1)

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

# Class to hold the number of empty and full places, and location
class GarageStatus:
    def __init__(self, emptyplaces, fullplaces, location):
        self.emptyplaces = int(emptyplaces)  # Ensure conversion to int
        self.fullplaces = int(fullplaces)    # Ensure conversion to int
        self.location = location

    # Method to convert object to a dictionary for JSON serialization
    def to_dict(self):
        return {
            "emptyplaces": self.emptyplaces,  # Already converted to int
            "fullplaces": self.fullplaces,    # Already converted to int
            "location": self.location
        }

# Creating 3 instances for 3 different garages, including location URLs
garage1 = GarageStatus(emptyplaces=int(0), fullplaces=int(6), location="https://maps.app.goo.gl/ADbYHBEvx9AT3J54A")
garage2 = GarageStatus(emptyplaces=int(3), fullplaces=int(3), location="https://maps.app.goo.gl/2aAutYdkyRy2NjAF8")
garage3 = GarageStatus(emptyplaces=int(4), fullplaces=int(2), location="https://maps.app.goo.gl/bibvj2RMA3KZg6DG8")

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

# Stop the loop and disconnect from the broker
client.loop_stop()
client.disconnect()
print("Disconnected from Broker.")