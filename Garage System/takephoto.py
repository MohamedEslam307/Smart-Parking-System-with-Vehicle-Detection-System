import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Give the camera time to warm up (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Capture a frame
ret, frame = cap.read()

# If frame is read correctly, save the image
if ret:
    # Save the photo to your desired path
    cv2.imwrite('/home/raspi/Smart-Parking-System-with-Vehicle-Detection-System/test.jpg', frame)
    print("Photo captured successfully!")
else:
    print("Error: Could not read frame.")

# Release the camera
cap.release()
cv2.destroyAllWindows()


