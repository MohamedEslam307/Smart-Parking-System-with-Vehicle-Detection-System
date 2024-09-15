import yaml
import numpy as np
import cv2
import time

# Path references
fn = "CarPark.mp4"
fn_yaml = "CarPark.yml"
fn_out = "CarPark.avi"
cascade_src = 'CarPark.xml'

# Initialize the car classifier
car_cascade = cv2.CascadeClassifier(cascade_src)
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
    'start_frame': 0,
    'show_ids': True,
    'classifier_used': True,
    'save_video': False
}

# Set up video capture
cap = cv2.VideoCapture(fn)
video_info = {
    'fps': cap.get(cv2.CAP_PROP_FPS),
    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.6),
    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.6),
    'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
    'num_of_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
}

cap.set(cv2.CAP_PROP_POS_FRAMES, dict['start_frame'])  # Jump to frame number specified

def run_classifier(img, id):
    cars = car_cascade.detectMultiScale(img, 1.1, 1)
    return len(cars) > 0

# Define codec and create VideoWriter object
if dict['save_video']:
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter(fn_out, fourcc, video_info['fps'], (video_info['width'], video_info['height']))

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
parking_data_motion = []

if parking_data:
    for park in parking_data:
        points = np.array(park['points'])
        rect = cv2.boundingRect(points)
        points_shifted = points.copy()
        points_shifted[:, 0] = points[:, 0] - rect[0]  # Shift contour to region of interest
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
    cv2.putText(frame_rev, str(park['id']), (centroid[0] + 1, centroid[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), (centroid[0] - 1, centroid[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), (centroid[0] + 1, centroid[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), (centroid[0] - 1, centroid[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

while cap.isOpened():
    start_time = time.time()  # Start the timer
    video_cur_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    video_cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    ret, frame_initial = cap.read()
    
    if ret:
        frame = cv2.resize(frame_initial, None, fx=0.6, fy=0.6)
    if not ret:
        print("Video ended")
        break

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

            if status != parking_status[ind] and parking_buffer[ind] is None:
                parking_buffer[ind] = video_cur_pos
                change_pos = video_cur_pos
            elif status != parking_status[ind] and parking_buffer[ind] is not None:
                if video_cur_pos - parking_buffer[ind] > dict['park_sec_to_wait']:
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
                rect = parking_bounding_rects[ind]
                roi_gray_ov = frame_gray[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
                parking_data_motion.append(roi_gray_ov)
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
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if dict['save_video']:
        out.write(frame_out)

    # Calculate the delay needed to match the frame rate
    frame_delay = int(1000 / video_info['fps'])
    elapsed_time = time.time() - start_time
    delay = max(frame_delay - int(elapsed_time * 1000), 0)
    
    # Print free parking slots
    free_slots = [i for i, status in enumerate(parking_status) if status]
    free_slots_text = f"Free slots: {len(free_slots)}"
    cv2.putText(frame_out, free_slots_text, (5, frame_out.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow('Video', frame_out)

    # Wait for the appropriate amount of time
    cv2.waitKey(delay)

    # Remove this if you want the video to play without waiting for 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if dict['save_video']:
    out.release()
cv2.destroyAllWindows()
