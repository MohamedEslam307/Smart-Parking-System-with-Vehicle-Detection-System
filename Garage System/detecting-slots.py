import cv2 as cv
import numpy as np

# Load the image of the garage (or use a video feed)
image = cv.imread("slots-test-image.jpg")

# Step 1: Preprocessing the image (convert to grayscale and detect edges)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)
edges = cv.Canny(blurred, 50, 150)

# Step 2: Detect lines using Hough Line Transform to find edges of parking slots
lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)

# Create a copy of the original image to draw the detected lines
line_image = np.copy(image)

# Draw the detected lines on the image (optional)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Step 3: Detect contours to find parking slots
# Apply another level of thresholding to get the regions of interest (potential parking slots)
thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

# Find contours based on the thresholded image
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Filter out contours that do not resemble parking slots (based on area/shape)
min_area = 5000  # Adjust this value based on the image scale
parking_slots = []

for contour in contours:
    area = cv.contourArea(contour)
    if area > min_area:  # Assuming parking slots have larger areas
        # Approximate the contour to reduce the number of vertices
        approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)

        # Consider it a parking slot if it has 4 vertices (a rectangle or square)
        if len(approx) == 4:
            parking_slots.append(approx)
            # Draw the detected parking slots on the image
            cv.drawContours(image, [approx], -1, (0, 255, 0), 3)

# Step 4: Display the results
# Show the original image with the detected parking slots
cv.imshow('Detected Parking Slots', image)

# Show the image with detected lines (optional)
# cv.imshow('Detected Lines', line_image)

cv.waitKey(0)
cv.destroyAllWindows()

