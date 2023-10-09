import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for green in HSV
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # Create a mask to isolate the green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Perform morphological operations to clean up the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store ball detection status
    ball_detected = False

    # Check if any contours are found
    if len(contours) > 0:
        # Find the largest contour (assumed to be the tennis ball)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the center and radius of the detected ball
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        # Draw the circle around the detected ball
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        
        # Set the flag to indicate ball detection
        ball_detected = True

    # Display the frame with or without ball detection
    if ball_detected:
        cv2.putText(frame, 'Tennis Ball Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Tennis Ball Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

