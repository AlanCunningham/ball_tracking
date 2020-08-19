from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import requests


# https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=0, help="Max buffer size")
    args = vars(ap.parse_args())

    # Set the lower and upper boundries of "green" and the max buffer of tracked
    # points.
    green_lower = (29, 86, 6)
    green_upper = (64, 255, 255)
    points = deque(maxlen=args["buffer"])

    # Use webcam if video not provided
    if not args.get("video", False):
        print("Webcam")
        video_stream = VideoStream(src=0).start()
    else:
        print(f"Video: {args['video']}")
        video_stream = cv2.VideoCapture(args["video"])
    time.sleep(2)

    while True:
        # Get the current frame
        frame = video_stream.read()
        # Handle the frame from the video file or the webcam
        frame = frame[1] if args.get("video", False) else frame
        # If we reach no frames, we have reached the end of the video
        if frame is None:
            print("Frame is none")
            break

        # Resize, blur and convert to HSV colour space
        frame = imutils.resize(frame, width=240)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Create a mask for the colour green
        mask = cv2.inRange(hsv, green_lower, green_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours and the centre of the ball
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        center = None

        # Contours found
        if len(contours) > 0:
            # Find the largers contour in the mask and compute the min enclosing
            # circle and centroid
            largest_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            M = cv2.moments(largest_contour)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Only proceed if the radius meets the minimum size
            MIN_SIZE = 10
            if radius > MIN_SIZE:
                print(f"X: {x} | Y: {y}")
                params = {"x": x, "y": y}
                url = "http://192.168.0.244:8000/ball"
                requests.get(url, params=params)
                # Draw the circle and centroid on frame
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # Update the points queue
        # points.appendleft(center)
        # for i in range(1, len(points)):
        #     if points[i - 1] is None or points[i] is None:
        #         continue
        #     thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        #     cv2.line(frame, points[i - 1], points[i], (0, 0, 255), thickness)

        # Show the frame on screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Clean up the video file / webcam
    if not args.get("video", False):
        video_stream.stop()
    else:
        video_stream.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()