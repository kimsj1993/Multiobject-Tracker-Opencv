from __future__ import print_function
import sys
import cv2
from random import randint
import json
import time
import pandas as pd
import numpy as np




if __name__ == '__main__':

    # Set video to load
    videoPath = "videos/TestVideo.MOV"

    # Create a video capture object to read videos
    cap = cv2.VideoCapture(videoPath)

    # Read first frame
    success, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    # quit if unable to read the video file
    if not success:
        print('Failed to read video')
        sys.exit(1)

    ## Select boxes
    bboxes = []
    colors = []
    pointsForKeys = []

    # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # So we will call this function in a loop till we are done selecting all objects
    while True:
        # draw bounding boxes over objects
        # selectROI's default behaviour is to draw box starting from the center
        # when fromCenter is set to false, you can draw box starting from top left corner
        bbox = cv2.selectROI('MultiTracker', frame)
        bboxes.append(bbox)
        colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv2.waitKey(0) & 0xFF
        if k == 113:  # q is pressed
            break

    print('Selected bounding boxes {}'.format(bboxes))



    # Create MultiTracker object
    multiTracker = cv2.MultiTracker_create()


    # Initialize MultiTracker
    for bbox in bboxes:
        multiTracker.add(cv2.TrackerCSRT_create(), frame, bbox)



    # Process video and track objects
    start = time.time()
    while cap.isOpened():
        success, frame = cap.read()
        try:
            frame = cv2.resize(frame, (640, 480))
        except Exception as e:
            print(str(e))

        if not success:
            break

        # get updated location of objects in subsequent frames
        success, boxes = multiTracker.update(frame)

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
            end = time.time()
            b = {'id': i, 'x': newbox[0] + newbox[2]/2, 'y': newbox[1] + newbox[3]/2, 'time': end - start}
            pointsForKeys.append(dict(b))


        # show frame
        cv2.imshow('MultiTracker', frame)

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            pointsForKeys = sorted(pointsForKeys, key=lambda k: k['time'])
            with open('result.json', 'w') as fp:
                json.dump(pointsForKeys, fp)

    pointsForKeys = sorted(pointsForKeys, key=lambda k: k['id'])
    df = pd.DataFrame(pointsForKeys,  columns = ['id', 'x', 'y', 'time'])
    df.to_csv('TestVideo.csv', encoding='utf-8')
