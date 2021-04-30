import matplotlib.pyplot as plt
import cv2
import numpy as np

from utils import (
    get_detector,
    detect_vehicles,
    draw_detections
)


def main():
    detector = get_detector()
    video_stream = cv2.VideoCapture("/video/video.mkv")
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    video_writer = cv2.VideoWriter("/output/output.mp4", fourcc, 15, (640, 480))
    while True:
        try:
            ret, frame = video_stream.read()
            if not ret:
                break
            detections = detect_vehicles(detector, frame)
            frame = draw_detections(frame, detections)
            video_writer.write(frame)
        except KeyboardInterrupt:
            video_writer.release()


if __name__ == "__main__":
    main()
