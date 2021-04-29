import matplotlib.pyplot as plt
import cv2

from utils import (
    get_detector,
    detect_vehicles
)


def main():
    detector = get_detector()
    video_stream = cv2.VideoCapture("/video/video.mkv")
    ret, frame = video_stream.read()
    out = detect_vehicles(detector, frame)


if __name__ == "__main__":
    main()
