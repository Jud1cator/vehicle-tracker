import cv2

from utils import (
    get_detector,
    detect_vehicles
)
from tracker import Tracker


def main():
    detector = get_detector()
    video_stream = cv2.VideoCapture("/video/video.mkv")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("/output/output.mp4", fourcc, 15, (640, 480))
    tracker = Tracker((640, 480))

    while True:
        try:
            ret, frame = video_stream.read()
            if not ret:
                break
            detections = detect_vehicles(detector, frame)
            tracker.update(detections)
            frame = tracker.draw_tracks(frame)
            video_writer.write(frame)
        except KeyboardInterrupt:
            video_writer.release()


if __name__ == "__main__":
    main()
