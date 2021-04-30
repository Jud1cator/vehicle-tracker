import argparse
import cv2
import tqdm
import yaml

from utils import (
    get_video_stream,
    get_video_writer,
)
from detector import Detector
from tracker import Tracker


def main(config):
    video_stream = get_video_stream(**config['input'])
    frame_shape = (
        int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    frames_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = Detector(**config['detector'])
    video_writer = get_video_writer(frame_shape, **config['output'])
    tracker = Tracker(frame_shape, **config['tracker'])

    for _ in tqdm.tqdm(range(frames_count)):
        ret, frame = video_stream.read()
        if not ret:
            break
        detections = detector.detect_vehicles(frame)
        tracker.update(detections)
        frame = tracker.draw_tracks(frame)
        video_writer.write(frame)
    video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        help="Path to yaml config file",
        required=True
    )
    args = parser.parse_args()
    stream = open(args.config, "r")
    config = yaml.safe_load(stream.read())
    main(config)
