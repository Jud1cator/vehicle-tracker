import cv2
import numpy as np

from utils import get_bbox, IMG_PADDING, compute_iou


class Tracker:
    def __init__(self, frame_shape, max_search_dist=-0.2, ttl=10):
        self.frame_shape = frame_shape
        self.max_search_dist = max_search_dist
        self.ttl = ttl
        self.tracks = dict()
        self.new_track_id = 0

    def update(self, detections):
        """
        Update current tracks with detections from the next frame.
        Uses distance to estimated trajectory as metric for matching boxes with tracks.
        :param detections: list of bounding boxes of detected vehicles on the current frame
        :type detections: List
        """
        distances = np.zeros((len(self.tracks), len(detections)))
        mapping = dict(zip(range(len(self.tracks)), self.tracks.keys()))
        updated_tracks = []
        assigned_detections = []

        # Build matrix of distances between current detections and tracks
        for i in range(len(self.tracks)):
            x, y = self.tracks[mapping[i]]['points'][-1]
            track_bbox = self.tracks[mapping[i]]['last_bbox']
            for j in range(len(detections)):
                det_bbox = get_bbox(
                    detections[j], self.frame_shape)
                distances[i, j] = -compute_iou(track_bbox, det_bbox)

        # Assign detections to closest tracks
        if len(self.tracks) > 0 and len(detections) > 0:
            while np.min(distances) < self.max_search_dist:
                track_idx, det_idx = divmod(
                    distances.argmin(), distances.shape[1])
                bbox = get_bbox(detections[det_idx], self.frame_shape)
                x_min, y_min, x_max, y_max = bbox
                x_cp = int(x_min + (x_max - x_min) / 2)
                y_cp = int(y_min + (y_max - y_min) / 2)
                self.tracks[mapping[track_idx]]['last_bbox'] = bbox
                self.tracks[mapping[track_idx]]['points'].append((x_cp, y_cp))
                self.tracks[mapping[track_idx]]['dead_frames'] = 0
                distances[track_idx, :] = np.inf
                distances[:, det_idx] = np.inf
                updated_tracks.append(track_idx)
                assigned_detections.append(det_idx)

        # Update count of dead frames for lost tracks and delete old ones
        for i in mapping.keys():
            if i in updated_tracks:
                continue
            self.tracks[mapping[i]]['dead_frames'] += 1
            if self.tracks[mapping[i]]['dead_frames'] > self.ttl:
                self.tracks.pop(mapping[i])

        # Create new tracks for unassigned detections
        for i in range(len(detections)):
            if i in assigned_detections:
                continue
            x_min, y_min, x_max, y_max = get_bbox(
                detections[i], self.frame_shape)
            bbox = (x_min, y_min, x_max, y_max)
            x_cp = int(x_min + (x_max - x_min) / 2)
            y_cp = int(y_min + (y_max - y_min) / 2)
            self.tracks[self.new_track_id] = dict()
            self.tracks[self.new_track_id]['points'] = [(x_cp, y_cp)]
            self.tracks[self.new_track_id]['last_bbox'] = bbox
            self.tracks[self.new_track_id]['dead_frames'] = 0
            self.new_track_id += 1

    def draw_tracks(self, frame):
        """
        Draw tracks on a given frame
        :param frame: input frame
        :type frame: np.array
        :return: frame with tracks and last bounding boxes
        :rtype: np.array
        """
        for track_id, data in self.tracks.items():
            if data['dead_frames'] != 0:
                continue
            # Draw current bounding box
            x_min, y_min, x_max, y_max = data['last_bbox']
            cv2.rectangle(
                frame,
                (x_min, y_min + IMG_PADDING[1]),
                (x_max, y_max + IMG_PADDING[1]),
                (0, 255, 0), 2
            )
            # Draw track
            for i in range(1, len(data['points'])):
                x1, y1 = data['points'][i-1][0], data['points'][i-1][1]
                x2, y2 = data['points'][i][0], data['points'][i][1]
                cv2.line(
                    frame,
                    (x1, y1 + IMG_PADDING[1]),
                    (x2, y2 + IMG_PADDING[1]),
                    (0, 255, 0), 2
                )
        return frame
