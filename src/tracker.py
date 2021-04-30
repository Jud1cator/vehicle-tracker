import cv2

from utils import get_bboxes, IMG_PADDING


class Tracker:
    def __init__(self, frame_shape, max_search_dist=100, ttl=15):
        self.frame_shape = frame_shape
        self.max_search_dist = max_search_dist
        self.ttl = ttl
        self.tracks = dict()
        self.new_track_id = 0

    def update(self, detections):
        """
        Update current tracks with detections from the next frame.
        :param detections: list of bounding boxes of detected vehicles on the current frame
        :type detections: List
        """
        unassigned_detections = detections
        to_delete_idxs = []

        for track_id, data in self.tracks.items():
            # If track was dead for sufficient number of frames, delete it
            if data['dead_frames'] > self.ttl:
                to_delete_idxs.append(track_id)
                continue

            x, y = data['points'][-1]
            best_det_idx = -1
            best_dist = self.max_search_dist
            best_cp = (0, 0)
            best_bbox = (0, 0, 0, 0)

            for i in range(len(unassigned_detections)):
                x_min, y_min, x_max, y_max = get_bboxes(
                    unassigned_detections[i], self.frame_shape)
                x_cp = int(x_min + (x_max - x_min) / 2)
                y_cp = int(y_min + (y_max - y_min) / 2)
                dist = (x-x_cp)**2+(y-y_cp)**2
                if dist < best_dist and dist < self.max_search_dist:
                    best_det_idx = i
                    best_dist = dist
                    best_cp = (x_cp, y_cp)
                    best_bbox = (x_min, y_min, x_max, y_max)

            # If no update found for this track, increase it's dead_frames
            if best_det_idx == -1:
                self.tracks[track_id]['dead_frames'] += 1
            else:
                unassigned_detections.pop(best_det_idx)
                self.tracks[track_id]['points'].append(best_cp)
                self.tracks[track_id]['last_bbox'] = best_bbox
                self.tracks[track_id]['dead_frames'] = 0

        # Delete all dead tracks
        for i in to_delete_idxs:
            self.tracks.pop(i)

        # For every unassigned detection left, create new track
        for det in unassigned_detections:
            x_min, y_min, x_max, y_max = get_bboxes(det, self.frame_shape)
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
