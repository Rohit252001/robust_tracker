import numpy as np
from typing import List, Tuple, Optional
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque

class RobustFaceTracker:
    """A robust face tracking model using Kalman filtering and IoU-based association.

    Args:
        max_age (int): Maximum age (frames) before a track is removed. Default: 30.
        min_hits (int): Minimum detections required to confirm a track. Default: 3.
        iou_threshold (float): IoU threshold for matching detections to tracks. Default: 0.3.
        buffer_size (int): Size of the buffer for inactive tracks. Default: 50.
    """
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3, buffer_size: int = 50):
        self.next_id = 1
        self.tracks = {}
        self.inactive_tracks = {}
        self.buffer = deque(maxlen=buffer_size)
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def update(self, detections: List[np.ndarray]) -> List[Tuple[int, np.ndarray]]:
        """Update the tracker with new detections.

        Args:
            detections: List of NumPy arrays, each [x1, y1, x2, y2, score].

        Returns:
            List of tuples (track_id, bbox), where bbox is [x1, y1, x2, y2].
        """
        # Predict for all tracks
        for track_id in list(self.tracks.keys()):
            kf = self.tracks[track_id]['kf']
            kf.predict()
            self.tracks[track_id]['bbox'] = kf.x[:4].reshape(-1)

        # Associate detections to tracks
        unmatched_tracks, unmatched_detections = self.associate_detections_to_tracks(detections)

        # Update matched tracks
        matched_tracks = [tid for tid in self.tracks if tid not in unmatched_tracks]
        for track_id in matched_tracks:
            self.tracks[track_id]['hits'] += 1
            self.tracks[track_id]['age'] = 0

        # Age unmatched tracks
        for track_id in unmatched_tracks:
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                self.inactive_tracks[track_id] = self.tracks.pop(track_id)
                self.buffer.append(track_id)

        # Initialize new tracks
        for detection in unmatched_detections:
            self.init_new_track(detection)

        # Clean buffer
        while len(self.buffer) > self.buffer.maxlen:
            old_id = self.buffer.popleft()
            self.inactive_tracks.pop(old_id, None)

        # Return confirmed tracks
        return [(tid, track['bbox']) for tid, track in self.tracks.items() if track['hits'] >= self.min_hits]

    @staticmethod
    def init_kalman_filter() -> KalmanFilter:
        """Initialize a Kalman filter for tracking."""
        kf = KalmanFilter(dim_x=6, dim_z=4)
        kf.F = np.array([[1, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1],
                         [0, 0, 1, 0, 1, 0],
                         [0, 0, 0, 1, 0, 1],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]], dtype=np.float32)
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0]], dtype=np.float32)
        kf.R = np.eye(4, dtype=np.float32) * 10
        kf.Q = np.eye(6, dtype=np.float32) * 0.1
        return kf

    def init_new_track(self, detection: np.ndarray):
        """Initialize a new track with a detection."""
        kf = self.init_kalman_filter()
        kf.x[:4] = detection[:4].reshape(-1, 1)
        self.tracks[self.next_id] = {
            'kf': kf,
            'bbox': detection[:4],
            'hits': 1,
            'age': 0,
            'score': detection[4]
        }
        self.next_id += 1

    def associate_detections_to_tracks(self, detections: List[np.ndarray]) -> Tuple[List[int], List[np.ndarray]]:
        """Associate detections to existing tracks using IoU."""
        active_tracks = list(self.tracks.keys())
        all_tracks = active_tracks + list(self.inactive_tracks.keys())

        if not all_tracks or not detections:
            return active_tracks, detections

        predicted_bboxes = np.array([self.tracks[tid]['bbox'] if tid in self.tracks else self.inactive_tracks[tid]['bbox']
                                     for tid in all_tracks], dtype=np.float32)
        det_bboxes = np.array([det[:4] for det in detections], dtype=np.float32)

        # Vectorized IoU computation
        cost_matrix = 1 - self.calculate_iou_vectorized(predicted_bboxes, det_bboxes)

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_indices = list(zip(row_ind, col_ind))

        unmatched_tracks = active_tracks.copy()
        unmatched_detections = list(range(len(detections)))
        matches = []

        for i, j in matched_indices:
            if cost_matrix[i, j] <= 1 - self.iou_threshold:
                track_id = all_tracks[i]
                if track_id in active_tracks:
                    unmatched_tracks.remove(track_id)
                unmatched_detections.remove(j)
                matches.append((track_id, j))

        for track_id, det_idx in matches:
            detection = detections[det_idx]
            if track_id in self.inactive_tracks:
                self.tracks[track_id] = self.inactive_tracks.pop(track_id)
                self.buffer.remove(track_id)
            self.tracks[track_id]['kf'].update(detection[:4])
            self.tracks[track_id]['bbox'] = detection[:4]
            self.tracks[track_id]['score'] = detection[4]

        return unmatched_tracks, [detections[j] for j in unmatched_detections]

    @staticmethod
    def calculate_iou_vectorized(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
        """Vectorized IoU calculation for multiple bounding boxes."""
        x1 = np.maximum(bboxes1[:, 0:1], bboxes2[:, 0:1].T)
        y1 = np.maximum(bboxes1[:, 1:2], bboxes2[:, 1:2].T)
        x2 = np.minimum(bboxes1[:, 2:3], bboxes2[:, 2:3].T)
        y2 = np.minimum(bboxes1[:, 3:4], bboxes2[:, 3:4].T)

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection

        return intersection / np.where(union > 0, union, 1e-6)

if __name__ == "__main__":
    # Example usage
    tracker = RobustFaceTracker()
    detections = [np.array([100, 150, 140, 190, 0.95], dtype=np.float32),
                  np.array([200, 250, 240, 290, 0.90], dtype=np.float32)]
    tracked_faces = tracker.update(detections)
    print("Tracked faces:", tracked_faces)