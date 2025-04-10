# Robust Tracker

A lightweight, optimized face tracking model using Kalman filtering and IoU-based association.


### Input
- **Method**: `update(detections)`
- **Type**: `List[np.ndarray]`
  - Each `np.ndarray` is of shape `(5,)` with elements `[x1, y1, x2, y2, score]` (dtype: `np.float32`).
  - Example: `[np.array([100, 150, 140, 190, 0.95]), np.array([200, 250, 240, 290, 0.90])]`

### Output
- **Type**: `List[Tuple[int, np.ndarray]]`
  - Each tuple contains:
    - `track_id`: Integer ID of the tracked face.
    - `bbox`: `np.ndarray` of shape `(4,)` with coordinates `[x1, y1, x2, y2]` (dtype: `np.float32`).
  - Example: `[(1, np.array([100, 150, 140, 190])), (2, np.array([200, 250, 240, 290]))]`

### How It Works
1. **Initialization**: Sets up tracker with configurable parameters (`max_age`, `min_hits`, etc.).
2. **Prediction**: Uses Kalman filters to predict face positions for active tracks.
3. **Association**: Matches detections to tracks using IoU and the Hungarian algorithm.
4. **Track Management**: Updates matched tracks, ages unmatched ones, and initializes new tracks for unmatched detections.
5. **Output**: Returns confirmed tracks with sufficient hits.

### Usage
```python
tracker = RobustFaceTracker()
detections = [np.array([100, 150, 140, 190, 0.95]), np.array([200, 250, 240, 290, 0.90])]
tracked_faces = tracker.update(detections)
print("Tracked faces:", tracked_faces)

## Installation
```bash
pip install git+https://github.com/Rohit252001/robust_tracker.git
