import mos.utils

from scipy.spatial import distance as dist
from collections import OrderedDict, deque

import numpy as np

__all__ = [
    'CentroidTracker'
]

class CentroidTracker:
    """
    Track centroids from frame to frame by inputting detection bounding boxes.

    Requires updates to be sent from detectors.

    Also computes a moving average of velocities, and computes a 'confidence'
    score for each tracked centroid.

    Modified from:
    pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
    """
    def __init__(self,
                 max_missing_count=50,
                 max_distance=50,
                 velocity_window=10):
        self.next_id = 0
        self.tracked_centroids = OrderedDict()

        self.velocities = OrderedDict()
        self.velocity_average_handlers = OrderedDict()
        self.velocity_window = velocity_window

        self.confidences = OrderedDict()

        # Counts number of times an object has been missing from an update
        # Remove object if count > max_missing_count
        self.missing_count = OrderedDict()
        self.max_missing_count = max_missing_count
        self.max_distance = max_distance

    def register(self, centroid): # Register centroid
        self.tracked_centroids[self.next_id] = centroid
        self.missing_count[self.next_id] = 0

        self.velocities[self.next_id] = np.zeros(centroid.shape)
        self.velocity_average_handlers[self.next_id] = mos.utils.MovingAverage(
            window=self.velocity_window,
            initial_entry=np.zeros(centroid.shape)
        )

        self.confidences[self.next_id] = 0

        self.next_id += 1

    def deregister(self, centroid_id):
        del self.tracked_centroids[centroid_id]
        del self.missing_count[centroid_id]

        del self.velocities[centroid_id]
        del self.velocity_average_handlers[centroid_id]
        del self.confidences[centroid_id]

    def update(self, rects):
        # If update is empty, mark all existing objects missing
        if len(rects) == 0:
            for centroid_id in list(self.missing_count.keys()):
                self.missing_count[centroid_id] += 1
                self.confidences[centroid_id] -= 1

                # Deregister centroids if missing for a long time
                if self.missing_count[centroid_id] > self.max_missing_count:
                    self.deregister(centroid_id)

            return self.tracked_centroids

        # Get centroids from input boxes
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x_1, y_1, x_2, y_2)) in enumerate(rects):
            input_centroids[i] = (
                mos.utils.midpoint_from_points([[x_1, y_1],[x_2, y_2]])
            )

        # Register new centroids
        if len(self.tracked_centroids) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        # Else, try to match input centroids to currently tracked centroids
        else:
            centroid_ids = list(self.tracked_centroids.keys())
            tracked_centroids = list(self.tracked_centroids.values())

            # Compare distance between each pair of tracked and input centroids
            D = dist.cdist(np.array(tracked_centroids), input_centroids)

            # Sort for ease of computing
            # These are indices to the sorted input and tracked centroids
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                elif D[row, col] > self.max_distance:
                    continue

                centroid_id = centroid_ids[row]

                # If matched, reset missing count and update tracked centroid
                #
                # [Update Procedure]
                # 1. Update moving average of velocity
                # 2. Update centroid
                # 3. Reset missing count and increase confidence

                # Update velocities
                current_velocity = (self.tracked_centroids[centroid_id]
                                    - input_centroids[col])
                self.velocities[centroid_id] = (
                    self.velocity_average_handlers[centroid_id].update(
                        current_velocity
                    )
                )

                # Update centroids
                self.tracked_centroids[centroid_id] = input_centroids[col]

                # Update missing count and confidence
                self.missing_count[centroid_id] = 0
                self.confidences[centroid_id] += 2

                used_rows.add(row)
                used_cols.add(col)

            # Handle unhandled elements
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # Handle un-updated tracked centroids
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    centroid_id = centroid_ids[row]
                    self.missing_count[centroid_id] += 1
                    self.confidences[centroid_id] -= 1

                    if self.missing_count[centroid_id] > self.max_missing_count:
                        self.deregister(centroid_id)

            # Else, register unmatched input centroids
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.tracked_centroids, self.confidences, self.velocities
