from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np

import math

__all__ = [
    'meanshift_cluster'
]

################################################################################
# UTILITIES ====================================================================
################################################################################
def meanshift_cluster(data, bandwidth=None, bin_seeding=True, as_arg=False):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    if bandwidth is None:
        bandwidth = estimate_bandwidth(data)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)
    ms.fit(data)

    labels = ms.labels_
    n_clusters = len(np.unique(labels))

    if as_arg:
        return [np.argwhere(labels == k).flatten() for k in range(n_clusters)]
    else:
        return [data[labels == k, 0] for k in range(n_clusters)]
