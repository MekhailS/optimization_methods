import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


def orthogonality_of_curve_segments(curve_x):
    curve_x = np.asarray(curve_x)
    curve_segments = [curve_x[i] - curve_x[i-1] for i in range(1, len(curve_x))]

    segments_pairs_dot_products = [curve_segments[i] @ curve_segments[i-1]
                                   for i in range(1, len(curve_segments))]
    return segments_pairs_dot_products

