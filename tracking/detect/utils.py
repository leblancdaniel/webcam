import numpy as np

def expand_box(box: np.ndarray, factor: float, x_max: int, y_max: int) -> np.ndarray:
    """ Expand bounding box by a constant factor.
        Arguments
        ---------
        box: Numpy array, input bounding box, should have format (x1, y1, x2, y2)
        factor: float, factor to expand equally in both dimensions
        x_max: int, maximum value in the x dimension
        y_max: int, maximum value in the y dimension
    
        Returns
        -------
        box: Numpy array with expanded bounding box with format (x1, y1, x2, y2)
    """
    xs = box[[0, 2]]
    ys = box[[1, 3]]

    box_width = xs[1] - xs[0]
    box_height = ys[1] - ys[0]

    dx = box_width / 2 * (factor - 1)
    dy = box_height / 2 * (factor - 1)

    # Need to make sure the expanded boxes don't go beyond the
    # bounds of the original image.
    expanded_xs = np.clip(xs + np.array([-dx, dx]), 0, x_max)
    expanded_ys = np.clip(ys + np.array([-dy, dy]), 0, y_max)

    # Interleave x and y position
    expanded = np.empty(box.size, dtype=box.dtype)
    expanded[0::2] = expanded_xs
    expanded[1::2] = expanded_ys

    return expanded
