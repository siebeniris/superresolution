# import tensorflow as tf
import numpy as np
from typing import List


def augment_rotate90(
    xmins: List[int],
    xmaxs: List[int],
    ymins: List[int],
    ymaxs: List[int],
    width: int,
    height: int,
):
    """
    Rotates the definition of bounding boxes by 90°

    We transform the coordinate system of the image to its center.
    Apply the transformation matrix on the max xy point and min xy point:
    [ 0 ;-1 ]  *  [ x ]
    [ 1 ; 0 ]     [ y ]
    Afterwards the coordinate system is transformed back again.

    >>> augment_rotate90([3],[9],[2],[6],14,14)
    ([2.0], [6.0], [4.0], [10.0])
    >>> augment_rotate90([2.0], [6.0], [4.0], [10.0],14,14)
    ([4.0], [10.0], [7.0], [11.0])
    >>> augment_rotate90([4.0], [10.0], [7.0], [11.0],14,14)
    ([7.0], [11.0], [3.0], [9.0])
    >>> augment_rotate90([7.0], [11.0], [3.0], [9.0],14,14)
    ([3.0], [9.0], [2.0], [6.0])
    """

    rotation_matrix = np.array([[0, 1], [-1, 0]], dtype=float)

    min_values = np.array([xmins, ymins], dtype=float)
    max_values = np.array([xmaxs, ymaxs], dtype=float)
    xoffset = (width - 1) / 2
    yoffset = (height - 1) / 2
    min_values[0] -= xoffset
    min_values[1] -= yoffset
    max_values[0] -= xoffset
    max_values[1] -= yoffset

    min_values = np.matmul(rotation_matrix, min_values)
    max_values = np.matmul(rotation_matrix, max_values)

    # invert offsets since image was rotated
    min_values[0] += yoffset
    min_values[1] += xoffset
    max_values[0] += yoffset
    max_values[1] += xoffset

    # x min and max remain min and max after rotation
    xmins = min_values[0,].tolist()
    xmaxs = max_values[0,].tolist()
    # y min and max switch places after rotation
    ymins = max_values[1,].tolist()
    ymaxs = min_values[1,].tolist()

    return xmins, xmaxs, ymins, ymaxs


def augment_flip(
    xmins: List[int],
    xmaxs: List[int],
    ymins: List[int],
    ymaxs: List[int],
    width: int,
    height: int,
    vh: str,
):
    """
    Flips bounding boxes along a mirror axis.

    :param vh
        if contains v -> flips UP and DOWN
        if contains h -> flips LEF and RIGHT


    >>> augment_flip([1],[5],[2],[4], 10,10,'h')
    ([4.0], [8.0], [2.0], [4.0])
    >>> augment_flip([4],[8],[2],[4], 10,10,'h')
    ([1.0], [5.0], [2.0], [4.0])
    >>> augment_flip([1],[5],[2],[4], 10,10,'v')
    ([1.0], [5.0], [5.0], [7.0])
    >>> augment_flip([1],[5],[5],[7],10,10,'v')
    ([1.0], [5.0], [2.0], [4.0])
    """
    min_values = np.array([xmins, ymins], dtype=float)
    max_values = np.array([xmaxs, ymaxs], dtype=float)

    xoffset = (width - 1) / 2
    yoffset = (height - 1) / 2
    min_values[0] -= xoffset
    min_values[1] -= yoffset
    max_values[0] -= xoffset
    max_values[1] -= yoffset

    if "h" in vh:
        new_x_maxs = np.multiply(-1, min_values[0,])
        new_x_mins = np.multiply(-1, max_values[0,])
        min_values[0,] = new_x_mins
        max_values[0,] = new_x_maxs

    if "v" in vh:
        new_y_maxs = np.multiply(-1, min_values[1,])
        new_y_mins = np.multiply(-1, max_values[1,])
        min_values[1,] = new_y_mins
        max_values[1,] = new_y_maxs

    min_values[0] += xoffset
    min_values[1] += yoffset
    max_values[0] += xoffset
    max_values[1] += yoffset

    return (
        min_values[0,].tolist(),
        max_values[0,].tolist(),
        min_values[1,].tolist(),
        max_values[1,].tolist(),
    )
