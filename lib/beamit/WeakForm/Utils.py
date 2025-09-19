# !/usr/bin/env python3
#
# Copyright (c) 2025, the beamit authors, all rights reserved.
#

import numpy as np
from enum import Enum


def skew_symmetric_matrices(vectors):
    """
    Given an (n, 3) array of **vectors**, return an (n, 3, 3) array of their skew-symmetric matrix form.
    """
    # check if vectors have shape (n, 3)
    if (vectors.ndim != 2 or vectors.shape[1] != 3):
        raise ValueError("Input must be an (n, 3) array of vectors.")

    n = vectors.shape[0]
    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]

    skew = np.zeros((n, 3, 3), dtype=vectors.dtype)
    skew[:, 0, 1] = -z
    skew[:, 0, 2] = y
    skew[:, 1, 0] = z
    skew[:, 1, 2] = -x
    skew[:, 2, 0] = -y
    skew[:, 2, 1] = x

    return skew


class SolutionUpdateType(Enum):
    """
    Enum for the type of update to be applied to the solution.

    Attributes:
        ADD_TNS_ADD_ROT: Additive update to both translations and rotations.
        ADD_TNS_MUL_ROT: Additive update to translations and multiplicative update to rotations.
    """
    ADD_TNS_ADD_ROT = 1
    ADD_TNS_MUL_ROT = 2
