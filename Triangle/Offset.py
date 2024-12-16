import numpy as np

from Triangle.Constants import *


class OffsetFunctions:
    """
    returns the dictionaries of functions for:
    1. intrinsic laser frequency offsets
    2. clock frequency offsets
    3. clock time offsets
    """

    def __init__(self, laser_flag=False, clock_flag=False):
        self.laser_flag = laser_flag
        self.clock_flag = clock_flag

    def LaserOffsets(self, O0=default_laser_O0):
        """
        laser frequency offsets
        """
        flag = self.laser_flag
        func = {}
        if flag:
            for label in MOSA_labels:
                func[label] = np.poly1d(
                    [
                        O0[label],
                    ]
                )
        else:
            for label in MOSA_labels:
                func[label] = self._zero_function
        return func

    def ClockFreqOffsets(self, y0=default_clock_y0, y1=default_clock_y1, y2=default_clock_y2):
        """
        clock frequency offsets [ffd]
        """
        flag = self.clock_flag
        func = {}
        if flag:
            for label in SC_labels:
                func[label] = np.poly1d([y2[label], y1[label], y0[label]])
        else:
            for label in SC_labels:
                func[label] = self._zero_function
        return func

    def ClockOffsets(
        self,
        t0=default_clock_t0,
        y0=default_clock_y0,
        y1=default_clock_y1,
        y2=default_clock_y2,
    ):
        """
        clock offsets [s]
        """
        flag = self.clock_flag
        func = {}
        if flag:
            for label in SC_labels:
                func[label] = np.poly1d([1.0 / 3.0 * y2[label], 0.5 * y1[label], y0[label], t0[label]])
        else:
            for label in SC_labels:
                func[label] = self._zero_function
        return func

    def _zero_function(self, t):
        return np.zeros_like(t)
