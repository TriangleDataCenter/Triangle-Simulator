import logging

import numpy as np
import scipy.sparse
from scipy.integrate import cumulative_trapezoid
from scipy.signal import firwin, kaiserord, lfilter

from Triangle.Constants import *
from Triangle.Noise import *

logger = logging.getLogger(__name__)

sci_downsample_kwargs = dict(fsample=4, downsample=40, kaiser_filter_coef=[240, 0.02, 0.08])


def downsampling(data, fsample, downsample, kaiser_filter_coef):
    # teleport sampling rate
    f_nyquist = fsample / 2.0
    teleport_fsample = fsample / downsample
    logger.debug("Downsampling to " + str(teleport_fsample) + " Hz.")

    # kaiser window parameters
    attenuation, freq1, freq2 = kaiser_filter_coef

    # filter parameters
    Ntaps, beta = kaiserord(attenuation, (freq2 - freq1) / f_nyquist)
    taps = firwin(Ntaps, (freq1 + freq2) / fsample, window=("kaiser", beta))
    logger.debug("Anti-aliase filtering with " + str(Ntaps) + "-order kaiser window.")

    # aafiltering & downsampling
    aafiltered_data = lfilter(taps, 1, data)
    downsampled_data = aafiltered_data[::downsample]
    logger.debug("Downsampled to " + str(teleport_fsample) + " Hz.")
    return downsampled_data


def detrending(data, order=2):
    x = np.arange(len(data))
    p = np.polyfit(x, data, order)
    y_fit = np.polyval(p, x)
    return data - y_fit


def DataSlice(datadict, time, time1, time2):
    start_idx = np.where(time >= time1)[0][0]
    end_idx = np.where(time <= time2)[0][-1]
    new_datadict = dict()
    for k, v in datadict.items():
        new_datadict[k] = v[start_idx : end_idx + 1]
    return new_datadict


def SliceDataDictionary(data_dict, time, time_start, time_end, buffer_time=80, channels=["A2", "E2", "T2"]):
    indmin = np.where(time > max(time_start + buffer_time, time[0]))[0][0]
    indmax = np.where(time < min(time_end - buffer_time, time[-1]))[0][-1]
    sliced_data_dict = dict()
    for chan in data_dict.keys():
        sliced_data_dict[chan] = (data_dict[chan]).copy()
        if chan in channels:
            sliced_data_dict[chan][:indmin] = 0.0
            sliced_data_dict[chan][indmax:] = 0.0
    return sliced_data_dict


# ================ methods for general dict =============================


def Dict2Array(d, keys=None):
    """
    the generated array must follow the order given by keys
    """
    if isinstance(keys, list):
        if set(keys) == set(d.keys()):
            a = []
            for key in keys:
                a.append(d[key])
            return np.array(a)
    else:
        return np.array(np.array(d.values()))


def Array2Dict(a, keys):
    """
    the array must follow the order given by keys
    """
    d = {}
    for v, k in zip(a, keys):
        d[k] = v
    return d


def AddDicts(dict_list, coef_list=1.0):
    """
    calculate linear combination of dicts
    the keys of dicts should be the same
    """
    if np.isscalar(coef_list):
        coef_list = np.ones(len(dict_list)) * coef_list
    key_list = []
    for d in dict_list:
        key_list.append(set(d.keys()))
    if all(x == key_list[0] for x in key_list[1:]):
        cd0 = {k: coef_list[0] * v for k, v in dict_list[0].items()}
        for d, c in zip(dict_list[1:], coef_list[1:]):
            for k in cd0.keys():
                cd0[k] += c * d[k]
    else:
        raise ValueError("keys of dictionaries mismatch.")
    return cd0


def MultiplyDicts(dict1, dict2, coef=1.0):
    if set(dict1.keys()) == set(dict2.keys()):
        dict3 = {}
        for k in dict1.keys():
            dict3[k] = dict1[k] * dict2[k] * coef
        return dict3
    else:
        raise ValueError("keys of dictionaries mismatch.")


# ========================== MOSA/SC dictionary wrapper ============================
class MOSADict:
    """
    NOTE: be careful when manipulating array or dict elements
    NOTE: use B = A.copy() instead of B = A
    the keys of MOSADict must be MOSA_labels
    the values of MOSADict are numpy arrays of the same dim or scalars
    """

    dict_keys = MOSA_labels

    def __init__(self, data_dict):
        if isinstance(data_dict, dict):
            if set(data_dict.keys()) == set(self.dict_keys):
                self.data = data_dict
            else:
                raise ValueError("keys of dictionaries mismatch.")
        else:
            raise TypeError("data_dict must be a dict.")

    def __repr__(self):
        return str(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        if key in self.dict_keys:
            self.data[key] = value
        else:
            raise ValueError(str(key) + " is not a surpported key.")

    def __add__(self, other):
        if isinstance(other, MOSADict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] + other.data[key]
            return MOSADict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] + other
            return MOSADict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'MOSADict' and '{}'".format(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, MOSADict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] * other.data[key]
            return MOSADict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] * other
            return MOSADict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'MOSADict' and '{}'".format(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, MOSADict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] - other.data[key]
            return MOSADict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] - other
            return MOSADict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'MOSADict' and '{}'".format(type(other)))

    def __rsub__(self, other):
        if isinstance(other, MOSADict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other.data[key] - self.data[key]
            return MOSADict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other - self.data[key]
            return MOSADict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'MOSADict' and '{}'".format(type(other)))

    def __neg__(self):
        return MOSADict({key: -self.data[key] for key in self.dict_keys})

    def __truediv__(self, other):
        if isinstance(other, MOSADict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] / other.data[key]
            return MOSADict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] / other
            return MOSADict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'MOSADict' and '{}'".format(type(other)))

    def __rtruediv__(self, other):
        if isinstance(other, MOSADict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other.data[key] / self.data[key]
            return MOSADict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other / self.data[key]
            return MOSADict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'MOSADict' and '{}'".format(type(other)))

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def copy(self):
        if type(self.data[MOSA_labels[0]]) is np.ndarray:
            d = {}
            for key in MOSA_labels:
                d[key] = np.copy(self.data[key])
            return MOSADict(d)
        else:
            return MOSADict(self.data.copy())

    def reverse(self):
        d = {}
        for k in MOSA_labels:
            k_re = k[1] + k[0]
            d[k] = self.data[k_re]
        return MOSADict(d)

    def adjacent(self):
        d = {}
        for k in MOSA_labels:
            k_ad = adjacent_MOSA_labels[k]
            d[k] = self.data[k_ad]
        return MOSADict(d)

    def timedelay(self, fsample, delay, doppler=None, order=31, pool=None):
        """
        delay and doppler (if not None) must be MOSADicts
        """
        if isinstance(delay, MOSADict):
            d = {}
            if isinstance(doppler, MOSADict):
                if pool is None:
                    for k in MOSA_labels:
                        d[k] = timeshift(self.data[k], -delay[k] * fsample, order=order)
                        # d[k] = timeshift(self.data[k], -delay[k] * fsample, order=order) * (1. - doppler[k])
                else:
                    param_arr = [(self.data[k], -delay[k] * fsample, order) for k in MOSA_labels]
                    d_arr = pool.starmap(timeshift, param_arr)
                    for j, k in enumerate(MOSA_labels):
                        d[k] = d_arr[j]
                return MOSADict(d) * (1.0 - doppler)
            elif doppler is None:
                if pool is None:
                    for k in MOSA_labels:
                        d[k] = timeshift(self.data[k], -delay[k] * fsample, order=order)
                else:
                    param_arr = [(self.data[k], -delay[k] * fsample, order) for k in MOSA_labels]
                    d_arr = pool.starmap(timeshift, param_arr)
                    for j, k in enumerate(MOSA_labels):
                        d[k] = d_arr[j]
                return MOSADict(d)
            else:
                raise TypeError("unsurpported doppler type.")
        else:
            raise TypeError("unsurpported delay type.")

    def integrate(self, fsample):
        d = {}
        for k in MOSA_labels:
            d[k] = cumulative_trapezoid(np.insert(self.data[k], 0, 0), dx=1 / fsample)
        return MOSADict(d)

    def derivative(self, fsample):
        d = {}
        for k in MOSA_labels:
            d[k] = np.gradient(self.data[k], 1 / fsample)
        return MOSADict(d)

    def downsampled(self, fsample, downsample, kaiser_filter_coef):
        d = {}
        for k in MOSA_labels:
            d[k] = downsampling(self.data[k], fsample, downsample, kaiser_filter_coef)
        return MOSADict(d)

    def detrended(self, order=2):
        d = {}
        for k in MOSA_labels:
            d[k] = detrending(self.data[k], order=order)
        return MOSADict(d)

    def drop_edge_points(self, points1=0, points2=0):
        d = {}
        if points2 == 0:
            for k in MOSA_labels:
                d[k] = self.data[k][points1:]
        else:
            for k in MOSA_labels:
                d[k] = self.data[k][points1:-points2]
        return MOSADict(d)

    def get_data_by_idx(self, idx):
        d = {}
        for key in MOSA_labels:
            d[key] = self.data[key][idx]
        return MOSADict(d)


class SCDict:
    """
    NOTE: be careful when manipulating array or dict elements
    NOTE: use B = A.copy() instead of B = A
    the keys of SCDict must be SC_labels
    the values of SCDict are numpy arrays of the same dim or scalars
    """

    dict_keys = SC_labels

    def __init__(self, data_dict):
        if isinstance(data_dict, dict):
            if set(data_dict.keys()) == set(self.dict_keys):
                self.data = data_dict
            else:
                raise ValueError("keys of dictionaries mismatch.")
        else:
            raise TypeError("data_dict must be a dict.")

    def __repr__(self):
        return str(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        if key in self.dict_keys:
            self.data[key] = value
        else:
            raise ValueError(str(key) + " is not a surpported key.")

    def __add__(self, other):
        if isinstance(other, SCDict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] + other.data[key]
            return SCDict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] + other
            return SCDict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'SCDict' and '{}'".format(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, SCDict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] * other.data[key]
            return SCDict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] * other
            return SCDict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'SCDict' and '{}'".format(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, SCDict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] - other.data[key]
            return SCDict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] - other
            return SCDict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'SCDict' and '{}'".format(type(other)))

    def __rsub__(self, other):
        if isinstance(other, SCDict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other.data[key] - self.data[key]
            return SCDict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other - self.data[key]
            return SCDict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'SCDict' and '{}'".format(type(other)))

    def __neg__(self):
        return SCDict({key: -self.data[key] for key in self.dict_keys})

    def __truediv__(self, other):
        if isinstance(other, SCDict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] / other.data[key]
            return SCDict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] / other
            return SCDict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'SCDict' and '{}'".format(type(other)))

    def __rtruediv__(self, other):
        if isinstance(other, SCDict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other.data[key] / self.data[key]
            return SCDict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other / self.data[key]
            return SCDict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'SCDict' and '{}'".format(type(other)))

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def copy(self):
        if type(self.data[SC_labels[0]]) is np.ndarray:
            d = {}
            for key in SC_labels:
                d[key] = np.copy(self.data[key])
            return SCDict(d)
        else:
            return SCDict(self.data.copy())

    def toMOSA(self):
        d = {}
        for key in MOSA_labels:
            d[key] = self.data[key[0]]
        return MOSADict(d)

    def integrate(self, fsample):
        d = {}
        for k in SC_labels:
            d[k] = cumulative_trapezoid(np.insert(self.data[k], 0, 0), dx=1 / fsample)
        return SCDict(d)

    def derivative(self, fsample):
        d = {}
        for k in SC_labels:
            d[k] = np.gradient(self.data[k], 1 / fsample)
        return SCDict(d)

    def timedelay(self, fsample, delay, order=31, pool=None):
        """
        delay must be SCDicts
        """
        if isinstance(delay, SCDict):
            d = {}
            if pool is None:
                for k in SC_labels:
                    d[k] = timeshift(self.data[k], -delay[k] * fsample, order=order)
            else:
                param_arr = [(self.data[k], -delay[k] * fsample, order) for k in SC_labels]
                d_arr = pool.starmap(timeshift, param_arr)
                for j, k in enumerate(SC_labels):
                    d[k] = d_arr[j]
            return SCDict(d)
        else:
            raise TypeError("unsurpported delay type.")

    def downsampled(self, fsample, downsample, kaiser_filter_coef):
        d = {}
        for k in SC_labels:
            d[k] = downsampling(self.data[k], fsample, downsample, kaiser_filter_coef)
        return SCDict(d)

    def detrended(self, order=2):
        d = {}
        for k in SC_labels:
            d[k] = detrending(self.data[k], order=order)
        return SCDict(d)

    def drop_edge_points(self, points1=0, points2=0):
        d = {}
        if points2 == 0:
            for k in SC_labels:
                d[k] = self.data[k][points1:]
        else:
            for k in SC_labels:
                d[k] = self.data[k][points1:-points2]
        return SCDict(d)


def assign_noise_for_MOSAs(arrays_or_psds, fsample, size):
    """
    if arrays_or_psds is numpy array or psd function, it will be filled in to a dictionary (arrays are copied)
    """
    if isinstance(arrays_or_psds, np.ndarray):
        arrays_or_psds_dict = {key: arrays_or_psds.copy() for key in MOSA_labels}  # use copy to avoid shared storage
    elif callable(arrays_or_psds):
        arrays_or_psds_dict = {key: arrays_or_psds for key in MOSA_labels}
    elif isinstance(arrays_or_psds, dict):
        if set(arrays_or_psds.keys()) == set(MOSA_labels):
            arrays_or_psds_dict = arrays_or_psds.copy()
        else:
            raise ValueError("keys of dictionaries mismatch.")
    else:
        raise TypeError("arrays_or_psds must be dict or callable or array.")

    logger.debug("arrays_or_psds:")
    logger.debug(arrays_or_psds_dict)

    noise_dict = {}
    for key in MOSA_labels:
        array_or_psd = arrays_or_psds_dict[key]
        noise = GeneralNoise(array_or_psd=array_or_psd)
        value = noise(fsample=fsample, size=size)
        noise_dict[key] = value
    return MOSADict(noise_dict)


def assign_noise_for_SCs(arrays_or_psds, fsample, size):
    """
    if arrays_or_psds is numpy array or psd function, it will be filled in to a dictionary (arrays are copied)
    """
    if isinstance(arrays_or_psds, np.ndarray):
        arrays_or_psds_dict = {key: arrays_or_psds.copy() for key in SC_labels}  # use copy to avoid shared storage
    elif callable(arrays_or_psds):
        arrays_or_psds_dict = {key: arrays_or_psds for key in SC_labels}
    elif isinstance(arrays_or_psds, dict):
        if set(arrays_or_psds.keys()) == set(SC_labels):
            arrays_or_psds_dict = arrays_or_psds.copy()
        else:
            raise ValueError("keys of dictionaries mismatch.")
    else:
        raise TypeError("arrays_or_psds must be dict or callable or array.")

    logger.debug("arrays_or_psds:")
    logger.debug(arrays_or_psds_dict)

    noise_dict = {}
    for key in SC_labels:
        array_or_psd = arrays_or_psds_dict[key]
        noise = GeneralNoise(array_or_psd=array_or_psd)
        value = noise(fsample=fsample, size=size)
        noise_dict[key] = value
    return SCDict(noise_dict)


def assign_function_for_MOSAs(functions, proper_time):
    """
    if functions is a callable function, it will be filled into a dict
    proper time can both be a SC dict or a numpy array
    """
    if callable(functions):
        functions_dict = {key: functions for key in MOSA_labels}
    elif isinstance(functions, dict):
        if set(functions.keys()) == set(MOSA_labels):
            functions_dict = functions.copy()
        else:
            raise ValueError("keys of dictionaries mismatch.")
    else:
        raise TypeError("functions must be dict or callable.")

    logger.debug("functions:")
    logger.debug(functions_dict)

    data_dict = {}
    if isinstance(proper_time, SCDict) or isinstance(proper_time, dict):
        for key in MOSA_labels:
            function = functions_dict[key]
            # Note: the 1st number in the key represents the label of SC by convention
            data_dict[key] = function(proper_time[key[0]])
    else:
        for key in MOSA_labels:
            function = functions_dict[key]
            data_dict[key] = function(proper_time)
    return MOSADict(data_dict)


def assign_function_for_SCs(functions, proper_time):
    """
    if functions is a callable function, it will be filled into a dict
    proper time can both be a SC dict or a numpy array
    """
    if callable(functions):
        functions_dict = {key: functions for key in SC_labels}
    elif isinstance(functions, dict):
        if set(functions.keys()) == set(SC_labels):
            functions_dict = functions.copy()
        else:
            raise ValueError("keys of dictionaries mismatch.")
    else:
        raise TypeError("functions must be dict or callable.")

    logger.debug("functions:")
    logger.debug(functions_dict)

    data_dict = {}
    if isinstance(proper_time, SCDict) or isinstance(proper_time, dict):
        for key in SC_labels:
            function = functions_dict[key]
            data_dict[key] = function(proper_time[key])
    else:
        for key in SC_labels:
            function = functions_dict[key]
            data_dict[key] = function(proper_time)
    return SCDict(data_dict)


def assign_const_for_MOSAs(val):
    if np.isscalar(val):
        return MOSADict({k: val for k in MOSA_labels})
    else:
        return MOSADict({k: val[i] for i, k in enumerate(MOSA_labels)})


def assign_const_for_SCs(val):
    if np.isscalar(val):
        return MOSADict({k: val for k in SC_labels})
    else:
        return MOSADict({k: val[i] for i, k in enumerate(SC_labels)})


# ======================= lagrange interpolation ============================
def uniform_lagrange_timeshift(data, shifts, order=31):
    """
    An implementation of the lagrange interpolation, optimized for uniformly spaced time series.
    Time shift by non-causal filter.
    """
    if order % 2 == 0 or not isinstance(order, int):
        raise ValueError("order must be an odd integer.")
    p = round((order + 1) / 2)
    Ndata = len(data)
    delay = np.arange(Ndata) + shifts
    d = np.round(np.ceil(delay) - 1).astype(int)
    eps = 1.0 + d - delay
    signal_delay = np.zeros_like(data)

    # pad data
    pad1 = max([0, -round(min(d)) + p - 1])
    pad2 = max([0, round(max(d)) + p - Ndata + 1])
    signal_pad = np.pad(data, (pad1, pad2)).astype(np.float64)

    B = 1.0 - eps
    C = eps
    D = B * C
    for i in range(Ndata):
        filter_coef = np.zeros(order + 1)
        A = 1.0 / p
        E = np.zeros(p - 1)
        F = np.zeros(p - 1)
        G = np.zeros(p - 1)
        for j in range(1, p):
            A *= (1.0 + eps[i] / j) * (1.0 + (1.0 - eps[i]) / j)
            E[j - 1] = (-1.0) ** j * np.math.factorial(p - 1) * np.math.factorial(p) / np.math.factorial(p - 1 - j) / np.math.factorial(p + j)
            F[j - 1] = j + eps[i]
            G[j - 1] = j + 1.0 - eps[i]
        filter_coef[: p - 1] = A * D[i] * np.flipud(E / G)
        filter_coef[p - 1] = A * C[i]
        filter_coef[p] = A * B[i]
        filter_coef[p + 1 :] = A * D[i] * E / F

        interp_data_points = signal_pad[pad1 + d[i] - p + 1 : pad1 + d[i] + p + 1]
        signal_delay[i] = np.dot(interp_data_points, filter_coef)
    return signal_delay


def lagrange_timeshift(data, shifts, order=31):
    """
    The original lagrange interpolation (not optimized).
    order (odd) is the number of data points used for interpolation.
    """
    delays = -shifts
    if len(data) != len(delays):
        raise ValueError("the size of data and timeshifts must be the same.")
    if order % 2 != 1:
        raise ValueError("order must be odd.")
    size = len(data)

    # pad data
    halfN = int((order + 1) / 2)
    maxdelay = int(np.max(delays)) + 1
    mindelay = int(np.min(delays))
    pad1 = max([0, maxdelay + halfN])
    pad2 = max([0, halfN - mindelay])
    data_padded = np.pad(data, (pad1, pad2)).astype(np.float64)

    # set the int and fractional part of delays
    ND = np.round(delays).astype("int")
    D = delays - ND
    data_delayed = np.zeros_like(data, dtype=np.float64)

    k = range(-int((order - 1) / 2), int((order + 1) / 2))
    for i in range(size):
        interp_data_points = data_padded[pad1 + i - ND[i] + k[0] : pad1 + i - ND[i] + k[-1] + 1]
        interp_coeffs = np.ones_like(interp_data_points, dtype=np.float64)
        for j in range(len(interp_coeffs)):
            for m in k:
                if m != k[j]:
                    interp_coeffs[j] *= (m + D[i]) / (m - k[j])

        data_delayed[i] = np.dot(interp_data_points, interp_coeffs)
    return data_delayed


def timeshift(data, shifts, order=31):
    """
    Another implementation of the lagrange interpolation (by Jean-Baptiste Bayle, Refs: PhysRevD.107.083019, LISAInstrument), turned out to be the fastest.
    The correctness has been confirmed by comparing with the original one.
    Args:
        data: input array
        shifts: array of time shifts [samples]
        order: interpolation order
    """
    # logger.debug("Time shifting data '%s' by '%s' samples (order=%d)", data, shifts, order)

    # Handle constant input and vanishing shifts
    data = np.asarray(data)
    shifts = np.asarray(shifts)
    if data.size == 1:
        # logger.debug("Input data is constant, skipping time-shift operation")
        return data
    if np.all(shifts == 0):
        # logger.debug("Time shifts are vanishing, skipping time-shift operation")
        return data

    # logger.debug("Computing Lagrange coefficients")
    halfp = (order + 1) // 2
    shift_ints = np.floor(shifts).astype(int)
    shift_fracs = shifts - shift_ints
    taps = lagrange_taps(shift_fracs, halfp)

    # Handle constant shifts
    if shifts.size == 1:
        # logger.debug("Constant shifts, using correlation method")
        i_min = shift_ints - (halfp - 1)
        i_max = shift_ints + halfp + data.size
        if i_max - 1 < 0:
            return np.repeat(data[0], data.size)
        if i_min > data.size - 1:
            return np.repeat(data[-1], data.size)
        # logger.debug("Padding data (left=%d, right=%d)", max(0, -i_min), max(0, i_max - data.size))
        data_trimmed = data[max(0, i_min) : min(data.size, i_max)]
        data_padded = np.pad(data_trimmed, (max(0, -i_min), max(0, i_max - data.size)), mode="edge")
        # logger.debug("Computing correlation product")
        return np.correlate(data_padded, taps[0], mode="valid")

    # Check that sizes or compatible
    if data.size != shifts.size:
        raise ValueError(f"`data` and `shift` must be of the same size (got {data.size}, {shifts.size})")

    # Handle time-varying shifts
    # logger.debug("Time-varying shifts, using matrix method")
    indices = np.arange(data.size)
    i_min = np.min(shift_ints - (halfp - 1) + indices)
    i_max = np.max(shift_ints + halfp + indices + 1)
    csr_ind = np.tile(np.arange(order + 1), data.size) + np.repeat(shift_ints + indices, order + 1) - (halfp - 1)
    csr_ptr = (order + 1) * np.arange(data.size + 1)
    mat = scipy.sparse.csr_matrix((taps.reshape(-1), csr_ind - i_min, csr_ptr), shape=(data.size, i_max - i_min))
    # logger.debug("Padding data (left=%d, right=%d)", max(0, -i_min), max(0, i_max - data.size))
    data_trimmed = data[max(0, i_min) : min(data.size, i_max)]
    data_padded = np.pad(data_trimmed, (max(0, -i_min), max(0, i_max - data.size)))
    # logger.debug("Computing matrix-vector product")
    return mat.dot(data_padded)


def lagrange_taps(shift_fracs, halfp):
    """
    Args:
        shift_fracs: array of fractional time shifts [samples]
        halfp: number of points on each side, equivalent to (order + 1) // 2
    """
    taps = np.zeros((2 * halfp, shift_fracs.size))

    if halfp > 1:
        factor = np.ones(shift_fracs.size, dtype=np.float64)
        factor *= shift_fracs * (1 - shift_fracs)

        for j in range(1, halfp):
            factor *= (-1) * (1 - j / halfp) / (1 + j / halfp)
            taps[halfp - 1 - j] = factor / (j + shift_fracs)
            taps[halfp + j] = factor / (j + 1 - shift_fracs)

        taps[halfp - 1] = 1 - shift_fracs
        taps[halfp] = shift_fracs

        for j in range(2, halfp):
            taps *= 1 - (shift_fracs / j) ** 2

        taps *= (1 + shift_fracs) * (1 - shift_fracs / halfp)
    else:
        taps[halfp - 1] = 1 - shift_fracs
        taps[halfp] = shift_fracs

    return taps.T
