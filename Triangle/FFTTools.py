import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import csd, welch, windows

tukey_dict_default = dict(alpha=0.4)
kaiser_dict_default = dict(beta=28)
window_dict_default = dict()


def FFT_window(data_array, fsample, window_type=None, window_args_dict=window_dict_default):
    """
    calculate windowed fft at f > 0.
    """
    n_data = len(data_array)
    if n_data % 2 == 0:
        n_f = int(n_data / 2) + 1
    else:
        n_f = int((n_data + 1) / 2) + 1
    f = rfftfreq(n_data, 1.0 / fsample)[1:n_f]

    if window_type is None:
        fft_array = rfft(data_array)[1:n_f] / fsample
    else:
        if window_type == "tukey":
            window = windows.tukey
            scale_factor = 1.110  # NOTE: not suitable for all the values of alpha
        elif window_type == "kaiser":
            window = windows.kaiser
            scale_factor = 2.480
        elif window_type == "hann":
            window = windows.hann
            scale_factor = 2.0
        elif window_type == "hamming":
            window = windows.hamming
            scale_factor = 1.852
        elif window_type == "blackman":
            window = windows.blackman
            scale_factor = 2.381
        else:
            raise ValueError("Unsupported window type.")

        fft_array = rfft(data_array * window(M=n_data, **window_args_dict))[1:n_f] / fsample * scale_factor

    return f, fft_array


def PSD_window(
    data_array,
    fsample,
    nbin=1,
    window_type=None,
    window_args_dict=window_dict_default,
    detrend="constant",
):
    """
    calculate PSD at f > 0.
    """
    nperseg = int(len(data_array) / nbin)
    if window_type is None:
        window_array = np.ones(nperseg)
    else:
        if window_type == "tukey":
            window = windows.tukey
        elif window_type == "kaiser":
            window = windows.kaiser
        elif window_type == "hann":
            window = windows.hann
        elif window_type == "hamming":
            window = windows.hamming
        elif window_type == "blackman":
            window = windows.blackman
        else:
            raise ValueError("Unsupported window type.")
        window_array = window(M=nperseg, **window_args_dict)
    f, Sf = welch(
        x=data_array,
        fs=fsample,
        window=window_array,
        return_onesided=True,
        nperseg=nperseg,
        detrend=detrend,
    )
    return f[1:], Sf[1:]


def CSD_window(
    data_array1,
    data_array2,
    fsample,
    nbin=1,
    window_type=None,
    window_args_dict=window_dict_default,
    detrend="constant",
):
    """
    calculate CSD at f > 0.
    """
    if len(data_array1) != len(data_array2):
        raise ValueError("the sizes of data must coincide.")
    nperseg = int(len(data_array1) / nbin)
    if window_type is None:
        window_array = np.ones(nperseg)
    else:
        if window_type == "tukey":
            window = windows.tukey
        elif window_type == "kaiser":
            window = windows.kaiser
        elif window_type == "hann":
            window = windows.hann
        elif window_type == "hamming":
            window = windows.hamming
        elif window_type == "blackman":
            window = windows.blackman
        else:
            raise ValueError("Unsupported window type.")
        window_array = window(M=nperseg, **window_args_dict)
    f, Sf = csd(
        x=data_array1,
        y=data_array2,
        fs=fsample,
        window=window_array,
        return_onesided=True,
        nperseg=nperseg,
        detrend=detrend,
    )
    return f[1:], Sf[1:]


def plot_ASDs(
    data_arrays,
    labels,
    fsample,
    nbin=1,
    window_type=None,
    window_args_dict=window_dict_default,
    psd_funcs=None,
    file=None,
):
    if not (isinstance(data_arrays, list) and isinstance(labels, list)):
        raise ValueError("data and labels must be lists.")
    plt.figure(figsize=(12, 5))
    ymin = None
    ymax = None
    if isinstance(psd_funcs, list):
        for data, label, psd_func in zip(data_arrays, labels, psd_funcs):
            f, Sf = PSD_window(data, fsample, nbin, window_type, window_args_dict)
            f = f[1:]
            Af = np.sqrt(Sf)[1:]
            color = np.random.rand(3)
            plt.loglog(f, Af, label=label, linewidth=1, color=color, alpha=0.6)
            plt.loglog(f, np.sqrt(psd_func(f)), color=color)
            if ymin is None:
                ymin = np.min(Af)
            else:
                ymin = np.min((ymin, np.min(Af)))
            if ymax is None:
                ymax = np.max(Af)
            else:
                ymax = np.max((ymax, np.max(Af)))
    else:
        for data, label in zip(data_arrays, labels):
            f, Sf = PSD_window(data, fsample, nbin, window_type, window_args_dict)
            f = f[1:]
            Af = np.sqrt(Sf)[1:]
            color = np.random.rand(3)
            plt.loglog(f, Af, label=label, linewidth=1, color=color, alpha=0.6)
            if ymin is None:
                ymin = np.min(Af)
            else:
                ymin = np.min((ymin, np.min(Af)))
            if ymax is None:
                ymax = np.max(Af)
            else:
                ymax = np.max((ymax, np.max(Af)))

    plt.ylim(ymin / 10.0, ymax * 10.0)
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.legend(loc="lower right")
    plt.xlabel(r"$f \ [{\rm Hz}]$")
    plt.ylabel(r"${\rm ASD}(f)$")
    if isinstance(file, str):
        plt.savefig(file)
