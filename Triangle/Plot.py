import matplotlib.pyplot as plt

from Triangle.Constants import *
from Triangle.FFTTools import *

def PlotTimeSeriesFromDict(data_dict, fsample, t_start=0, xlim=None, ylim=None, title=None, file=None):
    plt.figure(figsize=(10, 5))
    for k, v in data_dict.items():
        t = np.arange(len(v)) / fsample + t_start
        color = np.random.rand(3)
        plt.plot(t, v, label = k, color = color)
    if isinstance(xlim, list):
        plt.xlim(xlim[0], xlim[1])
    if isinstance(ylim, list):
        plt.ylim(ylim[0], ylim[1])
    plt.grid(which='major')
    plt.grid(which='minor', linestyle = ':', linewidth = 0.5)
    plt.legend(loc = 'upper right')
    plt.xlabel(r'${\rm Time} \ ({\rm s})$')
    plt.ylabel(r'${\rm signal}$')
    if isinstance(title, str):
        plt.title(title)
    if isinstance(file, str):
        plt.savefig(file)


def PlotASDFromDict(
        data_dict, fsample, nbin = 1, window_type = 'kaiser', window_args_dict = kaiser_dict_default, psd_funcs = None, xlim=None, title = None, file = None
        ):
    plt.figure(figsize=(10, 5))
    ymin = None
    ymax = None
    if isinstance(psd_funcs, dict):
        for k, v in data_dict.items():
            f, Sf = PSD_window(v, fsample, nbin, window_type, window_args_dict)
            f = f[1:]
            Af = np.sqrt(Sf)[1:]
            psd_func = psd_funcs[k]
            color = np.random.rand(3)
            plt.loglog(f, Af, label = k, linewidth = 1, color = color, alpha = 0.8)
            plt.loglog(f, np.sqrt(psd_func(f)), color = color)
            if ymin == None:
                ymin = np.min(Af)
            else:
                ymin = np.min((ymin, np.min(Af)))
            if ymax == None:
                ymax = np.max(Af)
            else:
                ymax = np.max((ymax, np.max(Af)))
    else:
        for k, v in data_dict.items():
            f, Sf = PSD_window(v, fsample, nbin, window_type, window_args_dict)
            f = f[1:]
            Af = np.sqrt(Sf)[1:]
            color = np.random.rand(3)
            plt.loglog(f, Af, label = k, linewidth = 1, color = color, alpha = 0.8)
            if ymin == None:
                ymin = np.min(Af)
            else:
                ymin = np.min((ymin, np.min(Af)))
            if ymax == None:
                ymax = np.max(Af)
            else:
                ymax = np.max((ymax, np.max(Af)))
    if callable(psd_funcs):
        color = np.random.rand(3)
        plt.loglog(f, np.sqrt(psd_funcs(f)), color = color)

    if isinstance(xlim, list):
        plt.xlim(xlim[0], xlim[1])
    plt.ylim(ymin / 10., ymax * 10.)        
    plt.grid(which='major')
    plt.grid(which='minor', linestyle = ':', linewidth = 0.5)
    plt.legend(loc = 'lower right')
    plt.xlabel(r'${\rm Frequency} \ ({\rm Hz})$')
    plt.ylabel(r'${\rm ASD}(f)$')
    if isinstance(title, str):
        plt.title(title)
    if isinstance(file, str):
        plt.savefig(file)

def PlotASD(
        data_array, fsample, nbin = 1, window_type = 'kaiser', window_args_dict = kaiser_dict_default, psd_func = None, xlim=None, ylim=None
        ):
    plt.figure(figsize=(10, 5))
    f, sf = PSD_window(data_array, fsample, nbin, window_type, window_args_dict)
    f = f[1:]
    af = np.sqrt(sf)[1:]
    plt.loglog(f, af, linewidth=1)
    if callable(psd_func):
        plt.loglog(f, np.sqrt(psd_func(f)), linewidth=1)
    if isinstance(xlim, list):
        plt.xlim(xlim[0], xlim[1])
    if isinstance(ylim, list):
        plt.ylim(ylim[0], ylim[1])
    plt.grid(which='major')
    plt.grid(which='minor', linestyle = ':', linewidth = 0.5)
    plt.xlabel(r'$Frequency \ ({\rm Hz})$')
    plt.ylabel(r'${\rm ASD}(f)$')
        


        
