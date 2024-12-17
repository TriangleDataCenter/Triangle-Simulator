import numpy as np
from scipy.integrate import cumulative_trapezoid

from Triangle.Constants import *


class Glitch:
    short_glitch_kwargs = dict(dv=2.2e-12, tau1=10.0, tau2=11.0)
    long_glitch_kwargs = dict(dv=1.18e-12, tau1=5661.65, tau2=5661.71)

    def __init__(self, fsample=1.0):
        self.fsample = fsample

    def LPF_legacy_glitch_model(self, t, dv, tau1, tau2, t0=0):
        """
        LPF glitch model in the unit of acceleration [m/s2]
        """
        res = np.zeros_like(t)
        inds = np.where(t >= t0)[0]
        t = t[inds]
        res[inds] = dv / (tau1 - tau2) * (np.exp(-(t - t0) / tau1) - np.exp(-(t - t0) / tau2)) * np.heaviside(t - t0, 1)
        return res

    def acc2ffd(self, data):
        """
        convert acceleration to fractional frequency difference, which can be then used as Interferometer.BasicNoise["acc_noise"]
        """
        return cumulative_trapezoid(np.insert(data, 0, 0), dx=1 / self.fsample) / C
