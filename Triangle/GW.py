import logging

# to mute the warnings of lal
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal

import numpy as np
import pycbc.waveform as wf
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

from Triangle.Constants import *
from Triangle.Data import *
from Triangle.FFTTools import *
from Triangle.Orbit import *

logger = logging.getLogger(__name__)


# utils
def eta_q(q):
    return q / (1.0 + q) ** 2


def q_eta(eta):
    if eta == 1.0 / 4.0:
        return 1.0
    else:
        return (1.0 - 2.0 * eta - np.sqrt(1.0 - 4.0 * eta)) / 2.0 / eta


def M_m1_q(m1, q):
    return m1 * (1.0 + q)


def Mc_m1_q(m1, q):
    m2 = m1 * q
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


def Mc_m1_m2(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def mu_m1_q(m1, q):
    m2 = m1 * q
    return m1 * m2 / (m1 + m2)


def q_Mc_mu(Mc, mu):
    temp = (Mc / mu) ** (5.0 / 2.0)
    return np.abs((temp - 2.0) - np.sqrt(temp * np.abs(temp - 4.0))) / 2


def m1_Mc_mu(Mc, mu):
    q = q_Mc_mu(Mc, mu)
    return mu * (1.0 + q) / q


def m1_Mc_q(Mc, q):
    return Mc / (q**3 / (1.0 + q)) ** 0.2


def mu_Mc_q(Mc, q):
    m1 = m1_Mc_q(Mc, q)
    return q / (1.0 + q) * m1


def Mc_M_eta(M, eta):
    return eta**0.6 * M


def SSBTimetoDetectorTime(time_ssb, orbit, longitude, latitude):
    wave_vector = -np.array(
        [
            np.cos(longitude) * np.cos(latitude),
            np.sin(longitude) * np.cos(latitude),
            np.sin(latitude),
        ]
    )
    p1 = orbit.Positionfunctions()["1"](time_ssb)
    p2 = orbit.Positionfunctions()["2"](time_ssb)
    p3 = orbit.Positionfunctions()["3"](time_ssb)
    p0 = (p1 + p2 + p3) / 3.0
    return time_ssb + np.dot(wave_vector, p0) / C


class GW:
    def __init__(self, orbit, ext_params, t0=0, GWDir=None, GWwaveform=None):
        """
        Args:
            GW can be initialized by either waveform data stored in a directory GWDir, or a GWwaveform object, which has at least two functions hpfunc and hcfunc, representing hp^SSB and hc^SSB.
            ext_params = [longitude, latitude, polarization]
        """
        if GWDir is not None:
            GWdata = np.loadtxt(GWDir)
            self.tdata = GWdata[:, 0] + t0
            self.hpdata = GWdata[:, 1]
            self.hcdata = GWdata[:, 2]
            self.hpfunc = InterpolatedUnivariateSpline(self.tdata, self.hpdata, k=5, ext="zeros")
            self.hcfunc = InterpolatedUnivariateSpline(self.tdata, self.hcdata, k=5, ext="zeros")
        elif GWwaveform is not None:
            self.hpfunc = GWwaveform.hpfunc
            self.hcfunc = GWwaveform.hcfunc

        self.orbit = orbit
        self.ext_params = ext_params

    def CalculateResponse(self, time_dict):
        """
        calculate single-arm responses in the fractional frequency difference unit
        """
        # calculate wave vector
        l, b, p = self.ext_params
        wave_vector = -np.array([np.cos(l) * np.cos(b), np.sin(l) * np.cos(b), np.sin(b)])
        logger.debug("Wave vector calculated.")

        # calculate (source frame) polarization tensors
        O = np.zeros((3, 3))
        O[0][0] = np.sin(l) * np.cos(p) - np.cos(l) * np.sin(b) * np.sin(p)
        O[0][1] = -np.sin(l) * np.sin(p) - np.cos(l) * np.sin(b) * np.cos(p)
        O[0][2] = -np.cos(l) * np.cos(b)
        O[1][0] = -np.cos(l) * np.cos(p) - np.sin(l) * np.sin(b) * np.sin(p)
        O[1][1] = np.cos(l) * np.sin(p) - np.sin(l) * np.sin(b) * np.cos(p)
        O[1][2] = -np.sin(l) * np.cos(b)
        O[2][0] = np.cos(b) * np.sin(p)
        O[2][1] = np.cos(b) * np.cos(p)
        O[2][2] = -np.sin(b)
        OT = O.transpose()
        ep_0 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        ec_0 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        polar_tensor_p = np.dot(np.dot(O, ep_0), OT)
        polar_tensor_c = np.dot(np.dot(O, ec_0), OT)
        logger.debug("Polarization basis calculated.")

        # calculate single-arm response
        y = {}
        pos_data = assign_function_for_SCs(self.orbit.Positionfunctions(), time_dict)
        arm_data = assign_function_for_MOSAs(self.orbit.ArmVectorfunctions(), time_dict)
        ltt_data = assign_function_for_MOSAs(self.orbit.LTTfunctions(), time_dict)
        logger.debug("Orbit calculated.")

        for label in MOSA_labels:
            recv = label[0]
            send = label[1]
            t_recv = time_dict[recv] - np.dot(pos_data[recv], wave_vector) / C  # (N) array
            t_send = time_dict[send] - np.dot(pos_data[send], wave_vector) / C - ltt_data[label]  # (N) array
            hp_recv = self.hpfunc(t_recv)  # (N) array
            hp_send = self.hpfunc(t_send)  # (N) array
            hc_recv = self.hcfunc(t_recv)  # (N) array
            hc_send = self.hcfunc(t_send)  # (N) array
            Fp_factor = np.sum(np.dot(arm_data[label], polar_tensor_p) * arm_data[label], axis=1)  # (N) array
            Fc_factor = np.sum(np.dot(arm_data[label], polar_tensor_c) * arm_data[label], axis=1)  # (N) array
            Denominator = (np.dot(wave_vector, (arm_data[label]).T) - 1.0) * (-2.0)  # (N) array

            y[label] = (Fp_factor * (hp_send - hp_recv) + Fc_factor * (hc_send - hc_recv)) / Denominator
        logger.info("Single-arm responses calculated.")
        return MOSADict(y)


class GWParallelGenerator:
    def __init__(self, time_dict):
        self.time_dict = time_dict

    def __call__(self, gw_class):
        return gw_class.CalculateResponse(self.time_dict)


class GB:
    """
    the GB waveform class for Triangle
    """

    def __init__(self, A, f, fdot, iota, phi0):
        self.A = A
        self.f = f
        self.fdot = fdot
        self.phi0 = phi0
        self.cosiota = np.cos(iota)
        self.fddot = 11.0 / 3.0 * fdot**2 / f

    def hpfunc(self, t):
        phase = 2 * np.pi * (self.f * t + 1.0 / 2.0 * self.fdot * t**2 + 1.0 / 6.0 * self.fddot * t**3) + self.phi0
        hp = np.cos(phase) * self.A * (1.0 + self.cosiota**2)
        return hp

    def hcfunc(self, t):
        phase = 2 * np.pi * (self.f * t + 1.0 / 2.0 * self.fdot * t**2 + 1.0 / 6.0 * self.fddot * t**3) + self.phi0
        hc = np.sin(phase) * 2.0 * self.A * self.cosiota
        return hc


class MBHB:
    """
    the MBHB waveform wrapper for PyCBC
    """

    def __init__(self, approx_method="IMRPhenomD", modes=None, buffer=False, verbose=0):
        """
        Args:
            approx_method: waveform model
            modes = None or a list [(2, 2), (2, 1), ...] specifying the harmonic modes of waveform
            buffer decides whether to extend f_lower to lower frequencies
        """
        self.approx_method = approx_method
        self.modes = modes
        self.buffer = buffer
        self.verbose = verbose

    def __call__(
        self,
        Mc,
        q,
        spin1z,
        spin2z,
        tc,
        phic,
        D,
        inc,
        dt=10.0,
        f_lower=None,
        mass_scale=None,
    ):
        """
        Args:
            Mc in [MSUN]
            q, spin1z, spin2z are dimensionless
            tc in [s]
            D in [Mpc]
            dt is the sampling time in [s]
            f_lower is the lowest frequency of waveform
            mass_scale is used to avoid the error of PyCBC caused by too large masses (doesn't always work though)
        """
        # For some values of the masses PyCBC might returns error, thus we use the rescaling rule to avoid this error.
        # For PhenomD waveform, the validity of this method is tested using another frequency-domain code.
        if mass_scale is None:
            mass_scale = max(Mc / 50, 1.0)

        # get rescaled masses
        m1 = m1_Mc_q(Mc, q) / mass_scale
        m2 = m1 * q

        # get the maximum m
        if self.modes is not None:
            m_array = [emm for (ell, emm) in self.modes]
            m_max = max(m_array)
        else:
            m_max = 2

        # set f_lower
        Tobs = tc / YEAR
        if f_lower is None:  # calculate the lower limit of frequency to the leading order
            f_lower = 1.75e-5 * (Mc / 1e6) ** (-5.0 / 8.0) * (Tobs / 10.0) ** (-3.0 / 8.0)
            if self.buffer:  # the buffered f_lower is at least 1.5 times lower
                f_lower = min(f_lower * 2.0 / m_max, f_lower / 1.5)
            f_lower = max(f_lower, 1e-5)  # set a lower limit of 1e-5 Hz
        f_lower *= mass_scale  # get rescaled lower frequency
        if self.verbose > 0:
            print("minimum frequency:", f_lower)

        # calculate waveform
        hp, hc = wf.get_td_waveform_from_fd(
            approximant=self.approx_method,
            mass1=m1,
            mass2=m2,
            spin1z=spin1z,
            spin2z=spin2z,
            coa_phase=phic,
            distance=D,
            inclination=inc,
            delta_t=dt / mass_scale,
            f_lower=f_lower,
            mode_array=self.modes,
        )

        # scale back to the desired waveform
        hp, hc = hp.trim_zeros(), hc.trim_zeros()
        hSp_data = np.array(hp) * mass_scale
        hSc_data = np.array(hc) * mass_scale
        t_data = np.array(hp.sample_times) * mass_scale + tc
        self.tend = t_data[-1]
        if self.verbose > 0:
            print("length of data:", len(t_data), len(hSp_data))

        # get interpolation functions
        self.hpfunc = InterpolatedUnivariateSpline(x=t_data, y=hSp_data, k=5, ext="zeros")
        self.hcfunc = InterpolatedUnivariateSpline(x=t_data, y=hSc_data, k=5, ext="zeros")
