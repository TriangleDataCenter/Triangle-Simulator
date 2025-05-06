import logging

import numpy as np
import scipy.interpolate as interp 
try:
    import cupy as xp
    import cupyx.scipy.interpolate as xinterp
    print("has cupy")
    HAS_GPU = True
except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    import scipy.interpolate as xinterp  
    print("no cupy ")
    HAS_GPU = False

# to mute the warnings of lal
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal
import pycbc.waveform as wf
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm import tqdm

from Triangle.Constants import *
from Triangle.Data import *
from Triangle.FFTTools import *
from Triangle.Orbit import *
from Triangle.TDI import AETfromXYZ



    

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
    """ this function is usually used to remove the invalied data points after the end of waveform data """
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

def SSB2Detector(lS, bS, pS, orbit, orbit_time):
    # detector positons 
    R1 = orbit.Positionfunctions()["1"](orbit_time) # (3,)
    R2 = orbit.Positionfunctions()["2"](orbit_time)
    R3 = orbit.Positionfunctions()["3"](orbit_time)
    R0 = (R1 + R2 + R3) / 3. 
    # print(R0.shape)
    
    # detector frame basis 
    xD = (R3 - R1) / np.linalg.norm(R3 - R1)
    x1D = (R3 - R2) / np.linalg.norm(R3 - R2)
    zD = np.cross(x1D, xD) / np.linalg.norm(np.cross(x1D, xD))
    yD = np.cross(zD, xD) / np.linalg.norm(np.cross(zD, xD))
    # print("inner of xy =", np.dot(xD, yD))
    # print("inner of yz =", np.dot(yD, zD))
    # print("inner of zx =", np.dot(zD, xD))
    
    # sky position 
    k = -np.array([np.cos(bS) * np.cos(lS), np.cos(bS) * np.sin(lS), np.sin(bS)])
    # print("wavevector in SSB:", k)
    bD = np.arcsin(-np.dot(k, zD))
    tmp = -np.dot(k, yD) / np.cos(bD) # sin(lD), sin(lD) > 0 -> 0 to PI 
    if tmp > 0.: 
        lD = np.arccos(-np.dot(k, xD) / np.cos(bD))
    else: 
        lD = TWOPI - np.arccos(-np.dot(k, xD) / np.cos(bD))
    
    # polarization angle 
    uS = np.array([np.sin(lS), -np.cos(lS), 0.])
    vS = np.array([-np.sin(bS) * np.cos(lS), -np.sin(bS) * np.sin(lS), np.cos(bS)])
    p = np.cos(pS) * uS + np.sin(pS) * vS 
    uD = np.sin(lD) * xD - np.cos(lD) * yD 
    vD = -np.sin(bD) * np.cos(lD) * xD - np.sin(bD) * np.sin(lD) * yD + np.cos(bD) * zD 
    pD = np.arctan2(np.dot(p, vD), np.dot(p, uD))
    if pD < 0.:
        pD += PI 
    
    return lD, bD, pD 


def Detector2SSB(lD, bD, pD, orbit, orbit_time): 
    # SSB frame basis 
    xS = np.array([1., 0., 0.])
    yS = np.array([0., 1., 0.])
    zS = np.array([0., 0., 1.])
    
    # detector positons 
    R1 = orbit.Positionfunctions()["1"](orbit_time) # (3,)
    R2 = orbit.Positionfunctions()["2"](orbit_time)
    R3 = orbit.Positionfunctions()["3"](orbit_time)
    R0 = (R1 + R2 + R3) / 3. 
    
    # detector frame basis 
    xD = (R3 - R1) / np.linalg.norm(R3 - R1)
    x1D = (R3 - R2) / np.linalg.norm(R3 - R2)
    zD = np.cross(x1D, xD) / np.linalg.norm(np.cross(x1D, xD))
    yD = np.cross(zD, xD) / np.linalg.norm(np.cross(zD, xD))
    
    # sky position 
    k_coef = -np.array([np.cos(bD) * np.cos(lD), np.cos(bD) * np.sin(lD), np.sin(bD)])
    k = k_coef[0] * xD + k_coef[1] * yD + k_coef[2] * zD 
    # print("wavevector in SSB:", k)
    bS = np.arcsin(-k[2])
    tmp = -k[1] / np.cos(bS)
    if tmp > 0.:
        lS = np.arccos(-k[0] / np.cos(bS))
    else: 
        lS = TWOPI - np.arccos(-k[0] / np.cos(bS))
    
    # polarization angle 
    uD_coef = np.array([np.sin(lD), -np.cos(lD), 0.])
    vD_coef = np.array([-np.sin(bD) * np.cos(lD), -np.sin(bD) * np.sin(lD), np.cos(bD)])
    uD = uD_coef[0] * xD + uD_coef[1] * yD + uD_coef[2] * zD 
    vD = vD_coef[0] * xD + vD_coef[1] * yD + vD_coef[2] * zD 
    p = np.cos(pD) * uD + np.sin(pD) * vD 
    uS = np.sin(lS) * xS - np.cos(lS) * yS 
    vS = -np.sin(bS) * np.cos(lS) * xS - np.sin(bS) * np.sin(lS) * yS + np.cos(bS) * zS 
    pS = np.arctan2(np.dot(p, vS), np.dot(p, uS))
    if pS < 0.:
        pS += PI 
    
    return lS, bS, pS 


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
        dt=None,
        f_lower=None,
    ):
        """
        Args:
            Mc in [MSUN]
            q, spin1z, spin2z are dimensionless
            tc in [s]
            D in [Mpc]
            dt is the sampling cadance of waveform in [s]
            f_lower is the lowest frequency of waveform in [Hz]
        """
        # For some values of the masses PyCBC might returns error, thus we use the rescaling rule to avoid this error.
        # For PhenomD waveform, the validity of this method is tested using another frequency-domain code.
        mass_scale = max(Mc / 50, 1.0)

        # get rescaled masses
        m1 = m1_Mc_q(Mc, q) / mass_scale
        m2 = m1 * q

        # get the maximum m
        if self.modes is not None:
            m_array = [emm for (ell, emm) in self.modes]
            m_max = max(m_array)
        else:
            if "HM" in self.approx_method:
                m_max = 4
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

        # conservative choice for the sampling cadance
        # if dt is None:
        #     if Mc <= 1e5:
        #         dt = 0.5
        #     elif Mc <= 1e6:
        #         dt = 5.
        #     elif Mc <= 1e7:
        #         dt = 50.
        #     else:
        #         dt = 100.
        if dt is None:
            dt = 1e-5 * Mc

        # calculate waveform
        # hp, hc = wf.get_td_waveform_from_fd(
        hp, hc = wf.get_td_waveform(
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


class GeneralWaveform:
    def __init__(self, tdata, hpdata, hcdata, t0=0):
        tdata_int = tdata - tdata[0] + t0  # shift the starting time to t0
        self.hpfunc = InterpolatedUnivariateSpline(x=tdata_int, y=hpdata, k=5, ext="zeros")
        self.hcfunc = InterpolatedUnivariateSpline(x=tdata_int, y=hcdata, k=5, ext="zeros")


def Initialize_GW_response(parameters, signal_type="MBHB", orbit=None, approximant=None, data=None):
    """
    Args:
        parameters: a dictionary storing the parameters of signal. Each item can be either a float number or a numpy array.
        1) For MBHB, the keys are 'chirp_mass' (in solar mass), 'mass_ratio', 'spin_1z', 'spin_2z', 'coalescence_time' (in day), 'coalescence_phase', 'luminosity_distance' (in MPC), 'inclination', 'longitude', 'latitude', 'psi';
        2) for GB, the keys are 'A', 'f0', 'fdot0', 'phase0', 'inclination', 'longitude', 'latitude', 'psi';
        3) for general GW, the keys are "longitude", "latitude", "psi" (only extrinsic parameters);
        orbit should be an "Orbit" object.
        approximant is only required for MBHB.
        data is only required for general, data = [[tdata, hpdata, hcdata], [tdata, hpdata, hcdata], ], tdata in second unit.

    Returns:
        a list of GW objects, which can be passed to Interferometer().
    """
    if signal_type == "MBHB":
        param_names = ["chirp_mass", "mass_ratio", "spin_1z", "spin_2z", "coalescence_time", "coalescence_phase", "luminosity_distance", "inclination", "longitude", "latitude", "psi"]
        params = dict()
        for key in param_names:
            params[key] = np.atleast_1d(parameters[key])
        N_source = len(params["chirp_mass"])
        if approximant == None:
            approx = "IMRPhenomD"
        else:
            approx = approximant

        print("initializing responses.")
        response_list = []
        for i in tqdm(range(N_source)):
            Mc_i = params["chirp_mass"][i]
            mbhb_i = MBHB(approx_method=approx, buffer=True)
            mbhb_i(
                Mc=Mc_i,
                q=params["mass_ratio"][i],
                spin1z=params["spin_1z"][i],
                spin2z=params["spin_2z"][i],
                tc=params["coalescence_time"][i] * DAY,  # convert the unit from day to second
                phic=params["coalescence_phase"][i],
                D=params["luminosity_distance"][i],
                inc=params["inclination"][i],
            )
            GW_i = GW(GWwaveform=mbhb_i, orbit=orbit, ext_params=[params["longitude"][i], params["latitude"][i], params["psi"][i]])
            response_list.append(GW_i)
        print("responses initialized.")

    elif signal_type == "GB":
        param_names = ["A", "f0", "fdot0", "phase0", "inclination", "longitude", "latitude", "psi"]
        params = dict()
        for key in param_names:
            params[key] = np.atleast_1d(parameters[key])
        N_source = len(params["A"])

        print("initializing responses.")
        response_list = []
        for i in tqdm(range(N_source)):
            gb_i = GB(A=params["A"][i], f=params["f0"][i], fdot=params["fdot0"][i], iota=params["inclination"][i], phi0=params["phase0"][i])
            GW_i = GW(GWwaveform=gb_i, orbit=orbit, ext_params=[params["longitude"][i], params["latitude"][i], params["psi"][i]])
            response_list.append(GW_i)
        print("responses initialized.")

    elif signal_type == "general":
        param_names = ["longitude", "latitude", "psi"]
        params = dict()
        for key in param_names:
            params[key] = np.atleast_1d(parameters[key])
        N_source = len(params["longitude"])
        N_source1 = len(data)
        if N_source != N_source1:
            raise ValueError("numbers of sources mismatch.")

        print("initializing responses.")
        response_list = []
        for i in tqdm(range(N_source)):
            general_i = GeneralWaveform(tdata=data[i][0], hpdata=data[i][1], hcdata=data[i][2])
            GW_i = GW(GWwaveform=general_i, orbit=orbit, ext_params=[params["longitude"][i], params["latitude"][i], params["psi"][i]])
            response_list.append(GW_i)
        print("responses initialized.")

    else:
        raise NotImplementedError("type of source not implemented.")

    return response_list


# ========================= fast GW TDI response injection ===========================
class General_FastLISA:
    def __init__(self, t_data, hp_data, hc_data):
        self.hpfunc = InterpolatedUnivariateSpline(x=t_data, y=hp_data, k=5, ext="zeros")
        self.hcfunc = InterpolatedUnivariateSpline(x=t_data, y=hc_data, k=5, ext="zeros")

    def __call__(self, psi, T=1.0, dt=10.0):
        t = np.arange(0.0, T * YEAR, dt)
        hSp = self.hpfunc(t)
        hSc = self.hcfunc(t)
        cos2psi = np.cos(2.0 * psi)
        sin2psi = np.sin(2.0 * psi)
        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi
        return hp + 1j * hc


class General_Injection:
    def __init__(self, t_data, hp_data, hc_data):
        self.t_data = t_data
        self.hp_data = hp_data
        self.hc_data = hc_data

    def __call__(self, params, times):
        remain_time_idx = np.where((self.t_data >= times[0]) & (self.t_data <= times[-1]))[0]
        t_data = self.t_data[remain_time_idx]
        hSp_data = self.hp_data[remain_time_idx]
        hSc_data = self.hc_data[remain_time_idx]
        return t_data, hSp_data + 1.0j * hSc_data


class MBHB_FastLISA:
    def __init__(self, approx_method="IMRPhenomD", modes=None, buffer=True, verbose=0):
        """
        Args:
            modes: None or a list [(2, 2), (2, 1), ...] specifying the harmonic modes of waveform
            buffer: decides whether to extend f_lower to lower frequencies
        """
        self.approx_method = approx_method
        self.modes = modes
        self.buffer = buffer
        self.verbose = verbose

    def __call__(self, Mc, q, spin1z, spin2z, tc, phic, D, inc, psi, T=1.0, dt=10.0):
        """
        Args:
            Mc: redshifted chirp mass in [MSUN]
            q, spin1z, spin2z: dimensionless mass ratio, spin of BH1 and BH2 along the z-axis
            tc: coalescence time in [s]
            phic: coalescence phase in [rad]
            D: luminosity distance in [Mpc]
            inc: inclination angle in [rad]
            psi: polarization angle in [rad]
            dt: the sampling time in [s]
            T: the total observation time in [year]
        Returns:
            hp + i hc in T, with a sampling cadance of dt
        """
        mass_scale = max(1.0, Mc / 50.0)

        if self.verbose == 1:
            print("mass scale:", mass_scale)

        # if Mc <= 1e5:
        #     dt_wf = 0.5
        # elif Mc <= 1e6:
        #     dt_wf = 5.
        # elif Mc <= 1e7:
        #     dt_wf = 50.
        # else:
        #     dt_wf = 100.
        dt_wf = 1e-5 * Mc

        # get rescaled masses
        m1 = m1_Mc_q(Mc, q) / mass_scale
        m2 = m1 * q

        # get the maximum m
        if self.modes is not None:
            m_array = [emm for (ell, emm) in self.modes]
            m_max = max(m_array)
        else:
            if "HM" in self.approx_method:
                m_max = 4
            else:
                m_max = 2

        # set f_lower
        Tobs = tc / YEAR
        f_lower = 1.75e-5 * (Mc / 1e6) ** (-5.0 / 8.0) * (Tobs / 10.0) ** (-3.0 / 8.0)
        if self.buffer:  # the buffered f_lower is at least 1.5 times lower
            f_lower = min(f_lower * 2.0 / m_max, f_lower / 1.5)
        f_lower = max(f_lower, 1e-5)  # set a lower limit of 1e-5 Hz
        f_lower *= mass_scale  # get rescaled lower frequency
        if self.verbose == 1:
            print("minimum frequency before rescale:", f_lower / mass_scale)
            print("minimum frequency after rescale:", f_lower)

        # calculate waveform
        hp, hc = wf.get_td_waveform(approximant=self.approx_method, mass1=m1, mass2=m2, spin1z=spin1z, spin2z=spin2z, coa_phase=phic, distance=D, inclination=inc, delta_t=dt_wf / mass_scale, f_lower=f_lower, mode_array=self.modes)

        # scale back to the desired waveform
        hp, hc = hp.trim_zeros(), hc.trim_zeros()
        hSp_data = np.array(hp) * mass_scale
        hSc_data = np.array(hc) * mass_scale
        t_data = np.array(hp.sample_times) * mass_scale + tc
        self.tend = t_data[-1]
        if self.verbose == 1:
            print("length of data:", len(t_data))

        # get interpolation functions
        self.hpfunc = InterpolatedUnivariateSpline(x=t_data, y=hSp_data, k=5, ext="zeros")
        self.hcfunc = InterpolatedUnivariateSpline(x=t_data, y=hSc_data, k=5, ext="zeros")

        # calculate hp and hc, considering the polarization angle
        t = np.arange(0.0, T * YEAR, dt)
        hSp = self.hpfunc(t)
        hSc = self.hcfunc(t)
        cos2psi = np.cos(2.0 * psi)
        sin2psi = np.sin(2.0 * psi)
        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi
        return hp + 1j * hc


class MBHB_Injection:
    def __init__(self, approx_method="IMRPhenomD", modes=None, buffer=True, verbose=0):
        """
        Args:
            modes: None or a list [(2, 2), (2, 1), ...] specifying the harmonic modes of waveform
            buffer: decides whether to extend f_lower to lower frequencies
        """
        self.approx_method = approx_method
        self.modes = modes
        self.buffer = buffer
        self.verbose = verbose

    def __call__(self, params, times):
        """
        Args:
            param: parameter dict, with keys:
                "chirp_mass": [Msun]
                "mass_ratio": [1]
                "spin_1z": [1]
                "spin_2z": [1]
                "coalescence_time": [day]
                "coalescence_phase": [rad]
                "luminosity_distance": [MPC]
                "inclination": [rad]
            times: numpy array 
        Returns:
            t, hp + i hc
        """
        Mc = params["chirp_mass"]
        q = params["mass_ratio"]
        spin1z = params["spin_1z"]
        spin2z = params["spin_2z"]
        tc = params["coalescence_time"] * DAY
        phic = params["coalescence_phase"]
        D = params["luminosity_distance"]
        inc = params["inclination"]

        mass_scale = max(1.0, Mc / 50.0)

        if self.verbose == 1:
            print("mass scale:", mass_scale)

        # if Mc <= 1e5:
        #     dt_wf = 0.5
        # elif Mc <= 1e6:
        #     dt_wf = 5.
        # elif Mc <= 1e7:
        #     dt_wf = 50.
        # else:
        #     dt_wf = 100.
        dt_wf = 1e-5 * Mc

        # get rescaled masses
        m1 = m1_Mc_q(Mc, q) / mass_scale
        m2 = m1 * q

        # get the maximum m
        if self.modes is not None:
            m_array = [emm for (ell, emm) in self.modes]
            m_max = max(m_array)
        else:
            if "HM" in self.approx_method:
                m_max = 4
            else:
                m_max = 2

        # set f_lower
        # Tobs = tc / YEAR
        Tobs = (tc - times[0]) / YEAR
        f_lower = 1.75e-5 * (Mc / 1e6) ** (-5.0 / 8.0) * (Tobs / 10.0) ** (-3.0 / 8.0)
        if self.buffer:  # the buffered f_lower is at least 1.5 times lower
            f_lower = min(f_lower * 2.0 / m_max, f_lower / 1.5)
        f_lower = max(f_lower, 1e-5)  # set a lower limit of 1e-5 Hz
        f_lower *= mass_scale  # get rescaled lower frequency
        if self.verbose == 1:
            print("minimum frequency before rescale:", f_lower / mass_scale)
            print("minimum frequency after rescale:", f_lower)

        # calculate waveform
        hp, hc = wf.get_td_waveform(approximant=self.approx_method, mass1=m1, mass2=m2, spin1z=spin1z, spin2z=spin2z, coa_phase=phic, distance=D, inclination=inc, delta_t=dt_wf / mass_scale, f_lower=f_lower, mode_array=self.modes)

        # scale back to the desired waveform
        hp, hc = hp.trim_zeros(), hc.trim_zeros()
        hSp_data = np.array(hp) * mass_scale
        hSc_data = np.array(hc) * mass_scale
        t_data = np.array(hp.sample_times) * mass_scale + tc
        self.tend = t_data[-1]

        # reduce the computation time of interpolation 
        remain_time_idx = np.where((t_data >= times[0]) & (t_data <= times[-1]))[0]
        t_data = t_data[remain_time_idx]
        hSp_data = hSp_data[remain_time_idx]
        hSc_data = hSc_data[remain_time_idx]

        if self.verbose == 1:
            print("data length:", len(t_data))

        return t_data, hSp_data + 1.0j * hSc_data


try:
    from pyseobnr.generate_waveform import GenerateWaveform, generate_modes_opt

    # print("Has pyseobnr")
except ImportError:
    # print("No pyseobnr")
    pass


class MBHB_v5_Injection:
    def __init__(self, approximant="SEOBNRv5HM", verbose=0):
        """
        str approximant:
        * ``SEOBNRv5HM`` (default)
        * ``SEOBNRv5PHM``
        * ``SEOBNRv5EHM``
        """
        if approximant not in ["SEOBNRv5HM", "SEOBNRv5PHM", "SEOBNRv5EHM"]:
            raise ValueError("approximant not implemented.")

        self.approximant = approximant
        self.verbose = verbose

    def __call__(self, params, times):
        """
        Args:
            param: parameter dict, with keys:
                "chirp_mass": [Msun]
                "mass_ratio": [1]
                "spin_1z": [1]
                "spin_2z": [1]
                "coalescence_time": [day]
                "reference_phase": [rad]
                "luminosity_distance": [MPC]
                "inclination": [rad]
                "eccentricity": [1], if EHM waveform
            times: numpy array 
        Returns:
            t, hp + i hc
        """
        Mc = params["chirp_mass"]
        q = params["mass_ratio"]
        spin1z = params["spin_1z"]
        spin2z = params["spin_2z"]
        tc = params["coalescence_time"] * DAY
        phi_ref = params["reference_phase"]
        D = params["luminosity_distance"]
        inc = params["inclination"]
        if self.approximant == "SEOBNRv5EHM":
            ecc = params["eccentricity"]

        # set sampling rate
        # if Mc <= 1e5:
        #     sampling_rate = 2.
        # elif Mc <= 1e6:
        #     sampling_rate = 0.2
        # elif Mc <= 1e7:
        #     sampling_rate = 0.02
        # else:
        #     sampling_rate = 0.01
        sampling_rate = 1.0 / (1e-5 * Mc)

        # set rescaled parameters
        mass_scale = max(Mc / 50.0, 1.0)
        m1 = m1_Mc_q(Mc, q) / mass_scale
        m2 = m1 * q
        Mt = m1 + m2
        dt = 1.0 / sampling_rate / mass_scale
        distance = D
        inclination = inc
        phiRef = phi_ref
        fRef = 10e-3 * mass_scale  # so that the original f_ref is 10 mHz
        # fRef = 100., # so that the original f_ref is 100 Hz / mass_scale
        approximant = self.approximant
        s1x = s1y = s2x = s2y = 0.0
        s1z = spin1z
        s2z = spin2z
        f_max = 1024.0  # Hz
        f_min = 0.0157 / (Mt * np.pi * lal.MTSUN_SI) / 10.0  # Hz
        deltaF = 0.125
        params_dict = {
            "mass1": m1,
            "mass2": m2,
            "spin1x": s1x,
            "spin1y": s1y,
            "spin1z": s1z,
            "spin2x": s2x,
            "spin2y": s2y,
            "spin2z": s2z,
            "deltaT": dt,
            "deltaF": deltaF,
            "f22_start": f_min,
            "f_ref": fRef,
            "phi_ref": phiRef,
            "distance": distance,
            "inclination": inclination,
            "f_max": f_max,
            "approximant": approximant,
            "postadiabatic": False,
        }
        if self.approximant == "SEOBNRv5EHM":
            params_dict["eccentricity"] = ecc
            params_dict.pop("f_ref")
            fRef = f_min # for eccentric case, f_ref is decided by f_min hence not fixed 

        if self.verbose == 1:
            print("mass scale:", mass_scale)
            print("rescaled parameters m1, m2, f_min, f_ref:", m1, m2, f_min, fRef)
        wfm_gen = GenerateWaveform(params_dict)  # call the generator with the parameters

        hp, hc = wfm_gen.generate_td_polarizations()  # calculate rescaled hp and hc in the source frame
        t, _ = wfm_gen.generate_td_modes()  # only the rescaled time will be used

        self.hp_data = np.array(hp.data.data) * mass_scale
        self.hc_data = np.array(hc.data.data) * mass_scale
        self.t_data = t * mass_scale + tc
        self.sampling_rate = 1.0 / (self.t_data[1] - self.t_data[0])
        self.tend = self.t_data[-1]
        self.f_ref = fRef / mass_scale

        if self.verbose == 1:
            print("original f_ref:", self.f_ref)
            print("length of data in day:", (self.t_data[-1] - self.t_data[0]) / DAY)
            print("end time of data in day:", self.tend / DAY)
            print("sampling rate:", self.sampling_rate)

        remain_time_idx = np.where((self.t_data >= times[0]) & (self.t_data <= times[-1]))[0]
        self.t_data = self.t_data[remain_time_idx]
        self.hp_data = self.hp_data[remain_time_idx]
        self.hc_data = self.hc_data[remain_time_idx]

        self.params = params.copy()
        self.params["reference_frequency"] = self.f_ref

        # self.hpfunc = InterpolatedUnivariateSpline(x=self.t_data, y=self.hp_data, k=5, ext='zeros')
        # self.hcfunc = InterpolatedUnivariateSpline(x=self.t_data, y=self.hc_data, k=5, ext='zeros')

        return self.t_data, self.hp_data + 1.0j * self.hc_data


class GB_FastLISA:
    """
    the GB waveform class for FastLISAResponse
    """

    def __init__(self, use_gpu=False):
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0):
        # get the t array
        t = self.xp.arange(0.0, T * YEAR, dt)
        cos2psi = self.xp.cos(2.0 * psi)
        sin2psi = self.xp.sin(2.0 * psi)
        cosiota = self.xp.cos(iota)

        fddot = 11.0 / 3.0 * fdot**2 / f

        # fast lisa response tutorial:
        # phase = (
        #     2 * np.pi * (f * t + 1.0 / 2.0 * fdot * t ** 2 + 1.0 / 6.0 * fddot * t ** 3)
        #     + phi0
        # )
        # hSp = -self.xp.cos(phase) * A * (1.0 + cosiota * cosiota)
        # hSc = -self.xp.sin(phase) * 2.0 * A * cosiota
        # hp = hSp * cos2psi - hSc * sin2psi
        # hc = hSp * sin2psi + hSc * cos2psi

        # TDC:
        phase = 2 * np.pi * (f * t + 1.0 / 2.0 * fdot * t**2 + 1.0 / 6.0 * fddot * t**3) + phi0
        hSp = self.xp.cos(phase) * A * (1.0 + cosiota * cosiota)
        hSc = self.xp.sin(phase) * 2.0 * A * cosiota
        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi

        return hp + 1j * hc


class GB_Injection:
    def __init__(self, use_gpu=False):
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

    def __call__(self, params, times):
        A = params["A"]
        f = params["f0"]
        fdot = params["fdot0"]
        phi0 = params["phase0"]
        cosiota = self.xp.cos(params["inclination"])
        fddot = 11.0 / 3.0 * fdot**2 / f
        times_in = self.xp.arange(times[0], times[-1], self.xp.float64(min(3e-2 / f, times[1] - times[0]))) 
        phase = 2 * self.xp.pi * (f * times_in + 1.0 / 2.0 * fdot * times_in**2 + 1.0 / 6.0 * fddot * times_in**3) + phi0

        hp = self.xp.cos(phase) * A * (1.0 + cosiota**2)
        hc = self.xp.sin(phase) * 2.0 * A * cosiota

        return times_in, hp + 1.0j * hc


class GeneralTDIResponse:
    eta_strings = {"12": [(1.0, [])], "13": [(1.0, [])], "23": [(1.0, [])], "21": [(1.0, [])], "31": [(1.0, [])], "32": [(1.0, [])]}
    X2_strings = {
        "12": [(1.0, []), (-1.0, ["13", "31"]), (-1.0, ["13", "31", "12", "21"]), (1.0, ["12", "21", "13", "31", "13", "31"])],
        "23": [],
        "31": [(-1.0, ["13"]), (1.0, ["12", "21", "13"]), (1.0, ["12", "21", "13", "31", "13"]), (-1.0, ["13", "31", "12", "21", "12", "21", "13"])],
        "21": [(1.0, ["12"]), (-1.0, ["13", "31", "12"]), (-1.0, ["13", "31", "12", "21", "12"]), (1.0, ["12", "21", "13", "31", "13", "31", "12"])],
        "32": [],
        "13": [(-1.0, []), (1.0, ["12", "21"]), (1.0, ["12", "21", "13", "31"]), (-1.0, ["13", "31", "12", "21", "12", "21"])],
    }
    GB_param_names = ["A", "f0", "fdot0", "phase0", "inclination", "longitude", "latitude", "psi"]
    MBHB_param_names = ["chirp_mass", "mass_ratio", "spin_1z", "spin_2z", "coalescence_time", "coalescence_phase", "luminosity_distance", "inclination", "longitude", "latitude", "psi"]
    MBHB_v5_param_names = ["chirp_mass", "mass_ratio", "spin_1z", "spin_2z", "coalescence_time", "reference_phase", "luminosity_distance", "inclination", "longitude", "latitude", "psi", "eccentricity"]
    general_param_names = ["longitude", "latitude", "psi"]

    def __init__(self, orbit, Pstring, tcb_times, use_gpu=False, drop_points=0, linear_interp=True, return_eta=False):
        """
        Args:
            and the times at which the polarizations are calculated, which do not have to be identical to the input times.
            orbit: an orbit object
            Pstring: a string specifying the TDI channel
            tcb_times: TCB times at which the TDI responses will be calculated
            use_gpu: if True, the waveform generator should takes cupy arraies as inputs and outputs cupy arraies
            gpu acceleration can only be used when linear_interp=True, while the accuracy of linear interpolation is no sufficient for complex waveforms such as HM MBHB and EMRI
        """
        self.orbit_object = orbit
        self.tcb_times = tcb_times.copy()
        self.Ntime = len(tcb_times)
        self.use_gpu = use_gpu
        self.drop_points = drop_points
        self.linear_interp = linear_interp
        self.return_eta = return_eta
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

        # the orbit functions use numpy array as input
        if isinstance(tcb_times, xp.ndarray) and HAS_GPU:
            tcb_times = tcb_times.get()

        # calculate time series associated with time delays
        self.Ndelay_dict = {}  # each item is an integer
        for key in MOSA_labels:
            self.Ndelay_dict[key] = len(Pstring[key])

        self.delay_factor_dict = {}  # each item is a xp array of shape (Ndelay)
        for key in MOSA_labels:
            self.delay_factor_dict[key] = []
            for Idelay in range(self.Ndelay_dict[key]):
                self.delay_factor_dict[key].append(Pstring[key][Idelay][0])
            self.delay_factor_dict[key] = self.xp.array(self.delay_factor_dict[key])

        self.delay_dict = {}  # each item is a xp array of shape (Ndelay, Ntime)
        for key in MOSA_labels:
            self.delay_dict[key] = []
            Pij = Pstring[key]
            for Idelay in range(self.Ndelay_dict[key]):
                d_Idelay = np.zeros_like(tcb_times)
                N_single_delay = len(Pij[Idelay][1])
                for I_single_delay in range(N_single_delay):
                    if Pij[Idelay][1][I_single_delay][0] == "-":
                        d_Idelay += -orbit.LTTfunctions()[Pij[Idelay][1][I_single_delay][1:]](tcb_times)
                    else:
                        d_Idelay += orbit.LTTfunctions()[Pij[Idelay][1][I_single_delay]](tcb_times)
                self.delay_dict[key].append(d_Idelay)

        self.delayed_dij_dict = {}  # dij delayed by d_Idelay, each item is a xp array of shape (Ndelay, Ntime)
        for key in MOSA_labels:
            self.delayed_dij_dict[key] = []
            for Idelay in range(self.Ndelay_dict[key]):
                self.delayed_dij_dict[key].append(orbit.LTTfunctions()[key](tcb_times - self.delay_dict[key][Idelay]))
            self.delayed_dij_dict[key] = self.xp.array(self.delayed_dij_dict[key])

        # calculate time series associated with orbit
        self.arm_vector_dict = assign_function_for_MOSAs(
            functions=orbit.ArmVectorfunctions(),
            proper_time=tcb_times,
        )  # each item is a xp array of shape (Ntime, 3)
        for key in MOSA_labels:
            self.arm_vector_dict[key] = self.xp.array(self.arm_vector_dict[key])

        self.delayed_send_position_vector_dict = {}  # positions of sending SCs delayed by d_Idelay, each item (Ndelay, Ntime, 3)
        for key in MOSA_labels:
            self.delayed_send_position_vector_dict[key] = []
            for Idelay in range(self.Ndelay_dict[key]):
                self.delayed_send_position_vector_dict[key].append(orbit.Positionfunctions()[key[1]](tcb_times - self.delay_dict[key][Idelay]))
            self.delayed_send_position_vector_dict[key] = self.xp.array(self.delayed_send_position_vector_dict[key])

        self.delayed_recv_position_vector_dict = {}  # positions of receiving SCs delayed by d_Idelay, each item (Ndelay, Ntime, 3)
        for key in MOSA_labels:
            self.delayed_recv_position_vector_dict[key] = []
            for Idelay in range(self.Ndelay_dict[key]):
                self.delayed_recv_position_vector_dict[key].append(orbit.Positionfunctions()[key[0]](tcb_times - self.delay_dict[key][Idelay]))
            self.delayed_recv_position_vector_dict[key] = self.xp.array(self.delayed_recv_position_vector_dict[key])

        # convert delay dict to xp array at last since it is used in the calculation of other dicts
        for key in MOSA_labels:
            self.delay_dict[key] = self.xp.array(self.delay_dict[key])

        self.ep_0 = self.xp.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        self.ec_0 = self.xp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

    def __call__(self, parameters, waveform_generator):
        """
        Args:
            parameters: a dictionary storing the source parameters
            waveform_generator: a waveform object, which has a __call__ function that returns source-frame polarizations hp + ihc for given parameters
        Returns:
            the time series of TDI responses
        """
        self.waveform_generator = waveform_generator

        # calculate wave vector and polar bases using the extrinsic parameters
        l = parameters["longitude"]
        b = parameters["latitude"]
        p = parameters["psi"]

        wave_vector = -self.xp.array([self.xp.cos(l) * self.xp.cos(b), self.xp.sin(l) * self.xp.cos(b), self.xp.sin(b)])  # (3)

        O = self.xp.zeros((3, 3))
        O[0][0] = self.xp.sin(l) * self.xp.cos(p) - self.xp.cos(l) * self.xp.sin(b) * self.xp.sin(p)
        O[0][1] = -self.xp.sin(l) * self.xp.sin(p) - self.xp.cos(l) * self.xp.sin(b) * self.xp.cos(p)
        O[0][2] = -self.xp.cos(l) * self.xp.cos(b)
        O[1][0] = -self.xp.cos(l) * self.xp.cos(p) - self.xp.sin(l) * self.xp.sin(b) * self.xp.sin(p)
        O[1][1] = self.xp.cos(l) * self.xp.sin(p) - self.xp.sin(l) * self.xp.sin(b) * self.xp.cos(p)
        O[1][2] = -self.xp.sin(l) * self.xp.cos(b)
        O[2][0] = self.xp.cos(b) * self.xp.sin(p)
        O[2][1] = self.xp.cos(b) * self.xp.cos(p)
        O[2][2] = -self.xp.sin(b)
        OT = O.transpose()
        e_p = self.xp.dot(self.xp.dot(O, self.ep_0), OT)  # (3, 3)
        e_c = self.xp.dot(self.xp.dot(O, self.ec_0), OT)  # (3, 3)

        # calculate fiducial waveforms at tcb_times, and the delayed ones will be obtained via interpolation
        times_interp, hphc0 = self.waveform_generator(parameters, self.tcb_times)  # times_interp does not has to be the same as tcb_times, it only acts as the x value of interpolation

        # calculate TDI response
        res = self.xp.zeros((6, self.Ntime))

        if self.linear_interp:
            for ikey, key in enumerate(MOSA_labels):
                if self.Ndelay_dict[key] == 0:
                    continue
                else:
                    # terms that do not need to be delayed
                    Fp = self.xp.sum(self.xp.matmul(self.arm_vector_dict[key], e_p) * self.arm_vector_dict[key], axis=1)  # (Ntime)
                    Fc = self.xp.sum(self.xp.matmul(self.arm_vector_dict[key], e_c) * self.arm_vector_dict[key], axis=1)  # (Ntime)
                    Denominator = (self.xp.matmul(self.arm_vector_dict[key], wave_vector) - 1.0) * (-2.0)  # (Ntime)

                    # terms that need to be delayed (waveforms)
                    Fp_delta_hp = self.xp.zeros(self.Ntime)
                    Fc_delta_hc = self.xp.zeros(self.Ntime)
                    for Idelay in range(self.Ndelay_dict[key]):
                        delayed_tcb_times = self.tcb_times - self.delay_dict[key][Idelay]
                        t_send = delayed_tcb_times - self.xp.matmul(self.delayed_send_position_vector_dict[key][Idelay], wave_vector) / C - self.delayed_dij_dict[key][Idelay]  # (Ntime)
                        t_recv = delayed_tcb_times - self.xp.matmul(self.delayed_recv_position_vector_dict[key][Idelay], wave_vector) / C  # (Ntime)
                        hphc_send = self.xp.interp(x=t_send, xp=times_interp, fp=hphc0, left=0.0, right=0.0)  # each (Ntime)
                        hphc_recv = self.xp.interp(x=t_recv, xp=times_interp, fp=hphc0, left=0.0, right=0.0)  # each (Ntime)
                        hp_send = self.xp.real(hphc_send)
                        hc_send = self.xp.imag(hphc_send)
                        hp_recv = self.xp.real(hphc_recv)
                        hc_recv = self.xp.imag(hphc_recv)
                        Fp_delta_hp += self.delay_factor_dict[key][Idelay] * (hp_send - hp_recv)  # (Ntime)
                        Fc_delta_hc += self.delay_factor_dict[key][Idelay] * (hc_send - hc_recv)  # (Ntime)
                    Fp_delta_hp *= Fp  # (Ntime)
                    Fc_delta_hc *= Fc  # (Ntime)
                    res[ikey] = (Fp_delta_hp + Fc_delta_hc) / Denominator

        else:
            hp_func = InterpolatedUnivariateSpline(x=times_interp, y=self.xp.real(hphc0), k=5, ext="zeros")
            hc_func = InterpolatedUnivariateSpline(x=times_interp, y=self.xp.imag(hphc0), k=5, ext="zeros")

            for ikey, key in enumerate(MOSA_labels):
                if self.Ndelay_dict[key] == 0:
                    continue
                else:
                    # terms that do not need to be delayed
                    Fp = self.xp.sum(self.xp.matmul(self.arm_vector_dict[key], e_p) * self.arm_vector_dict[key], axis=1)  # (Ntime)
                    Fc = self.xp.sum(self.xp.matmul(self.arm_vector_dict[key], e_c) * self.arm_vector_dict[key], axis=1)  # (Ntime)
                    Denominator = (self.xp.matmul(self.arm_vector_dict[key], wave_vector) - 1.0) * (-2.0)  # (Ntime)

                    # terms that need to be delayed (waveforms)
                    Fp_delta_hp = self.xp.zeros(self.Ntime)
                    Fc_delta_hc = self.xp.zeros(self.Ntime)
                    for Idelay in range(self.Ndelay_dict[key]):
                        delayed_tcb_times = self.tcb_times - self.delay_dict[key][Idelay]
                        t_send = delayed_tcb_times - self.xp.matmul(self.delayed_send_position_vector_dict[key][Idelay], wave_vector) / C - self.delayed_dij_dict[key][Idelay]  # (Ntime)
                        t_recv = delayed_tcb_times - self.xp.matmul(self.delayed_recv_position_vector_dict[key][Idelay], wave_vector) / C  # (Ntime)
                        hp_send = hp_func(t_send)
                        hc_send = hc_func(t_send)
                        hp_recv = hp_func(t_recv)
                        hc_recv = hc_func(t_recv)
                        Fp_delta_hp += self.delay_factor_dict[key][Idelay] * (hp_send - hp_recv)  # (Ntime)
                        Fc_delta_hc += self.delay_factor_dict[key][Idelay] * (hc_send - hc_recv)  # (Ntime)
                    Fp_delta_hp *= Fp  # (Ntime)
                    Fc_delta_hc *= Fc  # (Ntime)
                    res[ikey] = (Fp_delta_hp + Fc_delta_hc) / Denominator

        if self.return_eta:
            res[:, : self.drop_points] = 0.0
            res[:, -self.drop_points :] = 0.0
        else:
            res = self.xp.sum(res, axis=0)
            res[: self.drop_points] = 0.0
            res[-self.drop_points :] = 0.0
        return res
    
    

class FastMichelsonTDIResponse:
    GB_param_names = ["A", "f0", "fdot0", "phase0", "inclination", "longitude", "latitude", "psi"]
    MBHB_param_names = ["chirp_mass", "mass_ratio", "spin_1z", "spin_2z", "coalescence_time", "coalescence_phase", "luminosity_distance", "inclination", "longitude", "latitude", "psi"]
    MBHB_v5_param_names = ["chirp_mass", "mass_ratio", "spin_1z", "spin_2z", "coalescence_time", "reference_phase", "luminosity_distance", "inclination", "longitude", "latitude", "psi", "eccentricity"]
    general_param_names = ["longitude", "latitude", "psi"]

    def __init__(self, orbit, tcb_times, use_gpu=False, drop_points=0, interp_method="linear", complex_waveform_interp_order=5):
        """
        Initialize orbit-specified time delays, which will be used to calculate single-link responses and TDIs 
        Args:
            orbit: an orbit object
            tcb_times: TCB times at which the TDI responses will be calculated, numpy or cupy arraies 
            use_gpu: if True, the waveform generator should takes cupy arraies as inputs and outputs cupy arraies
            interp_method shoude be chosen from "linear", "Akima", "Spline3", "Spline4", "Spline5"
            choise of interpolation method: 
                1) for GBs, linear interpolation is generally enough 
                2) for MBHBs spline interpolation is a safe choise. if the speed of linear interpolation is necessary, make sure to turn on complex_waveform
        """
        self.orbit_object = orbit
        self.Ntime = len(tcb_times)
        self.use_gpu = use_gpu
        self.drop_points = drop_points
        
        if use_gpu: 
            self.xp = xp 
            self.xinterp = xinterp
        else: 
            self.xp = np 
            self.xinterp = interp
        self.SIN = self.xp.sin
        self.COS = self.xp.cos 
        self.MATMUL = self.xp.matmul
        self.REAL = self.xp.real
        self.IMAG = self.xp.imag
        
        if interp_method == "linear": 
            self.linear_interp = True 
            self.LIN_INTERP = self.xp.interp
        else: 
            self.linear_interp = False 
            if interp_method == "Akima": 
                self.spline_interp = False 
                self.interp_class = self.xinterp.Akima1DInterpolator
                self.interp_kwargs = dict()
            elif interp_method == "Spline3": 
                self.spline_interp = True 
                self.interp_class = self.xinterp.make_interp_spline
                self.interp_k_order = 3 
                self.interp_kwargs = dict(k=self.interp_k_order)
            elif interp_method == "Spline4": 
                self.spline_interp = True
                self.interp_class = self.xinterp.make_interp_spline
                self.interp_k_order = 4 
                self.interp_kwargs = dict(k=self.interp_k_order)
            elif interp_method == "Spline5": 
                self.spline_interp = True
                self.interp_class = self.xinterp.make_interp_spline
                self.interp_k_order = 5 
                self.interp_kwargs = dict(k=self.interp_k_order)
            else: 
                raise NotImplementedError("Interpolation method not implemented.")
            
        self.complex_waveform_interp_order = complex_waveform_interp_order
            
        # the orbit functions use numpy array as input
        if isinstance(tcb_times, xp.ndarray) and HAS_GPU:
            self.tcb_times = tcb_times.get() # now self.tcb_times is a numpy array 
        else: 
            self.tcb_times = tcb_times.copy() # now self.tcb_times is a numpy array 
            
        # calculate TDI delays 
        # 1-link delays 
        self.d12 = self.xp.array(orbit.LTTfunctions()["12"](self.tcb_times)) # (Nt,)
        self.d23 = self.xp.array(orbit.LTTfunctions()["23"](self.tcb_times))
        self.d31 = self.xp.array(orbit.LTTfunctions()["31"](self.tcb_times))
        self.d21 = self.xp.array(orbit.LTTfunctions()["21"](self.tcb_times))
        self.d32 = self.xp.array(orbit.LTTfunctions()["32"](self.tcb_times))
        self.d13 = self.xp.array(orbit.LTTfunctions()["13"](self.tcb_times))
        # 2-link delays 
        self.d212 = self.d21 + self.d12 # (Nt,) some of the variables are duplicated, but it won't affect the speed of response calculation 
        self.d232 = self.d23 + self.d32 
        self.d323 = self.d32 + self.d23 
        self.d313 = self.d31 + self.d13 
        self.d131 = self.d13 + self.d31 
        self.d121 = self.d12 + self.d21 
        # 4-link delays 
        self.d23212 = self.d232 + self.d212 # (Nt,)
        self.d31323 = self.d313 + self.d323 
        self.d12131 = self.d121 + self.d131 

        # calculate orbit variables 
        # arm vectors 
        self.arm_vector_dict = assign_function_for_MOSAs(
            functions=orbit.ArmVectorfunctions(),
            proper_time=self.tcb_times,
        )  
        for key in MOSA_labels:
            self.arm_vector_dict[key] = self.xp.array(self.arm_vector_dict[key]) # each item is a xp array of shape (Ntime, 3)
        # position vectors 
        self.position_vector_dict = assign_function_for_SCs(
            functions=orbit.Positionfunctions(), 
            proper_time=self.tcb_times, 
        )
        for key in SC_labels: 
            self.position_vector_dict[key] = self.xp.array(self.position_vector_dict[key]) # each item is a xp array of shape (Ntime, 3)

        self.tcb_times = self.xp.array(self.tcb_times) 
        self.ep_0 = self.xp.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        self.ec_0 = self.xp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

    def __call__(self, parameters, waveform_generator, optimal_combination=False, complex_waveform=False):
        """
        Args:
            parameters: a dictionary storing the source parameters
            waveform_generator: a waveform object, which has a __call__ function that returns source-frame polarizations hp + ihc for given parameters
            if complex_waveform is True, the waveform interpolation will always be done via splines, regardless of the interp_method setting.
        Returns:
            the time series of TDI responses XYZ (optimal_combination=False) or AET (optimal_combination=True), shape is (3, Nt) 
        """
        self.waveform_generator = waveform_generator

        # calculate wave vector and polar bases using the extrinsic parameters
        l = parameters["longitude"]
        b = parameters["latitude"]
        p = parameters["psi"]

        wave_vector = -self.xp.array([self.COS(l) * self.COS(b), self.SIN(l) * self.COS(b), self.SIN(b)])  # (3)

        O = self.xp.zeros((3, 3))
        O[0][0] = self.SIN(l) * self.COS(p) - self.COS(l) * self.SIN(b) * self.SIN(p)
        O[0][1] = -self.SIN(l) * self.SIN(p) - self.COS(l) * self.SIN(b) * self.COS(p)
        O[0][2] = -self.COS(l) * self.COS(b)
        O[1][0] = -self.COS(l) * self.COS(p) - self.SIN(l) * self.SIN(b) * self.SIN(p)
        O[1][1] = self.COS(l) * self.SIN(p) - self.SIN(l) * self.SIN(b) * self.COS(p)
        O[1][2] = -self.SIN(l) * self.COS(b)
        O[2][0] = self.COS(b) * self.SIN(p)
        O[2][1] = self.COS(b) * self.COS(p)
        O[2][2] = -self.SIN(b)
        OT = O.transpose()
        e_p = self.xp.dot(self.xp.dot(O, self.ep_0), OT)  # (3, 3)
        e_c = self.xp.dot(self.xp.dot(O, self.ec_0), OT)  # (3, 3)

        # calculate fiducial waveforms at tcb_times, and the delayed ones will be obtained via interpolation
        times_interp, hphc0 = self.waveform_generator(parameters, self.tcb_times)  # times_interp does not has to be the same as tcb_times, it only acts as the x value of interpolation

        # calculate pattern function 
        Fp12 = self.xp.sum(self.MATMUL(self.arm_vector_dict["12"], e_p) * self.arm_vector_dict["12"], axis=1)  # Fp12=Fp21 (Ntime)
        Fp23 = self.xp.sum(self.MATMUL(self.arm_vector_dict["23"], e_p) * self.arm_vector_dict["23"], axis=1)
        Fp31 = self.xp.sum(self.MATMUL(self.arm_vector_dict["31"], e_p) * self.arm_vector_dict["31"], axis=1)
        Fc12 = self.xp.sum(self.MATMUL(self.arm_vector_dict["12"], e_c) * self.arm_vector_dict["12"], axis=1)  # Fc12=Fc21 (Ntime)
        Fc23 = self.xp.sum(self.MATMUL(self.arm_vector_dict["23"], e_c) * self.arm_vector_dict["23"], axis=1)
        Fc31 = self.xp.sum(self.MATMUL(self.arm_vector_dict["31"], e_c) * self.arm_vector_dict["31"], axis=1)
        
        Denominator12 = (self.MATMUL(self.arm_vector_dict["12"], wave_vector) - 1.0) * (-2.0)  # (Ntime)
        Denominator23 = (self.MATMUL(self.arm_vector_dict["23"], wave_vector) - 1.0) * (-2.0) 
        Denominator31 = (self.MATMUL(self.arm_vector_dict["31"], wave_vector) - 1.0) * (-2.0)  
        Denominator21 = (self.MATMUL(self.arm_vector_dict["21"], wave_vector) - 1.0) * (-2.0)  
        Denominator32 = (self.MATMUL(self.arm_vector_dict["32"], wave_vector) - 1.0) * (-2.0) 
        Denominator13 = (self.MATMUL(self.arm_vector_dict["13"], wave_vector) - 1.0) * (-2.0)  
        
        kR1overC = self.MATMUL(self.position_vector_dict["1"], wave_vector) / C # (Ntime)
        kR2overC = self.MATMUL(self.position_vector_dict["2"], wave_vector) / C 
        kR3overC = self.MATMUL(self.position_vector_dict["3"], wave_vector) / C 
            
        if self.linear_interp:      
            if complex_waveform: 
                hp_func = self.xinterp.make_interp_spline(x=times_interp, y=self.REAL(hphc0), k=self.complex_waveform_interp_order)
                hc_func = self.xinterp.make_interp_spline(x=times_interp, y=self.IMAG(hphc0), k=self.complex_waveform_interp_order)
                
                t_send = self.tcb_times - self.d12 - kR2overC
                t_recv = self.tcb_times - kR1overC
                dhp12 = hp_func(t_send, extrapolate=False) - hp_func(t_recv, extrapolate=False) # (Ntime)
                dhc12 = hc_func(t_send, extrapolate=False) - hc_func(t_recv, extrapolate=False)
                dh12 = self.xp.nan_to_num(dhp12 + 1.j * dhc12, nan=0.) # (Ntime), complex 
                
                t_send = self.tcb_times - self.d23 - kR3overC
                t_recv = self.tcb_times - kR2overC
                dhp23 = hp_func(t_send, extrapolate=False) - hp_func(t_recv, extrapolate=False) # (Ntime)
                dhc23 = hc_func(t_send, extrapolate=False) - hc_func(t_recv, extrapolate=False)
                dh23 = self.xp.nan_to_num(dhp23 + 1.j * dhc23, nan=0.) # (Ntime), complex 
                
                t_send = self.tcb_times - self.d31 - kR1overC
                t_recv = self.tcb_times - kR3overC
                dhp31 = hp_func(t_send, extrapolate=False) - hp_func(t_recv, extrapolate=False) # (Ntime)
                dhc31 = hc_func(t_send, extrapolate=False) - hc_func(t_recv, extrapolate=False)
                dh31 = self.xp.nan_to_num(dhp31 + 1.j * dhc31, nan=0.) # (Ntime), complex 
                
                t_send = self.tcb_times - self.d21 - kR1overC
                t_recv = self.tcb_times - kR2overC
                dhp21 = hp_func(t_send, extrapolate=False) - hp_func(t_recv, extrapolate=False) # (Ntime)
                dhc21 = hc_func(t_send, extrapolate=False) - hc_func(t_recv, extrapolate=False)
                dh21 = self.xp.nan_to_num(dhp21 + 1.j * dhc21, nan=0.) # (Ntime), complex 
                
                t_send = self.tcb_times - self.d32 - kR2overC
                t_recv = self.tcb_times - kR3overC
                dhp32 = hp_func(t_send, extrapolate=False) - hp_func(t_recv, extrapolate=False) # (Ntime)
                dhc32 = hc_func(t_send, extrapolate=False) - hc_func(t_recv, extrapolate=False)
                dh32 = self.xp.nan_to_num(dhp32 + 1.j * dhc32, nan=0.) # (Ntime), complex 
                
                t_send = self.tcb_times - self.d13 - kR3overC
                t_recv = self.tcb_times - kR1overC
                dhp13 = hp_func(t_send, extrapolate=False) - hp_func(t_recv, extrapolate=False) # (Ntime)
                dhc13 = hc_func(t_send, extrapolate=False) - hc_func(t_recv, extrapolate=False)
                dh13 = self.xp.nan_to_num(dhp13 + 1.j * dhc13, nan=0.) # (Ntime), complex 

            else: 
                dh12 = self.LIN_INTERP(x=self.tcb_times - self.d12 - kR2overC, xp=times_interp, fp=hphc0, left=0., right=0.) - self.LIN_INTERP(x=self.tcb_times - kR1overC, xp=times_interp, fp=hphc0, left=0., right=0.) # (Ntime), complex 
                dh23 = self.LIN_INTERP(x=self.tcb_times - self.d23 - kR3overC, xp=times_interp, fp=hphc0, left=0., right=0.) - self.LIN_INTERP(x=self.tcb_times - kR2overC, xp=times_interp, fp=hphc0, left=0., right=0.)
                dh31 = self.LIN_INTERP(x=self.tcb_times - self.d31 - kR1overC, xp=times_interp, fp=hphc0, left=0., right=0.) - self.LIN_INTERP(x=self.tcb_times - kR3overC, xp=times_interp, fp=hphc0, left=0., right=0.)
                dh21 = self.LIN_INTERP(x=self.tcb_times - self.d21 - kR1overC, xp=times_interp, fp=hphc0, left=0., right=0.) - self.LIN_INTERP(x=self.tcb_times - kR2overC, xp=times_interp, fp=hphc0, left=0., right=0.)
                dh32 = self.LIN_INTERP(x=self.tcb_times - self.d32 - kR2overC, xp=times_interp, fp=hphc0, left=0., right=0.) - self.LIN_INTERP(x=self.tcb_times - kR3overC, xp=times_interp, fp=hphc0, left=0., right=0.)
                dh13 = self.LIN_INTERP(x=self.tcb_times - self.d13 - kR3overC, xp=times_interp, fp=hphc0, left=0., right=0.) - self.LIN_INTERP(x=self.tcb_times - kR1overC, xp=times_interp, fp=hphc0, left=0., right=0.)
            
            eta12 = (Fp12 * self.REAL(dh12) + Fc12 * self.IMAG(dh12)) / Denominator12 # (Ntime)
            eta23 = (Fp23 * self.REAL(dh23) + Fc23 * self.IMAG(dh23)) / Denominator23
            eta31 = (Fp31 * self.REAL(dh31) + Fc31 * self.IMAG(dh31)) / Denominator31
            eta21 = (Fp12 * self.REAL(dh21) + Fc12 * self.IMAG(dh21)) / Denominator21
            eta32 = (Fp23 * self.REAL(dh32) + Fc23 * self.IMAG(dh32)) / Denominator32
            eta13 = (Fp31 * self.REAL(dh13) + Fc31 * self.IMAG(dh13)) / Denominator13
            
            X0plus = eta12 + self.LIN_INTERP(x=self.tcb_times - self.d12, xp=self.tcb_times, fp=eta21, left=0., right=0.)
            X0minus = eta13 + self.LIN_INTERP(x=self.tcb_times - self.d13, xp=self.tcb_times, fp=eta31, left=0., right=0.)
            X1plus = X0plus - self.LIN_INTERP(x=self.tcb_times - self.d131, xp=self.tcb_times, fp=X0plus, left=0., right=0.)
            X1minus = X0minus - self.LIN_INTERP(x=self.tcb_times - self.d121, xp=self.tcb_times, fp=X0minus, left=0., right=0.) 
            X2_tmp = X1plus - X1minus
            X2 = X2_tmp - self.LIN_INTERP(x=self.tcb_times - self.d12131, xp=self.tcb_times, fp=X2_tmp, left=0., right=0.) # (Ntime)
            
            Y0plus = eta23 + self.LIN_INTERP(x=self.tcb_times - self.d23, xp=self.tcb_times, fp=eta32, left=0., right=0.)
            Y0minus = eta21 + self.LIN_INTERP(x=self.tcb_times - self.d21, xp=self.tcb_times, fp=eta12, left=0., right=0.)
            Y1plus = Y0plus - self.LIN_INTERP(x=self.tcb_times - self.d212, xp=self.tcb_times, fp=Y0plus, left=0., right=0.)
            Y1minus = Y0minus - self.LIN_INTERP(x=self.tcb_times - self.d232, xp=self.tcb_times, fp=Y0minus, left=0., right=0.) 
            Y2_tmp = Y1plus - Y1minus
            Y2 = Y2_tmp - self.LIN_INTERP(x=self.tcb_times - self.d23212, xp=self.tcb_times, fp=Y2_tmp, left=0., right=0.) # (Ntime)

            Z0plus = eta31 + self.LIN_INTERP(x=self.tcb_times - self.d31, xp=self.tcb_times, fp=eta13, left=0., right=0.)
            Z0minus = eta32 + self.LIN_INTERP(x=self.tcb_times - self.d32, xp=self.tcb_times, fp=eta23, left=0., right=0.)
            Z1plus = Z0plus - self.LIN_INTERP(x=self.tcb_times - self.d323, xp=self.tcb_times, fp=Z0plus, left=0., right=0.)
            Z1minus = Z0minus - self.LIN_INTERP(x=self.tcb_times - self.d313, xp=self.tcb_times, fp=Z0minus, left=0., right=0.) 
            Z2_tmp = Z1plus - Z1minus
            Z2 = Z2_tmp - self.LIN_INTERP(x=self.tcb_times - self.d31323, xp=self.tcb_times, fp=Z2_tmp, left=0., right=0.) # (Ntime)

        else:
            if complex_waveform: 
                hp_func = self.xinterp.make_interp_spline(x=times_interp, y=self.REAL(hphc0), k=self.complex_waveform_interp_order)
                hc_func = self.xinterp.make_interp_spline(x=times_interp, y=self.IMAG(hphc0), k=self.complex_waveform_interp_order)
            else: 
                hp_func = self.interp_class(x=times_interp, y=self.REAL(hphc0), **self.interp_kwargs)
                hc_func = self.interp_class(x=times_interp, y=self.IMAG(hphc0), **self.interp_kwargs)
                
            t_send = self.tcb_times - self.d12 - kR2overC
            t_recv = self.tcb_times - kR1overC
            dhp12 = hp_func(t_send, extrapolate=False) - hp_func(t_recv, extrapolate=False) # (Ntime)
            dhc12 = hc_func(t_send, extrapolate=False) - hc_func(t_recv, extrapolate=False)
            
            t_send = self.tcb_times - self.d23 - kR3overC
            t_recv = self.tcb_times - kR2overC
            dhp23 = hp_func(t_send, extrapolate=False) - hp_func(t_recv, extrapolate=False) # (Ntime)
            dhc23 = hc_func(t_send, extrapolate=False) - hc_func(t_recv, extrapolate=False)
            
            t_send = self.tcb_times - self.d31 - kR1overC
            t_recv = self.tcb_times - kR3overC
            dhp31 = hp_func(t_send, extrapolate=False) - hp_func(t_recv, extrapolate=False) # (Ntime)
            dhc31 = hc_func(t_send, extrapolate=False) - hc_func(t_recv, extrapolate=False)
            
            t_send = self.tcb_times - self.d21 - kR1overC
            t_recv = self.tcb_times - kR2overC
            dhp21 = hp_func(t_send, extrapolate=False) - hp_func(t_recv, extrapolate=False) # (Ntime)
            dhc21 = hc_func(t_send, extrapolate=False) - hc_func(t_recv, extrapolate=False)
            
            t_send = self.tcb_times - self.d32 - kR2overC
            t_recv = self.tcb_times - kR3overC
            dhp32 = hp_func(t_send, extrapolate=False) - hp_func(t_recv, extrapolate=False) # (Ntime)
            dhc32 = hc_func(t_send, extrapolate=False) - hc_func(t_recv, extrapolate=False)
            
            t_send = self.tcb_times - self.d13 - kR3overC
            t_recv = self.tcb_times - kR1overC
            dhp13 = hp_func(t_send, extrapolate=False) - hp_func(t_recv, extrapolate=False) # (Ntime)
            dhc13 = hc_func(t_send, extrapolate=False) - hc_func(t_recv, extrapolate=False)
            
            eta12 = self.xp.nan_to_num((Fp12 * dhp12 + Fc12 * dhc12) / Denominator12, nan=0.) # (Ntime)
            eta23 = self.xp.nan_to_num((Fp23 * dhp23 + Fc23 * dhc23) / Denominator23, nan=0.)
            eta31 = self.xp.nan_to_num((Fp31 * dhp31 + Fc31 * dhc31) / Denominator31, nan=0.)
            eta21 = self.xp.nan_to_num((Fp12 * dhp21 + Fc12 * dhc21) / Denominator21, nan=0.)
            eta32 = self.xp.nan_to_num((Fp23 * dhp32 + Fc23 * dhc32) / Denominator32, nan=0.)
            eta13 = self.xp.nan_to_num((Fp31 * dhp13 + Fc31 * dhc13) / Denominator13, nan=0.)
            
            eta12_func = self.interp_class(x=self.tcb_times, y=eta12, **self.interp_kwargs)
            eta23_func = self.interp_class(x=self.tcb_times, y=eta23, **self.interp_kwargs)
            eta31_func = self.interp_class(x=self.tcb_times, y=eta31, **self.interp_kwargs)
            eta21_func = self.interp_class(x=self.tcb_times, y=eta21, **self.interp_kwargs)
            eta32_func = self.interp_class(x=self.tcb_times, y=eta32, **self.interp_kwargs)
            eta13_func = self.interp_class(x=self.tcb_times, y=eta13, **self.interp_kwargs)
            
            # X2 = eta12 - eta12_func(self.tcb_times-self.d131, extrapolate=False) - eta12_func(self.tcb_times-self.d12131, extrapolate=False) + eta12_func(self.tcb_times-self.d12131-self.d131, extrapolate=False) \
            #     - eta31_func(self.tcb_times - self.d13, extrapolate=False) + eta31_func(self.tcb_times - self.d121 - self.d13, extrapolate=False) + eta31_func(self.tcb_times - self.d12131 - self.d13, extrapolate=False) - eta13_func(self.tcb_times - self.d12131 - self.d121 - self.d13, extrapolate=False) \
            #     + eta21_func(self.tcb_times - self.d12, extrapolate=False) - eta21_func(self.tcb_times - self.d131 - self.d12, extrapolate=False) - eta21_func(self.tcb_times - self.d12131 - self.d12, extrapolate=False) + eta21_func(self.tcb_times - self.d12131 - self.d131 - self.d12, extrapolate=False) \
            #     - eta13 + eta13_func(self.tcb_times - self.d121, extrapolate=False) + eta13_func(self.tcb_times - self.d12131, extrapolate=False) - eta13_func(self.tcb_times - self.d12131 - self.d121, extrapolate=False)

            # Y2 = eta23 - eta23_func(self.tcb_times-self.d212, extrapolate=False) - eta23_func(self.tcb_times-self.d23212, extrapolate=False) + eta23_func(self.tcb_times-self.d23212-self.d212, extrapolate=False) \
            #     - eta12_func(self.tcb_times - self.d21, extrapolate=False) + eta12_func(self.tcb_times - self.d232 - self.d21, extrapolate=False) + eta12_func(self.tcb_times - self.d23212 - self.d21, extrapolate=False) - eta21_func(self.tcb_times - self.d23212 - self.d232 - self.d21, extrapolate=False) \
            #     + eta32_func(self.tcb_times - self.d23, extrapolate=False) - eta32_func(self.tcb_times - self.d212 - self.d23, extrapolate=False) - eta32_func(self.tcb_times - self.d23212 - self.d23, extrapolate=False) + eta32_func(self.tcb_times - self.d23212 - self.d212 - self.d23, extrapolate=False) \
            #     - eta21 + eta21_func(self.tcb_times - self.d232, extrapolate=False) + eta21_func(self.tcb_times - self.d23212, extrapolate=False) - eta21_func(self.tcb_times - self.d23212 - self.d232, extrapolate=False)

            # Z2 = eta31 - eta31_func(self.tcb_times-self.d323, extrapolate=False) - eta31_func(self.tcb_times-self.d31323, extrapolate=False) + eta31_func(self.tcb_times-self.d31323-self.d323, extrapolate=False) \
            #     - eta23_func(self.tcb_times - self.d32, extrapolate=False) + eta23_func(self.tcb_times - self.d313 - self.d32, extrapolate=False) + eta23_func(self.tcb_times - self.d31323 - self.d32, extrapolate=False) - eta32_func(self.tcb_times - self.d31323 - self.d313 - self.d32, extrapolate=False) \
            #     + eta13_func(self.tcb_times - self.d31, extrapolate=False) - eta13_func(self.tcb_times - self.d323 - self.d31, extrapolate=False) - eta13_func(self.tcb_times - self.d31323 - self.d31, extrapolate=False) + eta13_func(self.tcb_times - self.d31323 - self.d323 - self.d31, extrapolate=False) \
            #     - eta32 + eta32_func(self.tcb_times - self.d313, extrapolate=False) + eta32_func(self.tcb_times - self.d31323, extrapolate=False) - eta32_func(self.tcb_times - self.d31323 - self.d313, extrapolate=False)
            
            X0plus = self.xp.nan_to_num(eta12 + eta21_func(self.tcb_times - self.d12, extrapolate=False), nan=0.)
            X0minus = self.xp.nan_to_num(eta13 + eta31_func(self.tcb_times - self.d13, extrapolate=False), nan=0.)
            X0plus_func = self.interp_class(x=self.tcb_times, y=X0plus, **self.interp_kwargs)
            X0minus_func = self.interp_class(x=self.tcb_times, y=X0minus, **self.interp_kwargs)
            X1plus = X0plus - X0plus_func(self.tcb_times - self.d131, extrapolate=False)
            X1minus = X0minus - X0minus_func(self.tcb_times - self.d121, extrapolate=False)
            X2_tmp = self.xp.nan_to_num(X1plus - X1minus, nan=0.)
            X2_tmp_func = self.interp_class(x=self.tcb_times, y=X2_tmp, **self.interp_kwargs)
            X2 = X2_tmp - X2_tmp_func(self.tcb_times - self.d12131, extrapolate=False) # (Ntime)
            
            Y0plus = self.xp.nan_to_num(eta23 + eta32_func(self.tcb_times - self.d23, extrapolate=False), nan=0.)
            Y0minus = self.xp.nan_to_num(eta21 + eta12_func(self.tcb_times - self.d21, extrapolate=False), nan=0.)
            Y0plus_func = self.interp_class(x=self.tcb_times, y=Y0plus, **self.interp_kwargs)
            Y0minus_func = self.interp_class(x=self.tcb_times, y=Y0minus, **self.interp_kwargs)
            Y1plus = Y0plus - Y0plus_func(self.tcb_times - self.d212, extrapolate=False)
            Y1minus = Y0minus - Y0minus_func(self.tcb_times - self.d232, extrapolate=False)
            Y2_tmp = self.xp.nan_to_num(Y1plus - Y1minus, nan=0.)
            Y2_tmp_func = self.interp_class(x=self.tcb_times, y=Y2_tmp, **self.interp_kwargs)
            Y2 = Y2_tmp - Y2_tmp_func(self.tcb_times - self.d23212, extrapolate=False) # (Ntime)
            
            Z0plus = self.xp.nan_to_num(eta31 + eta13_func(self.tcb_times - self.d31, extrapolate=False), nan=0.)
            Z0minus = self.xp.nan_to_num(eta32 + eta23_func(self.tcb_times - self.d32, extrapolate=False), nan=0.)
            Z0plus_func = self.interp_class(x=self.tcb_times, y=Z0plus, **self.interp_kwargs)
            Z0minus_func = self.interp_class(x=self.tcb_times, y=Z0minus, **self.interp_kwargs)
            Z1plus = Z0plus - Z0plus_func(self.tcb_times - self.d323, extrapolate=False)
            Z1minus = Z0minus - Z0minus_func(self.tcb_times - self.d313, extrapolate=False)
            Z2_tmp = self.xp.nan_to_num(Z1plus - Z1minus, nan=0.)
            Z2_tmp_func = self.interp_class(x=self.tcb_times, y=Z2_tmp, **self.interp_kwargs)
            Z2 = Z2_tmp - Z2_tmp_func(self.tcb_times - self.d31323, extrapolate=False) # (Ntime)
                
        if optimal_combination: 
            A2, E2, T2 = AETfromXYZ(X2, Y2, Z2)
            res_arr = self.xp.array([A2, E2, T2])
        else: 
            res_arr = self.xp.array([X2, Y2, Z2])
        res_arr[:, :self.drop_points] *= 0. 
        res_arr[:, -self.drop_points:] *= 0. 
        return self.xp.nan_to_num(res_arr, nan=0.)
