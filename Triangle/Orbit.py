import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

from Triangle.Constants import *

class Orbit:
    """
    return dictionaries of functions.
    all the functions take time array (N) as input and return (N, dim) array
    the start time of TCB is set to t = 0, and TCB/TPS coincide at t = 0
    """

    def __init__(self, OrbitDir, max_rows=None, tstart=0, dt=DAY, pn_order=2):

        # ---- step 1: read in orbit data (each item is N × 3) ----
        if max_rows is None:
            self.rdata = {key: np.loadtxt(OrbitDir + "/SCP" + key + ".dat") * AU for key in SC_labels}
            self.vdata = {key: np.loadtxt(OrbitDir + "/SCV" + key + ".dat") * AU / DAY for key in SC_labels}
        else:
            self.rdata = {key: np.loadtxt(OrbitDir + "/SCP" + key + ".dat", max_rows=max_rows) * AU for key in SC_labels}
            self.vdata = {key: np.loadtxt(OrbitDir + "/SCV" + key + ".dat", max_rows=max_rows) * AU / DAY for key in SC_labels}

        self.tstart = tstart
        self.N = len(self.rdata[SC_labels[0]])
        self.tdata = np.arange(self.N) * dt - self.tstart
        self.dt = dt

        # ---- step 2: PN-based post-processing (shared with subclasses) ----
        self._post_process(pn_order)

    # -----------------------------------------------------------------
    #  Post-processing: LTT, ARM, TCB↔TPS, PPR, Position/Velocity
    #  Requires: self.rdata, self.vdata, self.tdata, self.dt (set by __init__)
    # -----------------------------------------------------------------
    def _post_process(self, pn_order):

        LTTdata = {}
        self._LTTfunctions = {}
        self._Dopplerfunctions = {}
        self.LTT_time_int = dict()  # used in MBHB response calculation
        self.LTT_data_int = dict()  # used in MBHB response calculation

        ARMdata = {}
        self._ArmVectorfunctions = {}
        self.ARM_time_int = dict()  # used in MBHB response calculation
        self.ARM_data_int = dict()  # used in MBHB response calculation

        self._Positionfunctions = {}
        self._Velocityfunctions = {}
        self.POS_time_int = dict()  # used in MBHB response calculation
        self.POS_data_int = dict()  # used in MBHB response calculation

        for label in MOSA_labels:
            send_SC = label[1]
            receive_SC = label[0]

            L = self.rdata[receive_SC] - self.rdata[send_SC]
            # 0-th order
            LTT0 = np.sqrt(np.sum(L * L, axis=1)) / C
            ARM0 = L / LTT0[:, np.newaxis] / C
            # 1/2-th order
            nv_recv = np.sum(ARM0 * self.vdata[receive_SC], axis=1)
            LTT1 = np.sum(L * self.vdata[receive_SC], axis=1) / C**2
            ARM1 = self.vdata[receive_SC] / C - ARM0 * nv_recv[:, np.newaxis] / C
            # 1-st order
            LTT2 = 0.5 * (np.sum((self.vdata[receive_SC]) ** 2, axis=1) / C**2 + (nv_recv / C) ** 2) * LTT0
            nx_send = np.sum(ARM0 * self.rdata[send_SC], axis=1)
            nx_recv = np.sum(ARM0 * self.rdata[receive_SC], axis=1)
            K2 = np.sum((self.rdata[send_SC]) ** 2, axis=1) - nx_send**2
            P = (self.rdata[send_SC] - ARM0 * nx_send[:, np.newaxis]) / K2[:, np.newaxis]
            r_recv = np.sqrt(np.sum((self.rdata[receive_SC]) ** 2, axis=1))
            r_send = np.sqrt(np.sum((self.rdata[send_SC]) ** 2, axis=1))
            chi = P * (r_recv - r_send)[:, np.newaxis] + ARM0 * (np.log((nx_recv + r_recv) / (nx_send + r_send)))[:, np.newaxis]
            LTT2 += np.sum(
                (2.0 * G * MSUN / C**3 * chi - (G * MSUN / 2.0 / C / r_recv**3 * LTT0**2)[:, np.newaxis] * self.rdata[receive_SC]) * ARM0,
                axis=1,
            )

            if pn_order == 2:
                LTTdata[label] = LTT0 + LTT1 + LTT2
            elif pn_order == 1:
                LTTdata[label] = LTT0 + LTT1
            elif pn_order == 0:
                LTTdata[label] = LTT0
            else:
                raise NotImplementedError("PN order not implemented.")

            # LTTs are calculated at the emission times, while in the simulation we use LTTs at the reception times
            self._LTTfunctions[label] = InterpolatedUnivariateSpline(self.tdata + LTTdata[label], LTTdata[label], k=5, ext="extrapolate")
            self._Dopplerfunctions[label] = self._LTTfunctions[label].derivative()
            self.LTT_time_int[label] = self.tdata + LTTdata[label]  # (Ntime) used in MBHB response calculation
            self.LTT_data_int[label] = LTTdata[label].copy()  # (Ntime) in [s] used in MBHB response calculation

            ARMdata[label] = ARM0 + ARM1
            # arm vectors are calculated at the emission times, while in the simulation we use arm vectors at the reception times
            self._ArmVectorfunctions[label] = interp1d(
                self.tdata + LTTdata[label],
                ARMdata[label],
                axis=0,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.ARM_time_int[label] = self.tdata + LTTdata[label]  # (Ntime) used in MBHB response calculation
            self.ARM_data_int[label] = ARMdata[label].copy()  # (Ntime, 3) used in MBHB response calculation

        self.TCBinTPSfunctions = {}
        self.TPSinTCBfunctions = {}
        self.TPSwrtTCBfunctions = {}  # TPS - TCB in TCB
        TPSdata = {}
        for label in SC_labels:
            vSC = np.sqrt(np.sum((self.vdata[label]) ** 2, axis=1))
            rSC = np.sqrt(np.sum((self.rdata[label]) ** 2, axis=1))
            rel_diff = -G * MSUN / rSC / C**2 - vSC**2 / 2.0 / C**2
            rel_diff = cumulative_trapezoid(np.insert(rel_diff, 0, 0), dx=self.dt)  # proper time = tcb at the start time
            TPSdata[label] = self.tdata + rel_diff
            self.TCBinTPSfunctions[label] = InterpolatedUnivariateSpline(TPSdata[label], self.tdata, k=5, ext="extrapolate")
            self.TPSinTCBfunctions[label] = InterpolatedUnivariateSpline(self.tdata, TPSdata[label], k=5, ext="extrapolate")
            self.TPSwrtTCBfunctions[label] = InterpolatedUnivariateSpline(self.tdata, TPSdata[label] - self.tdata, k=5, ext="extrapolate")

        # calculate ppr data, recv proper time: TPSdata, recv tcb: tdata, send tcb: tdata - LTT(tdata), send proper time: TPSinTCB(send tcb)
        self._PPRfunctions = {}
        self._DPPRfunctions = {}
        for label in MOSA_labels:
            send_SC = label[1]
            receive_SC = label[0]
            recv_tps = TPSdata[receive_SC]
            recv_tcb = self.tdata
            send_tcb = recv_tcb - self._LTTfunctions[label](recv_tcb)
            send_tps = self.TPSinTCBfunctions[send_SC](send_tcb)
            ppr = recv_tps - send_tps
            self._PPRfunctions[label] = InterpolatedUnivariateSpline(recv_tps, ppr, k=5, ext="extrapolate")
            self._DPPRfunctions[label] = self._PPRfunctions[label].derivative()

        for label in SC_labels:
            self._Positionfunctions[label] = interp1d(
                self.tdata,
                self.rdata[label],
                axis=0,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            self._Velocityfunctions[label] = interp1d(
                self.tdata,
                self.vdata[label],
                axis=0,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.POS_time_int[label] = self.tdata.copy()  # (Ntime) used in MBHB response calculation
            self.POS_data_int[label] = self.rdata[label] / C  # (Ntime, 3) in [s] used in MBHB response calculation

    def ListMembers(self):
        for name, value in vars(self).items():
            print("%s=%s" % (name, value))

    def LTTfunctions(self):
        return self._LTTfunctions

    def Dopplerfunctions(self):
        return self._Dopplerfunctions

    def PPRfunctions(self):
        return self._PPRfunctions

    def DPPRfunctions(self):
        return self._DPPRfunctions

    def ArmVectorfunctions(self):
        return self._ArmVectorfunctions

    def Positionfunctions(self):
        return self._Positionfunctions

    def Velocityfunctions(self):
        return self._Velocityfunctions
    
    
class HeliocentricEqualArmAnalyticOrbit(Orbit):
    """
    Orbit subclass that internally computes the equal-arm analytic
    heliocentric constellation model instead of reading orbit files.

    All interpolation functions (LTT, Doppler, PPR, DPPR, arm vectors,
    positions, velocities, TCB/TPS conversions) are built via the same
    procedure as the parent ``Orbit`` class.

    Parameters
    ----------
    L : float
        Nominal arm length [m].  Default: ``L_nominal``.
    a : float
        Guiding-centre semi-major axis [m].  Default: ``AU``.
    kap : float
        Initial orbital phase [rad].  Default: 0.
    lam : float
        Initial constellation orientation [rad].  Default: 0.
    tstart : float
        Start-time offset [s].  Default: 0.
    dt : float
        Time step [s].  Default: ``DAY``.
    pn_order : int
        PN order * 2 for LTT (0, 1, 2).  Default: 2.
    Tobs : float or None
        Total duration [s].  If *None*, ``5 * YEAR`` is used.
    """

    def __init__(
        self,
        L=L_nominal,
        a=AU,
        kap=0.0,
        lam=0.0,
        tstart=0.0,
        dt=DAY,
        pn_order=2,
        Tobs=None,
    ):
        if Tobs is None:
            Tobs = 5 * YEAR  # nominal operation time for Taiji

        # ---- step 1: analytic equal-arm orbit data ----
        self.tstart = tstart
        self.dt = dt
        self.N = int(Tobs / dt)
        self.tdata = np.arange(self.N) * dt - self.tstart

        e = L / (2.0 * a * np.sqrt(3.0))
        omega = TWOPI / YEAR  # guiding-centre angular frequency

        self.rdata = {}
        self.vdata = {}

        for idx, label in enumerate(SC_labels):
            n = idx
            Bn = n * TWOPI / 3.0 + lam
            A = omega * self.tdata + kap

            sinA = np.sin(A)
            cosA = np.cos(A)
            sinBn = np.sin(Bn)
            cosBn = np.cos(Bn)

            # --- positions (SI) ---
            xn = (
                a * cosA
                + a * e * (sinA * cosA * sinBn - (1.0 + sinA**2) * cosBn)
            )
            yn = (
                a * sinA
                + a * e * (sinA * cosA * cosBn - (1.0 + cosA**2) * sinBn)
            )
            zn = -np.sqrt(3.0) * a * e * np.cos(A - Bn)
            self.rdata[label] = np.column_stack([xn, yn, zn])

            # --- velocities (analytic time derivative) ---
            vx = omega * (
                -a * sinA
                + a * e * (np.cos(2.0 * A) * sinBn - np.sin(2.0 * A) * cosBn)
            )
            vy = omega * (
                a * cosA
                + a * e * (np.cos(2.0 * A) * cosBn + np.sin(2.0 * A) * sinBn)
            )
            vz = omega * (np.sqrt(3.0) * a * e * np.sin(A - Bn))
            self.vdata[label] = np.column_stack([vx, vy, vz])

        # ---- step 2: PN-based post-processing (inherited from Orbit) ----
        self._post_process(pn_order)
        
        
class GeocentricEqualArmAnalyticOrbit(Orbit):
    """
    Orbit subclass for a geocentric equal-arm analytic constellation
    (e.g., TianQin-like detectors).

    Three SCs form an equilateral triangle orbiting the Earth in a
    detector plane whose orientation is given by (phi_det, theta_det).
    The Earth itself follows a circular Kepler orbit in the ecliptic.

    All post-processing (LTT, ARM, TCB↔TPS, PPR, interpolation) is
    inherited from ``Orbit._post_process()``.

    Parameters
    ----------
    L : float
        Arm length [m].  Default: √3 × 10⁸ (TianQin).
    kappa0 : float
        Initial orbital phase of SC1 in the detector plane [rad].
        Default: 0.
    kappa_earth : float
        Initial ecliptic longitude of Earth at t=0 [rad].  Default: 0.
    phi_det : float
        Ecliptic longitude of the detector-plane normal [rad].
        Default: 2.10205135 (J0806, TianQin).
    theta_det : float
        Ecliptic latitude of the detector-plane normal [rad]
        (= θ_s in TianQinOrbit).  Default: −0.08209992 (J0806, TianQin).
    tstart : float
        Start-time offset [s].  Default: 0.
    dt : float
        Time step [s].  Default: ``DAY``.
    pn_order : int
        PN order for LTT (0, 1, 2).  Default: 2.
    Tobs : float or None
        Total duration [s].  If *None*, ``5 * YEAR`` is used.
    """

    def __init__(
        self,
        L=np.sqrt(3.0) * 1e8,
        kappa0=0.0,
        kappa_earth=0.0,
        phi_det=2.102051345707588,       # J0806 ecliptic longitude
        theta_det=-0.0820999173027218,  # ecliptic latitude (J0806, TianQin)
        tstart=0.0,
        dt=DAY,
        pn_order=2,
        Tobs=None,
    ):
        if Tobs is None:
            Tobs = 5 * YEAR
            
        # Earth mass / gravitational parameter
        M_EARTH = 5.972e24
        MU_EARTH = G * M_EARTH  # ≈ 3.986e14 m³/s²

        # ---- step 1: time grid ----
        self.tstart = tstart
        self.dt = dt
        self.N = int(Tobs / dt)
        self.tdata = np.arange(self.N) * dt - self.tstart

        # ---- step 2: Earth heliocentric orbit (circular, ecliptic) ----
        omega_e = TWOPI / YEAR
        alpha_e = omega_e * self.tdata + kappa_earth

        R_e_x = AU * np.cos(alpha_e)
        R_e_y = AU * np.sin(alpha_e)
        R_e_z = np.zeros(self.N)

        V_e_x = -AU * omega_e * np.sin(alpha_e)
        V_e_y = AU * omega_e * np.cos(alpha_e)
        V_e_z = np.zeros(self.N)

        # ---- step 3: SC orbits around Earth (circular, in detector plane) ----
        R_orbit = L / np.sqrt(3.0)
        omega_sc = np.sqrt(MU_EARTH / R_orbit**3)  # 2π·f₀

        cp, sp = np.cos(phi_det), np.sin(phi_det)
        ct, st = np.cos(theta_det), np.sin(theta_det)

        self.rdata = {}
        self.vdata = {}

        for idx, label in enumerate(SC_labels):
            n = idx
            kappa_n = n * TWOPI / 3.0 + kappa0
            alpha = omega_sc * self.tdata + kappa_n

            sa, ca = np.sin(alpha), np.cos(alpha)

            # relative position w.r.t. Earth (detector-plane → SSB frame)
            x_rel = R_orbit * (sp * ca + cp * st * sa)
            y_rel = R_orbit * (-cp * ca + sp * st * sa)
            z_rel = R_orbit * (-ct * sa)

            # relative velocity w.r.t. Earth
            vx_rel = R_orbit * omega_sc * (-sp * sa + cp * st * ca)
            vy_rel = R_orbit * omega_sc * (cp * sa + sp * st * ca)
            vz_rel = R_orbit * omega_sc * (-ct * ca)

            # total position / velocity in SSB frame
            self.rdata[label] = np.column_stack([
                R_e_x + x_rel, R_e_y + y_rel, R_e_z + z_rel,
            ])
            self.vdata[label] = np.column_stack([
                V_e_x + vx_rel, V_e_y + vy_rel, V_e_z + vz_rel,
            ])


        # ---- step 4: PN post-processing (inherited) ----
        self._post_process(pn_order)