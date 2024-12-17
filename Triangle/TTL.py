import numpy as np

from Triangle.Constants import *
from Triangle.Data import *
from Triangle.TDI import *


class TTL:
    """
    model TTL, generate TTL coupling noises and DWS readouts
    Refs:
    [1] PHYSICAL REVIEW D 107, 022005 (2023)
    [2] PHYS. REV. D 106, 042005 (2022)
    """

    S_SC_ETA = 5e-9  # [rad]
    S_SC_PHI = 5e-9  # [rad]
    S_SC_THETA = 5e-9  # [rad]

    S_MOSA_ETA = 1e-9  # [rad]
    S_MOSA_PHI = 2e-9  # [rad]

    S_DWS_ETA = 7e-8 / 335  # [rad]
    S_DWS_PHI = 7e-8 / 335  # [rad]

    C_TTL_TX_PHI = 2.3e-3  # [m/rad]
    C_TTL_TX_ETA = 2.3e-3  # [m/rad]
    C_TTL_RX_PHI = 2.3e-3  # [m/rad]
    C_TTL_RX_ETA = 2.3e-3  # [m/rad]

    def __init__(self, fsample, delays, random_coef=True, scale_factor=1):
        self.fsample = fsample
        self.delays = delays
        self.size = len(delays[MOSA_labels[0]])
        # generate ttl coupling coefficients
        self.coef_TX_phi = {}
        self.coef_TX_eta = {}
        self.coef_RX_phi = {}
        self.coef_RX_eta = {}
        if random_coef:
            for key in MOSA_labels:
                self.coef_TX_phi[key] = np.random.normal() * self.C_TTL_TX_PHI * scale_factor
                self.coef_TX_eta[key] = np.random.normal() * self.C_TTL_TX_ETA * scale_factor
                self.coef_RX_phi[key] = np.random.normal() * self.C_TTL_RX_PHI * scale_factor
                self.coef_RX_eta[key] = np.random.normal() * self.C_TTL_RX_ETA * scale_factor
        else:
            for key in MOSA_labels:
                self.coef_TX_phi[key] = self.C_TTL_TX_PHI
                self.coef_TX_eta[key] = self.C_TTL_TX_ETA
                self.coef_RX_phi[key] = self.C_TTL_RX_PHI
                self.coef_RX_eta[key] = self.C_TTL_RX_ETA
        self.coef_TX_phi = MOSADict(self.coef_TX_phi)
        self.coef_TX_eta = MOSADict(self.coef_TX_eta)
        self.coef_RX_phi = MOSADict(self.coef_RX_phi)
        self.coef_RX_eta = MOSADict(self.coef_RX_eta)

    def PSD_angular_jitters(self, f, asd, unit="angular_velocity"):
        if unit == "rad":
            return asd**2 * (1.0 + (8e-4 / f) ** 4)
        elif unit == "angular_velocity":
            return asd**2 * (1.0 + (8e-4 / f) ** 4) * (2.0 * np.pi * f) ** 2
        else:
            raise NotImplementedError("unit not implemented.")

    def PSD_DWS_readouts(self, f, asd, unit="angular_velocity"):  # the same shape as isi readout noise
        if unit == "rad":
            return asd**2 * (1.0 + (2e-3 / f) ** 4)
        elif unit == "angular_velocity":
            return asd**2 * (1.0 + (2e-3 / f) ** 4) * (2.0 * np.pi * f) ** 2
        else:
            raise NotImplementedError("unit not implemented.")

    def TTL_model(self, phi_jitters, eta_jitters):
        """
        returns TTL optical path noises in each scientific interferometries [s/s]
        """
        TTL_local = self.coef_RX_phi * phi_jitters + self.coef_RX_eta * eta_jitters
        TTL_remote = self.coef_TX_phi * phi_jitters + self.coef_TX_eta * eta_jitters
        return (TTL_local + (TTL_remote.reverse()).timedelay(fsample=self.fsample, delay=self.delays, order=31)) / C

    def SimulateAngularJitters(self):
        """
        simulate the angular jitters of MOSAs w.r.t inertial frames [rad/s]
        """
        logger.info("Simulating angular jitters.")
        SC_phi_jitters = assign_noise_for_SCs(
            lambda f: self.PSD_angular_jitters(f, asd=self.S_SC_PHI, unit="angular_velocity"),
            fsample=self.fsample,
            size=self.size,
        )
        SC_eta_jitters = assign_noise_for_SCs(
            lambda f: self.PSD_angular_jitters(f, asd=self.S_SC_ETA, unit="angular_velocity"),
            fsample=self.fsample,
            size=self.size,
        )
        SC_theta_jitters = assign_noise_for_SCs(
            lambda f: self.PSD_angular_jitters(f, asd=self.S_SC_THETA, unit="angular_velocity"),
            fsample=self.fsample,
            size=self.size,
        )
        MOSA_phi_jitters = assign_noise_for_MOSAs(
            lambda f: self.PSD_angular_jitters(f, asd=self.S_MOSA_PHI, unit="angular_velocity"),
            fsample=self.fsample,
            size=self.size,
        )
        MOSA_eta_jitters = assign_noise_for_MOSAs(
            lambda f: self.PSD_angular_jitters(f, asd=self.S_MOSA_ETA, unit="angular_velocity"),
            fsample=self.fsample,
            size=self.size,
        )

        # combine the jitters of SCs and MOSAs
        self.phi_jitters = SC_phi_jitters.toMOSA() + MOSA_phi_jitters
        self.eta_jitters = dict()
        for key in ["12", "23", "31"]:
            self.eta_jitters[key] = SC_eta_jitters[key[0]] * np.cos(np.pi / 6.0) - SC_theta_jitters[key[0]] * np.sin(np.pi / 6.0) + MOSA_eta_jitters[key]
        for key in ["21", "32", "13"]:
            self.eta_jitters[key] = SC_eta_jitters[key[0]] * np.cos(np.pi / 6.0) + SC_theta_jitters[key[0]] * np.sin(np.pi / 6.0) + MOSA_eta_jitters[key]
        self.eta_jitters = MOSADict(self.eta_jitters)

    def SimulateDWSReadouts(self):
        """
        simulate the readout signals of DWS [rad/s]
        """
        logger.info("Simulating DWS readouts.")
        DWS_phi_noises = assign_noise_for_MOSAs(
            lambda f: self.PSD_DWS_readouts(f, asd=self.S_DWS_PHI, unit="angular_velocity"),
            fsample=self.fsample,
            size=self.size,
        )
        DWS_eta_noises = assign_noise_for_MOSAs(
            lambda f: self.PSD_DWS_readouts(f, asd=self.S_DWS_ETA, unit="angular_velocity"),
            fsample=self.fsample,
            size=self.size,
        )
        self.DWS_phi = DWS_phi_noises + self.phi_jitters
        self.DWS_eta = DWS_eta_noises + self.eta_jitters

    def SimulateTTL(self):
        """
        simulate TTL optical path noises [s/s]
        """
        logger.info("Simulating TTL noises.")
        self.TTL_noises = self.TTL_model(self.phi_jitters, self.eta_jitters)
