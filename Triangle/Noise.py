import numpy as np
from numpy.fft import fft, fftfreq, ifft

from Triangle.Constants import *


class GeneralNoise:
    """
    generate noise array based on the psd function or data array.
    """

    def __init__(self, array_or_psd):
        if callable(array_or_psd):
            self.noise_type = "psd"
            self.noise_psd = array_or_psd
        elif isinstance(array_or_psd, np.ndarray):
            self.noise_type = "array"
            self.noise_data = array_or_psd
        else:
            raise ValueError("array_or_psd should be a filename or a callable psd function.")

    def _TimeDomainNoiseGeneration(self, fsample, size):
        sigma0 = np.sqrt(fsample / 2.0)
        n0 = np.random.normal(0, sigma0, size)

        n0f = fft(n0)
        f_arr = fftfreq(size, d=1.0 / fsample)
        asd_arr = np.sqrt(self.noise_psd(np.abs(f_arr[1:])))
        asd_arr = np.insert(asd_arr, 0, 0)
        n1f = n0f * asd_arr
        n1 = np.real(ifft(n1f))
        return n1

    def _PadNoiseData(self, size):
        N = len(self.noise_data)
        if N >= size:
            return self.noise_data[:size]
        else:
            return np.pad(self.noise_data, (0, size - N), "constant")

    def FrequencyDomainNoiseGeneration(self, freqs):
        """
        noise_psd must be a psd function: freq array -> psd array
        """
        df = freqs[1] - freqs[0]
        size = len(freqs)
        psd = self.noise_psd(freqs)
        noise_Re = np.random.randn(size) * np.sqrt(psd / 4.0 / df)
        noise_Im = np.random.randn(size) * np.sqrt(psd / 4.0 / df)
        noise_total = noise_Re + 1.0j * noise_Im
        return noise_total

    def __call__(self, fsample, size):
        if self.noise_type == "psd":
            return self._TimeDomainNoiseGeneration(fsample, size)
        else:
            return self._PadNoiseData(size)


class InstrumentalPSDs:
    """
    the default unit is ffd (fractional frequency difference)
    ffd is also the default unit used by the inteferometer class and the TDI noise class
    f can be either a float number or a numpy array
    """

    def __init__(
        self,
        L=L_nominal,
        unit="ffd",
    ):
        self.L = L
        self.unit = unit

    def PSD_ACC(self, f, sacc=SACC_nominal):
        if self.unit == "ffd":
            u = TWOPI * f * self.L / C
            return (sacc * self.L / u / C**2) ** 2 * (1.0 + (0.4e-3 / f) ** 2) * (1.0 + (f / 8e-3) ** 4)
        elif self.unit == "frequency":
            u = TWOPI * f * self.L / C
            return (sacc * self.L / u / C**2) ** 2 * (1.0 + (0.4e-3 / f) ** 2) * (1.0 + (f / 8e-3) ** 4) * F_LASER**2

    def PSD_RO(self, f, sro=SOMS_nominal):
        if self.unit == "ffd":
            u = TWOPI * f * self.L / C
            return (u * sro / self.L) ** 2 * (1.0 + (2e-3 / f) ** 4)
        elif self.unit == "frequency":
            u = TWOPI * f * self.L / C
            return (u * sro / self.L) ** 2 * (1.0 + (2e-3 / f) ** 4) * F_LASER**2

    def PSD_LASER(self, f, slaser=SLASER_nominal):  # white in frequency
        if self.unit == "ffd":
            if isinstance(f, np.ndarray):
                return np.ones_like(f) * (slaser / F_LASER) ** 2
            else:
                return (slaser / F_LASER) ** 2
        elif self.unit == "frequency":
            if isinstance(f, np.ndarray):
                return np.ones_like(f) * (slaser) ** 2
            else:
                return (slaser) ** 2

    def PSD_OB(self, f, sob=SOB_nominal):  # white in phase
        if self.unit == "ffd":
            return (sob * f / F_LASER) ** 2
        elif self.unit == "frequency":
            return (sob * f) ** 2

    def PSD_CLOCK(self, f, sclock=SCLOCK_nominal):
        if self.unit == "ffd":
            result = sclock**2 / f
        elif self.unit == "frequency":
            result = sclock**2 / f * F_LASER**2
        return result

    def PSD_BL(self, f, sbl=SBL_nominal):
        if self.unit == "ffd":
            return (TWOPI * f * sbl / C) ** 2 * (1.0 + (2e-3 / f) ** 4)
        elif self.unit == "frequency":
            return (TWOPI * f * sbl / C) ** 2 * (1.0 + (2e-3 / f) ** 4) * F_LASER**2

    def PSD_R(self, f, sr=SR_nominal):
        if self.unit == "ffd":
            return (TWOPI * f * sr / C) ** 2
        elif self.unit == "frequency":
            return (TWOPI * f * sr / C) ** 2 * F_LASER**2

    def PSD_OP(self, f, sop=SOP_OTHER_nominal):
        if self.unit == "ffd":
            return (TWOPI * f * sop / C) ** 2
        elif self.unit == "frequency":
            return (TWOPI * f * sop / C) ** 2 * F_LASER**2

    def PSD_M(self, f, sm=SM_RIGHT_nominal):
        if self.unit == "ffd":
            return sm**2 * f ** (2.0 / 3.0)
        elif self.unit == "frequency":
            return sm**2 * f ** (2.0 / 3.0) * F_LASER**2


class TDIPSDs:
    """
    TDI PSDs in [fractional frequency difference], considering only the optical metrology system noise and test-mass acceleration noise
    """

    def __init__(self, sacc=SACC_nominal, sro=SOMS_nominal, L=L_nominal):  # default Taiji [sacc] = acceleration, [sro] = distance
        self.sa = sacc
        self.so = sro
        self.L = L
        self.BasicNoisePSDs = InstrumentalPSDs(L=L, unit="ffd")

    def PSD_Sa(self, f):
        return self.BasicNoisePSDs.PSD_ACC(f=f, sacc=self.sa)

    def PSD_So(self, f):
        return self.BasicNoisePSDs.PSD_RO(f=f, sro=self.so)

    def PSD_X(self, f):  # self.sa, self.so are the asd of acc and opt noise, WTN and wang and Vallisneri
        u = TWOPI * f * self.L / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        return Sa * (8.0 * (np.sin(2.0 * u)) ** 2 + 32.0 * (np.sin(u)) ** 2) + 16.0 * So * (np.sin(u)) ** 2

    def PSD_X2(self, f):
        u = TWOPI * f * self.L / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        return 64.0 * (np.sin(2.0 * u)) ** 2 * (np.sin(u)) ** 2 * (So + (3.0 + np.cos(2.0 * u)) * Sa)

    def PSD_X2_unequal(self, f, arms):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f)
        Sa = self.PSD_Sa(f)
        u121 = TWOPI * f * (arms["12"] + arms["21"])
        u131 = TWOPI * f * (arms["13"] + arms["31"])
        u12131 = u121 + u131
        s121 = np.sin(u121 / 2.0) ** 2
        s131 = np.sin(u131 / 2.0) ** 2
        c121 = np.cos(u121 / 2.0) ** 2
        c131 = np.cos(u131 / 2.0) ** 2
        s12131 = np.sin(u12131 / 2.0) ** 2
        OMS_term = 32.0 * So * s12131 * (s121 + s131)
        ACC_term = 64.0 * Sa * s12131 * (s121 * (c131 + 1) + s131 * (c121 + 1))
        return OMS_term + ACC_term

    def PSD_Y2_unequal(self, f, arms):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f)
        Sa = self.PSD_Sa(f)
        u232 = TWOPI * f * (arms["23"] + arms["32"])
        u212 = TWOPI * f * (arms["21"] + arms["12"])
        u23212 = u232 + u212
        s232 = np.sin(u232 / 2.0) ** 2
        s212 = np.sin(u212 / 2.0) ** 2
        c232 = np.cos(u232 / 2.0) ** 2
        c212 = np.cos(u212 / 2.0) ** 2
        s23212 = np.sin(u23212 / 2.0) ** 2
        OMS_term = 32.0 * So * s23212 * (s232 + s212)
        ACC_term = 64.0 * Sa * s23212 * (s232 * (c212 + 1) + s212 * (c232 + 1))
        return OMS_term + ACC_term

    def PSD_Z2_unequal(self, f, arms):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f)
        Sa = self.PSD_Sa(f)
        u313 = TWOPI * f * (arms["31"] + arms["13"])
        u323 = TWOPI * f * (arms["32"] + arms["23"])
        u31323 = u313 + u323
        s313 = np.sin(u313 / 2.0) ** 2
        s323 = np.sin(u323 / 2.0) ** 2
        c313 = np.cos(u313 / 2.0) ** 2
        c323 = np.cos(u323 / 2.0) ** 2
        s31323 = np.sin(u31323 / 2.0) ** 2
        OMS_term = 32.0 * So * s31323 * (s313 + s323)
        ACC_term = 64.0 * Sa * s31323 * (s313 * (c323 + 1) + s323 * (c313 + 1))
        return OMS_term + ACC_term

    def PSD_X2Y2star_unequal(self, f, arms):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f)
        Sa = self.PSD_Sa(f)
        u121 = TWOPI * f * (arms["12"] + arms["21"])
        u131 = TWOPI * f * (arms["13"] + arms["31"])
        u12131 = u121 + u131
        u232 = TWOPI * f * (arms["23"] + arms["32"])
        u212 = TWOPI * f * (arms["21"] + arms["12"])
        u23212 = u232 + u212
        D21 = np.exp(-1.0j * TWOPI * f * arms["21"])
        D12 = np.exp(-1.0j * TWOPI * f * arms["12"])
        D131 = np.exp(-1.0j * u131)
        D12131 = np.exp(-1.0j * u12131)
        D232 = np.exp(-1.0j * u232)
        D23212 = np.exp(-1.0j * u23212)
        delay_factor = (1.0 - D131) * (1.0 - D12131) * (1.0 - D232.conjugate()) * (1.0 - D23212.conjugate()) * (D21.conjugate() + D12)
        OMS_term = -So
        ACC_term = -4.0 * Sa
        return (OMS_term + ACC_term) * delay_factor

    def PSD_Y2Z2star_unequal(self, f, arms):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f)
        Sa = self.PSD_Sa(f)
        u232 = TWOPI * f * (arms["23"] + arms["32"])
        u212 = TWOPI * f * (arms["21"] + arms["12"])
        u23212 = u232 + u212
        u313 = TWOPI * f * (arms["31"] + arms["13"])
        u323 = TWOPI * f * (arms["32"] + arms["23"])
        u31323 = u313 + u323
        D32 = np.exp(-1.0j * TWOPI * f * arms["32"])
        D23 = np.exp(-1.0j * TWOPI * f * arms["23"])
        D212 = np.exp(-1.0j * u212)
        D23212 = np.exp(-1.0j * u23212)
        D313 = np.exp(-1.0j * u313)
        D31323 = np.exp(-1.0j * u31323)
        delay_factor = (1.0 - D212) * (1.0 - D23212) * (1.0 - D313.conjugate()) * (1.0 - D31323.conjugate()) * (D32.conjugate() + D23)
        OMS_term = -So
        ACC_term = -4.0 * Sa
        return (OMS_term + ACC_term) * delay_factor

    def PSD_Z2X2star_unequal(self, f, arms):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f)
        Sa = self.PSD_Sa(f)
        u313 = TWOPI * f * (arms["31"] + arms["13"])
        u323 = TWOPI * f * (arms["32"] + arms["23"])
        u31323 = u313 + u323
        u121 = TWOPI * f * (arms["12"] + arms["21"])
        u131 = TWOPI * f * (arms["13"] + arms["31"])
        u12131 = u121 + u131
        D13 = np.exp(-1.0j * TWOPI * f * arms["13"])
        D31 = np.exp(-1.0j * TWOPI * f * arms["31"])
        D323 = np.exp(-1.0j * u323)
        D31323 = np.exp(-1.0j * u31323)
        D121 = np.exp(-1.0j * u121)
        D12131 = np.exp(-1.0j * u12131)
        delay_factor = (1.0 - D323) * (1.0 - D31323) * (1.0 - D121.conjugate()) * (1.0 - D12131.conjugate()) * (D13.conjugate() + D31)
        OMS_term = -So
        ACC_term = -4.0 * Sa
        return (OMS_term + ACC_term) * delay_factor

    def PSD_A(self, f):  # WTN
        u = TWOPI * f * self.L / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        return 8.0 * So * (2.0 + np.cos(u)) * (np.sin(u)) ** 2 + 16.0 * Sa * (3.0 + 2.0 * np.cos(u) + np.cos(2.0 * u)) * (np.sin(u)) ** 2

    def PSD_A2(self, f):
        u = TWOPI * f * self.L / C
        return np.maximum(4.0 * (np.sin(2.0 * u)) ** 2 * self.PSD_A(f), 1e-50)

    def PSD_A2_unequal(self, f, arms):
        return 0.5 * self.PSD_X2_unequal(f, arms) + 0.5 * self.PSD_Z2_unequal(f, arms) - np.real(self.PSD_Z2X2star_unequal(f, arms))

    def PSD_E2_unequal(self, f, arms):
        return (
            self.PSD_X2_unequal(f, arms)
            + 4.0 * self.PSD_Y2_unequal(f, arms)
            + self.PSD_Z2_unequal(f, arms)
            - 4.0 * np.real(self.PSD_X2Y2star_unequal(f, arms))
            + 2.0 * np.real(self.PSD_Z2X2star_unequal(f, arms))
            - 4.0 * np.real(self.PSD_Y2Z2star_unequal(f, arms))
        ) / 6.0

    def PSD_T2_unequal(self, f, arms):
        return (
            self.PSD_X2_unequal(f, arms)
            + self.PSD_Y2_unequal(f, arms)
            + self.PSD_Z2_unequal(f, arms)
            + 2.0 * np.real(self.PSD_X2Y2star_unequal(f, arms))
            + 2.0 * np.real(self.PSD_Z2X2star_unequal(f, arms))
            + 2.0 * np.real(self.PSD_Y2Z2star_unequal(f, arms))
        ) / 3.0

    def PSD_X2_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        """
        arms, amplitudes should be given by MOSADict objects
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f) / self.so**2  # oms noise with unit amplitude
        Sa = self.PSD_Sa(f) / self.sa**2  # acc noise with unit amplitude
        u121 = TWOPI * f * (arms["12"] + arms["21"])
        u131 = TWOPI * f * (arms["13"] + arms["31"])
        u12131 = u121 + u131
        s121 = np.sin(u121 / 2.0) ** 2
        s131 = np.sin(u131 / 2.0) ** 2
        c121 = np.cos(u121 / 2.0) ** 2
        c131 = np.cos(u131 / 2.0) ** 2
        s12131 = np.sin(u12131 / 2.0) ** 2
        OMS_term = 16.0 * s12131 * (s131 * (A_OMS["12"] ** 2 + A_OMS["21"] ** 2) + s121 * (A_OMS["13"] ** 2 + A_OMS["31"] ** 2))
        ACC_term = 64.0 * s12131 * (s131 * (c121 * A_ACC["12"] ** 2 + A_ACC["21"] ** 2) + s121 * (c131 * A_ACC["13"] ** 2 + A_ACC["31"] ** 2))
        return OMS_term * So + ACC_term * Sa

    def PSD_Y2_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f) / self.so**2  # oms noise with unit amplitude
        Sa = self.PSD_Sa(f) / self.sa**2  # acc noise with unit amplitude
        u232 = TWOPI * f * (arms["23"] + arms["32"])
        u212 = TWOPI * f * (arms["21"] + arms["12"])
        u23212 = u232 + u212
        s232 = np.sin(u232 / 2.0) ** 2
        s212 = np.sin(u212 / 2.0) ** 2
        c232 = np.cos(u232 / 2.0) ** 2
        c212 = np.cos(u212 / 2.0) ** 2
        s23212 = np.sin(u23212 / 2.0) ** 2
        OMS_term = 16.0 * s23212 * (s212 * (A_OMS["23"] ** 2 + A_OMS["32"] ** 2) + s232 * (A_OMS["21"] ** 2 + A_OMS["12"] ** 2))
        ACC_term = 64.0 * s23212 * (s212 * (c232 * A_ACC["23"] ** 2 + A_ACC["32"] ** 2) + s232 * (c212 * A_ACC["21"] ** 2 + A_ACC["12"] ** 2))
        return OMS_term * So + ACC_term * Sa

    def PSD_Z2_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f) / self.so**2  # oms noise with unit amplitude
        Sa = self.PSD_Sa(f) / self.sa**2  # acc noise with unit amplitude
        u313 = TWOPI * f * (arms["31"] + arms["13"])
        u323 = TWOPI * f * (arms["32"] + arms["23"])
        u31323 = u313 + u323
        s313 = np.sin(u313 / 2.0) ** 2
        s323 = np.sin(u323 / 2.0) ** 2
        c313 = np.cos(u313 / 2.0) ** 2
        c323 = np.cos(u323 / 2.0) ** 2
        s31323 = np.sin(u31323 / 2.0) ** 2
        OMS_term = 16.0 * s31323 * (s323 * (A_OMS["31"] ** 2 + A_OMS["13"] ** 2) + s313 * (A_OMS["32"] ** 2 + A_OMS["23"] ** 2))
        ACC_term = 64.0 * s31323 * (s323 * (c313 * A_ACC["31"] ** 2 + A_ACC["13"] ** 2) + s313 * (c323 * A_ACC["32"] ** 2 + A_ACC["23"] ** 2))
        return OMS_term * So + ACC_term * Sa

    def PSD_X2Y2star_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f) / self.so**2  # oms noise with unit amplitude
        Sa = self.PSD_Sa(f) / self.sa**2  # acc noise with unit amplitude
        u121 = TWOPI * f * (arms["12"] + arms["21"])
        u131 = TWOPI * f * (arms["13"] + arms["31"])
        u12131 = u121 + u131
        u232 = TWOPI * f * (arms["23"] + arms["32"])
        u212 = TWOPI * f * (arms["21"] + arms["12"])
        u23212 = u232 + u212
        D21 = np.exp(-1.0j * TWOPI * f * arms["21"])
        D12 = np.exp(-1.0j * TWOPI * f * arms["12"])
        D131 = np.exp(-1.0j * u131)
        D12131 = np.exp(-1.0j * u12131)
        D232 = np.exp(-1.0j * u232)
        D23212 = np.exp(-1.0j * u23212)
        OMS_term = -1.0 * (1.0 - D131) * (1.0 - D12131) * np.conjugate((1.0 - D232) * (1.0 - D23212)) * (np.conjugate(D21) * A_OMS["12"] ** 2 + D12 * A_OMS["21"] ** 2)
        ACC_term = -2.0 * (1.0 - D131) * (1.0 - D12131) * np.conjugate((1.0 - D232) * (1.0 - D23212)) * ((np.conjugate(D21) + D12) * (A_ACC["12"] ** 2 + A_ACC["21"] ** 2))
        return OMS_term * So + ACC_term * Sa

    def PSD_Y2Z2star_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f) / self.so**2  # oms noise with unit amplitude
        Sa = self.PSD_Sa(f) / self.sa**2  # acc noise with unit amplitude
        u232 = TWOPI * f * (arms["23"] + arms["32"])
        u212 = TWOPI * f * (arms["21"] + arms["12"])
        u23212 = u232 + u212
        u313 = TWOPI * f * (arms["31"] + arms["13"])
        u323 = TWOPI * f * (arms["32"] + arms["23"])
        u31323 = u313 + u323
        D32 = np.exp(-1.0j * TWOPI * f * arms["32"])
        D23 = np.exp(-1.0j * TWOPI * f * arms["23"])
        D212 = np.exp(-1.0j * u212)
        D23212 = np.exp(-1.0j * u23212)
        D313 = np.exp(-1.0j * u313)
        D31323 = np.exp(-1.0j * u31323)
        OMS_term = -1.0 * (1.0 - D212) * (1.0 - D23212) * np.conjugate((1.0 - D313) * (1.0 - D31323)) * (np.conjugate(D32) * A_OMS["23"] ** 2 + D23 * A_OMS["32"] ** 2)
        ACC_term = -2.0 * (1.0 - D212) * (1.0 - D23212) * np.conjugate((1.0 - D313) * (1.0 - D31323)) * ((np.conjugate(D32) + D23) * (A_ACC["23"] ** 2 + A_ACC["32"] ** 2))
        return OMS_term * So + ACC_term * Sa

    def PSD_Z2X2star_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f) / self.so**2  # oms noise with unit amplitude
        Sa = self.PSD_Sa(f) / self.sa**2  # acc noise with unit amplitude
        u313 = TWOPI * f * (arms["31"] + arms["13"])
        u323 = TWOPI * f * (arms["32"] + arms["23"])
        u31323 = u313 + u323
        u121 = TWOPI * f * (arms["12"] + arms["21"])
        u131 = TWOPI * f * (arms["13"] + arms["31"])
        u12131 = u121 + u131
        D13 = np.exp(-1.0j * TWOPI * f * arms["13"])
        D31 = np.exp(-1.0j * TWOPI * f * arms["31"])
        D323 = np.exp(-1.0j * u323)
        D31323 = np.exp(-1.0j * u31323)
        D121 = np.exp(-1.0j * u121)
        D12131 = np.exp(-1.0j * u12131)
        OMS_term = -1.0 * (1.0 - D323) * (1.0 - D31323) * np.conjugate((1.0 - D121) * (1.0 - D12131)) * (np.conjugate(D13) * A_OMS["31"] ** 2 + D31 * A_OMS["13"] ** 2)
        ACC_term = -2.0 * (1.0 - D323) * (1.0 - D31323) * np.conjugate((1.0 - D121) * (1.0 - D12131)) * ((np.conjugate(D13) + D31) * (A_ACC["31"] ** 2 + A_ACC["13"] ** 2))
        return OMS_term * So + ACC_term * Sa

    def PSD_A2_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        return 0.5 * self.PSD_X2_unequal_different_amps(f, arms, A_OMS, A_ACC) + 0.5 * self.PSD_Z2_unequal_different_amps(f, arms, A_OMS, A_ACC) - np.real(self.PSD_Z2X2star_unequal_different_amps(f, arms, A_OMS, A_ACC))

    def PSD_E2_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        return (
            self.PSD_X2_unequal_different_amps(f, arms, A_OMS, A_ACC)
            + 4.0 * self.PSD_Y2_unequal_different_amps(f, arms, A_OMS, A_ACC)
            + self.PSD_Z2_unequal_different_amps(f, arms, A_OMS, A_ACC)
            - 4.0 * np.real(self.PSD_X2Y2star_unequal_different_amps(f, arms, A_OMS, A_ACC))
            + 2.0 * np.real(self.PSD_Z2X2star_unequal_different_amps(f, arms, A_OMS, A_ACC))
            - 4.0 * np.real(self.PSD_Y2Z2star_unequal_different_amps(f, arms, A_OMS, A_ACC))
        ) / 6.0

    def PSD_T2_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        return (
            self.PSD_X2_unequal_different_amps(f, arms, A_OMS, A_ACC)
            + self.PSD_Y2_unequal_different_amps(f, arms, A_OMS, A_ACC)
            + self.PSD_Z2_unequal_different_amps(f, arms, A_OMS, A_ACC)
            + 2.0 * np.real(self.PSD_X2Y2star_unequal_different_amps(f, arms, A_OMS, A_ACC))
            + 2.0 * np.real(self.PSD_Z2X2star_unequal_different_amps(f, arms, A_OMS, A_ACC))
            + 2.0 * np.real(self.PSD_Y2Z2star_unequal_different_amps(f, arms, A_OMS, A_ACC))
        ) / 3.0

    def PSD_T(self, f):  # WTN
        u = TWOPI * f * self.L / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        return 16.0 * So * (1.0 - np.cos(u)) * (np.sin(u)) ** 2 + 128.0 * Sa * (np.sin(u)) ** 2 * (np.sin(u / 2.0)) ** 4

    def PSD_T2(self, f):
        u = TWOPI * f * self.L / C
        return 4.0 * (np.sin(2.0 * u)) ** 2 * self.PSD_T(f)

    def PSD_alpha(self, f):  # wang, Vallis
        u = TWOPI * f * self.L / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        return 8.0 * Sa * ((np.sin(1.5 * u)) ** 2 + 2.0 * (np.sin(0.5 * u)) ** 2) + 6.0 * So

    def PSD_alpha2(self, f):
        u = TWOPI * f * self.L / C
        return self.PSD_alpha(f) * 4.0 * np.sin(1.5 * u) ** 2

    def PSD_alpha2_unequal(self, f, arms):
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        u12 = TWOPI * f * arms["12"]
        u13 = TWOPI * f * arms["13"]
        u23 = TWOPI * f * arms["23"]
        u123 = u12 + u23
        u132 = u13 + u23
        u1231 = u123 + u13
        OMS_term = 24.0 * So * np.sin(u1231 / 2.0) ** 2
        ACC_term = 32.0 * Sa * np.sin(u1231 / 2.0) ** 2 * (np.sin(u1231 / 2.0) ** 2 + np.sin((u132 - u12) / 2.0) ** 2 + np.sin((u123 - u13) / 2.0) ** 2)
        return OMS_term + ACC_term

    def PSD_beta2_unequal(self, f, arms):
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        u23 = TWOPI * f * arms["23"]
        u21 = TWOPI * f * arms["21"]
        u31 = TWOPI * f * arms["31"]
        u231 = u23 + u31
        u213 = u21 + u31
        u2312 = u231 + u21
        OMS_term = 24.0 * So * np.sin(u2312 / 2.0) ** 2
        ACC_term = 32.0 * Sa * np.sin(u2312 / 2.0) ** 2 * (np.sin(u2312 / 2.0) ** 2 + np.sin((u213 - u23) / 2.0) ** 2 + np.sin((u231 - u21) / 2.0) ** 2)
        return OMS_term + ACC_term

    def PSD_gamma2_unequal(self, f, arms):
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        u31 = TWOPI * f * arms["31"]
        u32 = TWOPI * f * arms["32"]
        u12 = TWOPI * f * arms["12"]
        u312 = u31 + u12
        u321 = u32 + u12
        u3123 = u312 + u32
        OMS_term = 24.0 * So * np.sin(u3123 / 2.0) ** 2
        ACC_term = 32.0 * Sa * np.sin(u3123 / 2.0) ** 2 * (np.sin(u3123 / 2.0) ** 2 + np.sin((u321 - u31) / 2.0) ** 2 + np.sin((u312 - u32) / 2.0) ** 2)
        return OMS_term + ACC_term

    def PSD_alpha2gamma2star_unequal(self, f, arms):
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        u12 = TWOPI * f * arms["12"]
        u13 = TWOPI * f * arms["13"]
        u23 = TWOPI * f * arms["23"]
        u123 = u12 + u23
        u132 = u13 + u23
        u1231 = u123 + u13
        prefactor = 4.0 * np.sin(u1231 / 2.0) ** 2
        OMS_term = 2.0 * So * (2.0 * np.cos(u13) + np.cos(u123))
        ACC_term = 4.0 * Sa * (np.cos(u13) - np.cos(u23 - u12))
        return prefactor * (OMS_term + ACC_term)

    def PSD_P22_unequal(self, f, arms):
        return 0.5 * (self.PSD_alpha2_unequal(f, arms) + self.PSD_gamma2_unequal(f, arms) + 2.0 * self.PSD_alpha2gamma2star_unequal(f, arms))


def ResidualClockNoise(freqs, a, b, idx=0, channel="X2"):
    u = TWOPI * freqs * L_nominal / C
    psd = InstrumentalPSDs()
    psd_q = psd.PSD_CLOCK
    S_q = psd_q(freqs)

    if channel == "X2":
        term1 = (a["12"][idx] - a["13"][idx]) ** 2
        term2 = (a["21"][idx]) ** 2
        term3 = (a["31"][idx]) ** 2
        term4 = -4.0 * b["12"][idx] * (a["12"][idx] - a["13"][idx] - b["12"][idx]) * (np.sin(u)) ** 2
    if channel == "Y2":
        term1 = (a["23"][idx] - a["21"][idx]) ** 2
        term2 = (a["32"][idx]) ** 2
        term3 = (a["12"][idx]) ** 2
        term4 = -4.0 * b["23"][idx] * (a["23"][idx] - a["21"][idx] - b["23"][idx]) * (np.sin(u)) ** 2
    if channel == "Z2":
        term1 = (a["31"][idx] - a["32"][idx]) ** 2
        term2 = (a["13"][idx]) ** 2
        term3 = (a["23"][idx]) ** 2
        term4 = -4.0 * b["31"][idx] * (a["31"][idx] - a["32"][idx] - b["31"][idx]) * (np.sin(u)) ** 2
    return 16.0 * (np.sin(2.0 * u)) ** 2 * (np.sin(u)) ** 2 * (term1 + term2 + term3 + term4) * S_q


def PSD_laser_residual_from_bias(f, delay_bias, channel="X2"):
    PSD = InstrumentalPSDs(unit="frequency")
    psd = PSD.PSD_LASER(f)
    omega = TWOPI * f
    u = omega * L_nominal / C
    if channel == "X2":
        B12 = delay_bias["12"]
        B21 = delay_bias["21"]
        B13 = delay_bias["13"]
        B31 = delay_bias["31"]
        return 16.0 * np.sin(u) ** 2 * np.sin(2.0 * u) ** 2 * omega**2 * (B12**2 + B21**2 + B13**2 + B31**2) * psd
    elif channel == "Y2":
        B23 = delay_bias["23"]
        B32 = delay_bias["32"]
        B21 = delay_bias["21"]
        B12 = delay_bias["12"]
        return 16.0 * np.sin(u) ** 2 * np.sin(2.0 * u) ** 2 * omega**2 * (B23**2 + B32**2 + B21**2 + B12**2) * psd
    elif channel == "Z2":
        B31 = delay_bias["31"]
        B13 = delay_bias["13"]
        B32 = delay_bias["32"]
        B23 = delay_bias["23"]
        return 16.0 * np.sin(u) ** 2 * np.sin(2.0 * u) ** 2 * omega**2 * (B31**2 + B13**2 + B32**2 + B23**2) * psd
    elif channel == "A2":
        B12 = delay_bias["12"]
        B21 = delay_bias["21"]
        B32 = delay_bias["32"]
        B23 = delay_bias["23"]
        B13 = delay_bias["13"]
        B31 = delay_bias["31"]
        return 8.0 * np.sin(u) ** 2 * np.sin(2.0 * u) ** 2 * omega**2 * (B12**2 + B21**2 + B32**2 + B23**2 + 4.0 * np.cos(u / 2.0) ** 2 * (B31**2 + B13**2)) * psd
    elif channel == "E2":
        B12 = delay_bias["12"]
        B21 = delay_bias["21"]
        B32 = delay_bias["32"]
        B23 = delay_bias["23"]
        B13 = delay_bias["13"]
        B31 = delay_bias["31"]
        return 8.0 / 3.0 * np.sin(u) ** 2 * np.sin(2.0 * u) ** 2 * omega**2 * ((5.0 + 4.0 * np.cos(u)) * (B12**2 + B21**2 + B32**2 + B23**2) + 4.0 * np.sin(u / 2.0) ** 2 * (B31**2 + B13**2)) * psd
    elif channel == "T2":
        B12 = delay_bias["12"]
        B21 = delay_bias["21"]
        B32 = delay_bias["32"]
        B23 = delay_bias["23"]
        B13 = delay_bias["13"]
        B31 = delay_bias["31"]
        return 64.0 / 3.0 * np.sin(u) ** 2 * np.sin(2.0 * u) ** 2 * omega**2 * np.sin(u / 2.0) ** 2 * (B12**2 + B21**2 + B32**2 + B23**2 + B31**2 + B13**2) * psd
    else:
        raise NotImplementedError("channel not implemented.")


class TDIPSDFromPString:
    """
    PSD of arbitraty TDI channel, considering only OMS and ACC noises.
    """

    # examples for the P strings (NOTE that the sign of XYZ should be inverted to be consistent with the path strings and fast michelson calculations)
    X2_strings = {
        "12": [
            (-1.0, []),
            (1.0, ["13", "31"]),
            (1.0, ["13", "31", "12", "21"]),
            (-1.0, ["12", "21", "13", "31", "13", "31"]),
        ],
        "23": [],
        "31": [
            (1.0, ["13"]),
            (-1.0, ["12", "21", "13"]),
            (-1.0, ["12", "21", "13", "31", "13"]),
            (1.0, ["13", "31", "12", "21", "12", "21", "13"]),
        ],
        "21": [
            (-1.0, ["12"]),
            (1.0, ["13", "31", "12"]),
            (1.0, ["13", "31", "12", "21", "12"]),
            (-1.0, ["12", "21", "13", "31", "13", "31", "12"]),
        ],
        "32": [],
        "13": [
            (1.0, []),
            (-1.0, ["12", "21"]),
            (-1.0, ["12", "21", "13", "31"]),
            (1.0, ["13", "31", "12", "21", "12", "21"]),
        ],
    }
    Y2_strings = {
        "23": [
            (-1.0, []),
            (1.0, ["21", "12"]),
            (1.0, ["21", "12", "23", "32"]),
            (-1.0, ["23", "32", "21", "12", "21", "12"]),
        ],
        "31": [],
        "12": [
            (1.0, ["21"]),
            (-1.0, ["23", "32", "21"]),
            (-1.0, ["23", "32", "21", "12", "21"]),
            (1.0, ["21", "12", "23", "32", "23", "32", "21"]),
        ],
        "32": [
            (-1.0, ["23"]),
            (1.0, ["21", "12", "23"]),
            (1.0, ["21", "12", "23", "32", "23"]),
            (-1.0, ["23", "32", "21", "12", "21", "12", "23"]),
        ],
        "13": [],
        "21": [
            (1.0, []),
            (-1.0, ["23", "32"]),
            (-1.0, ["23", "32", "21", "12"]),
            (1.0, ["21", "12", "23", "32", "23", "32"]),
        ],
    }
    Z2_strings = {
        "31": [
            (-1.0, []),
            (1.0, ["32", "23"]),
            (1.0, ["32", "23", "31", "13"]),
            (-1.0, ["31", "13", "32", "23", "32", "23"]),
        ],
        "12": [],
        "23": [
            (1.0, ["32"]),
            (-1.0, ["31", "13", "32"]),
            (-1.0, ["31", "13", "32", "23", "32"]),
            (1.0, ["32", "23", "31", "13", "31", "13", "32"]),
        ],
        "13": [
            (-1.0, ["31"]),
            (1.0, ["32", "23", "31"]),
            (1.0, ["32", "23", "31", "13", "31"]),
            (-1.0, ["31", "13", "32", "23", "32", "23", "31"]),
        ],
        "21": [],
        "32": [
            (1.0, []),
            (-1.0, ["31", "13"]),
            (-1.0, ["31", "13", "32", "23"]),
            (1.0, ["32", "23", "31", "13", "31", "13"]),
        ],
    }
    C_12_3_strings = {
        "13": [(1.0, ["32", "23", "-31"]), (-1.0, ["31", "-12", "23", "-31"])],
        "21": [(-1.0, ["31", "-12"]), (1.0, ["31", "-12", "23", "-31", "12"])],
        "12": [(1.0, ["31", "-12", "23", "-31"]), (-1.0, ["32", "23", "-31"])],
        "23": [(-1.0, ["32"]), (1.0, ["31", "-12"])],
        "31": [(1.0, []), (-1.0, ["32", "23", "-31", "12", "-23"])],
        "32": [(-1.0, []), (1.0, ["32", "23", "-31", "12", "-23"])],
    }
    C_14_3_strings = {
        "12": [
            (1.0, ["23", "-31"]),
            (-1.0, ["21", "12", "23", "-31"]),
            (1.0, ["23", "-31", "12", "-23", "31"]),
            (-1.0, ["21"]),
        ],
        "13": [(-1.0, ["23", "-31"]), (1.0, ["21", "12", "23", "-31"])],
        "32": [
            (-1.0, ["23", "-31", "12", "-23"]),
            (1.0, ["21", "12", "23", "-31", "12", "-23"]),
        ],
        "21": [(-1.0, []), (1.0, ["23", "-31", "12", "-23", "31", "12"])],
        "31": [
            (1.0, ["23", "-31", "12", "-23"]),
            (-1.0, ["21", "12", "23", "-31", "12", "-23"]),
        ],
        "23": [(1.0, []), (-1.0, ["21", "12"])],
    }
    C_16_28_strings = {
        "12": [
            (1.0, ["-12", "23", "-31"]),
            (-1.0, ["-13", "-31", "12", "23", "-31"]),
            (1.0, ["-12", "23", "-31", "12", "-23", "-31"]),
            (-1.0, ["-13", "-31"]),
        ],
        "32": [
            (-1.0, ["-12", "23", "-31", "12", "-23"]),
            (1.0, ["-13", "-31", "12", "23", "-31", "12", "-23"]),
        ],
        "23": [(1.0, ["-12"]), (-1.0, ["-13", "-31", "12"])],
        "13": [
            (1.0, ["-13", "-31", "12", "23", "-31"]),
            (-1.0, ["-12", "23", "-31"]),
            (-1.0, ["-12", "23", "-31", "12", "-23", "-31"]),
            (1.0, ["-13", "-31"]),
        ],
        "31": [(1.0, ["-13"]), (-1.0, ["-13", "-31", "12", "23", "-31", "12", "-23"])],
        "21": [(-1.0, ["-12"]), (1.0, ["-12", "23", "-31", "12", "-23", "-31", "12"])],
    }

    def __init__(self, dij, L=L_nominal):
        """
        args:
            dij: MOSADict, each item being a scalar
        """
        self.dij = dij
        self.inst_noise_class = InstrumentalPSDs(L=L, unit="ffd")

    def PSD_OMS(self, f, AOMS=SOMS_nominal):
        return self.inst_noise_class.PSD_RO(f=f, sro=AOMS)  # (Nf,)

    def PSD_ACC(self, f, AACC=SACC_nominal):
        return self.inst_noise_class.PSD_ACC(f=f, sacc=AACC)  # (Nf,)

    def TDI_P_ij(self, P_ij_strings, f):
        """
        Args:
            P_ij_strings is a MosaDict, each item like: [(1., ["12", "21"]), (-1., ["-13"]), ...], repersenting [D_12 D_21 - A_13 ...]
        Returns:
            P_ij: MosaDict, each item (Nf)
        """
        # calculate single delay operators
        tmp = -1.0j * TWOPI * f  # -2pif
        D_ij = dict()
        for key in MOSA_labels:
            D_ij[key] = np.exp(tmp * self.dij[key])  # (Nf)

        P_ij = dict()
        for key in MOSA_labels:
            P_ij[key] = np.zeros_like(f, dtype=np.complex128)  # (Nf)
            for D_strings in P_ij_strings[key]:  # D_strings = (sign, ["ij", "jk", ...])
                prefactor = D_strings[0] + 0.0j
                D_operators = np.ones_like(f, dtype=np.complex128)
                for D_string in D_strings[1]:  # D_string = "ij" or "-ij"
                    D_operator = D_ij[D_string[-2] + D_string[-1]]  # (Nf)
                    if D_string[0] == "-":
                        D_operator = np.conjugate(D_operator)
                    D_operators *= D_operator
                P_ij[key] += D_operators * prefactor  # (Nf)
        return P_ij

    def TDI_noise_contributor(self, f, P_ij=None, P_ij_strings=None):
        """
        calculate oms and acc noise PSD contributor for each MOSA from Pij or the strings to produce Pij
        """
        if P_ij is None:
            P_ij = self.TDI_P_ij(P_ij_strings=P_ij_strings, f=f)

        tmp = -1.0j * TWOPI * f  # -2pif
        D_ij = dict()
        for key in MOSA_labels:
            D_ij[key] = np.exp(tmp * self.dij[key])  # (Nf)

        noise_oms_ij = dict()
        noise_acc_ij = dict()
        for key in MOSA_labels:
            noise_oms_ij[key] = np.abs(P_ij[key]) ** 2  # (Nf)
            key_inverse = key[1] + key[0]
            noise_acc_ij[key] = np.abs(P_ij[key] + D_ij[key_inverse] * P_ij[key_inverse]) ** 2  # (Nf)
        return noise_oms_ij, noise_acc_ij

    def TDI_noise(self, f, P_ij=None, P_ij_strings=None, AOMS=SOMS_nominal, AACC=SACC_nominal):
        """
        calculate total noise PSD from Pij or the strings to produce Pij, unit [ffd]
        """
        noise_oms_ij, noise_acc_ij = self.TDI_noise_contributor(f=f, P_ij=P_ij, P_ij_strings=P_ij_strings)
        SOMS = self.PSD_OMS(f=f, AOMS=AOMS)
        SACC = self.PSD_ACC(f=f, AACC=AACC)
        total_noise = np.zeros_like(f)
        for key in MOSA_labels:
            total_noise += noise_oms_ij[key] * SOMS + noise_acc_ij[key] * SACC  # (Nf)
        return total_noise
