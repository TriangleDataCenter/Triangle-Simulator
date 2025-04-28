import logging

import numpy as np

from Triangle.Constants import *
from Triangle.Data import *
from Triangle.FFTTools import *
from Triangle.GW import *
from Triangle.Noise import *
from Triangle.Offset import *
from Triangle.Orbit import *
from Triangle.Plot import *

logger = logging.getLogger(__name__)


class Interferometers:
    r"""
    Initialize the Interferometer simulation.
    Parameters:
        fsample (int): Sampling frequency in Hz. Default is 4.
        size (int): Size of the data. Default is 40000.
        t0 (float): Start time of the simulation. Default is 0.
        time_frame (str): Time frame for the simulation, either 'ProperTime' or 'ClockTime'. Default is 'ProperTime'.
        garbage_time1 (float): Initial garbage time. Default is 100.
        garbage_time2 (float): Final garbage time. Default is 100.
        telemetry_downsample (int or None): Downsampling factor for telemetry. Default is None.
        aafilter_coef (list): Anti-aliasing filter coefficients for downsampling. Default is [240, 1.1, 2.9].
        detrend_order (int or None): Order of the detrending polynomial. Default is None.
        acc_noise (bool): Flag to include accelerometer noise. Default is True.
        ro_noise (bool): Flag to include readout noise. Default is True.
        laser_noise (bool): Flag to include laser noise. Default is True.
        ob_noise (bool): Flag to include optical bench noise. Default is True.
        clock_noise (bool): Flag to include clock noise. Default is True.
        bl_noise (bool): Flag to include baseline noise. Default is True.
        ranging_noise (bool): Flag to include ranging noise. Default is True.
        op_noise (bool): Flag to include optical path noise. Default is True.
        modulation_noise (bool): Flag to include modulation noise. Default is True.
        noise_class (object or None): Class for noise settings. Default is None.
        offset_class (object or None): Class for offset settings. Default is None.
        orbit_class (object or None): Class for orbit settings. Default is None.
        gw_class (object or None): Class for gravitational wave settings. Default is None.
        modulation_freqs (dict or None): Dictionary of modulation frequencies. Default is None.
        fplan (dict or None): Frequency plan settings. Default is None.
        order (int): Interpolation order. Default is 31.
        pool (object or None): Multiprocessing pool. Default is None.
        clean_memory (bool): Flag to clean memory after simulation. Default is True.
    """

    def __init__(
        self,
        # size of data
        fsample=4,
        size=40000,
        # the start time of simulation
        t0=0,
        time_frame="ProperTime",
        garbage_time1=100.0,
        garbage_time2=100.0,
        telemetry_downsample=None,
        aafilter_coef=[240, 1.1, 2.9],  # downsampling filter used for 16Hz -> 4Hz
        detrend_order=None,
        # noise flags
        acc_noise=True,
        ro_noise=True,
        laser_noise=True,
        ob_noise=True,
        clock_noise=True,
        bl_noise=True,
        ranging_noise=True,
        op_noise=True,
        modulation_noise=True,
        # noise, offset, orbit settings
        noise_class=None,
        offset_class=None,
        orbit_class=None,
        gw_class=None,
        # modulation freqs
        modulation_freqs=modulation_freqs,
        # fplan setting
        fplan=fplan,
        # interpolation order
        order=31,
        # multiprocessing
        pool=None,
        # clean memory flag
        clean_memory=True,
    ):
        # basic settings
        self.fsample = fsample
        self.size = size
        logger.info("Simulating data with sampling frequency " + str(self.fsample) + " Hz.")
        logger.info("size = " + str(size))

        self.garbage_time1 = garbage_time1
        self.garbage_time2 = garbage_time2
        self.telemetry_downsample = telemetry_downsample
        self.aafilter_coef = aafilter_coef
        self.detrend_order = detrend_order
        self.order = order
        self.pool = pool
        self.modulation_freqs = MOSADict(modulation_freqs)

        if fplan is None:
            self.fplan_flag = False
        elif isinstance(fplan, dict) or isinstance(fplan, MOSADict):
            self.fplan_flag = True
            self.fplan = fplan
            logger.info("Set frequency plan.")
        else:
            raise ValueError("fplan should be None or dictionary.")

        # allocate times
        self.t0 = t0
        self.proper_time = None
        self.total_clock_deviation = None
        self.total_clock_freq_deviation = None
        self.proper_wrt_measured_time = None
        if time_frame in ["ClockTime", "ProperTime"]:
            self.time_frame = time_frame
        else:
            raise NotImplementedError("time frame not implemented.")
        logger.info("time frame is " + self.time_frame)

        # allocate offsets
        self.offset_class = offset_class
        self.clock_offset = None
        self.clock_freq_offset = None
        self.laser_offset = None  # the carrier laser offset, capital O term, or \nu^o_c
        self.laser_offset_sb = None  # the modulated laser offset, \nu^o_sb

        # allocate noises
        # the units of noise asds should be [ffd]
        self.noise_class = noise_class
        self.acc_noise_flag = acc_noise
        self.ro_noise_flag = ro_noise
        self.laser_noise_flag = laser_noise
        self.ob_noise_flag = ob_noise
        self.clock_noise_flag = clock_noise
        self.bl_noise_flag = bl_noise
        self.ranging_noise_flag = ranging_noise
        self.op_noise_flag = op_noise
        self.modulation_noise_flag = modulation_noise
        self.BasicNoise = {}
        if acc_noise:
            self.BasicNoise["acc_noise"] = None
        if ro_noise:
            self.BasicNoise["ro_sci_c_noise"] = None
            self.BasicNoise["ro_sci_sb_noise"] = None
            self.BasicNoise["ro_ref_c_noise"] = None
            self.BasicNoise["ro_ref_sb_noise"] = None
            self.BasicNoise["ro_tm_c_noise"] = None
        if laser_noise:
            self.BasicNoise["laser_noise"] = None
        if ob_noise:
            self.BasicNoise["ob_noise"] = None
        if clock_noise:
            self.BasicNoise["clock_noise"] = None
        if bl_noise:
            self.BasicNoise["bl_noise"] = None
        if ranging_noise:
            self.BasicNoise["ranging_noise"] = None
        if op_noise:
            self.BasicNoise["op_sci_local_noise"] = None
            self.BasicNoise["op_sci_distant_noise"] = None
            self.BasicNoise["op_ref_local_noise"] = None
            self.BasicNoise["op_ref_adjacent_noise"] = None
            self.BasicNoise["op_tm_local_noise"] = None
            self.BasicNoise["op_tm_adjacent_noise"] = None
        if modulation_noise:
            self.BasicNoise["modulation_noise"] = None
        logger.info("Noise types:")
        for key in self.BasicNoise.keys():
            logger.info("\t" + key)

        # set GW
        if gw_class is None:
            logger.info("No GW signal.")
            self.gw_flag = False
        else:
            logger.info("The simulation contains GW signal.")
            self.gw_class = gw_class
            self.gw_flag = True

        # allocate delays
        self.orbit_class = orbit_class
        self.ppr = None
        self.dppr = None
        self.mpr = None

        # allocate ifo measurements
        self.sci_c_ifo = dict(offset=None, fluctuation=None, total=None)
        self.sci_sb_ifo = dict(offset=None, fluctuation=None, total=None)
        self.ref_c_ifo = dict(offset=None, fluctuation=None, total=None)
        self.ref_sb_ifo = dict(offset=None, fluctuation=None, total=None)
        self.tm_c_ifo = dict(offset=None, fluctuation=None, total=None)
        #         self.tm_sb_ifo = dict(offset=None, fluctuation=None, total=None)

        # set clean memory flag
        self.clean_memory = clean_memory

    def SimulateProperTimes(self):
        """
        Simulates the proper times for the spacecraft (SC) in the interferometer.
        This method generates uniform and universal proper times for all spacecraft
        and stores them in a dictionary. It also calculates the TCB (Barycentric Coordinate Time)
        with respect to the proper times, which is used to calculate gravitational wave (GW) responses.
        Attributes:
            proper_time (SCDict): A dictionary containing the proper times for each spacecraft.
            tcb_time (dict): A dictionary containing the TCB times for each spacecraft.
        Logs:
            Logs the start and completion of the proper time generation process.
        """
        logger.info("Generating proper times.")

        # Proper times are set to be uniform and universal for all SCs
        time = np.arange(self.size) / self.fsample + self.t0

        self.proper_time = {}
        for label in SC_labels:
            self.proper_time[label] = time
        self.proper_time = SCDict(self.proper_time)

        # get tcb wrt proper times, used to calculate GW responses
        self.tcb_time = assign_function_for_SCs(functions=self.orbit_class.TCBinTPSfunctions, proper_time=self.proper_time)

        logger.info("Proper time generated.")

    def SimulateProperRanges(self):
        """
        Simulates the proper ranges for the interferometer.

        This method generates the proper delays using the unperturbed proper
        time delays (ppr) and their derivatives (dppr). These values are
        assumed to be already converted to the proper times of the spacecraft (SCs).

        The method performs the following steps:
        1. Logs the start of proper delay generation.
        2. Assigns the proper time delays (ppr) using the orbit class's PPR functions.
        3. Assigns the derivative of proper time delays (dppr) using the orbit class's DPPR functions.
        4. Copies the proper time delays (ppr) to the member variable `mpr`.
        5. Logs the completion of proper delay generation.

        The generated delays are stored in the instance variables `ppr`, `dppr`, and `mpr`.

        Returns:
            None
        """
        # Delays are performed using the unperturbed ppr i.e. d^o(\tau)
        # these values are assumed to be already converted to the proper times of SCs
        logger.info("Generating proper delays.")
        self.ppr = assign_function_for_MOSAs(functions=self.orbit_class.PPRfunctions(), proper_time=self.proper_time)
        self.dppr = assign_function_for_MOSAs(functions=self.orbit_class.DPPRfunctions(), proper_time=self.proper_time)
        self.mpr = self.ppr.copy()
        logger.info("Proper delays generated.")

    def SimulateBasicNoise(self):
        """
        noises are generated and converted to the units used in the measurement equations.
        NOTE: the unit of noise asds should be ffd
        """
        logger.info("Generating basic instrumental noises.")

        if self.acc_noise_flag:
            self.BasicNoise["acc_noise"] = assign_noise_for_MOSAs(self.noise_class.PSD_ACC, fsample=self.fsample, size=self.size)

        if self.ro_noise_flag:
            # readout noises are converted to freq unit
            self.BasicNoise["ro_sci_c_noise"] = (
                assign_noise_for_MOSAs(
                    lambda f: self.noise_class.PSD_RO(f, sro=SRO_SCI_C_nominal),
                    fsample=self.fsample,
                    size=self.size,
                )
                * F_LASER
            )
            self.BasicNoise["ro_sci_sb_noise"] = (
                assign_noise_for_MOSAs(
                    lambda f: self.noise_class.PSD_RO(f, sro=SRO_SCI_SB_nominal),
                    fsample=self.fsample,
                    size=self.size,
                )
                * F_LASER
            )
            if SRO_REF_C_nominal == 0.0:
                self.BasicNoise["ro_ref_c_noise"] = assign_noise_for_MOSAs(np.zeros(self.size), self.fsample, self.size)
            else:
                self.BasicNoise["ro_ref_c_noise"] = (
                    assign_noise_for_MOSAs(
                        lambda f: self.noise_class.PSD_RO(f, sro=SRO_REF_C_nominal),
                        fsample=self.fsample,
                        size=self.size,
                    )
                    * F_LASER
                )
            if SRO_REF_SB_nominal == 0.0:
                self.BasicNoise["ro_ref_sb_noise"] = assign_noise_for_MOSAs(np.zeros(self.size), self.fsample, self.size)
            else:
                self.BasicNoise["ro_ref_sb_noise"] = (
                    assign_noise_for_MOSAs(
                        lambda f: self.noise_class.PSD_RO(f, sro=SRO_REF_SB_nominal),
                        fsample=self.fsample,
                        size=self.size,
                    )
                    * F_LASER
                )
            if SRO_TM_C_nominal == 0.0:
                self.BasicNoise["ro_tm_c_noise"] = assign_noise_for_MOSAs(np.zeros(self.size), self.fsample, self.size)
            else:
                self.BasicNoise["ro_tm_c_noise"] = (
                    assign_noise_for_MOSAs(
                        lambda f: self.noise_class.PSD_RO(f, sro=SRO_TM_C_nominal),
                        fsample=self.fsample,
                        size=self.size,
                    )
                    * F_LASER
                )

        ############# NOTE: laser noises will be re-calculated for locked lasers #############
        # set laser noise = 0 to create an interface, even laser_noise_flag is false
        if self.fplan_flag:
            self.BasicNoise["laser_noise"] = assign_noise_for_MOSAs(np.zeros(self.size), self.fsample, self.size)
        if self.laser_noise_flag:
            self.BasicNoise["laser_noise"] = assign_noise_for_MOSAs(self.noise_class.PSD_LASER, fsample=self.fsample, size=self.size)
            self.BasicNoise["laser_noise"] *= F_LASER  # convert to freq

        if self.ob_noise_flag:
            self.BasicNoise["ob_noise"] = assign_noise_for_MOSAs(self.noise_class.PSD_OB, fsample=self.fsample, size=self.size)

        if self.clock_noise_flag:
            self.BasicNoise["clock_noise"] = assign_noise_for_SCs(self.noise_class.PSD_CLOCK, fsample=self.fsample, size=self.size)

        if self.bl_noise_flag:
            self.BasicNoise["bl_noise"] = assign_noise_for_MOSAs(self.noise_class.PSD_BL, fsample=self.fsample, size=self.size)

        if self.ranging_noise_flag:
            R_in_s = lambda f: self.noise_class.PSD_R(f) / (2.0 * np.pi * f) ** 2
            self.BasicNoise["ranging_noise"] = assign_noise_for_MOSAs(R_in_s, fsample=self.fsample, size=self.size)

        if self.op_noise_flag:
            if SOP_OTHER_nominal == 0.0:
                self.BasicNoise["op_sci_local_noise"] = assign_noise_for_MOSAs(np.zeros(self.size), self.fsample, self.size)
            else:
                self.BasicNoise["op_sci_local_noise"] = assign_noise_for_MOSAs(
                    lambda f: self.noise_class.PSD_OP(f, sop=SOP_OTHER_nominal),
                    fsample=self.fsample,
                    size=self.size,
                )
            if SOP_OTHER_nominal == 0.0:
                self.BasicNoise["op_sci_distant_noise"] = assign_noise_for_MOSAs(np.zeros(self.size), self.fsample, self.size)
            else:
                self.BasicNoise["op_sci_distant_noise"] = assign_noise_for_MOSAs(
                    lambda f: self.noise_class.PSD_OP(f, sop=SOP_OTHER_nominal),
                    fsample=self.fsample,
                    size=self.size,
                )
            if SOP_REF_LOCAL_nominal == 0.0:
                self.BasicNoise["op_ref_local_noise"] = assign_noise_for_MOSAs(np.zeros(self.size), self.fsample, self.size)
            else:
                self.BasicNoise["op_ref_local_noise"] = assign_noise_for_MOSAs(
                    lambda f: self.noise_class.PSD_OP(f, sop=SOP_REF_LOCAL_nominal),
                    fsample=self.fsample,
                    size=self.size,
                )
            if SOP_OTHER_nominal == 0.0:
                self.BasicNoise["op_ref_adjacent_noise"] = assign_noise_for_MOSAs(np.zeros(self.size), self.fsample, self.size)
            else:
                self.BasicNoise["op_ref_adjacent_noise"] = assign_noise_for_MOSAs(
                    lambda f: self.noise_class.PSD_OP(f, sop=SOP_OTHER_nominal),
                    fsample=self.fsample,
                    size=self.size,
                )
            if SOP_TM_LOCAL_nominal == 0.0:
                self.BasicNoise["op_tm_local_noise"] = assign_noise_for_MOSAs(np.zeros(self.size), self.fsample, self.size)
            else:
                self.BasicNoise["op_tm_local_noise"] = assign_noise_for_MOSAs(
                    lambda f: self.noise_class.PSD_OP(f, sop=SOP_TM_LOCAL_nominal),
                    fsample=self.fsample,
                    size=self.size,
                )
            if SOP_OTHER_nominal == 0.0:
                self.BasicNoise["op_tm_adjacent_noise"] = assign_noise_for_MOSAs(np.zeros(self.size), self.fsample, self.size)
            else:
                self.BasicNoise["op_tm_adjacent_noise"] = assign_noise_for_MOSAs(
                    lambda f: self.noise_class.PSD_OP(f, sop=SOP_OTHER_nominal),
                    fsample=self.fsample,
                    size=self.size,
                )

        if self.modulation_noise_flag:
            mnoise = {}
            for key in left_MOSA_labels:
                noise_generator = GeneralNoise(array_or_psd=(lambda f: self.noise_class.PSD_M(f, sm=SM_LEFT_nominal)))
                mnoise[key] = noise_generator(fsample=self.fsample, size=self.size)
            for key in right_MOSA_labels:
                noise_generator = GeneralNoise(array_or_psd=(lambda f: self.noise_class.PSD_M(f, sm=SM_RIGHT_nominal)))
                mnoise[key] = noise_generator(fsample=self.fsample, size=self.size)
            self.BasicNoise["modulation_noise"] = MOSADict(mnoise)

        logger.info("Basic instrumental noises generated.")

    def CleanBasicNoise(self, noise_type):
        """
        Cleans the specified basic noise type by reassigning it with zero noise.

        Parameters:
        noise_type (str): The type of noise to be cleaned. Must be a key in self.BasicNoise.

        Raises:
        ValueError: If the specified noise_type is not found in self.BasicNoise.

        Logs:
        Info: Logs a message indicating that the specified noise type has been cleaned.
        """
        if noise_type not in self.BasicNoise.keys():
            raise ValueError("noise type" + noise_type + "not simulated.")
        else:
            self.BasicNoise[noise_type] = assign_noise_for_MOSAs(np.zeros(self.size), self.fsample, self.size)
            logger.info("Noise type " + noise_type + " cleaned.")

    def SimulateGW(self):
        r"""
        the calculation of GW response is conducted in TCB
        \dot{H}_{ij} = -y_{ij} where H is the variation rate of optical path and y is the ffd
        self.gw = \dot{H}_{ij}
        """
        if isinstance(self.gw_class, list):
            NGW = len(self.gw_class)
            logger.info(str(NGW) + " GW signals will be generated.")
            if self.pool is None:
                self.gw = assign_const_for_MOSAs(0.0)
                for i in range(NGW):
                    self.gw += -self.gw_class[i].CalculateResponse(self.tcb_time)
            else:
                tmp = self.pool.map(GWParallelGenerator(self.tcb_time), self.gw_class)
                tmp = np.array(tmp)
                self.gw = -np.sum(tmp, axis=0)
        else:
            self.gw = -self.gw_class.CalculateResponse(self.tcb_time)
        logger.info("GW responses generated.")

    def SimulateClocks(self):
        """
        Simulates the clock offsets and deviations for the spacecraft.
        This method generates the clock offsets and frequency offsets for the spacecraft
        using the provided offset class. It also calculates the total clock deviations
        (clock times with respect to proper times) by considering both the offsets and
        any additional clock noise if the clock noise flag is set.
        Attributes:
            clock_offset (dict): The clock offsets for the spacecraft.
            clock_freq_offset (dict): The clock frequency offsets for the spacecraft.
            total_clock_deviation (dict): The total clock deviations including offsets and noise.
            total_clock_freq_deviation (dict): The total clock frequency deviations including offsets and noise.
        Raises:
            None
        Logs:
            Info: 'Generating clock offsets.'
            Info: 'Clock offsets generated.'
        """
        logger.info("Generating clock offsets.")
        # clock offsets
        self.clock_offset = assign_function_for_SCs(functions=self.offset_class.ClockOffsets(), proper_time=self.proper_time)
        self.clock_freq_offset = assign_function_for_SCs(functions=self.offset_class.ClockFreqOffsets(), proper_time=self.proper_time)
        logger.info("Clock offsets generated.")

        # total clock deviations (clock times wrt proper times)
        # 1. offset
        self.total_clock_deviation = self.clock_offset.copy()
        self.total_clock_freq_deviation = self.clock_freq_offset.copy()
        # 2. jitter
        if self.clock_noise_flag:
            self.total_clock_deviation += (self.BasicNoise["clock_noise"]).integrate(fsample=self.fsample)
            self.total_clock_freq_deviation += self.BasicNoise["clock_noise"]

    def SimulateSources(self):
        # unlocked
        logger.info("Generating unlocked sources.")
        # 1. offsets
        self.laser_offset = assign_function_for_MOSAs(functions=self.offset_class.LaserOffsets(), proper_time=self.proper_time)
        # 2. fluctuations
        # already generated.
        logger.info("unlocked sources generated.")

        # locked
        # NOTE: Manipulate individual dict items rather than the dict class
        if self.fplan_flag:
            logger.info("Generating locked sources.")
            # 0. set all the locked lasers to 0
            key_primary = lock_order[0]
            for key in MOSA_labels:
                if key != key_primary:
                    self.laser_offset[key] = np.zeros(self.size)
                    if self.laser_noise_flag:
                        self.BasicNoise["laser_noise"][key] = np.zeros(self.size)
                    logger.debug("laser_" + str(key) + "removed.")
            for key in lock_order:
                if lock_topology[key] == "primary":
                    logger.debug("primary laser " + str(key) + " passed.")
                elif lock_topology[key] == "adjacent":
                    # 1. offset
                    key_ad = adjacent_MOSA_labels[key]
                    self.laser_offset[key] = self.laser_offset[key_ad] - self.fplan[key] * (1.0 + self.clock_freq_offset[key[0]])
                    # 2. fluctuation
                    self.BasicNoise["laser_noise"][key] += self.BasicNoise["laser_noise"][key_ad]
                    if self.clock_noise_flag:
                        self.BasicNoise["laser_noise"][key] += -self.fplan[key] * self.BasicNoise["clock_noise"][key[0]]
                    if self.bl_noise_flag:
                        self.BasicNoise["laser_noise"][key] += -(F_LASER + self.laser_offset[key_ad]) * self.BasicNoise["bl_noise"][key]
                    if self.op_noise_flag:
                        self.BasicNoise["laser_noise"][key] += -(F_LASER + self.laser_offset[key_ad]) * self.BasicNoise["op_ref_adjacent_noise"][key] + (F_LASER + self.laser_offset[key]) * self.BasicNoise["op_ref_local_noise"][key]
                    if self.ro_noise_flag:
                        self.BasicNoise["laser_noise"][key] += self.BasicNoise["ro_ref_c_noise"][key]
                    logger.debug("laser " + str(key) + " locked to adjacent.")
                elif lock_topology[key] == "distant":
                    # 1. offset
                    key_ds = key[1] + key[0]
                    delayed_distant = timeshift(
                        self.laser_offset[key_ds],
                        -self.ppr[key] * self.fsample,
                        self.order,
                    )
                    doppler_delayed_distant = delayed_distant * (1.0 - self.dppr[key])
                    self.laser_offset[key] = doppler_delayed_distant - F_LASER * self.dppr[key] - self.fplan[key] * (1.0 + self.clock_freq_offset[key[0]])
                    # 2. fluctuation
                    delayed_distant_ln = timeshift(
                        self.BasicNoise["laser_noise"][key_ds],
                        -self.ppr[key] * self.fsample,
                        self.order,
                    )
                    doppler_delayed_distant_ln = delayed_distant_ln * (1.0 - self.dppr[key])
                    self.BasicNoise["laser_noise"][key] += doppler_delayed_distant_ln
                    if self.clock_noise_flag:
                        self.BasicNoise["laser_noise"][key] += -self.fplan[key] * self.BasicNoise["clock_noise"][key[0]]
                    if self.ob_noise_flag:
                        delayed_distant_on = timeshift(
                            self.BasicNoise["ob_noise"][key_ds],
                            -self.ppr[key] * self.fsample,
                            self.order,
                        )
                        doppler_delayed_distant_on = delayed_distant_on * (1.0 - self.dppr[key])
                        self.BasicNoise["laser_noise"][key] += (F_LASER + delayed_distant) * (doppler_delayed_distant_on + self.BasicNoise["ob_noise"][key])
                    if self.op_noise_flag:
                        self.BasicNoise["laser_noise"][key] += -(F_LASER + delayed_distant) * self.BasicNoise["op_sci_distant_noise"][key] + (F_LASER + self.laser_offset[key]) * self.BasicNoise["op_sci_local_noise"][key]
                    if self.ro_noise_flag:
                        self.BasicNoise["laser_noise"][key] += self.BasicNoise["ro_sci_c_noise"][key]
                    if self.gw_flag:
                        self.BasicNoise["laser_noise"][key] += -(F_LASER + delayed_distant) * self.gw[key]
                    logger.debug("laser " + str(key) + " locked to distant.")
                else:
                    raise ValueError("locking method not surpported.")
            logger.info("Locked sources generated.")
        # side band
        self.laser_offset_sb = self.laser_offset + self.modulation_freqs * (1.0 + self.clock_freq_offset.toMOSA())

    def SimulateOffsets(self):
        logger.info("Generating offsets.")
        # doppler delay args
        dd_args = dict(fsample=self.fsample, delay=self.ppr, doppler=self.dppr, order=self.order)
        # delay args
        # d_args = dict(fsample=self.fsample, delay=self.ppr, order=self.order)

        # sci_c
        self.sci_c_ifo["offset"] = (self.laser_offset.reverse()).timedelay(**dd_args, pool=self.pool) - F_LASER * self.dppr - self.laser_offset
        # sci_sb
        # tmp = (self.modulation_freqs.reverse() * (1. + (self.clock_freq_offset.toMOSA()).reverse())).timedelay(**dd_args, pool=self.pool) - self.modulation_freqs * (1. + self.clock_freq_offset.toMOSA())
        nu_m_ji = self.modulation_freqs.reverse()
        nu_m_ij = self.modulation_freqs
        tmp = (nu_m_ji - nu_m_ij) - nu_m_ji * self.dppr + ((self.clock_freq_offset.toMOSA()).reverse()).timedelay(**dd_args, pool=self.pool) * nu_m_ji - self.clock_freq_offset.toMOSA() * nu_m_ij
        self.sci_sb_ifo["offset"] = self.sci_c_ifo["offset"] + tmp
        tmp = None
        # ref_c
        self.ref_c_ifo["offset"] = self.laser_offset.adjacent() - self.laser_offset
        # ref_sb
        self.ref_sb_ifo["offset"] = self.ref_c_ifo["offset"] + (self.modulation_freqs.adjacent() - self.modulation_freqs) * (1.0 + self.clock_freq_offset.toMOSA())
        # tm_c
        self.tm_c_ifo["offset"] = self.ref_c_ifo["offset"].copy()

        logger.info("Offsets generated.")

    def SimulateFluctuations(self):
        logger.info("Generating fluctuations.")
        self.sci_c_ifo["fluctuation"] = assign_noise_for_MOSAs(arrays_or_psds=np.zeros(self.size), fsample=self.fsample, size=self.size)
        self.sci_sb_ifo["fluctuation"] = assign_noise_for_MOSAs(arrays_or_psds=np.zeros(self.size), fsample=self.fsample, size=self.size)
        self.ref_c_ifo["fluctuation"] = assign_noise_for_MOSAs(arrays_or_psds=np.zeros(self.size), fsample=self.fsample, size=self.size)
        self.ref_sb_ifo["fluctuation"] = assign_noise_for_MOSAs(arrays_or_psds=np.zeros(self.size), fsample=self.fsample, size=self.size)
        self.tm_c_ifo["fluctuation"] = assign_noise_for_MOSAs(arrays_or_psds=np.zeros(self.size), fsample=self.fsample, size=self.size)

        # doppler delay args
        dd_args = dict(fsample=self.fsample, delay=self.ppr, doppler=self.dppr, order=self.order)
        # delay args
        d_args = dict(fsample=self.fsample, delay=self.ppr, order=self.order)

        # ############## NOTE: laser noise should be always calculated for fplan=True ####################
        # laser noise
        if self.laser_noise_flag or self.fplan_flag:
            sci = (self.BasicNoise["laser_noise"].reverse()).timedelay(**dd_args, pool=self.pool) - self.BasicNoise["laser_noise"]
            ref = self.BasicNoise["laser_noise"].adjacent() - self.BasicNoise["laser_noise"]
            self.sci_c_ifo["fluctuation"] += sci
            self.sci_sb_ifo["fluctuation"] += sci
            self.ref_c_ifo["fluctuation"] += ref
            self.ref_sb_ifo["fluctuation"] += ref
            self.tm_c_ifo["fluctuation"] += ref
            sci = None
            ref = None
            logger.debug("laser flutctuations generated.")
            if self.clean_memory:
                self.BasicNoise["laser_noise"] = None

        # readout noise
        if self.ro_noise_flag:
            self.sci_c_ifo["fluctuation"] += self.BasicNoise["ro_sci_c_noise"]
            self.sci_sb_ifo["fluctuation"] += self.BasicNoise["ro_sci_sb_noise"]
            self.ref_c_ifo["fluctuation"] += self.BasicNoise["ro_ref_c_noise"]
            self.ref_sb_ifo["fluctuation"] += self.BasicNoise["ro_ref_sb_noise"]
            self.tm_c_ifo["fluctuation"] += self.BasicNoise["ro_tm_c_noise"]
            logger.debug("readout flutctuations generated.")
            if self.clean_memory:
                self.BasicNoise["ro_sci_c_noise"] = None
                self.BasicNoise["ro_sci_sb_noise"] = None
                self.BasicNoise["ro_ref_c_noise"] = None
                self.BasicNoise["ro_ref_sb_noise"] = None
                self.BasicNoise["ro_tm_c_noise"] = None

        # tm noise
        if self.acc_noise_flag:
            self.sci_c_ifo["fluctuation"] += 0.0
            self.sci_sb_ifo["fluctuation"] += 0.0
            self.ref_c_ifo["fluctuation"] += 0.0
            self.ref_sb_ifo["fluctuation"] += 0.0
            self.tm_c_ifo["fluctuation"] += (F_LASER + self.laser_offset) * (-2.0) * self.BasicNoise["acc_noise"]
            logger.debug("tm flutctuations generated.")
            if self.clean_memory:
                self.BasicNoise["acc_noise"] = None

        # ob noise
        if self.ob_noise_flag:
            displacement = (self.BasicNoise["ob_noise"].reverse()).timedelay(**dd_args, pool=self.pool) + self.BasicNoise["ob_noise"]
            self.sci_c_ifo["fluctuation"] += (F_LASER + (self.laser_offset.reverse()).timedelay(**d_args, pool=self.pool)) * displacement
            self.sci_sb_ifo["fluctuation"] += (F_LASER + (self.laser_offset_sb.reverse()).timedelay(**d_args, pool=self.pool)) * displacement
            self.ref_c_ifo["fluctuation"] += 0.0
            self.ref_sb_ifo["fluctuation"] += 0.0
            self.tm_c_ifo["fluctuation"] += (F_LASER + self.laser_offset) * 2.0 * self.BasicNoise["ob_noise"]
            displacement = None
            logger.debug("ob flutctuations generated.")
            if self.clean_memory:
                self.BasicNoise["ob_noise"] = None

        # bl noise
        if self.bl_noise_flag:
            bl_c = -(F_LASER + self.laser_offset.adjacent()) * self.BasicNoise["bl_noise"]
            bl_sb = -(F_LASER + self.laser_offset_sb.adjacent()) * self.BasicNoise["bl_noise"]
            self.sci_c_ifo["fluctuation"] += 0.0
            self.sci_sb_ifo["fluctuation"] += 0.0
            self.ref_c_ifo["fluctuation"] += bl_c
            self.ref_sb_ifo["fluctuation"] += bl_sb
            self.tm_c_ifo["fluctuation"] += bl_c
            logger.debug("bl flutctuations generated.")
            if self.clean_memory:
                self.BasicNoise["bl_noise"] = None

        # op_noise
        if self.op_noise_flag:
            self.sci_c_ifo["fluctuation"] += -(F_LASER + (self.laser_offset.reverse()).timedelay(**d_args, pool=self.pool)) * self.BasicNoise["op_sci_distant_noise"] + (F_LASER + self.laser_offset) * self.BasicNoise["op_sci_local_noise"]
            self.sci_sb_ifo["fluctuation"] += -(F_LASER + (self.laser_offset_sb.reverse()).timedelay(**d_args, pool=self.pool)) * self.BasicNoise["op_sci_distant_noise"] + (F_LASER + self.laser_offset_sb) * self.BasicNoise["op_sci_local_noise"]
            self.ref_c_ifo["fluctuation"] += -(F_LASER + self.laser_offset.adjacent()) * self.BasicNoise["op_ref_adjacent_noise"] + (F_LASER + self.laser_offset) * self.BasicNoise["op_ref_local_noise"]
            self.ref_sb_ifo["fluctuation"] += -(F_LASER + self.laser_offset_sb.adjacent()) * self.BasicNoise["op_ref_adjacent_noise"] + (F_LASER + self.laser_offset_sb) * self.BasicNoise["op_ref_local_noise"]
            self.tm_c_ifo["fluctuation"] += -(F_LASER + self.laser_offset.adjacent()) * self.BasicNoise["op_tm_adjacent_noise"] + (F_LASER + self.laser_offset) * self.BasicNoise["op_tm_local_noise"]
            logger.debug("op flutctuations generated.")
            if self.clean_memory:
                self.BasicNoise["op_sci_distant_noise"] = None
                self.BasicNoise["op_sci_local_noise"] = None
                self.BasicNoise["op_ref_adjacent_noise"] = None
                self.BasicNoise["op_ref_local_noise"] = None
                self.BasicNoise["op_tm_adjacent_noise"] = None
                self.BasicNoise["op_tm_local_noise"] = None

        # clock noise modulation to the sidebands
        if self.clock_noise_flag:
            sideband_modulation = self.modulation_freqs * (self.BasicNoise["clock_noise"].toMOSA())
            self.sci_sb_ifo["fluctuation"] += (sideband_modulation.reverse()).timedelay(**dd_args, pool=self.pool) - sideband_modulation
            self.ref_sb_ifo["fluctuation"] += ((self.modulation_freqs).adjacent() - self.modulation_freqs) * (self.BasicNoise["clock_noise"].toMOSA())
            sideband_modulation = None
            logger.debug("sideband clock noise modulation generated.")

        # modulation noise
        if self.modulation_noise_flag:
            sideband_modulation = self.modulation_freqs * self.BasicNoise["modulation_noise"]
            self.sci_sb_ifo["fluctuation"] += (sideband_modulation.reverse()).timedelay(**dd_args, pool=self.pool) - sideband_modulation
            self.ref_sb_ifo["fluctuation"] += sideband_modulation.adjacent() - sideband_modulation
            sideband_modulation = None
            logger.debug("sideband modulation fluctuations generated.")
            if self.clean_memory:
                self.BasicNoise["modulation_noise"] = None

        # gw signal
        if self.gw_flag:
            self.sci_c_ifo["fluctuation"] += -(F_LASER + (self.laser_offset.reverse()).timedelay(**d_args, pool=self.pool)) * self.gw
            self.sci_sb_ifo["fluctuation"] += -(F_LASER + (self.laser_offset_sb.reverse()).timedelay(**d_args, pool=self.pool)) * self.gw
            logger.debug("GW fluctuations generated.")
            if self.clean_memory:
                self.gw = None

        # add ranging noise to  mpr
        if self.ranging_noise_flag:
            self.mpr += self.BasicNoise["ranging_noise"]
            logger.debug("PRN ranging fluctuations generated.")
            if self.clean_memory:
                self.BasicNoise["ranging_noise"] = False

        # add clock deviation to mpr
        self.mpr += self.total_clock_deviation.toMOSA() - ((self.total_clock_deviation.toMOSA()).reverse()).timedelay(**d_args, pool=self.pool)
        logger.debug("mpr generated.")

        logger.info("Flutctuations generated.")

    def SimulateTotal(self):
        self.sci_c_ifo["total"] = self.sci_c_ifo["fluctuation"] + self.sci_c_ifo["offset"]
        self.sci_sb_ifo["total"] = self.sci_sb_ifo["fluctuation"] + self.sci_sb_ifo["offset"]
        self.ref_c_ifo["total"] = self.ref_c_ifo["fluctuation"] + self.ref_c_ifo["offset"]
        self.ref_sb_ifo["total"] = self.ref_sb_ifo["fluctuation"] + self.ref_sb_ifo["offset"]
        self.tm_c_ifo["total"] = self.tm_c_ifo["fluctuation"] + self.tm_c_ifo["offset"]
        logger.info("Total measurements constructed.")

    def SimulateTimeShift(self):
        """
        total freq and the decomposed freqs are shifted to clock times separately:
        1. total: time shift by the total clock deviations, clock noise is added by time shift
        2. decomposed: add clock noise to fluctuation, and then shift offsets and fluctuations by the clock offsets
        """
        logger.info("Calculating time shift.")
        # calculate proper times wrt clock times by iteration
        dtau0 = self.total_clock_deviation.copy()
        dtau1 = self.total_clock_deviation.timedelay(fsample=self.fsample, delay=dtau0, order=self.order, pool=self.pool)
        dtau2 = self.total_clock_deviation.timedelay(fsample=self.fsample, delay=dtau1, order=self.order, pool=self.pool)
        self.proper_wrt_measured_time = -dtau2

        # shift interferometers, use total dot q for total freqs, use offset dot q for offsets and fluctuations
        d_args = dict(fsample=self.fsample, delay=dtau2.toMOSA(), order=self.order)
        clock_factor = (1.0 / (1.0 + self.total_clock_freq_deviation)).toMOSA()
        clock_offset_factor = (1.0 / (1.0 + self.clock_freq_offset)).toMOSA()
        # 1. shift total measurements with the total clock deviation
        self.sci_c_ifo["total"] = (self.sci_c_ifo["total"] * clock_factor).timedelay(**d_args, pool=self.pool)
        self.sci_sb_ifo["total"] = (self.sci_sb_ifo["total"] * clock_factor).timedelay(**d_args, pool=self.pool)
        self.ref_c_ifo["total"] = (self.ref_c_ifo["total"] * clock_factor).timedelay(**d_args, pool=self.pool)
        self.ref_sb_ifo["total"] = (self.ref_sb_ifo["total"] * clock_factor).timedelay(**d_args, pool=self.pool)
        self.tm_c_ifo["total"] = (self.tm_c_ifo["total"] * clock_factor).timedelay(**d_args, pool=self.pool)
        # 2. shift offsets and fluctuations with the clock offsets
        # 1) add clock noise
        if self.clock_noise_flag:
            cfactor = self.BasicNoise["clock_noise"].toMOSA() / (1.0 + self.clock_freq_offset.toMOSA())
            self.sci_c_ifo["fluctuation"] += -self.sci_c_ifo["offset"] * cfactor
            self.sci_sb_ifo["fluctuation"] += -self.sci_sb_ifo["offset"] * cfactor
            self.ref_c_ifo["fluctuation"] += -self.ref_c_ifo["offset"] * cfactor
            self.ref_sb_ifo["fluctuation"] += -self.ref_sb_ifo["offset"] * cfactor
            self.tm_c_ifo["fluctuation"] += -self.tm_c_ifo["offset"] * cfactor
            cfactor = None
            logger.debug("clock flutctuations generated.")
        # 2) time shift
        for key in ["offset", "fluctuation"]:
            self.sci_c_ifo[key] = (self.sci_c_ifo[key] * clock_offset_factor).timedelay(**d_args, pool=self.pool)
            self.sci_sb_ifo[key] = (self.sci_sb_ifo[key] * clock_offset_factor).timedelay(**d_args, pool=self.pool)
            self.ref_c_ifo[key] = (self.ref_c_ifo[key] * clock_offset_factor).timedelay(**d_args, pool=self.pool)
            self.ref_sb_ifo[key] = (self.ref_sb_ifo[key] * clock_offset_factor).timedelay(**d_args, pool=self.pool)
            self.tm_c_ifo[key] = (self.tm_c_ifo[key] * clock_offset_factor).timedelay(**d_args, pool=self.pool)

        # shift mpr
        self.mpr = self.mpr.timedelay(**d_args, pool=self.pool)
        self.ppr = self.ppr.timedelay(**d_args, pool=self.pool)
        logger.info("Time frame shifted.")

    def SimulateTelemetry(self):
        logger.info("Downsampling to telemetry sampling frequency.")
        downsample = round(self.telemetry_downsample)
        # interferometers
        self.sci_c_ifo_tel = (self.sci_c_ifo["total"]).downsampled(self.fsample, downsample, self.aafilter_coef)
        self.sci_sb_ifo_tel = (self.sci_sb_ifo["total"]).downsampled(self.fsample, downsample, self.aafilter_coef)
        self.ref_c_ifo_tel = (self.ref_c_ifo["total"]).downsampled(self.fsample, downsample, self.aafilter_coef)
        self.ref_sb_ifo_tel = (self.ref_sb_ifo["total"]).downsampled(self.fsample, downsample, self.aafilter_coef)
        self.tm_c_ifo_tel = (self.tm_c_ifo["total"]).downsampled(self.fsample, downsample, self.aafilter_coef)
        # mpr
        self.mpr_tel = (self.mpr).downsampled(self.fsample, downsample, self.aafilter_coef)
        self.ppr_tel = (self.ppr).downsampled(self.fsample, downsample, self.aafilter_coef)
        # time stamp
        # if shifted to clock times, time_stamp represents on-board clock time
        # otherwise, time_stamp represents spacecraft proper time
        if not hasattr(self, "time_stamp"):
            self.time_stamp = self.proper_time.copy()
        time_stamp_tel = {}
        for key in SC_labels:
            time_stamp_tel[key] = self.time_stamp[key][::downsample]
        self.time_stamp_tel = SCDict(time_stamp_tel)
        # set telemetry fsample and size
        self.fsample_tel = self.fsample / downsample
        self.size_tel = len(self.sci_c_ifo_tel["12"])
        logger.info("Measurements downsampled.")

    def SimulateDropEdgePoints(self):
        """
        drop starting or/and ending points before detrending (both on board and telemetry data)
        1. garbage points at the start may be caused by time delay (or shifting time frame)
        2. garbage points at the end may be caused by shifting time frame
        """
        logger.info("Removing starting invalid points (" + str(self.garbage_time1) + " s) and ending invalid points (" + str(self.garbage_time2) + " s).")

        # 1. on board measurements
        drop_points1 = int(self.garbage_time1 * self.fsample)
        drop_points2 = int(self.garbage_time2 * self.fsample)
        # (1) interferometers
        for key in ["total", "offset", "fluctuation"]:
            self.sci_c_ifo[key] = (self.sci_c_ifo[key]).drop_edge_points(points1=drop_points1, points2=drop_points2)
            self.sci_sb_ifo[key] = (self.sci_sb_ifo[key]).drop_edge_points(points1=drop_points1, points2=drop_points2)
            self.ref_c_ifo[key] = (self.ref_c_ifo[key]).drop_edge_points(points1=drop_points1, points2=drop_points2)
            self.ref_sb_ifo[key] = (self.ref_sb_ifo[key]).drop_edge_points(points1=drop_points1, points2=drop_points2)
            self.tm_c_ifo[key] = (self.tm_c_ifo[key]).drop_edge_points(points1=drop_points1, points2=drop_points2)
        # (2) mpr
        self.mpr = self.mpr.drop_edge_points(points1=drop_points1, points2=drop_points2)
        self.ppr = self.ppr.drop_edge_points(points1=drop_points1, points2=drop_points2)
        # (3) time stamp
        if not hasattr(self, "time_stamp"):
            self.time_stamp = self.proper_time.copy()
        self.time_stamp = self.time_stamp.drop_edge_points(points1=drop_points1, points2=drop_points2)
        self.size = len(self.sci_c_ifo["total"]["12"])

        # 2. telemetry measurements
        if self.telemetry_downsample is not None:
            drop_points1 = int(self.garbage_time1 * self.fsample_tel)
            drop_points2 = int(self.garbage_time2 * self.fsample_tel)
            # (1) interferometers
            self.sci_c_ifo_tel = (self.sci_c_ifo_tel).drop_edge_points(points1=drop_points1, points2=drop_points2)
            self.sci_sb_ifo_tel = (self.sci_sb_ifo_tel).drop_edge_points(points1=drop_points1, points2=drop_points2)
            self.ref_c_ifo_tel = (self.ref_c_ifo_tel).drop_edge_points(points1=drop_points1, points2=drop_points2)
            self.ref_sb_ifo_tel = (self.ref_sb_ifo_tel).drop_edge_points(points1=drop_points1, points2=drop_points2)
            self.tm_c_ifo_tel = (self.tm_c_ifo_tel).drop_edge_points(points1=drop_points1, points2=drop_points2)
            # (2) mpr
            self.mpr_tel = self.mpr_tel.drop_edge_points(points1=drop_points1, points2=drop_points2)
            self.ppr_tel = self.ppr_tel.drop_edge_points(points1=drop_points1, points2=drop_points2)
            # (3) time stamp
            self.time_stamp_tel = self.time_stamp_tel.drop_edge_points(points1=drop_points1, points2=drop_points2)
            self.size_tel = len(self.sci_c_ifo_tel["12"])

        logger.info("Invalid points removed.")

    def SimulateDetrend(self):
        """
        This function simply detrend data with polynomials, only suitable for short-duration data (e.g. ~1 day).
        NOTE: detrend must be done **after** removing the problematic points
        """
        logger.info("Detrending measurements.")
        self.sci_c_ifo_det = (self.sci_c_ifo["total"]).detrended(order=self.detrend_order)
        self.sci_sb_ifo_det = (self.sci_sb_ifo["total"]).detrended(order=self.detrend_order)
        self.ref_c_ifo_det = (self.ref_c_ifo["total"]).detrended(order=self.detrend_order)
        self.ref_sb_ifo_det = (self.ref_sb_ifo["total"]).detrended(order=self.detrend_order)
        self.tm_c_ifo_det = (self.tm_c_ifo["total"]).detrended(order=self.detrend_order)
        if self.telemetry_downsample is not None:
            self.sci_c_ifo_tel_det = (self.sci_c_ifo_tel).detrended(order=self.detrend_order)
            self.sci_sb_ifo_tel_det = (self.sci_sb_ifo_tel).detrended(order=self.detrend_order)
            self.ref_c_ifo_tel_det = (self.ref_c_ifo_tel).detrended(order=self.detrend_order)
            self.ref_sb_ifo_tel_det = (self.ref_sb_ifo_tel).detrended(order=self.detrend_order)
            self.tm_c_ifo_tel_det = (self.tm_c_ifo_tel).detrended(order=self.detrend_order)
        logger.info("Measurements detrended.")

    def SimulateRangeAndClock(self):
        self.SimulateProperTimes()
        self.SimulateProperRanges()
        self.SimulateClocks()

    def SimulateMeasurements(self):
        self.SimulateSources()
        self.SimulateOffsets()
        self.SimulateFluctuations()
        self.SimulateTotal()
        if self.time_frame == "ClockTime":
            self.SimulateTimeShift()
        elif self.time_frame == "ProperTime":
            pass
        # shift to TCB should be done on ground
        else:
            raise NotImplementedError("time frame not implemented.")
        # downsample and telemetry
        if self.telemetry_downsample is not None:
            self.SimulateTelemetry()
        # remove starting invalid points
        self.SimulateDropEdgePoints()
        # detrend telemetry data, leaving only fluctuations
        if self.detrend_order is not None:
            self.SimulateDetrend()
        logger.info("Simulation completed.")

        logger.info("Sampling frequency of data = " + str(self.fsample) + " Hz.")
        logger.info("Data size = " + str(self.size))
        if self.telemetry_downsample:
            logger.info("Sampling frequency of telemetry data = " + str(self.fsample_tel) + " Hz.")
            logger.info("telemetry data size = " + str(self.size_tel))

    def SimulateInterferometers(self):
        self.SimulateBasicNoise()
        self.SimulateRangeAndClock()
        if self.gw_flag:
            self.SimulateGW()
        self.SimulateMeasurements()

    def OutputMeasurements(self, mode="decomposed"):
        """
        modes:
            decomposed
            total
            detrended
            telemetry_total
            telemetry_detrended
        """
        if mode == "decomposed":
            m = dict(
                sci_c=self.sci_c_ifo["fluctuation"],
                sci_sb=self.sci_sb_ifo["fluctuation"],
                ref_c=self.ref_c_ifo["fluctuation"],
                ref_sb=self.ref_sb_ifo["fluctuation"],
                tm_c=self.tm_c_ifo["fluctuation"],
                a=self.sci_c_ifo["offset"],
                b=self.ref_c_ifo["offset"],
                mpr=self.mpr,
                ppr=self.ppr,
                fsample=self.fsample,
                time=self.time_stamp,
            )
            return m

        elif mode == "total":
            m = dict(
                sci_c=self.sci_c_ifo["total"],
                sci_sb=self.sci_sb_ifo["total"],
                ref_c=self.ref_c_ifo["total"],
                ref_sb=self.ref_sb_ifo["total"],
                tm_c=self.tm_c_ifo["total"],
                a=self.sci_c_ifo["offset"],
                b=self.ref_c_ifo["offset"],
                mpr=self.mpr,
                ppr=self.ppr,
                fsample=self.fsample,
                time=self.time_stamp,
            )
            return m

        elif mode == "detrended":
            if self.detrend_order is None:
                raise ValueError("detrend not simulated.")
            else:
                m = dict(
                    sci_c=self.sci_c_ifo_det,
                    sci_sb=self.sci_sb_ifo_det,
                    ref_c=self.ref_c_ifo_det,
                    ref_sb=self.ref_sb_ifo_det,
                    tm_c=self.tm_c_ifo_det,
                    a=self.sci_c_ifo["offset"],
                    b=self.ref_c_ifo["offset"],
                    mpr=self.mpr,
                    ppr=self.ppr,
                    fsample=self.fsample,
                    time=self.time_stamp,
                )
                return m

        elif mode == "telemetry_total":
            if self.telemetry_downsample:
                m = dict(
                    sci_c=self.sci_c_ifo_tel,
                    sci_sb=self.sci_sb_ifo_tel,
                    ref_c=self.ref_c_ifo_tel,
                    ref_sb=self.ref_sb_ifo_tel,
                    tm_c=self.tm_c_ifo_tel,
                    a=self.sci_c_ifo_tel,
                    b=self.ref_c_ifo_tel,
                    mpr=self.mpr_tel,
                    ppr=self.ppr_tel,
                    fsample=self.fsample_tel,
                    time=self.time_stamp_tel,
                )

                return m
            else:
                raise ValueError("telemetry_total mode is only surpported for downsampled data.")

        elif mode == "telemetry_detrended":
            if self.detrend_order is None:
                raise ValueError("detrend not simulated.")
            else:
                if self.telemetry_downsample:
                    m = dict(
                        sci_c=self.sci_c_ifo_tel_det,
                        sci_sb=self.sci_sb_ifo_tel_det,
                        ref_c=self.ref_c_ifo_tel_det,
                        ref_sb=self.ref_sb_ifo_tel_det,
                        tm_c=self.tm_c_ifo_tel_det,
                        a=self.sci_c_ifo_tel,
                        b=self.ref_c_ifo_tel,
                        mpr=self.mpr_tel,
                        ppr=self.ppr_tel,
                        fsample=self.fsample_tel,
                        time=self.time_stamp_tel,
                    )

                    return m
                else:
                    raise ValueError("telemetry_detrended mode is only surpported for downsampled data.")

        else:
            raise ValueError("mode not supported.")

    def clean(self):
        self.BasicNoise = None
        self.sci_c_ifo = None
        self.sci_sb_ifo = None
        self.ref_c_ifo = None
        self.ref_sb_ifo = None
        self.tm_c_ifo = None
        self.sci_c_ifo_tel = None
        self.sci_sb_ifo_tel = None
        self.ref_c_ifo_tel = None
        self.ref_sb_ifo_tel = None
        self.tm_c_ifo_tel = None
        self.sci_c_ifo_tel_det = None
        self.sci_sb_ifo_tel_det = None
        self.ref_c_ifo_tel_det = None
        self.ref_sb_ifo_tel_det = None
        self.tm_c_ifo_tel_det = None
        self.sci_c_ifo_det = None
        self.sci_sb_ifo_det = None
        self.ref_c_ifo_det = None
        self.ref_sb_ifo_det = None
        self.tm_c_ifo_det = None
        logger.info("Memory released.")

    def ListMembers(self):
        for name, value in vars(self).items():
            print("%s=%s" % (name, value))


# Convert ifo measurements from proper times to tcb
def TPStoTCB(measurement, orbit_class, order=31, freq_unit=True, pool=None):
    m_tcb = {}
    m_tcb["time"] = measurement["time"]  # tcb is set to to the same uniform grid as proper time
    m_tcb["fsample"] = measurement["fsample"]
    m_tcb["ltt"] = assign_function_for_MOSAs(
        functions=orbit_class.LTTfunctions(),
        proper_time=m_tcb["time"],
    )
    tps_wrt_tcb = assign_function_for_SCs(
        functions=orbit_class.TPSwrtTCBfunctions,
        proper_time=m_tcb["time"],
    )
    d_args = dict(fsample=m_tcb["fsample"], delay=-tps_wrt_tcb.toMOSA(), order=order)
    if freq_unit is True:
        dtps_wrt_tcb = {}
        for key in SC_labels:
            dtps_wrt_tcb[key] = ((orbit_class.TPSinTCBfunctions[key]).derivative())(m_tcb["time"][key])
        dtps_wrt_tcb = SCDict(dtps_wrt_tcb).toMOSA()
        for key in ["sci_c", "sci_sb", "ref_c", "ref_sb", "tm_c", "a", "b"]:
            m_tcb[key] = (measurement[key].timedelay(**d_args, pool=pool)) * dtps_wrt_tcb
    else:
        for key in ["sci_c", "sci_sb", "ref_c", "ref_sb", "tm_c", "a", "b"]:
            m_tcb[key] = measurement[key].timedelay(**d_args, pool=pool)
    return m_tcb
