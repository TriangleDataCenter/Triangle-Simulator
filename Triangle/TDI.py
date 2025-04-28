import logging

import numpy as np
from tqdm import tqdm

from Triangle.Constants import *
from Triangle.Data import *
from Triangle.FFTTools import *
from Triangle.Noise import *
from Triangle.Offset import *
from Triangle.Orbit import *
from Triangle.Plot import *

logger = logging.getLogger(__name__)


class TDI:
    """
    NOTE: garbage points should be dropped after the calculation of TDI.
    """

    # examples for the path strings
    X1_plus = "12131"
    X1_minus = "13121"
    X2_plus = "121313121"
    X2_minus = "131212131"
    alpha1_plus = "1231"
    alpha1_minus = "1321"
    alpha2_plus = "1321231"
    alpha2_minus = "1231321"

    # examples for the P strings (NOTE that the sign of XYZ should be inverted to be consistent with the path strings and fast michelson calculations)
    X2_string = {
        "12": [(1.0, []), (-1.0, ["13", "31"]), (-1.0, ["13", "31", "12", "21"]), (1.0, ["12", "21", "13", "31", "13", "31"])],
        "23": [],
        "31": [(-1.0, ["13"]), (1.0, ["12", "21", "13"]), (1.0, ["12", "21", "13", "31", "13"]), (-1.0, ["13", "31", "12", "21", "12", "21", "13"])],
        "21": [(1.0, ["12"]), (-1.0, ["13", "31", "12"]), (-1.0, ["13", "31", "12", "21", "12"]), (1.0, ["12", "21", "13", "31", "13", "31", "12"])],
        "32": [],
        "13": [(-1.0, []), (1.0, ["12", "21"]), (1.0, ["12", "21", "13", "31"]), (-1.0, ["13", "31", "12", "21", "12", "21"])],
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

    def __init__(self, measurements, delays, fsample, order=31, delay_order=5):
        """
        the measurements dict contains following keys:
        "sci_c",
        "sci_sb",
        "ref_c",
        "ref_sb",
        "tm_c",
        "mpr",
        "a",
        "b",
        each is a MOSADict object
        delays is also a MOSADict object
        """
        self.measurements = measurements.copy()
        self.delays = delays.copy()
        self.fsample = fsample
        if "sci_c" in self.measurements.keys():
            self.size = len(measurements["sci_c"]["12"])
        elif "eta" in self.measurements.keys():
            self.size = len(measurements["eta"]["12"])
        else:
            raise ValueError("can not decide data size.")
        self.order = order
        self.delay_order = delay_order
        if "dpl" not in self.measurements.keys():
            self.measurements["dpl"] = self.delays.derivative(fsample)
        self.dpl = self.measurements["dpl"].copy()

        # construct other channels
        self.Y1_plus = self.index_cycle(self.X1_plus)
        self.Y1_minus = self.index_cycle(self.X1_minus)
        self.Z1_plus = self.index_cycle(self.Y1_plus)
        self.Z1_minus = self.index_cycle(self.Y1_minus)
        self.Y2_plus = self.index_cycle(self.X2_plus)
        self.Y2_minus = self.index_cycle(self.X2_minus)
        self.Z2_plus = self.index_cycle(self.Y2_plus)
        self.Z2_minus = self.index_cycle(self.Y2_minus)

        self.beta1_plus = self.index_cycle(self.alpha1_plus)
        self.beta1_minus = self.index_cycle(self.alpha1_minus)
        self.gamma1_plus = self.index_cycle(self.beta1_plus)
        self.gamma1_minus = self.index_cycle(self.beta1_minus)
        self.beta2_plus = self.index_cycle(self.alpha2_plus)
        self.beta2_minus = self.index_cycle(self.alpha2_minus)
        self.gamma2_plus = self.index_cycle(self.beta2_plus)
        self.gamma2_minus = self.index_cycle(self.beta2_minus)

    def CalculateXi(self, doppler=True, pool=None):
        """
        calculate intermediate variable xi
        """
        if doppler:
            d_args = dict(
                fsample=self.fsample,
                delay=self.delays,
                doppler=self.measurements["dpl"],
                order=self.order,
            )
        else:
            d_args = dict(fsample=self.fsample, delay=self.delays, order=self.order)
        ob_term = self.measurements["ref_c"] - self.measurements["tm_c"]
        delayed_reverse_ob_term = (ob_term.reverse()).timedelay(**d_args, pool=pool)
        xi = self.measurements["sci_c"]
        xi += 0.5 * ob_term
        xi += 0.5 * delayed_reverse_ob_term
        self.measurements["xi"] = xi
        # logger.info('Intervar Xi calculated.')

    def CalculateEta(self, doppler=True):
        """
        calculate intermediate variable eta
        """
        d = {}
        for k in left_MOSA_labels:
            k_re = k[1] + k[0]
            k_ad = adjacent_MOSA_labels[k_re]  # adjacent of reverse
            delayed_ref = timeshift(
                self.measurements["ref_c"][k_re] - self.measurements["ref_c"][k_ad],
                -self.fsample * self.delays[k],
                order=self.order,
            )
            if doppler:
                delayed_ref *= 1.0 - self.measurements["dpl"][k]
            d[k] = self.measurements["xi"][k] + 0.5 * delayed_ref
        for k in right_MOSA_labels:
            k_ad = adjacent_MOSA_labels[k]
            ref = self.measurements["ref_c"][k_ad] - self.measurements["ref_c"][k]
            d[k] = self.measurements["xi"][k] + 0.5 * ref
        self.measurements["eta"] = MOSADict(d)
        # logger.info('intervar Eta calculated.')

    def Path_string_interpret(self, Path_string):
        """
        interpret a virtual path as the combination of data
        """
        N_plus = len(Path_string)
        Path_string_inv = Path_string[::-1]
        delay_array = []
        for i in range(N_plus - 1):
            delay_array.append({})
            delay_array[i]["measurement"] = Path_string_inv[i] + Path_string_inv[i + 1]
            delay_array[i]["delay"] = []
            for j in range(i):
                delay_array[i]["delay"].append(Path_string_inv[j] + Path_string_inv[j + 1])
        return delay_array

    def index_cycle(self, Path_string):
        """
        get the path string of Y from X, Z from Y
        """
        cycle_pair = {1: 2, 2: 3, 3: 1}
        Path_string_cycled = ""
        for char in Path_string:
            idx = int(char)
            Path_string_cycled = Path_string_cycled + str(cycle_pair[idx])
        return Path_string_cycled

    def CalculatePath(self, Path_string, eta_dict, doppler=True):
        """
        calculate the result of a virtual path using the (eta) data and delays
        the path ends at current time
        """
        delay_array = self.Path_string_interpret(Path_string)
        N_measurement = len(delay_array)
        tdi = np.zeros(self.size)
        for i in range(N_measurement):
            N_delay = len(delay_array[i]["delay"])
            total_delay_time = np.zeros(self.size)
            for j in range(N_delay):
                delay_idx = delay_array[i]["delay"][j]
                total_delay_time = total_delay_time - timeshift(
                    self.delays[delay_idx],
                    total_delay_time * self.fsample,
                    order=self.delay_order,
                )  # delays can be calculated with low-order interpolation
            if doppler is True:
                total_delay_doppler = np.gradient(total_delay_time, 1.0 / self.fsample)
                tdi += (1.0 + total_delay_doppler) * timeshift(
                    eta_dict[delay_array[i]["measurement"]],
                    total_delay_time * self.fsample,
                    order=self.order,
                )
            else:
                tdi += timeshift(
                    eta_dict[delay_array[i]["measurement"]],
                    total_delay_time * self.fsample,
                    order=self.order,
                )
        return tdi

    def CalculatePathCombination(self, plus_paths, minus_paths, eta_dict, doppler=True):
        """
        plus path and minus path must be lists: e.g. plus path = [string1, string2, ...]
        """
        N_plus = len(plus_paths)
        N_minus = len(minus_paths)
        tdi_plus = np.zeros(self.size)
        tdi_minus = np.zeros(self.size)

        for i in range(N_plus):
            logger.debug("Calculating plus path " + plus_paths[i])
            tdi_plus += self.CalculatePath(Path_string=plus_paths[i], eta_dict=eta_dict, doppler=doppler)
        for i in range(N_minus):
            logger.debug("Calculating minus path " + minus_paths[i])
            tdi_minus += self.CalculatePath(Path_string=minus_paths[i], eta_dict=eta_dict, doppler=doppler)
        return tdi_plus - tdi_minus

    def CalculateNestedDelay(self, Delay_string, measurement, doppler=True):
        """
        calculate nested delay of any measurement
        """
        if Delay_string == "":
            return measurement
        N_delay = len(Delay_string) - 1
        delays = []
        for i in range(N_delay):
            delays.append(Delay_string[i] + Delay_string[i + 1])
        total_delay_time = np.zeros(self.size)
        for i in range(N_delay):
            delay_idx = delays[i]
            total_delay_time = total_delay_time - timeshift(
                self.delays[delay_idx],
                total_delay_time * self.fsample,
                order=self.delay_order,
            )
        if doppler:
            total_delay_doppler = np.gradient(total_delay_time, 1.0 / self.fsample)
            return (1.0 + total_delay_doppler) * timeshift(measurement, total_delay_time * self.fsample, order=self.order)
        else:
            return timeshift(measurement, total_delay_time * self.fsample, order=self.order)

    def CalculateNestedDelayTime(self, Delay_string):
        """
        calculate the total delay time of a multiple delay operator
        """
        N_delay = len(Delay_string) - 1
        delays = []
        for i in range(N_delay):
            delays.append(Delay_string[i] + Delay_string[i + 1])
        total_delay_time = np.zeros(self.size)
        for i in range(N_delay):
            delay_idx = delays[i]
            total_delay_time = total_delay_time - timeshift(
                self.delays[delay_idx],
                total_delay_time * self.fsample,
                order=self.delay_order,
            )
        return total_delay_time

    def CalculateNestedDelayCombination(self, Delay_string_plus, Delay_string_minus, measurement, doppler=True):
        N_plus = len(Delay_string_plus)
        N_minus = len(Delay_string_minus)
        plus_delay = np.zeros(self.size)
        minus_delay = np.zeros(self.size)
        for i in range(N_plus):
            plus_delay += self.CalculateNestedDelay(
                Delay_string=Delay_string_plus[i],
                measurement=measurement,
                doppler=doppler,
            )
        for i in range(N_minus):
            minus_delay += self.CalculateNestedDelay(
                Delay_string=Delay_string_minus[i],
                measurement=measurement,
                doppler=doppler,
            )
        return plus_delay - minus_delay

    def CalculateBasicTDI(self, channel="X2", doppler=True, channel_name="tdi_channel"):
        """
        channel can be either a string or a list
        1. string: 'X1', 'X2',...
        2. list: [plus_path, minus_path]
        3. to calculate a single path, set channel = [path, '']
        NOTE: calculatexi and calculateeta should be called before
        """
        if isinstance(channel, str):
            channel_name = channel
            if channel == "X1":
                plus_paths = [
                    self.X1_plus,
                ]
                minus_paths = [
                    self.X1_minus,
                ]
            elif channel == "X2":
                plus_paths = [
                    self.X2_plus,
                ]
                minus_paths = [
                    self.X2_minus,
                ]
            elif channel == "alpha1":
                plus_paths = [
                    self.alpha1_plus,
                ]
                minus_paths = [
                    self.alpha1_minus,
                ]
            elif channel == "alpha2":
                plus_paths = [
                    self.alpha2_plus,
                ]
                minus_paths = [
                    self.alpha2_minus,
                ]
            else:
                raise NotImplementedError("channel " + channel + " not implemented.")
        elif isinstance(channel, list):
            plus_paths = [
                channel[0],
            ]
            minus_paths = [
                channel[1],
            ]
        else:
            raise ValueError("channel should be string or list.")

        self.measurements[channel_name] = self.CalculatePathCombination(
            plus_paths=plus_paths,
            minus_paths=minus_paths,
            eta_dict=self.measurements["eta"],
            doppler=doppler,
        )
        logger.info("TDI channel " + channel_name + " calculated.")

    def CalculateClockTDI(
        self,
        channel="X2",
        doppler=True,
        modulation_correction=False,
        channel_name="tdi_channel",
    ):
        """
        channel should be string or list:
        1. string: 'X1', 'X2', ...
        2. list: [plus_path, minus_path]
        3. to calculate a single path, set channel = [path, '']
        the clock correction term will be named as channel_name + '_q'
        """
        if isinstance(channel, str):
            channel_name = channel
            if channel == "X1":
                plus_array = self.Path_string_interpret(self.X1_plus)
                minus_array = self.Path_string_interpret(self.X1_minus)
            elif channel == "X2":
                plus_array = self.Path_string_interpret(self.X2_plus)
                minus_array = self.Path_string_interpret(self.X2_minus)
            elif channel == "alpha1":
                plus_array = self.Path_string_interpret(self.alpha1_plus)
                minus_array = self.Path_string_interpret(self.alpha1_minus)
            elif channel == "alpha2":
                plus_array = self.Path_string_interpret(self.alpha2_plus)
                minus_array = self.Path_string_interpret(self.alpha2_minus)
            else:
                raise NotImplementedError("TDI channel " + channel + " not implemented.")
        elif isinstance(channel, list):
            plus_array = self.Path_string_interpret(channel[0])
            minus_array = self.Path_string_interpret(channel[1])
        else:
            raise ValueError("channel should be string or list.")

        # calculate r
        nu_m = MOSADict(modulation_freqs)
        r = (self.measurements["sci_sb"] - self.measurements["sci_c"]) / (nu_m.reverse())
        if modulation_correction:
            dM = {}
            for i, j, k in ["123", "231", "312"]:
                dM[i] = (self.measurements["ref_sb"][i + k] - self.measurements["ref_c"][i + k] - self.measurements["ref_sb"][i + j] + self.measurements["ref_c"][i + j]) / 2.0
            for i, j, k in ["123", "231", "312"]:
                delayed_dM = timeshift(dM[j], -self.fsample * self.delays[i + j], order=self.order)
                if doppler:
                    delayed_dM *= 1.0 - self.measurements["dpl"][i + j]
                r[i + j] += delayed_dM / nu_m[j + i]
                r[i + k] += -dM[i] / nu_m[k + i]
            logger.debug("modulation correction built.")
        logger.debug("intervar r calculated.")

        # construct TDI P arrays, each term will be interpreted as the path of q
        plus_array_summed = {key: [] for key in MOSA_labels}
        minus_array_summed = {key: [] for key in MOSA_labels}

        for i in range(len(plus_array)):
            tmp = plus_array[i]["delay"]
            tmp1 = ""
            for j in range(len(tmp)):
                if j == 0:
                    tmp1 += tmp[j]
                else:
                    tmp1 += tmp[j][1]
            plus_array_summed[plus_array[i]["measurement"]].append(tmp1)

        for i in range(len(minus_array)):
            tmp = minus_array[i]["delay"]
            tmp1 = ""
            for j in range(len(tmp)):
                if j == 0:
                    tmp1 += tmp[j]
                else:
                    tmp1 += tmp[j][1]
            minus_array_summed[minus_array[i]["measurement"]].append(tmp1)

        # calculate Pr
        Pr = {key: np.zeros(self.size) for key in MOSA_labels}
        for key in MOSA_labels:
            for delay in plus_array_summed[key]:
                if delay == "":
                    Pr[key] += r[key]
                else:
                    Pr[key] += self.CalculateNestedDelay(Delay_string=delay, measurement=r[key], doppler=doppler)
            for delay in minus_array_summed[key]:
                if delay == "":
                    Pr[key] += -r[key]
                else:
                    Pr[key] += -self.CalculateNestedDelay(Delay_string=delay, measurement=r[key], doppler=doppler)
        Pr = MOSADict(Pr)
        logger.debug("intervar P * r calculated.")

        # calculate R
        R = {key: np.zeros(self.size) for key in MOSA_labels}
        for key in MOSA_labels:
            paths = plus_array_summed[key]
            for path in paths:
                if path != "":
                    R[key] += self.CalculatePath(Path_string=path[::-1], eta_dict=r, doppler=doppler)
            paths = minus_array_summed[key]
            for path in paths:
                if path != "":
                    R[key] += -self.CalculatePath(Path_string=path[::-1], eta_dict=r, doppler=doppler)
        R = MOSADict(R)
        logger.debug("intervar R calculated.")

        # construct clock correction
        clock_correction = np.zeros(self.size)
        for i, j, k in ["123", "231", "312"]:
            clock_correction += (self.measurements["b"][j + k] - self.measurements["a"][i + j]) * R[i + j] - (self.measurements["b"][i + j] + self.measurements["a"][i + k]) * R[i + k] + self.measurements["b"][j + k] * Pr[i + j]
        self.measurements[channel_name + "_q"] = clock_correction
        logger.info("clock correction for channel " + channel_name + " calculated.")
        # return clock_correction

    def CalculateFullTDI(
        self,
        channel="X2",
        channel_name="tdi_channel",
        doppler=True,
        clock_correction=True,
        garbage_time=None,
    ):
        if isinstance(channel, str):
            channel_name = channel
        self.CalculateXi(doppler=doppler)
        self.CalculateEta(doppler=doppler)
        self.CalculateBasicTDI(channel=channel, doppler=doppler, channel_name=channel_name)
        if clock_correction:
            self.CalculateClockTDI(channel=channel, doppler=doppler, channel_name=channel_name)
            self.measurements[channel_name + "_c"] = self.measurements[channel_name] - self.measurements[channel_name + "_q"]
        if garbage_time is not None:
            drop_points = int(garbage_time * self.fsample)
            self.measurements[channel_name] = self.measurements[channel_name][drop_points:]
            if clock_correction:
                self.measurements[channel_name + "_c"] = self.measurements[channel_name + "_c"][drop_points:]

    def FastMichelson(self, doppler=True, channel="X"):
        """
        calculate the 2nd-generation Michelson TDI channels with an optimized algorithm
        """
        if channel in ["X", "XYZ", "AET"]:
            X0plus = (
                self.CalculateNestedDelay(
                    Delay_string="12",
                    measurement=self.measurements["eta"]["21"],
                    doppler=doppler,
                )
                + self.measurements["eta"]["12"]
            )
            X0minus = (
                self.CalculateNestedDelay(
                    Delay_string="13",
                    measurement=self.measurements["eta"]["31"],
                    doppler=doppler,
                )
                + self.measurements["eta"]["13"]
            )
            X0 = X0plus - X0minus
            X1plus = self.CalculateNestedDelay(Delay_string="131", measurement=X0plus, doppler=doppler) + X0minus
            X1minus = self.CalculateNestedDelay(Delay_string="121", measurement=X0minus, doppler=doppler) + X0plus
            X1 = X1plus - X1minus
            X2plus = self.CalculateNestedDelay(Delay_string="12131", measurement=X1plus, doppler=doppler) + X1minus
            X2minus = self.CalculateNestedDelay(Delay_string="13121", measurement=X1minus, doppler=doppler) + X1plus
            X2 = X2plus - X2minus
            self.measurements["X0"] = X0
            self.measurements["X1"] = X1
            self.measurements["X2"] = X2
            if channel in ["XYZ", "AET"]:
                X0plus = (
                    self.CalculateNestedDelay(
                        Delay_string="23",
                        measurement=self.measurements["eta"]["32"],
                        doppler=doppler,
                    )
                    + self.measurements["eta"]["23"]
                )
                X0minus = (
                    self.CalculateNestedDelay(
                        Delay_string="21",
                        measurement=self.measurements["eta"]["12"],
                        doppler=doppler,
                    )
                    + self.measurements["eta"]["21"]
                )
                Y0 = X0plus - X0minus
                X1plus = self.CalculateNestedDelay(Delay_string="212", measurement=X0plus, doppler=doppler) + X0minus
                X1minus = self.CalculateNestedDelay(Delay_string="232", measurement=X0minus, doppler=doppler) + X0plus
                Y1 = X1plus - X1minus
                X2plus = self.CalculateNestedDelay(Delay_string="23212", measurement=X1plus, doppler=doppler) + X1minus
                X2minus = self.CalculateNestedDelay(Delay_string="21232", measurement=X1minus, doppler=doppler) + X1plus
                Y2 = X2plus - X2minus
                self.measurements["Y0"] = Y0
                self.measurements["Y1"] = Y1
                self.measurements["Y2"] = Y2
                X0plus = (
                    self.CalculateNestedDelay(
                        Delay_string="31",
                        measurement=self.measurements["eta"]["13"],
                        doppler=doppler,
                    )
                    + self.measurements["eta"]["31"]
                )
                X0minus = (
                    self.CalculateNestedDelay(
                        Delay_string="32",
                        measurement=self.measurements["eta"]["23"],
                        doppler=doppler,
                    )
                    + self.measurements["eta"]["32"]
                )
                Z0 = X0plus - X0minus
                X1plus = self.CalculateNestedDelay(Delay_string="323", measurement=X0plus, doppler=doppler) + X0minus
                X1minus = self.CalculateNestedDelay(Delay_string="313", measurement=X0minus, doppler=doppler) + X0plus
                Z1 = X1plus - X1minus
                X2plus = self.CalculateNestedDelay(Delay_string="31323", measurement=X1plus, doppler=doppler) + X1minus
                X2minus = self.CalculateNestedDelay(Delay_string="32313", measurement=X1minus, doppler=doppler) + X1plus
                Z2 = X2plus - X2minus
                self.measurements["Z0"] = Z0
                self.measurements["Z1"] = Z1
                self.measurements["Z2"] = Z2
                if channel == "AET":
                    A0, E0, T0 = AETfromXYZ(X0, Y0, Z0)
                    A1, E1, T1 = AETfromXYZ(X1, Y1, Z1)
                    A2, E2, T2 = AETfromXYZ(X2, Y2, Z2)
                    self.measurements["A0"] = A0
                    self.measurements["E0"] = E0
                    self.measurements["T0"] = T0
                    self.measurements["A1"] = A1
                    self.measurements["E1"] = E1
                    self.measurements["T1"] = T1
                    self.measurements["A2"] = A2
                    self.measurements["E2"] = E2
                    self.measurements["T2"] = T2
        else:
            raise ValueError("Only X, XYZ or AET supported.")

    def CalculateTDIFromPStrings(self, PStrings, doppler=True):
        """
        args:
            eta, delays, PStrings should be MOSADicts
            shapes of each item:
            - eta: (Nt)
            - delays: (Nt)
            - PStrings: [(1, ["ij", "-jk"]), (-1, []), ...]
        """
        eta = self.measurements["eta"]  # MOSADict, each item (Nt)
        delays = self.delays  # MOSADict, each item (Nt)
        Nt = len(eta["12"])
        TDICombination = np.zeros(Nt)
        for key in MOSA_labels:
            TDIContributer_ij = np.zeros(Nt)
            for prefactor, DStrings in PStrings[key]:
                delayed_eta_ij = eta[key]
                for DString in DStrings[::-1]:
                    if DString[0] == "-":
                        delay_sign = 1.0
                    else:
                        delay_sign = -1.0
                    if doppler is True:
                        tmp_delay = delays[DString[-2] + DString[-1]]
                        tmp_doppler = np.gradient(tmp_delay) * self.fsample
                        delayed_eta_ij = timeshift(
                            data=delayed_eta_ij,
                            shifts=delay_sign * self.fsample * tmp_delay,
                            order=self.order,
                        ) * (1.0 - tmp_doppler)
                    else:
                        delayed_eta_ij = timeshift(
                            data=delayed_eta_ij,
                            shifts=delay_sign * self.fsample * delays[DString[-2] + DString[-1]],
                            order=self.order,
                        )
                TDIContributer_ij += prefactor * delayed_eta_ij
            TDICombination += TDIContributer_ij
        return TDICombination


class TDIStringManipulation:
    replacement_rule = {"1": "2", "2": "3", "3": "1"}

    @classmethod
    def replace_string(cls, s):
        return "".join(cls.replacement_rule[char] for char in s)

    @classmethod
    def replace_in_list(cls, lst):
        return [cls.replace_string(item) for item in lst]

    @classmethod
    def TDIStringCyc(cls, data):
        new_dict = {}
        for key, value in data.items():
            new_key = cls.replace_string(key)

            new_value = []
            for v in value:
                coefficient, str_list = v
                new_value.append((coefficient, cls.replace_in_list(str_list)))

            new_dict[new_key] = new_value
        return new_dict

    @classmethod
    def TDIStringScalarProduct(cls, X2_strings, a):
        new_X2_strings = dict()
        for key, value in X2_strings.items():
            new_X2_strings[key] = [(coef * a, seq) for coef, seq in value]
        return new_X2_strings

    @classmethod
    def AETStringsfromXString(cls, X2_strings):
        Y2_strings = cls.TDIStringCyc(X2_strings)
        Z2_strings = cls.TDIStringCyc(Y2_strings)
        A2_strings = dict()
        E2_strings = dict()
        T2_strings = dict()
        for key in MOSA_labels:
            A2_strings[key] = cls.TDIStringScalarProduct(X2_strings, -1.0 / np.sqrt(2.0))[key] + cls.TDIStringScalarProduct(Z2_strings, 1.0 / np.sqrt(2.0))[key]
            E2_strings[key] = cls.TDIStringScalarProduct(X2_strings, 1.0 / np.sqrt(6.0))[key] + cls.TDIStringScalarProduct(Y2_strings, -2.0 / np.sqrt(6.0))[key] + cls.TDIStringScalarProduct(Z2_strings, 1.0 / np.sqrt(6.0))[key]
            T2_strings[key] = cls.TDIStringScalarProduct(X2_strings, 1.0 / np.sqrt(3.0))[key] + cls.TDIStringScalarProduct(Y2_strings, 1.0 / np.sqrt(3.0))[key] + cls.TDIStringScalarProduct(Z2_strings, 1.0 / np.sqrt(3.0))[key]
        return A2_strings, E2_strings, T2_strings


def AETfromXYZ(X, Y, Z):
    A = (Z - X) / np.sqrt(2.0)
    E = (X - 2.0 * Y + Z) / np.sqrt(6.0)
    T = (X + Y + Z) / np.sqrt(3.0)
    return A, E, T


def XYZfromAET(A, E, T):
    X = (-np.sqrt(3.0) * A + E + np.sqrt(2.0) * T) / np.sqrt(6.0)
    Y = (T - np.sqrt(2.0) * E) / np.sqrt(3.0)
    Z = (np.sqrt(3.0) * A + E + np.sqrt(2.0) * T) / np.sqrt(6.0)
    return X, Y, Z


def RangingProcessing(m, modulation_correction=True, doppler=True, order=31, drop1=0, drop2=0):
    """
    process ranging information for the ``TDI without synchronization'' scheme
    """
    measurements = m.copy()
    raw_mpr = measurements["mpr"].copy()
    Delta = measurements["sci_c"] - measurements["sci_sb"]
    dpl = {}
    for i, j, k in ["123", "231", "312"]:
        dpl[i + j] = (Delta[i + j] + 1.0e6) / modulation_freqs[i + k]
        dpl[i + k] = (Delta[i + k] - 1.0e6) / modulation_freqs[i + j]
    measurements["dpl"] = MOSADict(dpl)

    int_dpl = measurements["dpl"].integrate(fsample=measurements["fsample"])
    res_dpl = raw_mpr - int_dpl
    d0 = {}
    for key in MOSA_labels:
        d0[key] = np.mean(res_dpl[key])
    d0 = MOSADict(d0)
    measurements["mpr"] = d0 + int_dpl

    if modulation_correction is True:
        dM = {}
        delta = measurements["ref_sb"] - measurements["ref_c"]
        for i, j, k in ["123", "231", "312"]:
            dM[i] = (delta[i + k] + 1.0e6) / 2.0 - (delta[i + j] - 1.0e6) / 2.0
        dM = SCDict(dM)
        for i, j, k in ["123", "231", "312"]:
            delayed_dM = timeshift(
                dM[j],
                -measurements["fsample"] * measurements["mpr"][i + j],
                order=order,
            )
            if doppler:
                delayed_dM *= 1.0 - measurements["dpl"][i + j]
            measurements["dpl"][i + j] += -delayed_dM / modulation_freqs[i + k]
            measurements["dpl"][i + k] += dM[i] / modulation_freqs[i + j]

        # refine the calculation of mpr
        int_dpl = measurements["dpl"].integrate(fsample=measurements["fsample"])
        res_dpl = raw_mpr - int_dpl
        d0 = {}
        for key in MOSA_labels:
            d0[key] = np.mean(res_dpl[key])
        d0 = MOSADict(d0)
        measurements["mpr"] = d0 + int_dpl

    # mpr and dpl will be used to calculate delays in tdi
    for key in [
        "mpr",
        # "ppr",
        "dpl",
        "sci_c",
        "sci_sb",
        "ref_c",
        "ref_sb",
        "tm_c",
        "time",
    ]:
        measurements[key] = measurements[key].drop_edge_points(drop1, drop2)
    return measurements


def TheoreticalGWTDI(y, channel="X2", fsample=4, T=10.0, order=5):
    """
    y = Interferometer.gw
    T is the norminal delay time
    """
    y = -y.copy()  # to match our TDI convention
    if channel == "X2":
        tmp1 = y["12"] + timeshift(y["21"], -fsample * T, order)
        tmp2 = y["13"] + timeshift(y["31"], -fsample * T, order)
        tmp3 = tmp1 + timeshift(tmp2, -fsample * 2.0 * T, order)
        tmp4 = tmp2 + timeshift(tmp1, -fsample * 2.0 * T, order)
        Xplus = tmp3 - timeshift(tmp3, -fsample * 4.0 * T, order)
        Xminus = tmp4 - timeshift(tmp4, -fsample * 4.0 * T, order)
        return Xplus - Xminus
    elif channel == "Y2":
        tmp1 = y["23"] + timeshift(y["32"], -fsample * T, order)
        tmp2 = y["21"] + timeshift(y["12"], -fsample * T, order)
        tmp3 = tmp1 + timeshift(tmp2, -fsample * 2.0 * T, order)
        tmp4 = tmp2 + timeshift(tmp1, -fsample * 2.0 * T, order)
        Yplus = tmp3 - timeshift(tmp3, -fsample * 4.0 * T, order)
        Yminus = tmp4 - timeshift(tmp4, -fsample * 4.0 * T, order)
        return Yplus - Yminus
    elif channel == "Z2":
        tmp1 = y["31"] + timeshift(y["13"], -fsample * T, order)
        tmp2 = y["32"] + timeshift(y["23"], -fsample * T, order)
        tmp3 = tmp1 + timeshift(tmp2, -fsample * 2.0 * T, order)
        tmp4 = tmp2 + timeshift(tmp1, -fsample * 2.0 * T, order)
        Zplus = tmp3 - timeshift(tmp3, -fsample * 4.0 * T, order)
        Zminus = tmp4 - timeshift(tmp4, -fsample * 4.0 * T, order)
        return Zplus - Zminus
    else:
        raise ValueError("channel not supported.")


def TDITPStoTCB(TDI_channel, SC_idx, time, fsample, orbit_class, order=31):
    tps_wrt_tcb = orbit_class.TPSwrtTCBfunctions[SC_idx](time)
    return timeshift(TDI_channel, tps_wrt_tcb * fsample, order=order)


class TDISensitivity:
    """
    calculate sensitivity for given orbit configuration and TDI scheme
    """

    def __init__(self, Ri, nij, dij, L=L_nominal, S_OMS=SOMS_nominal, S_ACC=SACC_nominal):
        """
        initialize with the orbit configuration.
        Args:
            Ri: SCDict, each item has shape (3,)
            nij: MosaDict, each item has shape (3,)
            dij: MosaDict, each item is a scalar
            L: nominal arm-length of the detector
            S_OMS: amplitude of optical metrology noise
            S_ACC: amplitude of test-mass acceleration noise
        """
        self.Ri, self.nij, self.dij = Ri, nij, dij
        R0 = (self.Ri["1"] + self.Ri["2"] + self.Ri["3"]) / 3.0  # (3,)
        self.qi = dict()
        for key in SC_labels:
            self.qi[key] = self.Ri[key] - R0
        self.qi = SCDict(self.qi)
        self.L = L
        self.S_OMS = S_OMS
        self.S_ACC = S_ACC

    def wave_vector(self, lam, beta):
        """
        calculate the wavevector k of indicent GW.
        """
        return np.array([-np.cos(beta) * np.cos(lam), -np.cos(beta) * np.sin(lam), -np.sin(beta)])  # (3,)

    def polar_basis(self, lam, beta):
        """
        calculate the polarization basis e+, ex.
        """
        u = np.array([np.sin(lam), -np.cos(lam), 0.0])
        v = np.array([-np.sin(beta) * np.cos(lam), -np.sin(beta) * np.sin(lam), np.cos(beta)])
        ep = np.outer(u, u) - np.outer(v, v)
        ec = np.outer(u, v) + np.outer(v, u)
        return ep, ec  # (3, 3)

    def Prefactor_ij(self, f):
        """
        Args:
            f: array of frequencies, (Nf)
        Returns:
            prefactor_ij: MosaDict, each item (Nf)
        """
        tmp = -1.0j * np.pi * f
        prefactor_ij = dict()
        for key in MOSA_labels:
            prefactor_ij[key] = tmp * self.dij[key]  # (Nf,)
        return MOSADict(prefactor_ij)

    def Exp_ij(self, k, f):
        tmp = -1.0j * np.pi * f
        exp_ij = dict()
        for key in MOSA_labels:
            exp_ij[key] = np.exp(tmp * (self.dij[key] + np.dot(k, (self.qi[key[0]] + self.qi[key[1]]) / C)))  # (Nf,)
        return MOSADict(exp_ij)

    def Sinc_ij(self, k, f):
        sinc_ij = dict()
        for key in MOSA_labels:
            sinc_ij[key] = np.sinc(f * self.dij[key] * (1.0 - np.dot(k, self.nij[key])))  # (Nf,)
        return MOSADict(sinc_ij)

    def PatternFunction_ij(self, ep, ec):
        Fp_ij = dict()
        Fc_ij = dict()
        for key in MOSA_labels:
            Fp_ij[key] = np.dot(self.nij[key], np.matmul(ep, self.nij[key]))  # scalar
            Fc_ij[key] = np.dot(self.nij[key], np.matmul(ec, self.nij[key]))  # scalar
        return MOSADict(Fp_ij), MOSADict(Fc_ij)

    def TDI_P_ij(self, P_ij_strings, f):
        """
        calculate the frequency-domain P_ij opetators.
        Args:
            P_ij_strings is a dictionary, and each item should be like: "12": [(1., ["12", "21"]), (-1., ["-13"]), ...], repersenting [D_12 D_21 - A_13 ...], representing the P_ij opetator.
        Returns:
            P_ij: MosaDict, each item (Nf)
        """
        # calculate single delay operators
        tmp = -1.0j * TWOPI * f  # -2pif
        D_ij = dict()
        for key in MOSA_labels:
            D_ij[key] = np.exp(tmp * self.dij[key])  # (Nf)
        D_ij = MOSADict(D_ij)

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
        return MOSADict(P_ij)

    def TDI_response_function(self, lam, beta, f, P_ij=None, P_ij_strings=None):
        """
        calculate R from P_ij or the strings representing P_ij.
        """
        if P_ij is None:
            P_ij = self.TDI_P_ij(P_ij_strings=P_ij_strings, f=f)  # mosa, (Nf,)

        k = self.wave_vector(lam=lam, beta=beta)  # (3,)
        ep, ec = self.polar_basis(lam=lam, beta=beta)  # (3, 3)
        PreFactor = self.Prefactor_ij(f=f)  # mosa, (Nf,)
        ExpFactor = self.Exp_ij(k=k, f=f)  # mosa, (Nf,)
        SincFactor = self.Sinc_ij(k=k, f=f)  # mosa, (Nf,)
        Fp, Fc = self.PatternFunction_ij(ep=ep, ec=ec)  # mosa, scalar

        R2p = np.zeros_like(f, dtype=np.complex128)
        R2c = np.zeros_like(f, dtype=np.complex128)
        for key in MOSA_labels:
            tmp = P_ij[key] * PreFactor[key] * ExpFactor[key] * SincFactor[key]
            R2p += tmp * Fp[key]
            R2c += tmp * Fc[key]
        R2p = np.abs(R2p) ** 2  # (Nf)
        R2c = np.abs(R2c) ** 2
        return (R2p + R2c) / 2.0

    def TDI_noise_contributor(self, f, P_ij=None, P_ij_strings=None):
        """
        calculate oms and acc noise contributor for each MOSA from Pij or the strings representing Pij.
        """
        if P_ij is None:
            P_ij = self.TDI_P_ij(P_ij_strings=P_ij_strings, f=f)

        tmp = -1.0j * TWOPI * f  # -2pif
        D_ij = dict()
        for key in MOSA_labels:
            D_ij[key] = np.exp(tmp * self.dij[key])  # (Nf)
        D_ij = MOSADict(D_ij)

        noise_oms_ij = dict()
        noise_acc_ij = dict()
        for key in MOSA_labels:
            noise_oms_ij[key] = np.abs(P_ij[key]) ** 2  # (Nf)
            key_inverse = key[1] + key[0]
            noise_acc_ij[key] = np.abs(P_ij[key] + D_ij[key_inverse] * P_ij[key_inverse]) ** 2  # (Nf)
        return noise_oms_ij, noise_acc_ij

    def TDI_noise(self, f, P_ij=None, P_ij_strings=None):
        """
        calculate total noise PSD from Pij or the Pij strings in the fractional frequency difference unit
        """
        PSDFunction = InstrumentalPSDs(L=self.L)
        noise_oms_ij, noise_acc_ij = self.TDI_noise_contributor(f=f, P_ij=P_ij, P_ij_strings=P_ij_strings)
        PSDOMS = PSDFunction.PSD_RO(f=f, sro=self.S_OMS)
        PSDACC = PSDFunction.PSD_ACC(f=f, sacc=self.S_ACC)
        total_noise = np.zeros_like(f)
        for key in MOSA_labels:
            total_noise += noise_oms_ij[key] * PSDOMS + noise_acc_ij[key] * PSDACC  # (Nf)
        return total_noise

    def TDI_sensitivity(self, f, P_ij=None, P_ij_strings=None, Nsource=1024):
        """
        calculate sensitivity from Pij or the Pij strings. the response funciton is averaged over N sources in random directions.
        """
        # calculate Pij if absent
        if P_ij is None:
            P_ij = self.TDI_P_ij(P_ij_strings=P_ij_strings, f=f)  # mosa, (Nf,)

        # calculate instrumental noise PSD
        PSD = self.TDI_noise(f=f, P_ij=P_ij)  # (Nf)

        # calculate the average response function
        Response_arr = []
        for _ in tqdm(range(Nsource)):
            longitude = np.random.uniform(0, TWOPI)
            latitude = np.arcsin(np.random.uniform(-1, 1))
            Response_arr.append(self.TDI_response_function(lam=longitude, beta=latitude, f=f, P_ij=P_ij))
        Response_arr = np.array(Response_arr)  # (Nsource, Nf)
        Response_avg = np.mean(Response_arr, axis=0)  # (Nf)

        # return sensitivity
        return PSD / Response_avg

    def TDI_noise_CSD(self, f, P_ij=None, Q_ij=None, P_ij_strings=None, Q_ij_strings=None, return_PSD=False):
        """calculate the noise CSD of channel P and Q, i.e. 2/T <P^* Q>, reduce to PSD if P = Q"""
        if P_ij is None:
            P_ij = self.TDI_P_ij(P_ij_strings=P_ij_strings, f=f)
        if Q_ij is None:
            Q_ij = self.TDI_P_ij(P_ij_strings=Q_ij_strings, f=f)

        Dij = dict()
        for key in MOSA_labels:
            Dij[key] = np.exp(-1.0j * TWOPI * f * self.dij[key])

        PSDFunction = InstrumentalPSDs(L=self.L)
        PSDOMS = PSDFunction.PSD_RO(f=f, sro=self.S_OMS)
        PSDACC = PSDFunction.PSD_ACC(f=f, sacc=self.S_ACC)

        CSD = np.zeros_like(f, dtype=np.complex128)
        for key in MOSA_labels:
            invk = key[1] + key[0]
            CSD += np.conjugate(P_ij[key]) * Q_ij[key] * PSDOMS + np.conjugate(P_ij[key] + P_ij[invk] * Dij[invk]) * (Q_ij[key] + Q_ij[invk] * Dij[invk]) * PSDACC

        if return_PSD:
            return np.real(CSD)
        else:
            return CSD
