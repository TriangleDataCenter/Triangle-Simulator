"""Tutorial 4 simulation pipeline, independent of the desktop UI."""

import copy
import json
import logging
import os
from pathlib import Path
import time
import traceback

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DAY = 86400.0
LINKS = ("12", "23", "31", "21", "32", "13")
NOISE_KEYS = ("acc_noise", "ro_noise", "laser_noise", "clock_noise", "ob_noise", "bl_noise", "op_noise", "ranging_noise", "modulation_noise")
METHOD_CHANNELS = {
    "fast_michelson": ("AET", "XYZ", "X"),
    "channel_name": ("X1", "X2", "alpha1", "alpha2"),
    "channel_path": ("X2",),
    "channel_operator": ("X2",),
}


def default_config():
    return {
        "fsample": 0.1, "data_time_days": 1.0, "interp_order": 31,
        "orbit_dir": str(ROOT / "OrbitData/MicroSateOrbitEclipticTCB"),
        "seed": 42, "trim_seconds": 1000.0, "acc_scale": 1.0, "ro_scale": 1.0,
        **{key: key in ("acc_noise", "ro_noise") for key in NOISE_KEYS},
        "gb_enable": True,
        "gb": {"A": 1.39879192e-22, "f0": 0.00984299, "fdot0": 4.31168731e-16,
               "phase0": 0.79960729, "inclination": 2.29680093, "longitude": 4.50950561,
               "latitude": -0.30076513, "psi": 3.08152927, "N": 1},
        "mbhb_enable": False,
        "mbhb": {"chirp_mass": 400000.0, "mass_ratio": 0.3640815267051051,
                 "spin_1z": 0.7899406190510447, "spin_2z": 0.015424596359292053,
                 "coalescence_time": 5.0, "coalescence_phase": 3.2535342675592966,
                 "luminosity_distance": 106039.10941463517, "inclination": 1.4495086550199021,
                 "longitude": 3.032776598141215, "latitude": -1.1817968279048852,
                 "psi": 0.2409146690555858, "approximant": "SEOBNRv4_opt"},
        "emri_enable": False,
        "emri": {"file": str(ROOT / "GWData/Demo_EMRI_waveform_data.npz"),
                 "longitude": 4.5, "latitude": -0.3, "psi": 0.5},
        "sgwb_enable": False,
        "sgwb": {"Omega0": 1e-11, "f0": 3.162e-3, "n": -1.0, "nside": 5},
        "tdi_method": "fast_michelson", "tdi_channel": "AET", "tdi_interp": 31, "tdi_doppler": True,
        "add_glitch": False, "glitch_tm": "12", "glitch_time": 43200.0,
        "glitch_dv": 2.2e-12, "glitch_tau1": 10.0, "glitch_tau2": 11.0,
    }


def validate_config(config):
    """Return a validated copy; never change the saved parameter snapshot."""
    defaults = default_config()
    p = copy.deepcopy(defaults)
    for key, value in config.items():
        if key in defaults:
            if isinstance(defaults[key], dict) and isinstance(value, dict):
                p[key].update(value)
            else:
                p[key] = value
    if "data_time" in config and "data_time_days" not in config:
        p["data_time_days"] = config["data_time"] / DAY
    if config.get("time_frame", "ProperTime") != "ProperTime":
        raise ValueError("此流程使用 ProperTime → TCB；暂不支持 ClockTime 校正。")

    def check_types(values, template, prefix=""):
        for key, expected in template.items():
            value = values[key]
            name = prefix + key
            if isinstance(expected, dict):
                if not isinstance(value, dict):
                    raise ValueError(f"{name} 必须是参数对象")
                check_types(value, expected, name + ".")
            elif isinstance(expected, bool):
                if not isinstance(value, bool):
                    raise ValueError(f"{name} 必须是布尔值")
            elif isinstance(expected, (int, float)):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f"{name} 必须是有限数值")
                if isinstance(expected, int) and int(value) != value:
                    raise ValueError(f"{name} 必须是整数")
                values[key] = type(expected)(value)
            elif not isinstance(value, str):
                raise ValueError(f"{name} 必须是文本")

    check_types(p, defaults)
    if p["fsample"] <= 0 or p["data_time_days"] <= 0:
        raise ValueError("采样率和仿真时长必须大于 0")
    n = int(p["fsample"] * p["data_time_days"] * DAY)
    if not 512 <= n <= 2_000_000:
        raise ValueError("单次仿真样本数须为 512–2,000,000，请调整时长或采样率")
    for key in ("interp_order", "tdi_interp"):
        if p[key] not in (7, 15, 31):
            raise ValueError("插值阶数请选择 7、15 或 31")
    if not 0 <= p["seed"] <= 2**32 - 1:
        raise ValueError("随机种子须在 0–4294967295 之间")
    if p["trim_seconds"] < 0 or n - 2 * int(p["trim_seconds"] * p["fsample"]) < 256:
        raise ValueError("裁边后至少保留 256 个样本，请增加时长或减少裁边")
    if p["acc_scale"] < 0 or p["ro_scale"] < 0:
        raise ValueError("噪声幅度倍率不能为负")
    if p["tdi_method"] not in METHOD_CHANNELS or p["tdi_channel"] not in METHOD_CHANNELS[p["tdi_method"]]:
        raise ValueError("TDI 方法与通道不匹配")
    for group in ("gb", "mbhb", "emri"):
        if not p[group + "_enable"]:
            continue
        if not -np.pi / 2 <= p[group]["latitude"] <= np.pi / 2:
            raise ValueError(f"{group}: 纬度须在 -π/2 至 π/2 之间（弧度）")
        if group != "emri" and not 0 <= p[group]["inclination"] <= np.pi:
            raise ValueError(f"{group}: 倾角须在 0 至 π 之间（弧度）")
    if p["gb_enable"]:
        g = p["gb"]
        f_end = g["f0"] + g["fdot0"] * p["data_time_days"] * DAY
        margin = 0.001 * g["f0"] if g["N"] > 1 else 0
        if g["A"] < 0 or min(g["f0"], f_end) - margin <= 0 or max(g["f0"], f_end) + margin >= p["fsample"] / 2:
            raise ValueError("GB 幅度须非负，整个观测期的频率须在 0 与 Nyquist 频率之间")
        if not 1 <= g["N"] <= 1000:
            raise ValueError("GB 数目须为 1–1000")
    if p["mbhb_enable"]:
        m = p["mbhb"]
        if m["chirp_mass"] <= 0 or m["luminosity_distance"] <= 0 or m["coalescence_time"] <= 0 or not 0 < m["mass_ratio"] <= 1:
            raise ValueError("MBHB 质量、距离和并合时间须为正，质量比须在 (0, 1] 内")
        if abs(m["spin_1z"]) >= 1 or abs(m["spin_2z"]) >= 1:
            raise ValueError("MBHB 自旋须在 (-1, 1) 内")
        if m["approximant"] not in ("IMRPhenomD", "IMRPhenomT", "SEOBNRv4_opt"):
            raise ValueError("不支持的 MBHB 波形模型")
    if p["sgwb_enable"]:
        s = p["sgwb"]
        if not 1 <= s["nside"] <= 8 or s["Omega0"] <= 0 or s["f0"] <= 0:
            raise ValueError("SGWB 的 NSIDE 须为 1–8，Omega0 和参考频率须为正")
        if n * 12 * s["nside"]**2 > 30_000_000:
            raise ValueError("SGWB 波形规模过大，请减少 NSIDE、时长或采样率")
    if p["add_glitch"]:
        if p["glitch_tm"] not in LINKS or not 0 <= p["glitch_time"] < p["data_time_days"] * DAY:
            raise ValueError("Glitch 注入位置或时间无效")
        if min(p["glitch_tau1"], p["glitch_tau2"]) <= 0 or p["glitch_tau1"] == p["glitch_tau2"]:
            raise ValueError("Glitch 的两个时间常数必须为正且不相等")
    for group, key in ((None, "orbit_dir"), ("emri", "file")):
        if group and not p["emri_enable"]:
            continue
        owner = p[group] if group else p
        path = Path(owner[key]).expanduser()
        if not path.is_absolute():
            path = Path(__file__).parent / path
        path = path.resolve()
        if not path.exists():
            raise ValueError(f"路径不存在：{path}")
        owner[key] = str(path)
    for sc in "123":
        for kind in ("SCP", "SCV"):
            if not (Path(p["orbit_dir"]) / f"{kind}{sc}.dat").is_file():
                raise ValueError(f"轨道目录缺少 {kind}{sc}.dat")
    return p


def simulate(config, report=lambda progress, message: None):
    """Run the same decomposed-measurement -> TCB -> Xi/Eta -> TDI path as Tutorial 4."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from Triangle.Constants import F_LASER, H0, MPC, L_nominal
    from Triangle.GW import Initialize_GW_response, GeneralWaveform, GW
    from Triangle.Glitch import Glitch
    from Triangle.Interferometer import Interferometers, TPStoTCB
    from Triangle.Noise import InstrumentalPSDs, GeneralNoise
    from Triangle.Offset import OffsetFunctions
    from Triangle.Orbit import Orbit
    from Triangle.TDI import TDI

    started = time.monotonic()
    p = validate_config(config)
    np.random.seed(p["seed"])
    fs = p["fsample"]
    size = int(p["data_time_days"] * DAY * fs)
    report(5, "载入数值轨道")
    orbit = Orbit(OrbitDir=p["orbit_dir"])
    if (size - 1) / fs > orbit.tdata[-1]:
        raise ValueError(f"仿真超出轨道范围（最大 {orbit.tdata[-1] / DAY:g} 天）")
    gw = []
    report(12, "生成引力波信号")
    if p["gb_enable"]:
        g = dict(p["gb"])
        count = g.pop("N")
        for _ in range(count):
            source = dict(g)
            if count > 1:
                source.update(f0=g["f0"] * np.random.uniform(0.999, 1.001), longitude=np.random.uniform(0, 2*np.pi),
                              latitude=np.arcsin(np.random.uniform(-1, 1)), psi=np.random.uniform(0, np.pi))
            gw.extend(Initialize_GW_response(signal_type="GB", parameters=source, orbit=orbit))
    if p["mbhb_enable"]:
        source = dict(p["mbhb"])
        approx = source.pop("approximant")
        gw.extend(Initialize_GW_response(signal_type="MBHB", parameters=source, orbit=orbit, approximant=approx))
    if p["emri_enable"]:
        source = dict(p["emri"])
        filename = source.pop("file")
        with np.load(filename, allow_pickle=False) as data:
            t, hp, hc = (np.asarray(data[key]) for key in ("t", "h1", "h2"))
            if t.ndim != 1 or hp.shape != t.shape or hc.shape != t.shape or len(t) < 6 or not np.all(np.diff(t) > 0):
                raise ValueError("EMRI 文件须含等长一维 t、h1、h2，且 t 严格递增")
            if not all(np.isfinite(a).all() for a in (t, hp, hc)):
                raise ValueError("EMRI 波形含非有限数值")
            gw.extend(Initialize_GW_response(signal_type="general", parameters=source, orbit=orbit, data=[[t, hp, -hc]]))
    if p["sgwb_enable"]:
        import healpy as hp
        s = p["sgwb"]
        npix = hp.nside2npix(s["nside"])
        colat, lon = hp.pix2ang(s["nside"], np.arange(npix))
        generator = GeneralNoise(lambda f: 3 * H0**2 / (4*np.pi**2*f**3) * s["Omega0"] * (f/s["f0"])**s["n"] * 1e6 / MPC**2)
        t = np.arange(size) / fs
        for i in range(npix):
            wf = GeneralWaveform(tdata=t, hpdata=generator(fs, size)/np.sqrt(npix), hcdata=generator(fs, size)/np.sqrt(npix))
            gw.append(GW(orbit=orbit, ext_params=[lon[i], np.pi/2-colat[i], 0.0], GWwaveform=wf))
        report(23, f"SGWB：{npix} 个天空方向")

    noise_flags = {key: p[key] for key in NOISE_KEYS}
    # A glitch still needs the acceleration coupling when stochastic ACC is off.
    noise_flags["acc_noise"] = p["acc_noise"] or p["add_glitch"]
    ifo = Interferometers(size=size, fsample=fs, time_frame="ProperTime", orbit_class=orbit,
                          offset_class=OffsetFunctions(laser_flag=False, clock_flag=False),
                          noise_class=InstrumentalPSDs(L=L_nominal), gw_class=gw or None,
                          fplan=None, order=p["interp_order"], pool=None, clean_memory=True, **noise_flags)
    report(28, "生成仪器噪声")
    ifo.SimulateBasicNoise()
    if noise_flags["acc_noise"]:
        for link in LINKS:
            ifo.BasicNoise["acc_noise"][link] *= p["acc_scale"] if p["acc_noise"] else 0.0
    if p["ro_noise"]:
        for name, values in ifo.BasicNoise.items():
            if name.startswith("ro_"):
                for link in LINKS:
                    values[link] *= p["ro_scale"]
    report(38, "计算测距与时钟")
    ifo.SimulateRangeAndClock()
    if p["add_glitch"]:
        glitch = Glitch(fsample=fs)
        link = p["glitch_tm"]
        acc = glitch.LPF_legacy_glitch_model(t=ifo.proper_time[link[0]], t0=p["glitch_time"],
                                            dv=p["glitch_dv"], tau1=p["glitch_tau1"], tau2=p["glitch_tau2"])
        ifo.BasicNoise["acc_noise"][link] += glitch.acc2ffd(acc)
        report(43, f"已注入 Glitch：TM {link}")
    if gw:
        report(46, f"计算 {len(gw)} 个信号的六链路响应")
        ifo.SimulateGW()
    report(57, "生成干涉仪测量")
    ifo.SimulateMeasurements()
    report(65, "ProperTime → TCB 同步")
    measurements = TPStoTCB(ifo.OutputMeasurements(mode="decomposed"), orbit_class=orbit, order=p["interp_order"])
    ifo.clean()
    tdi = TDI(measurements=measurements, delays=measurements["ltt"], fsample=fs, order=p["tdi_interp"])
    report(74, "TDI：计算 ξ 与 η")
    tdi.CalculateXi(doppler=p["tdi_doppler"])
    tdi.CalculateEta(doppler=p["tdi_doppler"])
    report(84, "TDI：组合延迟通道")
    method, channel = p["tdi_method"], p["tdi_channel"]
    if method == "fast_michelson":
        tdi.FastMichelson(doppler=p["tdi_doppler"], channel=channel)
        channels = {"AET": ["A2", "E2", "T2"], "XYZ": ["X2", "Y2", "Z2"], "X": ["X2"]}[channel]
    else:
        channels = [channel]
        if method == "channel_name":
            tdi.CalculateBasicTDI(channel=channel, doppler=p["tdi_doppler"])
        elif method == "channel_path":
            tdi.CalculateBasicTDI(channel=[TDI.X2_plus, TDI.X2_minus], channel_name="X2", doppler=p["tdi_doppler"])
        else:
            tdi.measurements["X2"] = tdi.CalculateTDIFromPStrings(TDI.X2_string, doppler=p["tdi_doppler"])
    trim = int(p["trim_seconds"] * fs)
    keep = slice(trim, -trim if trim else None)
    arrays = {"time_tcb_s": np.asarray(measurements["time"]["1"])[keep]}
    for name in ("sci_c", "sci_sb", "ref_c", "ref_sb", "tm_c", "ltt"):
        for link in LINKS:
            # Measurements are frequency differences in Hz; ltt stays in seconds.
            arrays[f"{name}_{link}"] = np.asarray(measurements[name][link])[keep] / (1 if name == "ltt" else F_LASER)
    for link in LINKS:
        arrays[f"eta_{link}"] = np.asarray(tdi.measurements["eta"][link])[keep] / F_LASER
    for name in channels:
        arrays[name] = np.asarray(tdi.measurements[name])[keep] / F_LASER
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("仿真产生非有限数值，请检查波形参数和插值设置")
    metadata = {"config": p, "channels": channels, "elapsed_seconds": time.monotonic()-started,
                "source_count": len(gw), "samples": len(arrays["time_tcb_s"]), "trim_samples_each_edge": trim,
                "measurement_unit": "fractional frequency (Hz / F_LASER)", "time_unit": "TCB seconds", "ltt_unit": "seconds",
                "tdi_convention": "Tutorial 4, native Triangle TDI methods"}
    report(98, "整理结果")
    return arrays, metadata


def save_result(path, arrays, metadata):
    np.savez_compressed(path, **arrays, metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))


def load_result(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        arrays = {key: data[key] for key in data.files if key != "metadata_json"}
    return arrays, metadata


def simulation_worker(config, events, output_path):
    """Spawn entry point. Only small status messages cross the process queue."""
    class QueueHandler(logging.Handler):
        def emit(self, record):
            events.put(("log", self.format(record)))

    handler = QueueHandler()
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logger = logging.getLogger("Triangle")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        arrays, metadata = simulate(config, lambda percent, message: events.put(("progress", percent, message)))
        save_result(output_path, arrays, metadata)
        events.put(("done", output_path))
    except Exception:
        events.put(("error", traceback.format_exc()))
    finally:
        logger.removeHandler(handler)
