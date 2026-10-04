"""Scientific integration checks: python -m unittest discover -s Test -p 'test_simulation_backend.py'."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from simulation_backend import default_config, validate_config, simulate, save_result, load_result


class SimulationTests(unittest.TestCase):
    def config(self, **overrides):
        p = default_config()
        p["data_time_days"] = 0.1
        p.update(overrides)
        return p

    def test_validation(self):
        for change in ({"fsample": 0}, {"seed": -1}, {"interp_order": 8}, {"trim_seconds": 90000},
                       {"data_time_days": float("nan")}, {"tdi_channel": "X2"},
                       {"add_glitch": True, "glitch_time": 100000}, {"gb": {"f0": 1}},
                       {"sgwb_enable": True, "sgwb": {"nside": 0}}, {"time_frame": "ClockTime"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(self.config(**change))

    def test_reproducibility_and_export(self):
        config = self.config()
        snapshot = copy.deepcopy(config)
        a, meta = simulate(config)
        b, _ = simulate(config)
        self.assertEqual(config, snapshot)
        self.assertEqual(meta["config"]["gb"]["N"], 1)
        self.assertEqual(meta["samples"], 664)
        self.assertEqual(a["time_tcb_s"][0], 1000)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        c, _ = simulate(self.config(seed=43))
        self.assertGreater(np.linalg.norm(a["A2"]-c["A2"]), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.npz"
            save_result(path, a, meta)
            loaded, metadata = load_result(path)
            self.assertEqual(meta, metadata)
            self.assertIn("sci_c_12", loaded)
            self.assertIn("ltt_12", loaded)
            for key in a:
                np.testing.assert_array_equal(a[key], loaded[key])

    def test_linear_signal_plus_noise(self):
        signal, _ = simulate(self.config(acc_noise=False, ro_noise=False))
        noise, _ = simulate(self.config(gb_enable=False))
        mixed, _ = simulate(self.config())
        for channel in ("A2", "E2", "T2", "sci_c_12"):
            error = np.linalg.norm(mixed[channel] - signal[channel] - noise[channel])
            self.assertLess(error / np.linalg.norm(mixed[channel]), 1e-10)

    def test_tdi_method_agreement(self):
        results = {}
        for method in ("fast_michelson", "channel_name", "channel_path", "channel_operator"):
            p = self.config(acc_noise=False, ro_noise=False, tdi_method=method,
                            tdi_channel="X" if method == "fast_michelson" else "X2")
            arrays, _ = simulate(p)
            results[method] = arrays["X2"]
        reference = results["fast_michelson"]
        for method, data in results.items():
            with self.subTest(method=method):
                self.assertLess(np.linalg.norm(data-reference) / np.linalg.norm(reference), 1e-10)
        np.testing.assert_array_equal(results["channel_name"], results["channel_path"])

    def test_glitch_without_stochastic_acc(self):
        p = self.config(gb_enable=False, acc_noise=False, ro_noise=False)
        empty, _ = simulate(p)
        np.testing.assert_array_equal(empty["A2"], 0)
        p.update(add_glitch=True, glitch_time=4000)
        glitch, _ = simulate(p)
        self.assertGreater(np.max(np.abs(glitch["A2"])), 0)
        before = glitch["time_tcb_s"] < 3000
        np.testing.assert_array_equal(glitch["A2"][before], 0)

    def test_all_noise_switches(self):
        from simulation_backend import NOISE_KEYS
        p = self.config(gb_enable=False, **{key: True for key in NOISE_KEYS})
        arrays, _ = simulate(p)
        self.assertTrue(all(np.isfinite(value).all() for value in arrays.values()))

    def test_other_channels_and_no_trim(self):
        for method, channel, expected in (("fast_michelson", "XYZ", ["X2", "Y2", "Z2"]),
                                          ("channel_name", "X1", ["X1"]), ("channel_name", "alpha1", ["alpha1"]),
                                          ("channel_name", "alpha2", ["alpha2"])):
            with self.subTest(channel=channel):
                arrays, meta = simulate(self.config(tdi_method=method, tdi_channel=channel, trim_seconds=0))
                self.assertEqual(meta["channels"], expected)
                self.assertEqual(len(arrays[expected[0]]), 864)

    def test_waveform_types(self):
        for source in ("mbhb", "emri", "sgwb"):
            with self.subTest(source=source):
                p = self.config(gb_enable=False, acc_noise=False, ro_noise=False)
                p[source + "_enable"] = True
                if source == "mbhb":
                    p["mbhb"].update(coalescence_time=0.06)
                if source == "sgwb":
                    p["sgwb"]["nside"] = 1
                arrays, _ = simulate(p)
                self.assertGreater(np.linalg.norm(arrays["A2"]), 0)


if __name__ == "__main__":
    unittest.main()
