"""Optional desktop smoke test; requires Tk. python Test/test_gui_workflow.py"""

import tempfile
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from gui_app import TriangleSimulatorGUI
from simulation_backend import load_result


def main():
    app = TriangleSimulatorGUI()
    app.withdraw()
    errors = []
    stage = [0]
    deadline = time.monotonic() + 90

    def check():
        try:
            if time.monotonic() > deadline:
                raise TimeoutError("GUI workflow timed out")
            if stage[0] == 0:
                app.variables["data_time_days"].set("0.1")
                app._run()
                assert app.running
                stage[0] = 1
            elif stage[0] == 1 and not app.running:
                assert app.result is not None, app.log.get("1.0", "end")
                assert app.process is None and app.tempdir is None
                for mode in ("TDI 输出", "六链路测量", "TDI 前后对比"):
                    app.view.set(mode)
                    app._refresh_channels()
                    assert len(app.figure.axes) == 2
                    assert len(app.figure.axes[0].lines) > 0
                for variable in app.channel_vars.values():
                    variable.set(False)
                app._plot()
                app.variables["tdi_method"].set("channel_name")
                assert app.variables["tdi_channel"].get() == "X1"
                app.variables["tdi_method"].set("channel_operator")
                assert app.variables["tdi_channel"].get() == "X2"
                with tempfile.TemporaryDirectory() as directory:
                    output = str(Path(directory) / "gui_result.npz")
                    with patch("gui_app.filedialog.asksaveasfilename", return_value=output):
                        app._export()
                    _, metadata = load_result(output)
                    assert metadata["channels"] == ["A2", "E2", "T2"]
                    config = str(Path(directory) / "config.json")
                    with patch("gui_app.filedialog.asksaveasfilename", return_value=config):
                        app._save_config()
                    app.variables["seed"].set("111")
                    with patch("gui_app.filedialog.askopenfilename", return_value=config):
                        app._load_config()
                    assert app.variables["seed"].get() == "42"
                app._run()
                app._cancel()
                assert not app.running and app.process is None and app.result is not None
                app._run()
                stage[0] = 2
            elif stage[0] == 2 and not app.running:
                assert app.result[1]["channels"] == ["X2"]
                orbit_file = Path(app.variables["orbit_dir"].get()) / "SCP1.dat"
                days = len(np.loadtxt(orbit_file)) + 1
                app.variables["data_time_days"].set(str(days))
                app.variables["fsample"].set(str(1024 / (days * 86400)))
                app.variables["gb_enable"].set(False)
                app._run()
                stage[0] = 3
            elif stage[0] == 3 and not app.running:
                assert "仿真失败" in app.status.get()
                assert app.result[1]["channels"] == ["X2"]
                assert app.process is None
                print("GUI workflow passed: run, plots, channel methods, export, config, cancel, rerun, failure recovery.", flush=True)
                app._close()
                return
            app.after(100, check)
        except Exception as exc:
            errors.append(exc)
            app._close()

    app.after(250, check)
    app.mainloop()
    if errors:
        raise errors[0]


if __name__ == "__main__":
    main()
