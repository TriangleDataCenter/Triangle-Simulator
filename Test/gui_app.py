#!/usr/bin/env python3
"""Desktop workbench for Tutorial 4. Run with: bash Test/run_gui.sh"""

import copy
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import platform
import queue
import tempfile
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from simulation_backend import (
    DAY, LINKS, METHOD_CHANNELS, NOISE_KEYS, default_config, validate_config,
    simulation_worker, load_result, save_result,
)

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.signal import periodogram

BG, WHITE, INK, MUTED = "#f5f7f8", "#ffffff", "#20272d", "#66737c"
LINE, ACCENT, GREEN, RED = "#e1e7e9", "#007f86", "#23805c", "#bc4545"
PLOT_COLORS = ("#00858a", "#d28a27", "#7669b2", "#4078b3", "#c65e79", "#689342")
FONT = "PingFang SC" if platform.system() == "Darwin" else "Microsoft YaHei" if platform.system() == "Windows" else "sans-serif"

GB_FIELDS = [("A", "振幅 A"), ("f0", "初始频率 / Hz"), ("fdot0", "频率导数 / Hz s⁻¹"),
             ("phase0", "初相位 / rad"), ("inclination", "倾角 / rad"), ("longitude", "黄经 / rad"),
             ("latitude", "黄纬 / rad"), ("psi", "偏振角 / rad"), ("N", "源数目")]
MBHB_FIELDS = [("chirp_mass", "啁啾质量 / M☉"), ("mass_ratio", "质量比 m₂/m₁"), ("spin_1z", "自旋 1z"),
               ("spin_2z", "自旋 2z"), ("coalescence_time", "并合时间 / day"), ("coalescence_phase", "并合相位 / rad"),
               ("luminosity_distance", "光度距离 / Mpc"), ("inclination", "倾角 / rad"),
               ("longitude", "黄经 / rad"), ("latitude", "黄纬 / rad"), ("psi", "偏振角 / rad")]
NOISE_LABELS = ["测试质量加速度 · ACC", "干涉仪读出 · RO", "激光频率", "时钟抖动", "光学平台位移",
                "光纤背链", "光路", "伪测距", "边带调制"]


class ToolTip:
    def __init__(self, widget, text):
        self.widget, self.text, self.window = widget, text, None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def show(self, event=None):
        if self.window:
            return
        self.window = tk.Toplevel(self.widget)
        self.window.overrideredirect(True)
        self.window.geometry(f"+{self.widget.winfo_rootx()+12}+{self.widget.winfo_rooty()+self.widget.winfo_height()+5}")
        tk.Label(self.window, text=self.text, font=(FONT, 11), background=INK, foreground=WHITE,
                 padx=10, pady=6, wraplength=300, justify="left").pack()

    def hide(self, event=None):
        if self.window:
            self.window.destroy()
            self.window = None


class ScrollPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, style="White.TFrame")
        self.canvas = tk.Canvas(self, background=WHITE, highlightthickness=0, width=365)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas, style="White.TFrame", padding=(18, 16))
        self.body.columnconfigure(1, weight=1)
        window = self.canvas.create_window(0, 0, window=self.body, anchor="nw")
        self.canvas.configure(yscrollcommand=bar.set)
        self.body.bind("<Configure>", lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(window, width=event.width))
        bar.pack(side="right", fill="y")
        self.canvas.pack(fill="both", expand=True)


class TriangleSimulatorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Triangle Simulator | 科学数据仿真")
        self.geometry("1320x880")
        self.minsize(1040, 720)
        self.configure(background=BG)
        self.defaults = default_config()
        self.variables, self.inputs, self.pages = {}, {}, []
        self.process = self.events = None
        self.result = None
        self.tempdir = None
        self.started = None
        self.running = False
        self._suppress_dirty = False
        self.channel_vars = {}
        self._style()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind_all("<MouseWheel>", self._scroll)
        self.bind_all("<Button-4>", self._scroll)
        self.bind_all("<Button-5>", self._scroll)
        self.after(150, self._poll)
        self.after(100, self._empty_plot)

    def _style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=(FONT, 12), background=BG, foreground=INK, bordercolor=LINE)
        style.configure("White.TFrame", background=WHITE)
        style.configure("TLabel", background=WHITE)
        style.configure("Muted.TLabel", foreground=MUTED, font=(FONT, 11))
        style.configure("Heading.TLabel", font=(FONT, 12, "bold"))
        style.configure("TButton", background=WHITE, padding=(12, 7), borderwidth=1, relief="flat")
        style.map("TButton", background=[("active", "#edf3f4")], foreground=[("disabled", "#a2abb0")])
        style.configure("Primary.TButton", background=ACCENT, foreground=WHITE, borderwidth=0)
        style.map("Primary.TButton", background=[("disabled", "#a2bfc0"), ("active", "#006a70")], foreground=[("disabled", WHITE)])
        style.configure("TCheckbutton", background=WHITE, padding=(0, 5))
        style.map("TCheckbutton", background=[("active", WHITE)])
        style.configure("TEntry", fieldbackground=WHITE, padding=(7, 5), borderwidth=1)
        style.configure("TCombobox", fieldbackground=WHITE, padding=(6, 5), arrowsize=12)
        style.map("TCombobox", fieldbackground=[("readonly", WHITE)], selectbackground=[("readonly", WHITE)], selectforeground=[("readonly", INK)])
        style.configure("TNotebook", background=WHITE, borderwidth=0, tabmargins=0)
        style.configure("TNotebook.Tab", padding=(16, 10), background=BG, foreground=MUTED, borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", WHITE)], foreground=[("selected", ACCENT)])
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=LINE, borderwidth=0, thickness=4)
        self.option_add("*TCombobox*Listbox.font", (FONT, 12))

    def _build(self):
        header = tk.Frame(self, bg=WHITE, padx=24, pady=16)
        header.pack(fill="x")
        brand = tk.Frame(header, bg=WHITE)
        brand.pack(side="left")
        tk.Label(brand, text="TRIANGLE", bg=WHITE, fg=INK, font=(FONT, 23, "bold")).pack(anchor="w")
        tk.Label(brand, text="科学数据仿真  /  SCIENTIFIC DATA LAB", bg=WHITE, fg=MUTED, font=(FONT, 10)).pack(anchor="w", pady=(2, 0))
        self.run_button = ttk.Button(header, text="开始仿真", style="Primary.TButton", command=self._run)
        self.run_button.pack(side="right", padx=(10, 0))
        self.cancel_button = ttk.Button(header, text="取消", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="right", padx=(10, 0))
        self.export_button = ttk.Button(header, text="导出结果", command=self._export, state="disabled")
        self.export_button.pack(side="right")
        ttk.Separator(self).pack(fill="x")
        footer = tk.Frame(self, bg=WHITE, padx=22, pady=8)
        footer.pack(side="bottom", fill="x")
        self.status = tk.StringVar(value="就绪")
        self.status_label = tk.Label(footer, textvariable=self.status, bg=WHITE, fg=MUTED, font=(FONT, 11), anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True)
        self.elapsed = tk.StringVar(value="ProperTime → TCB")
        tk.Label(footer, textvariable=self.elapsed, bg=WHITE, fg=MUTED, font=(FONT, 11)).pack(side="right")
        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.pack(side="bottom", fill="x")
        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=16, pady=16)
        sidebar = ttk.Frame(main, style="White.TFrame", width=390)
        main.add(sidebar, weight=0)
        presets = ttk.Frame(sidebar, style="White.TFrame", padding=(16, 12))
        presets.pack(fill="x")
        ttk.Label(presets, text="实验配置", style="Heading.TLabel").pack(side="left")
        self.preset = ttk.Combobox(presets, state="readonly", width=16,
                                   values=("快速实验", "教程综合仿真", "仅引力波", "仅仪器噪声", "激光噪声与 TDI", "Glitch 注入"))
        self.preset.set("快速实验")
        self.preset.pack(side="right")
        self.preset.bind("<<ComboboxSelected>>", self._preset)
        self.settings_notebook = ttk.Notebook(sidebar)
        self.settings_notebook.pack(fill="both", expand=True)
        for name in ("仿真", "信号", "噪声", "TDI"):
            page = ScrollPage(self.settings_notebook)
            self.settings_notebook.add(page, text=name)
            self.pages.append(page)
        self._settings()
        config_bar = ttk.Frame(sidebar, style="White.TFrame", padding=12)
        config_bar.pack(fill="x")
        self.load_button = ttk.Button(config_bar, text="载入配置", command=self._load_config)
        self.load_button.pack(side="left", expand=True, fill="x", padx=(0, 5))
        ttk.Button(config_bar, text="保存配置", command=self._save_config).pack(side="left", expand=True, fill="x", padx=(5, 0))
        output = ttk.Frame(main, style="White.TFrame", padding=(18, 14))
        main.add(output, weight=1)
        self.result_title = tk.StringVar(value="观测与处理")
        ttk.Label(output, textvariable=self.result_title, font=(FONT, 17, "bold")).pack(anchor="w")
        self.metrics = tk.StringVar(value="MicroSate 轨道    ·    6 条激光链路    ·    第二代 TDI")
        ttk.Label(output, textvariable=self.metrics, style="Muted.TLabel").pack(anchor="w", pady=(6, 14))
        self.output_notebook = ttk.Notebook(output)
        self.output_notebook.pack(fill="both", expand=True)
        self.plot_tab = ttk.Frame(self.output_notebook, style="White.TFrame")
        self.log_tab = ttk.Frame(self.output_notebook, style="White.TFrame")
        self.output_notebook.add(self.plot_tab, text="数据视图")
        self.output_notebook.add(self.log_tab, text="运行日志")
        controls = ttk.Frame(self.plot_tab, style="White.TFrame", padding=(0, 12))
        controls.pack(fill="x")
        self.view = tk.StringVar(value="TDI 输出")
        self.view_box = ttk.Combobox(controls, textvariable=self.view, values=("TDI 输出", "六链路测量", "TDI 前后对比"), state="readonly", width=14)
        self.view_box.pack(side="left", padx=(0, 12))
        self.view_box.bind("<<ComboboxSelected>>", lambda event: self._refresh_channels())
        self.channel_frame = ttk.Frame(controls, style="White.TFrame")
        self.channel_frame.pack(side="left", fill="x", expand=True)
        self.figure = Figure(figsize=(8, 7), dpi=100, facecolor=WHITE, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_tab)
        toolbar_frame = tk.Frame(self.plot_tab, bg=WHITE)
        toolbar_frame.pack(side="bottom", fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.configure(background=WHITE)
        self.toolbar.pack(fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        log_bar = ttk.Scrollbar(self.log_tab)
        log_bar.pack(side="right", fill="y")
        self.log = tk.Text(self.log_tab, bg=WHITE, fg=INK, relief="flat", font=("Menlo", 10),
                           padx=12, pady=12, wrap="word", state="disabled", yscrollcommand=log_bar.set)
        self.log.pack(fill="both", expand=True)
        log_bar.configure(command=self.log.yview)
        self._append_log("Triangle Simulator · Tutorial 4\n等待仿真任务。")
        for var in self.variables.values():
            var.trace_add("write", self._mark_dirty)
        self.variables["tdi_method"].trace_add("write", self._method_changed)
        self.after(50, lambda: main.sashpos(0, 400))

    def _section(self, parent, title):
        row = parent.grid_size()[1]
        ttk.Label(parent, text=title, style="Heading.TLabel").grid(row=row, column=0, columnspan=2, sticky="w", pady=(14 if row else 0, 9))

    def _field(self, parent, key, label, choices=None, tip=None):
        value = self.defaults
        for part in key.split("."):
            value = value[part]
        var = tk.BooleanVar(value=value) if isinstance(value, bool) else tk.StringVar(value=str(value))
        self.variables[key] = var
        row = parent.grid_size()[1]
        if isinstance(value, bool):
            widget = ttk.Checkbutton(parent, text=label, variable=var)
            widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        else:
            ttk.Label(parent, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=6)
            widget = ttk.Combobox(parent, textvariable=var, values=choices, state="readonly", width=18) if choices else ttk.Entry(parent, textvariable=var, width=18)
            widget.grid(row=row, column=1, sticky="ew", pady=4)
        self.inputs[key] = widget
        if tip:
            ToolTip(widget, tip)
        return widget

    def _path(self, parent, key, label, directory=False):
        self._section(parent, label)
        value = self.defaults["orbit_dir"] if directory else self.defaults["emri"]["file"]
        var = tk.StringVar(value=value)
        self.variables[key] = var
        frame = ttk.Frame(parent, style="White.TFrame")
        frame.grid(row=parent.grid_size()[1], column=0, columnspan=2, sticky="ew", pady=(0, 8))
        frame.columnconfigure(0, weight=1)
        entry = ttk.Entry(frame, textvariable=var, width=24)
        entry.grid(row=0, column=0, sticky="ew")
        def choose():
            filename = filedialog.askdirectory(parent=self) if directory else filedialog.askopenfilename(parent=self, filetypes=[("Waveform", "*.npz")])
            if filename:
                var.set(filename)
        button = ttk.Button(frame, text="…", width=3, command=choose)
        button.grid(row=0, column=1, padx=(6, 0))
        ToolTip(button, "选择轨道目录" if directory else "选择波形文件")
        self.inputs[key] = entry
        self.inputs[key + "_browse"] = button

    def _settings(self):
        basic, signal, noise, tdi = [p.body for p in self.pages]
        self._section(basic, "采样与观测")
        self._field(basic, "data_time_days", "仿真时长 / day")
        self._field(basic, "fsample", "采样率 / Hz")
        self._field(basic, "interp_order", "Lagrange 插值阶数", (7, 15, 31))
        self._field(basic, "seed", "随机种子")
        self._field(basic, "trim_seconds", "两端各裁去 / s", tip="应用于图形和导出数据；用于剔除插值与延迟的边界污染。教程使用 1000 秒。")
        self._path(basic, "orbit_dir", "数值轨道", directory=True)
        self._section(basic, "时间基准")
        ttk.Label(basic, text="ProperTime → TCB", foreground=ACCENT).grid(row=basic.grid_size()[1], column=0, columnspan=2, sticky="w")
        self._section(signal, "引力波源")
        for key, label in (("gb_enable", "银河双星 · GB"), ("mbhb_enable", "大质量黑洞双星 · MBHB"),
                           ("emri_enable", "自定义波形 · EMRI"), ("sgwb_enable", "随机引力波背景 · SGWB")):
            self._field(signal, key, label)
        source_tabs = ttk.Notebook(signal)
        source_tabs.grid(row=signal.grid_size()[1], column=0, columnspan=2, sticky="ew", pady=(14, 0))
        source_frames = {}
        for name in ("GB", "MBHB", "EMRI", "SGWB"):
            frame = ttk.Frame(source_tabs, style="White.TFrame", padding=(0, 10))
            frame.columnconfigure(1, weight=1)
            source_tabs.add(frame, text=name)
            source_frames[name] = frame
        for key, label in GB_FIELDS:
            self._field(source_frames["GB"], "gb."+key, label)
        for key, label in MBHB_FIELDS:
            self._field(source_frames["MBHB"], "mbhb."+key, label)
        self._field(source_frames["MBHB"], "mbhb.approximant", "波形模型", ("SEOBNRv4_opt", "IMRPhenomT", "IMRPhenomD"))
        self._path(source_frames["EMRI"], "emri.file", "波形 NPZ · t / h1 / h2")
        for key, label in (("longitude", "黄经 / rad"), ("latitude", "黄纬 / rad"), ("psi", "偏振角 / rad")):
            self._field(source_frames["EMRI"], "emri."+key, label)
        for key, label in (("Omega0", "Ω₀"), ("f0", "参考频率 / Hz"), ("n", "谱指数"), ("nside", "HEALPix NSIDE")):
            self._field(source_frames["SGWB"], "sgwb."+key, label)
        self._section(noise, "仪器噪声")
        for key, label in zip(NOISE_KEYS, NOISE_LABELS):
            self._field(noise, key, label)
        self._field(noise, "acc_scale", "ACC 幅度倍率")
        self._field(noise, "ro_scale", "RO 幅度倍率")
        self._section(noise, "瞬态扰动")
        self._field(noise, "add_glitch", "注入 Glitch")
        self._field(noise, "glitch_tm", "测试质量", LINKS)
        self._field(noise, "glitch_time", "注入时间 / s")
        self._field(noise, "glitch_dv", "速度增量 Δv / m s⁻¹")
        self._field(noise, "glitch_tau1", "τ₁ / s")
        self._field(noise, "glitch_tau2", "τ₂ / s")
        self._section(tdi, "时延干涉处理")
        self._field(tdi, "tdi_method", "组合方法", tuple(METHOD_CHANNELS))
        self._field(tdi, "tdi_channel", "输出通道", METHOD_CHANNELS["fast_michelson"])
        self._field(tdi, "tdi_interp", "TDI 插值阶数", (7, 15, 31))
        self._field(tdi, "tdi_doppler", "Doppler 延迟算子")
        self._section(tdi, "处理链")
        for label in ("01   干涉仪测量", "02   TCB 时间同步", "03   ξ → η", "04   延迟组合"):
            ttk.Label(tdi, text=label, style="Muted.TLabel").grid(row=tdi.grid_size()[1], column=0, columnspan=2, sticky="w", pady=7)
        ToolTip(self.inputs["tdi_method"], "channel_path 和 channel_operator 使用教程的 X2 路径与算子。")

    def _scroll(self, event):
        widget = self.winfo_containing(event.x_root, event.y_root)
        for page in self.pages:
            if widget and str(widget).startswith(str(page)):
                if getattr(event, "num", None) in (4, 5):
                    delta = -1 if event.num == 4 else 1
                else:
                    delta = -int(event.delta) if platform.system() == "Darwin" else -int(event.delta / 120)
                page.canvas.yview_scroll(delta, "units")
                break

    def _method_changed(self, *args):
        options = METHOD_CHANNELS.get(self.variables["tdi_method"].get(), ("AET",))
        self.inputs["tdi_channel"].configure(values=options)
        if self.variables["tdi_channel"].get() not in options:
            self.variables["tdi_channel"].set(options[0])

    def _mark_dirty(self, *args):
        if not self._suppress_dirty and not self.running:
            self.status.set("参数已修改 · 图中保留上次运行结果" if self.result else "参数已修改 · 就绪")

    def _collect(self):
        config = copy.deepcopy(self.defaults)
        for path, var in self.variables.items():
            keys = path.split(".")
            owner = config
            for key in keys[:-1]:
                owner = owner[key]
            expected = owner[keys[-1]]
            try:
                owner[keys[-1]] = type(expected)(var.get())
            except (ValueError, tk.TclError):
                self.inputs[path].focus_set()
                raise ValueError(f"{path} 的数值格式无效") from None
        return validate_config(config)

    def _apply(self, config):
        self._suppress_dirty = True
        try:
            for path, var in self.variables.items():
                value = config
                for key in path.split("."):
                    value = value[key]
                var.set(value)
            self._method_changed()
        finally:
            self._suppress_dirty = False
        self._mark_dirty()

    def _preset(self, event=None):
        p = default_config()
        preset = self.preset.get()
        if preset == "教程综合仿真":
            p.update(data_time_days=10.0, mbhb_enable=True, emri_enable=True, sgwb_enable=True, glitch_time=100000.0)
        elif preset == "仅引力波":
            p.update(acc_noise=False, ro_noise=False)
        elif preset == "仅仪器噪声":
            p["gb_enable"] = False
        elif preset == "激光噪声与 TDI":
            p.update(gb_enable=False, acc_noise=False, ro_noise=False, laser_noise=True, tdi_channel="X")
        elif preset == "Glitch 注入":
            p["add_glitch"] = True
        self._apply(p)

    def _save_config(self):
        try:
            config = self._collect()
            path = filedialog.asksaveasfilename(parent=self, defaultextension=".json", initialfile="triangle_config.json", filetypes=[("JSON", "*.json")])
            if path:
                Path(path).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
                self.status.set("配置已保存")
        except Exception as exc:
            messagebox.showerror("保存配置失败", str(exc), parent=self)

    def _load_config(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            config = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("配置根节点须是 JSON 对象")
            self._apply(validate_config(config))
            self.status.set("配置已载入" + (" · 图中保留上次运行结果" if self.result else ""))
        except Exception as exc:
            messagebox.showerror("载入配置失败", str(exc), parent=self)

    def _set_running(self, running):
        self.running = running
        self.run_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")
        self.export_button.configure(state="normal" if self.result and not running else "disabled")
        self.preset.configure(state="disabled" if running else "readonly")
        self.load_button.configure(state="disabled" if running else "normal")
        for widget in self.inputs.values():
            widget.configure(state="disabled" if running else "readonly" if isinstance(widget, ttk.Combobox) else "normal")

    def _run(self):
        if self.running:
            return
        try:
            config = self._collect()
        except Exception as exc:
            messagebox.showerror("请检查参数", str(exc), parent=self)
            return
        self._set_running(True)
        self.progress["value"] = 0
        self.started = time.monotonic()
        self.status.set("正在启动仿真进程")
        self.status_label.configure(fg=ACCENT)
        self._append_log("\n" + "─" * 48 + "\n" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f"  seed={config['seed']}")
        try:
            self.tempdir = tempfile.TemporaryDirectory(prefix="triangle_gui_")
            context = mp.get_context("spawn")
            self.events = context.Queue()
            self.process = context.Process(target=simulation_worker, args=(config, self.events, str(Path(self.tempdir.name) / "result.npz")), daemon=True)
            self.process.start()
        except Exception as exc:
            self._finish(False, str(exc))

    def _poll(self):
        if self.running:
            self.elapsed.set(f"运行中  {time.monotonic()-self.started:.1f} s")
            try:
                while self.events is not None:
                    event = self.events.get_nowait()
                    if event[0] == "progress":
                        self.progress["value"] = event[1]
                        self.status.set(event[2])
                        self._append_log(event[2])
                    elif event[0] == "log":
                        self._append_log(event[1])
                    elif event[0] == "done":
                        self.result = load_result(event[1])
                        self._finish(True)
                        self._refresh_channels()
                        self.output_notebook.select(self.plot_tab)
                        break
                    elif event[0] == "error":
                        self._finish(False, event[1])
                        break
            except queue.Empty:
                pass
            except Exception as exc:
                self._finish(False, str(exc))
            if self.running and self.process is not None and self.process.exitcode is not None:
                self._finish(False, f"后台进程意外退出（exit code {self.process.exitcode}）")
        self.after(150, self._poll)

    def _cleanup(self):
        if self.process is not None:
            if self.process.pid is not None:
                self.process.join(timeout=0.2)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=2)
            self.process.close()
            self.process = None
        if self.events is not None:
            self.events.close()
            self.events = None
        if self.tempdir is not None:
            self.tempdir.cleanup()
            self.tempdir = None

    def _finish(self, success, error=None):
        self._cleanup()
        self._set_running(False)
        seconds = time.monotonic() - self.started if self.started else 0
        self.elapsed.set(f"耗时 {seconds:.1f} s")
        if success:
            self.progress["value"] = 100
            self.status.set("仿真完成")
            self.status_label.configure(fg=GREEN)
            _, meta = self.result
            p = meta["config"]
            self.result_title.set("仿真结果")
            self.metrics.set(f"{p['data_time_days']:g} day    ·    {p['fsample']:g} Hz    ·    {meta['samples']:,} 样本    ·    {meta['source_count']} 个源")
            self._append_log(f"仿真完成：{', '.join(meta['channels'])}，裁边后 {meta['samples']:,} 个样本")
        else:
            self.progress["value"] = 0
            self.status.set("仿真失败 · 详见运行日志" + (" · 保留上次结果" if self.result else ""))
            self.status_label.configure(fg=RED)
            self._append_log(error or "未知错误")
            self.output_notebook.select(self.log_tab)

    def _cancel(self):
        if self.running:
            self._cleanup()
            self._set_running(False)
            self.progress["value"] = 0
            self.status.set("已取消" + (" · 保留上次结果" if self.result else ""))
            self.status_label.configure(fg=MUTED)
            self.elapsed.set("ProperTime → TCB")
            self._append_log("用户取消仿真，后台进程已结束。")

    def _append_log(self, message):
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 4000:
            self.log.delete("1.0", "1000.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _empty_plot(self):
        if self.result:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        directory = Path(self.variables["orbit_dir"].get())
        try:
            positions = np.array([np.loadtxt(directory / f"SCP{i}.dat")[0] for i in "123"])
            positions = (positions - positions.mean(axis=0)) * 149.5978707
            ring = positions[[0, 1, 2, 0]]
            ax.plot(ring[:, 0], ring[:, 1], color=ACCENT, linewidth=1.4)
            ax.scatter(positions[:, 0], positions[:, 1], s=80, c=[ACCENT, PLOT_COLORS[1], PLOT_COLORS[2]], zorder=3)
            for i, pos in enumerate(positions):
                ax.annotate(f"SC {i+1}", pos[:2], xytext=(10, 12), textcoords="offset points", color=INK, fontsize=11)
            ax.set(xlabel="X / million km", ylabel="Y / million km", title="MicroSate constellation · TCB = 0 s")
            ax.set_aspect("equal", adjustable="datalim")
            ax.margins(0.28)
        except (OSError, ValueError):
            ax.text(0.5, 0.5, "No simulation data", transform=ax.transAxes, ha="center", color=MUTED)
        self._style_axes(ax)
        self.canvas.draw_idle()

    def _refresh_channels(self):
        if not self.result:
            return
        for widget in self.channel_frame.winfo_children():
            widget.destroy()
        _, meta = self.result
        mode = self.view.get()
        choices = meta["channels"] if mode == "TDI 输出" else [f"sci_c_{link}" for link in LINKS] if mode == "六链路测量" else ["sci_c_12"] + meta["channels"]
        self.channel_vars = {}
        for name in choices:
            var = tk.BooleanVar(value=True)
            self.channel_vars[name] = var
            label = name.replace("sci_c_", "y")
            ttk.Checkbutton(self.channel_frame, text=label, variable=var, command=self._plot).pack(side="left", padx=(0, 5))
        self._plot()

    @staticmethod
    def _style_axes(ax):
        ax.set_facecolor(WHITE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color(LINE)
        ax.grid(True, color=LINE, linewidth=0.6, alpha=0.8)
        ax.tick_params(labelsize=9, colors=MUTED)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(INK)
        ax.title.set_fontsize(11)

    def _plot(self):
        if not self.result:
            return
        arrays, meta = self.result
        self.figure.clear()
        self.toolbar.update()
        ax_time, ax_asd = self.figure.subplots(2, 1)
        selected = [name for name, var in self.channel_vars.items() if var.get()]
        fs = meta["config"]["fsample"]
        t = arrays["time_tcb_s"] / DAY
        for name in selected:
            data = arrays[name]
            color = PLOT_COLORS[list(self.channel_vars).index(name) % len(PLOT_COLORS)]
            linestyle = "--" if self.view.get() == "TDI 前后对比" and name.startswith("sci_c") else "-"
            label = name.replace("sci_c_", "y")
            # Preserve transient extrema when reducing dense time series for display.
            if len(data) > 6000:
                bins = np.array_split(np.arange(len(data)), 2000)
                idx = np.sort(np.concatenate([[b[np.argmin(data[b])], b[np.argmax(data[b])]] for b in bins]))
            else:
                idx = np.arange(len(data))
            ax_time.plot(t[idx], data[idx], label=label, color=color, linewidth=0.75, alpha=0.85, linestyle=linestyle)
            freq, psd = periodogram(data, fs=fs, window="hann", detrend=False, scaling="density")
            positive = (freq > 0) & (psd > 0)
            if positive.any():
                ax_asd.loglog(freq[positive], np.sqrt(psd[positive]), label=label, color=color, linewidth=0.8, linestyle=linestyle)
        ax_time.set(title="Time series", xlabel="TCB time / day", ylabel="Fractional frequency")
        ax_time.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax_asd.set(title="Amplitude spectral density · Hann", xlabel="Frequency / Hz", ylabel="ASD / Hz$^{-1/2}$")
        if selected:
            ax_time.legend(loc="upper right", frameon=False, fontsize=9, ncol=min(3, len(selected)))
            if ax_asd.lines:
                ax_asd.legend(loc="upper right", frameon=False, fontsize=9, ncol=min(3, len(selected)))
            else:
                ax_asd.text(0.5, 0.5, "All selected channels are zero", transform=ax_asd.transAxes, ha="center", color=MUTED)
        else:
            ax_time.text(0.5, 0.5, "No channels selected", transform=ax_time.transAxes, ha="center", color=MUTED)
        for ax in (ax_time, ax_asd):
            self._style_axes(ax)
        self.canvas.draw_idle()

    def _export(self):
        if not self.result:
            return
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".npz", initialfile="triangle_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".npz", filetypes=[("NumPy archive", "*.npz")])
        if path:
            try:
                save_result(path, *self.result)
                self.status.set("结果已导出 · " + Path(path).name)
                self._append_log(f"结果已导出：{path}")
            except Exception as exc:
                messagebox.showerror("导出失败", str(exc), parent=self)

    def _close(self):
        self._cleanup()
        self.destroy()


def main():
    mp.freeze_support()
    app = TriangleSimulatorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
