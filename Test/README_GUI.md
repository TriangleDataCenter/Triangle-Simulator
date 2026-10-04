# Triangle 科学数据仿真界面

基于 `Tutorials/4_Demo_Simulation_for_Scientific_Data_Analysis.ipynb` 的本地桌面应用。
运行项目自身的 Triangle 仿真算法，使用 Tkinter + Matplotlib，无需 Web 服务或新增 GUI 框架。

## 启动

在项目根目录执行：

```bash
bash Test/run_gui.sh
```

脚本优先使用已激活的 virtualenv / conda 环境，其次使用项目 `.venv`。
环境应具备项目依赖以及 Tkinter。也可指定解释器：

```bash
TRIANGLE_PYTHON=/path/to/python bash Test/run_gui.sh
```

直接运行 `python Test/gui_app.py` 同样可用。数据路径默认根据代码位置解析，不依赖终端当前目录。

## 工作流程

1. 选择实验配置。默认“快速实验”使用 1 天、0.1 Hz、一个 GB 源及 ACC / RO 噪声。
2. 在“仿真 / 信号 / 噪声 / TDI”中编辑参数，点击“开始仿真”。
3. 查看 TDI 输出、六链路测量或前后对比。通过复选框选择通道；图形工具栏支持缩放、平移、复位和保存图片。
4. “导出结果”保存 NPZ；“保存配置 / 载入配置”使用 JSON。

“教程综合仿真”使用 10 天、GB + MBHB + EMRI + SGWB（NSIDE=5，共 300 个方向），计算量和内存开销明显增加。
另有纯信号、纯仪器噪声、激光噪声与 TDI、Glitch 预设。

## 仿真约定

- 信号：教程中的 GB、PyCBC MBHB、仓库内 EMRI 示例文件和 HEALPix SGWB；可同时启用。
- EMRI 文件包含 `t`、`h1`、`h2`；按教程使用 `h_plus=h1`、`h_cross=-h2`。
- GB 多源模式在初始频率的 ±0.1% 范围采样，并随机化天空位置和偏振角。
- 噪声：ACC、RO、激光、时钟、光学平台、背链、光路、伪测距和调制噪声。ACC / RO 倍率作用于时域幅度，其 PSD 按倍率平方缩放。
- Glitch 注入到所选测试质量的 ACC 耦合项；即使关闭随机 ACC 噪声，Glitch 仍可单独注入。`dv` 单位是 m/s。
- 处理链：`SimulateBasicNoise` → `SimulateRangeAndClock` → 可选 Glitch / GW → `SimulateMeasurements` → `OutputMeasurements(mode="decomposed")` → `TPStoTCB` → `CalculateXi` → `CalculateEta` → TDI。
- 与教程一致，使用 ProperTime 生成测量，再同步到 TCB；不提供未经校正的 ClockTime 模式。
- `fast_michelson` 输出 A2/E2/T2、X2/Y2/Z2 或 X2；`channel_name` 支持 X1、X2、alpha1、alpha2；路径法和算子法使用教程的 X2。
- TDI 使用轨道的真实光行时 `ltt`，不将伪测距噪声传播进延迟估计；本界面未实现时钟噪声校正和伪测距解算。启用这些噪声不代表相应校正已完成。
- 输出测量及 TDI 除以 `F_LASER`，单位为分数频率；时间保留真实的 TCB 网格，裁边后不会重置到 0。
- 两端默认各裁去 1000 s。允许设为 0 以诊断边界，此时端点可能受插值、延迟污染。
- ASD 使用 Hann 窗单段 periodogram，单边 PSD 的平方根，单位 Hz⁻¹/²。前后对比是不同观测量的对照，不是归一化后的灵敏度或 SNR。
- 固定 seed 可复现相同配置。请根据波形频段选择足够的采样率，特别是 MBHB 并合阶段；GB 会检查整个观测期的 Nyquist 条件。
- 为控制桌面实验的资源占用，单次限制 2,000,000 个样本，SGWB 限制 30,000,000 个方向-时间样本，NSIDE 为 1–8。

计算在独立后台进程中串行执行。取消或关闭窗口会结束该进程；失败和取消保留上次成功结果。
配置被修改时，状态栏会说明图中仍是上次运行的数据。导出总是包含结果实际使用的参数快照。

## 导出格式

NPZ 不含 pickle 对象，可用 `allow_pickle=False` 安全读取：

```python
import json
import numpy as np

with np.load("triangle_result.npz", allow_pickle=False) as result:
    metadata = json.loads(str(result["metadata_json"]))
    time = result["time_tcb_s"]
    channel = result[metadata["channels"][0]]
    link12 = result["sci_c_12"]
```

数组包括 `time_tcb_s`、六链路 `sci_c_*` / `sci_sb_*` / `ref_c_*` / `ref_sb_*` / `tm_c_*` / `eta_*`、单位为秒的 `ltt_*`，以及选定的 TDI 通道。
`metadata_json` 记录配置、通道、样本数、裁边数、耗时和单位。保存的是完整裁边结果；仅界面时域绘图为保持响应速度进行保留极值的抽样。

## 验证

```bash
.venv/bin/python -m unittest discover -s Test -p 'test_simulation_backend.py' -v
.venv/bin/python Test/test_gui_workflow.py
```

后端测试覆盖噪声开关、四类波形、TDI 方法一致性、信号与噪声线性叠加、随机种子、Glitch、边界参数及 NPZ 往返。
GUI 测试需要 Tk 环境，覆盖后台计算、绘图、通道联动、导出、配置往返、取消和重新运行。

文件：`gui_app.py` 为界面，`simulation_backend.py` 为可独立调用的仿真后端。
