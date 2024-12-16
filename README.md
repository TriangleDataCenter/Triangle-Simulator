
# Triangle-Simulator

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

**Triangle-Simulator** is a Python package designed for [brief description of your project, e.g., gravitational wave data analysis, interferometry simulations, etc.]. It offers a suite of tools and modules to facilitate [specific functionalities, e.g., efficient data handling, simulation execution, result visualization, etc.].

## Features

- **Data Management:** Efficient handling and processing of gravitational wave data and orbital configurations.
- **Simulation Tools:** Modules for conducting interferometry simulations, noise modeling, and more.
- **Visualization:** Comprehensive plotting tools for analyzing and visualizing simulation results.
- **Integration with PyCBC:** Utilizes PyCBC for advanced gravitational wave analysis (used only for demonstration purposes).

## Project Structure

```
TriangleProject/
├── Demo_GW_Injection.ipynb
├── Demo_Interferometry_Simulation_and_TDI.ipynb
├── Figures/
│   └── constellation.png
├── GWData/
│   └── Demo_MBHB_waveform_data.npy
├── LICENSE
├── OrbitData/
│   ├── LISALikeOrbitEclipticTCB/
│   │   ├── 20280322_LISA_2p5Mkm/
│   │   └── 20280322_LISA_3Mkm/
│   └── MicroSateOrbitEclipticTCB/
├── README.md
├── Triangle/
│   ├── Constants.py
│   ├── Cosmology.py
│   ├── Data.py
│   ├── FFTTools.py
│   ├── GW.py
│   ├── Glitch.py
│   ├── Interferometer.py
│   ├── Noise.py
│   ├── Offset.py
│   ├── Orbit.py
│   ├── Plot.py
│   ├── TDI.py
│   ├── TTL.py
│   └── __init__.py
└── requirements.txt
```

## Getting Started

### Prerequisites

- **[Anaconda](https://docs.anaconda.com/anaconda/install/):** Package and environment manager.
- **Python 3.9.19**
- **Git** (optional, for cloning the repository)

### Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/TriangleDataCenter/Triangle-Simulator
   cd Triangle-Simulator
   ```

2. **Create and Activate a Conda Environment**

   ```sh
   conda create -n tri_env python=3.9.19
   conda activate tri_env
   ```

3. **Install Required Packages**

   ```sh
   pip install -r requirements.txt
   ```

   *Note: Triangle itself does not depend on PyCBC, and PyCBC is only used for demonstration purposes in the provided Jupyter notebooks.*

4. **Install the Package Locally in Editable Mode**

   ```sh
   pip install -e .
   ```

## Usage

### Using the Package

After installation, you can utilize the package modules in your Python scripts or interactive sessions.

**Example:**

```python
import Triangle

# Example usage
from Triangle import GW

result = GW.add(3, 5)  # Replace with an actual function from GW.py
print(f"The result of add(3, 5) is {result}")
```

*Ensure that you replace `GW.add` with a valid function from your `GW.py` module.*

### Running Jupyter Notebooks

The repository includes Jupyter notebooks for demonstrations and simulations.

1. **Launch Jupyter Notebook:**

   ```sh
   jupyter notebook
   ```

2. **Navigate to the Project Directory:**

   Open the desired notebook:

   - `Demo_GW_Injection.ipynb`
   - `Demo_Interferometry_Simulation_and_TDI.ipynb`

3. **Run the Notebook:**

   Execute the cells to perform simulations and visualize results.

## Data

The project includes essential data files located in the `GWData/` and `OrbitData/` directories. These files are used by the modules for simulations and data analysis.

- **GWData:** Contains gravitational wave waveform data.
- **OrbitData:** Contains orbital configuration files and related data.

**Note:** Ensure that these data directories are present in the root directory after installation, as the package modules depend on them for proper functionality.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [PyCBC](https://pycbc.org/) for providing tools for gravitational wave analysis.
- [Healpy](https://healpy.readthedocs.io/en/latest/) for spherical data analysis.
- [Matplotlib](https://matplotlib.org/) for data visualization.

## Contact

For any questions or suggestions, please contact [your.email@example.com](mailto:your.email@example.com).
