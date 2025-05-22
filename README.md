
# Triangle-Simulator

## Overview

**Triangle-Simulator** is a full-Python prototype simulator for the data of space-based gravitational wave detectors.
For **Taiji Data Challenge II** users, please make sure to look at Tutorial 5. 
If you are interested how the noises and signals are simulated, Tutorial 4 can serve as a brief guide. 
Users who are curious about how the raw data look like can refer to Tutorial 1 - 3.    

## Features

- **Highly integrated:** The simulation of noises, gravitational wave signals, other instrumental effects and time-delay interferometry processing.
- **High flexibility:** custom numerial orbits, noise data, gravitational wave waveforms and time-delay interferometry configurations.

## Getting Started

### Prerequisites

- **[Anaconda](https://docs.anaconda.com/anaconda/install/):** Package and environment manager.
- **Git** (optional, for cloning the repository)

### Installation

(Tested platform: Ubuntu22.04, MacOS15)

1. **Clone the Repository**  
(neglect the git clone step if you install from the downloaded .zip file)
   ```sh
   git clone https://github.com/TriangleDataCenter/Triangle-Simulator.git
   cd Triangle-Simulator
   ```

2. **Create and Activate a Conda Environment**

   ```sh
   conda create -n tri_env -c conda-forge python=3.9.19 uv
   conda activate tri_env
   ```

<!-- 3. **Install Required Packages**

   ```sh
   uv pip install .
   ```

   *Note: Triangle itself does not depend on PyCBC, and PyCBC is only used for demonstration purposes in the provided Jupyter notebooks.* -->

3. **Install the Package Locally in Editable Mode**

   ```sh
   uv pip install -e .
   ```

## Usage

### Using the Package

After installation, you can utilize the package modules in your Python scripts or interactive sessions. Demonstrations for the usage of this code can be found at the Tutorial folder.

## Data

- **GWData:** Contains gravitational wave waveform data used as examples.
- **OrbitData:** Contains the data of numerical orbit.

**Note:** Ensure that these data directories are present in the root directory after installation, as the package modules depend on them for proper functionality.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

<!-- ## Acknowledgements -->

## Reference 

- Taiji Data Challenge II: [the TDC II paper](TBD), [the TDC II manual](TBD).
- The modeling of laser interfetormetry is greatly inspired by the [PhD thesis of O. Hartwig](https://repo.uni-hannover.de/items/4afe8e21-39a1-49a9-a85d-996e1c5dbe30), as well as related research works such as [Phys. Rev. D 107, 083019](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.107.083019). In the current version, the lagrange interpolation filter is adapted from the implementation of [LISAInstrument](https://gitlab.in2p3.fr/lisa-simulation/instrument), and more efficient approaches such as the [22-coefficient cosine-sum kernels](https://arxiv.org/html/2412.14884v1) will be tested in the future versions.   
- Research on the numerical simulation of drag-free control: 


## Contact

For any questions or suggestions, please contact the owner of this repository.
