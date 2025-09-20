# Computational modeling of fracture in geometrically exact beams based on the DG/CZM approach <br> [![Run Tests](https://github.com/aces-research/beamit/actions/workflows/ci.yml/badge.svg)](https://github.com/aces-research/beamit/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the code for computational modeling of large deformations and fracture in 3D beams. It uses 
the geometrically exact beam formulation (Simo et al., 1986) and its torsion-free Kirchhoff-Love variant (Meier, 2016) 
for beam large deformations, along with a discontinuous Galerkin/Cohesive Zone Model (DG/CZM) approach for fracture.

**Assumptions / Notes**:

- The code works for *single beams only*. If your structure is a spaghetti of beams, sorry, not supported yet.
- Beams are assumed to be *initially straight and parallel to the x-axis*. Curvy or rebellious beams are not allowed.
- Beam elements are *connected end-to-end (like a simple chain), with exactly two nodes per element*. So complex beam networks cannot be modeled.
- If your beam passes all the above "tests", you can go ahead and use the code!

## Requirements

- **Python**
  - Version `3.10.12` is recommended.  
  - You can use [pyenv](https://github.com/pyenv/pyenv) (Linux/macOS) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) (Windows) to install and switch Python versions.
  - If you don't have Python 3.10.12, the installation will still work with your system’s default Python, *but compatibility with other versions is not guaranteed*.

## Installation

Run the following commands in your terminal (or PowerShell on Windows):

```bash
# 1. Clone the repository
git clone https://github.com/aces-research/beamit.git
cd beamit

# 2. Create and activate a virtual environment
python -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1

# 3. Install dependencies and library (in editable mode)
python -m pip install --upgrade pip
pip install -e .

# 4. Verify installation (optional):
python -m pytest -v tests
```

## Structure

```
.
├── examples
│   ├── arc_segment_out_of_plane_load
│   ├── beam_blast
│   ├── beam_buckling
│   ├── DG_derivatives
│   ├── double_clamped_beam_fracture
│   ├── euler_bernoulli_dynamic_bending
│   ├── euler_bernoulli_dynamic_tension
│   ├── pure_bending_3d
│   ├── released_spaghetti_fracture
│   └── spall_problem
├── lib
│   └── beamit
│       ├── FunctionSpace.py
│       ├── Material.py
│       ├── PostProcess.py
│       ├── Solver.py
│       ├── System.py
│       └── WeakForm
│           ├── EulerBernoulliWeakForm.py
│           ├── ShearFlexibleGeometricallyExactWeakForm.py
│           ├── TFKLGeometricallyExactWeakForm.py
│           ├── Utils.py
│           └── WeakForm.py
├── LICENSE
├── README.md
├── requirements.txt
└── tests
    ├── euler-bernoulli
    ├── SFGE-beam-CG
    ├── TFKL-beam-CG
    └── TFKL-beam-DG
```

- The `examples/` directory contains various examples demonstrating the capabilities of the library.
- The `lib/beamit/` directory contains the main library code, organized into modules for different functionalities.
  - `FunctionSpace.py`: Defines the mathematical utilities such as shape functions and numerical integration for CG and DG discretizations of beams.
  - `Material.py`: Contains material models for beams with related constitutive computations.
  - `PostProcess.py`: Provides functions for post-processing and visualizing simulation results.
  - `Solver.py`: Implements the solvers for static and dynamic problems, including Newton-Raphson solver and explicit Newmark time integration scheme.
  - `System.py`: Handles the assembly of global system matrices and vectors from individual weak form contributions. Currently only one weak form is considered within the system.
  - `WeakForm/`: Contains implementations of different weak forms i.e., classes to compute stiffness and mass matrices, residuals, and other related quantities for various beam formulations.
    - `WeakForm.py`: Base class for weak form implementations.
    - `Utils.py`: Utility functions for weak form computations.
    - `EulerBernoulliWeakForm.py`: Weak form implementation for the Euler-Bernoulli beam formulation.
    - `ShearFlexibleGeometricallyExactWeakForm.py`: Weak form implementation for the shear-flexible geometrically exact beam formulation.
    - `TFKLGeometricallyExactWeakForm.py`: Weak form implementation for the torsion-free Kirchhoff-Love geometrically exact beam formulation.
- The `tests/` directory contains unit tests to verify the correctness of various components of the library.

## Usage

- To run an example, go to the desired example folder inside `examples/`, and execute the Python script. For instance, to run the beam buckling example, you would:

```bash
cd examples/beam_buckling
python beam_buckling.py
```

- The simulation will take some time to complete, depending on the complexity of the example and your system's performance.

- The results of the simulation will be saved in a `VTK` folder within the example directory in `*.vtu` format, which can be visualized using a software like [ParaView](https://www.paraview.org/). If you are using ParaView, we recommend you to increase the line width of the beam and enable "Render Lines as Tubes" for better visibility.

## Authors

This software has been developed by:  
- *Sai Kubair Kota* ([@saikubairkota](https://github.com/saikubairkota)), ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0000-0003-4285-4744](https://orcid.org/0000-0003-4285-4744), S.K.Kota@tudelft.nl, Technische Universiteit Delft
- *Bianca Giovanardi* ([@biancagi](https://github.com/biancagi)), ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0000-0001-7768-8542
](https://orcid.org/0000-0001-7768-8542), B.Giovanardi@tudelft.nl, Technische Universiteit Delft

## License

This code is licensed under an MIT License - you are free to use, modify, and distribute it under the terms of that license. See [LICENSE](LICENSE) for details.

Copyright notice:

Technische Universiteit Delft hereby disclaims all copyright interest in the program “Computational modeling of fracture in geometrically exact beams based on the DG/CZM approach” written by the Author(s).

Henri Werij, Dean of Faculty of Aerospace Engineering, Technische Universiteit Delft.

&copy; 2025, S. K. Kota, B. Giovanardi

## References

- Simo, J. C., & Vu-Quoc, L. (1986). [A three-dimensional finite-strain rod model. Part II: Computational aspects](https://www.sciencedirect.com/science/article/pii/0045782586900794). Computer methods in applied mechanics and engineering, 58(1), 79-116.

- Meier, C. A. (2016). [Geometrically exact finite element formulations for slender beams and their contact interaction](https://mediatum.ub.tum.de/doc/1306287/1306287.pdf) (Doctoral dissertation, Technische Universität München).

- Kota, S. K., Kumar, S., & Giovanardi, B. (2025). [A discontinuous Galerkin/cohesive zone model approach for the computational modeling of fracture in geometrically exact slender beams](https://link.springer.com/article/10.1007/s00466-024-02521-0). Computational Mechanics, 75(2), 595-612.

- *The above paper will be replaced with my PhD thesis.*

## Cite this repository

**How to cite this repository:** S. K. Kota, B. Giovanardi., 2025. Computational modeling of fracture in geometrically exact beams based on the DG/CZM approach. 4TU.ResearchData. Software. https://doi.org/10.4121/11b0b61a-0e81-4187-a0f6-5b64c79ccfba

## Would you like to contribute?

You are welcome to contribute! If you have any comments, feedback, or recommendations, feel free to reach out to us.

If you want to contribute directly, you are welcome to **fork** this repository.
