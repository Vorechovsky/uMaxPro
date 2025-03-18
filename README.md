# (u)MaxPro Design Generator

This repository provides tools for generating and analyzing MaxPro designs using Python. It includes utilities for design generation, plotting, and optimization, with support for Cython-based performance enhancements.

## Features

- **MaxPro Design Generation**: Generate MaxPro designs using Python scripts and Jupyter notebooks.
- **Cython Optimization**: Accelerated computations using Cython modules.
- **Plotting Utilities**: Visualize designs and optimization progress.
- **Simulated Annealing**: Tools for optimization with simulated annealing.

## Repository Structure

```
.gitignore
GenerateMaxProDesign.py       # Main script for generating MaxPro designs
MaxPro_Generator.ipynb        # Jupyter notebook for interactive design generation
MaxproTools_cython.pyx        # Cython source file for performance optimization
MaxproTools_python.py         # Python implementation of MaxPro tools
plotting.py                   # Utilities for plotting designs and progress
setup.py                      # Setup script for building Cython modules
data/                         # Directory for input/output data
build/                        # Build artifacts for Cython modules
__pycache__/                  # Compiled Python files
```

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Vorechovsky/uMaxPro
   cd uMaxPro
   ```

2. Build the Cython module:
   ```sh
   python setup.py build_ext --inplace
   ```

3. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Generate MaxPro Designs
Run the main script to generate designs:
```sh
python GenerateMaxProDesign.py
```

### Interactive Design Generation
Use the Jupyter notebook for interactive design generation:
```sh
jupyter notebook MaxPro_Generator.ipynb
```

### Plotting
Use the `plotting.py` script to visualize designs and optimization progress:
```sh
python plotting.py
```

## Cython Module

The repository includes a Cython implementation for performance-critical computations. The Cython source file is located at `MaxproTools_cython.pyx`. After building the module, it will be available as a `.pyd` or `.so` file depending on your platform.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

## References

- **2D View Comparison**: See `2D_view_comparison.pdf` for a comparison of 2D designs.
- **Simulated Annealing Progress**: See `SimulatedAnnealingProgress.pdf` for optimization progress visualizations.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## Contact

https://www.fce.vutbr.cz/STM/vorechovsky.m/

