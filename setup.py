from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np
import sys
import os

from distutils.sysconfig import get_config_vars

# Enable docstrings in the compiled code
Options.docstrings = True

# Define compiler and linker arguments
compile_args = []
link_args = []

if sys.platform == 'win32':
    # Windows (MSVC)
    compile_args.append('/openmp')
    # MSVC automatically links OpenMP, no need for extra_link_args
else:
    # Linux / macOS (assuming gcc/clang with OpenMP support)
    compile_args.append('-fopenmp')
    link_args.append('-fopenmp')

# Define extension module
ext_modules = [
    Extension(
        "MaxproTools_cython",                     # Name of the resulting module
        ["MaxproTools_cython.pyx"],               # Cython source file
        extra_compile_args=compile_args,          # Compiler flags
        extra_link_args=link_args,                # Linker flags
        include_dirs=[np.get_include()]           # Numpy headers
    )
]

# Setup function using setuptools
setup(
    name='MaxproTools_cython',
    ext_modules=cythonize(ext_modules, language_level=3),  # Ensure Python 3 compatibility
)
