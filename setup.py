from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup

import numpy as np

_NP_INCLUDE_DIRS = np.get_include()

import torch.cuda
from torch.utils.cpp_extension import CppExtension, CUDAExtension

ext_modules = []

#https://stackoverflow.com/questions/45600866/add-c-function-to-existing-python-module-with-pybind11

if torch.cuda.is_available():
    extension = CUDAExtension(
        name='padding._C',
        sources = [
            'src/gpu_ops.cpp',
            'src/padding_cpu.cpp', #-> padding_cpu.o
            'src/padding_gpu.cu',  #-> padding_gpu.o
            'src/crop_cpu.cpp',    #-> crop_cpu.o
            'src/crop_gpu.cu',     #-> crop_gpu.o
            'src/fused_gpu.cpp',   #-> fused_gpu.o
        ],
        extra_compile_args={'cxx': ['-g', '-fopenmp'],
                            'nvcc': ['-O2']})
else:
    extension = CppExtension(
        name='padding._C',
        sources = [
            'src/cpu_ops.cpp',
            'src/padding_cpu.cpp',
            'src/crop_cpu.cpp',
        ],
        extra_compile_args={'cxx': ['-g', '-fopenmp']})

ext_modules.append(extension)


setup(
    name='padding',
    packages=['padding', 'padding/pkgs'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension})
