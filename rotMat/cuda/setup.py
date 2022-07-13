# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rotMatcuda',
    ext_modules=[
        CUDAExtension('rotMatcuda', ['rotMatCuda.cpp','rotMatCudaFun.cu']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
    )
