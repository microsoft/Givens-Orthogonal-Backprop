# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='rotMatcpp',
    ext_modules=[
        CppExtension('rotMatcpp', ['rotMat.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
    )
