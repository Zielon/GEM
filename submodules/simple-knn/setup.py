#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == "nt":
    cxx_compiler_flags.append("/wd4624")

nvcc_args = ["-O3"]
nvcc_args.extend(
    [
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        # "-gencode=arch=compute_90,code=sm_90",
    ]
)

setup(
    version="1.0.0",
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=["spatial.cu", "simple_knn.cu", "ext.cpp"],
            extra_compile_args={"nvcc": nvcc_args, "cxx": cxx_compiler_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
