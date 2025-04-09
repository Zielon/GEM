from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

cxx_compiler_flags = ["-O3"]
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
    name="upfirdn2d",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="upfirdn2d",
            sources=[os.path.join("upfirdn2d.cpp"), os.path.join("upfirdn2d_kernel.cu")],
            extra_compile_args={"nvcc": nvcc_args, "cxx": cxx_compiler_flags},
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)

setup(
    version="1.0.0",
    name="fused",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="fused",
            sources=[os.path.join("fused_bias_act.cpp"), os.path.join("fused_bias_act_kernel.cu")],
            extra_compile_args={"nvcc": nvcc_args, "cxx": cxx_compiler_flags},
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
