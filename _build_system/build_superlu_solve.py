"""
CMake/SWIG interface for superlu_solve.
"""
import os

from build_utils import cmake, make_call

__all__ = ("cmake_superlu_solve", "generate_superlu_solve_wrapper")


def cmake_superlu_solve(bglb):
    """
    Build and install the shared interface using private static dependencies.
    """
    root = bglb.rootdir / "build" / "superlu"
    package = bglb.bdist_wheel_prefix / "petram" / "ext" / "superlu"
    cmake_opts = {
        "DCMAKE_BUILD_TYPE": "Release",
        "DCMAKE_INSTALL_PREFIX": str(package),
        "DCMAKE_INSTALL_LIBDIR": "lib",
        "DCMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "DBUILD_SHARED_LIBS": "ON",
        "DBUILD_TESTING": "OFF",
        "DMPI_CXX_COMPILER": bglb.mpicxx,
        "DNATIVE_PREFIX": str(root / "local"),
    }
    if bglb.cc:
        cmake_opts["DCMAKE_C_COMPILER"] = bglb.cc
    if bglb.cxx:
        cmake_opts["DCMAKE_CXX_COMPILER"] = bglb.cxx

    cmake("-S", bglb.rootdir / "superlu_solve", "-B", root / "interface",
          **cmake_opts)
    cmake("--build", root / "interface", "-j", str(min(os.cpu_count() or 1, 8)))
    cmake("--install", root / "interface")


def generate_superlu_solve_wrapper(bglb):
    """
    Generate the SWIG wrapper and Python proxy in the build directories.
    """
    source = bglb.rootdir / "superlu_solve"
    package = bglb.bdist_wheel_prefix / "petram" / "ext" / "superlu"
    wrapper = bglb.build_temp / "superlu_solve_wrap.cxx"
    make_call([bglb.swig, "-c++", "-python", "-I" + str(source),
               "-outdir", package, "-o", wrapper,
               source / "superlu_solve.i"])
    return wrapper
