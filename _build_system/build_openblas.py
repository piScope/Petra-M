"""
Build private single-threaded OpenBLAS for SuperLU_DIST
"""
import os

from build_utils import cmake, download

__all__ = ("clone_build_openblas",)


def _clone_openblas(bglb):
    """
    Fetch the pinned OpenBLAS source using the shared download utility.
    """
    download("openblas", bglb.rootdir / "build" / "superlu" / "sources")


def _cmake_openblas(bglb):
    """
    Configure static PIC BLAS kernels without LAPACK or worker threads.
    """
    root = bglb.rootdir / "build" / "superlu"
    cmake_opts = {
        "DCMAKE_BUILD_TYPE": "Release",
        "DCMAKE_INSTALL_PREFIX": str(root / "local"),
        "DCMAKE_INSTALL_LIBDIR": "lib",
        "DCMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "DBUILD_STATIC_LIBS": "ON",
        "DBUILD_SHARED_LIBS": "OFF",
        "DBUILD_WITHOUT_LAPACK": "ON",
        "DBUILD_TESTING": "OFF",
        "DNOFORTRAN": "ON",
        "DINTERFACE64": "OFF",
        "DFIXED_LIBNAME": "ON",
        "DUSE_THREAD": "OFF",
        "DUSE_OPENMP": "OFF",
        "DBUILD_SINGLE": "OFF",
        "DBUILD_COMPLEX": "OFF",
        "DBUILD_DOUBLE": "ON",
        "DBUILD_COMPLEX16": "ON",
    }
    if bglb.cc:
        cmake_opts["DCMAKE_C_COMPILER"] = bglb.cc
    cmake("-S", root / "sources" / "openblas", "-B", root / "openblas",
          **cmake_opts)


def _build_openblas(bglb):
    """
    Build the private archive using the existing bounded job count.
    """
    root = bglb.rootdir / "build" / "superlu"
    cmake("--build", root / "openblas", "-j", str(min(os.cpu_count() or 1, 8)))


def _install_openblas(bglb):
    """
    Install OpenBLAS into the private dependency prefix.
    """
    cmake("--install", bglb.rootdir / "build" / "superlu" / "openblas")


def clone_build_openblas(bglb):
    """
    Fetch, configure, build, and install the selected BLAS implementation.
    """
    _clone_openblas(bglb)
    _cmake_openblas(bglb)
    _build_openblas(bglb)
    _install_openblas(bglb)
