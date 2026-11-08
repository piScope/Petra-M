"""
Build pinned SuperLU_MT, bundled C BLAS, and METIS/GKlib dependencies.
"""
import os
from build_utils import cmake, download

__all__ = ("clone_build_superlu_mt",)


def _clone_superlu_mt(bglb):
    """
    Fetch and verify the pinned SuperLU_MT source.
    """
    root = bglb.rootdir / "build" / "superlu_mt"
    download("superlu_mt", root / "sources")


def _clone_ordering(bglb):
    """
    Fetch and verify the pinned METIS and GKlib sources.
    """
    root = bglb.rootdir / "build" / "superlu_mt"
    for name in ("gklib", "metis"):
        download(name, root / "sources")


def _cmake_superlu_mt(bglb):
    """
    Configure the private static dependencies.
    """
    root = bglb.rootdir / "build" / "superlu_mt"
    sources = {name: root / "sources" / name
               for name in ("gklib", "metis", "superlu_mt")}
    prefix = root / "local"
    common_opts = {
        "DCMAKE_BUILD_TYPE": "Release",
        "DCMAKE_INSTALL_PREFIX": str(prefix),
        "DCMAKE_INSTALL_LIBDIR": "lib",
        "DCMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "DBUILD_SHARED_LIBS": "OFF",
        "DBUILD_TESTING": "OFF",
    }
    if bglb.cc:
        common_opts["DCMAKE_C_COMPILER"] = bglb.cc
    if bglb.cxx:
        common_opts["DCMAKE_CXX_COMPILER"] = bglb.cxx

    cmake_opts = common_opts.copy()
    cmake_opts.update({
        "DFETCHCONTENT_SOURCE_DIR_GKLIB": str(sources["gklib"]),
        "DIDXTYPEWIDTH": "32",
        "DREALTYPEWIDTH": "32",
        "DOPENMP": "OFF",
    })
    cmake("-S", sources["metis"], "-B", root / "metis", **cmake_opts)

    cmake_opts = common_opts.copy()
    cmake_opts.update({
        "DTHREAD_API": "PTHREAD",
        "DLONGINT": "OFF",
        "Denable_internal_blaslib": "ON",
        "Denable_fortran": "OFF",
        "Denable_examples": "OFF",
        "Denable_tests": "OFF",
    })
    cmake("-S", sources["superlu_mt"], "-B", root / "superlu_mt", **cmake_opts)


def _build_superlu_mt(bglb):
    """
    Build METIS/GKlib and SuperLU_MT with bundled C BLAS.
    """
    root = bglb.rootdir / "build" / "superlu_mt"
    num_jobs = min(os.cpu_count() or 1, 8)
    for name in ("metis", "superlu_mt"):
        cmake("--build", root / name, "-j", str(num_jobs))


def _install_superlu_mt(bglb):
    """
    Install the private static dependencies into the local prefix.
    """
    root = bglb.rootdir / "build" / "superlu_mt"
    for name in ("metis", "superlu_mt"):
        cmake("--install", root / name)


def clone_build_superlu_mt(bglb):
    """
    Fetch the sources and build the private SuperLU_MT dependency stack.
    """
    _clone_superlu_mt(bglb)
    _clone_ordering(bglb)
    _cmake_superlu_mt(bglb)
    _build_superlu_mt(bglb)
    _install_superlu_mt(bglb)
