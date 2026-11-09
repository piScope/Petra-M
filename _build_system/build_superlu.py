"""
Build pinned SuperLU_DIST and private METIS/GKlib using the existing layout.
"""
import os
import subprocess

from build_utils import cmake, download

__all__ = ("clone_build_superlu",)


def _clone_superlu(bglb):
    """
    Fetch SuperLU_DIST and apply the bundled METIS and MC64-exclusion patches.
    """
    source = download("superlu_dist", bglb.rootdir / "build" / "superlu" / "sources")
    for name in ("superlu-dist-metis-only.patch", "superlu-dist-no-mc64.patch"):
        patch = bglb.rootdir / "_build_system" / "patches" / name
        applied = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)],
                                 cwd=source, capture_output=True, text=True)
        if applied.returncode != 0:
            subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
            subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)


def _clone_ordering(bglb):
    """
    Fetch the pinned METIS and GKlib sources.
    """
    root = bglb.rootdir / "build" / "superlu"
    for name in ("gklib", "metis"):
        download(name, root / "sources")


def _common_cmake_options(bglb):
    """
    Configure private static PIC dependencies with the selected compilers.
    """
    prefix = bglb.rootdir / "build" / "superlu" / "local"
    cmake_opts = {
        "DCMAKE_BUILD_TYPE": "Release",
        "DCMAKE_INSTALL_PREFIX": str(prefix),
        "DCMAKE_INSTALL_LIBDIR": "lib",
        "DCMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "DBUILD_SHARED_LIBS": "OFF",
        "DBUILD_TESTING": "OFF",
    }
    if bglb.cc:
        cmake_opts["DCMAKE_C_COMPILER"] = bglb.cc
    if bglb.cxx:
        cmake_opts["DCMAKE_CXX_COMPILER"] = bglb.cxx
    return cmake_opts


def _cmake_ordering(bglb):
    """
    Configure int32 METIS/GKlib without OpenMP.
    """
    root = bglb.rootdir / "build" / "superlu"
    cmake_opts = _common_cmake_options(bglb)
    cmake_opts.update({
        "DFETCHCONTENT_SOURCE_DIR_GKLIB": str(root / "sources" / "gklib"),
        "DIDXTYPEWIDTH": "32",
        "DREALTYPEWIDTH": "32",
        "DOPENMP": "OFF",
    })
    cmake("-S", root / "sources" / "metis", "-B", root / "metis", **cmake_opts)


def _build_ordering(bglb):
    """
    Build the private ordering libraries before SuperLU_DIST.
    """
    root = bglb.rootdir / "build" / "superlu"
    cmake("--build", root / "metis", "-j", str(min(os.cpu_count() or 1, 8)))


def _install_ordering(bglb):
    """
    Install METIS/GKlib for SuperLU_DIST's configure and link steps.
    """
    root = bglb.rootdir / "build" / "superlu"
    cmake("--install", root / "metis")


def _cmake_superlu(bglb):
    """
    Configure static PIC, int32 SuperLU_DIST with METIS and no OpenMP.
    """
    root = bglb.rootdir / "build" / "superlu"
    prefix = root / "local"
    archive = prefix / "lib" / "libopenblas.a"
    if not archive.is_file():
        raise RuntimeError("Build the private single-threaded OpenBLAS first: " + str(archive))

    cmake_opts = _common_cmake_options(bglb)
    cmake_opts.update({
        "DMPI_C_COMPILER": bglb.mpicc,
        "DMPI_CXX_COMPILER": bglb.mpicxx,
        "DXSDK_INDEX_SIZE": "32",
        "DXSDK_ENABLE_Fortran": "OFF",
        "Denable_openmp": "OFF",
        "Denable_python": "OFF",
        "Denable_single": "OFF",
        "Denable_double": "ON",
        "Denable_complex16": "ON",
        "Denable_examples": "OFF",
        "Denable_tests": "OFF",
        "DTPL_ENABLE_INTERNAL_BLASLIB": "OFF",
        "DTPL_BLAS_LIBRARIES": str(archive),
        "DTPL_ENABLE_METISLIB": "ON",
        "DTPL_METIS_INCLUDE_DIRS": str(prefix / "include"),
        "DTPL_METIS_LIBRARIES": ";".join(str(prefix / "lib" / name)
                                        for name in ("libmetis.a", "libGKlib.a")),
        "DTPL_ENABLE_PARMETISLIB": "OFF",
        "DTPL_ENABLE_LAPACKLIB": "OFF",
        "DTPL_ENABLE_COMBBLASLIB": "OFF",
        "DTPL_ENABLE_CUDALIB": "OFF",
        "DTPL_ENABLE_HIPLIB": "OFF",
    })
    cmake("-S", root / "sources" / "superlu_dist", "-B", root / "superlu_dist",
          **cmake_opts)


def _build_superlu(bglb):
    """
    Build SuperLU_DIST against the installed private dependencies.
    """
    root = bglb.rootdir / "build" / "superlu"
    cmake("--build", root / "superlu_dist", "-j", str(min(os.cpu_count() or 1, 8)))


def _install_superlu(bglb):
    """
    Install the private static SuperLU_DIST library.
    """
    root = bglb.rootdir / "build" / "superlu"
    cmake("--install", root / "superlu_dist")


def clone_build_superlu(bglb):
    """
    Fetch and build the MPI SuperLU dependency stack after OpenBLAS.
    """
    _clone_superlu(bglb)
    _clone_ordering(bglb)
    _cmake_ordering(bglb)
    _build_ordering(bglb)
    _install_ordering(bglb)
    _cmake_superlu(bglb)
    _build_superlu(bglb)
    _install_superlu(bglb)
