from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from codecs import open
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_build_system"))
from build_config import configure_build, print_config
from build_globals import bglb
from build_mpi import configure_mpi
from build_openblas import clone_build_openblas
from build_superlu import clone_build_superlu
from build_superlu_solve import (cmake_superlu_solve,
                                generate_superlu_solve_wrapper)


class BuildExt(build_ext):
    """
    Build the interface library, generate SWIG, and compile the Python extension.
    """

    def run(self):
        if not self.extensions:
            return

        import numpy
        import mpi4py

        bglb.bdist_wheel_prefix = Path(self.build_lib).resolve()
        bglb.build_temp = Path(self.build_temp).resolve()
        package = bglb.bdist_wheel_prefix / "petram" / "ext" / "superlu"
        package.mkdir(parents=True, exist_ok=True)
        bglb.build_temp.mkdir(parents=True, exist_ok=True)
        try:
            source = bglb.rootdir / "superlu_solve"
            configure_mpi(bglb)
            clone_build_openblas(bglb)
            clone_build_superlu(bglb)
            cmake_superlu_solve(bglb)
            wrapper = generate_superlu_solve_wrapper(bglb)
            extension = self.extensions[0]
            extension.sources = [os.path.relpath(wrapper, bglb.rootdir)]
            extension.include_dirs = [numpy.get_include(), mpi4py.get_include(), str(source)]
            extension.library_dirs = [str(package)]
            extension.libraries = ["superlu_solve"]
            extension.extra_compile_args = ["-std=c++17", "-fvisibility=hidden"]
            extension.runtime_library_dirs = ["@loader_path" if sys.platform == "darwin"
                                               else "$ORIGIN"]
            if sys.platform == "linux":
                extension.extra_link_args = [
                    "-Wl,--version-script=" + str(source / "python_exports.txt")]
            # Use the selected compilers for both CMake and the SWIG extension,
            # following Petra-M--DS's BuildExt.
            if bglb.mpicc:
                os.environ["CC"] = bglb.mpicc
            if bglb.mpicxx:
                os.environ["CXX"] = bglb.mpicxx
            super().run()
        except Exception as error:
            raise RuntimeError("SuperLU_DIST build failed. To omit the native component, "
                               "use -Cwith-superlu_dist=No. Original error: " + str(error)) from error


def long_description():
    rootdir = os.path.abspath(os.path.dirname(__file__))    
    with open(os.path.join(rootdir, 'README.md'), encoding='utf-8') as f:
        return f.read()

def run_setup():
    configure_build(bglb)
    print_config(bglb)
    extensions = []
    if bglb.with_superlu_dist:
        extensions.append(Extension(
            "petram.ext.superlu._superlu_solve",
            ["superlu_solve/superlu_solve.i"], language="c++"))
    setup(
       ext_modules = extensions,
       cmdclass = {"build_ext": BuildExt},
       long_description = long_description(),
       long_description_content_type = "text/markdown",)

def main():
    run_setup()


if __name__ == '__main__':
    main()

