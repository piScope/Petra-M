from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from codecs import open
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_build_system"))
from build_config import configure_build, print_config
from build_globals import bglb
from build_superlu_mt import clone_build_superlu_mt
from build_superlumt_solve import (cmake_superlumt_solve,
                                  generate_superlumt_solve_wrapper)


class BuildExt(build_ext):
    """
    Build the interface library, generate SWIG, and compile the Python extension.
    """

    def run(self):
        if not self.extensions:
            return

        import numpy

        bglb.bdist_wheel_prefix = Path(self.build_lib).resolve()
        bglb.build_temp = Path(self.build_temp).resolve()
        package = bglb.bdist_wheel_prefix / "petram" / "ext" / "superlu_mt"
        package.mkdir(parents=True, exist_ok=True)
        bglb.build_temp.mkdir(parents=True, exist_ok=True)
        try:
            clone_build_superlu_mt(bglb)
            cmake_superlumt_solve(bglb)
            wrapper = generate_superlumt_solve_wrapper(bglb)
            source = bglb.rootdir / "superlumt_solve"
            extension = self.extensions[0]
            extension.sources = [os.path.relpath(wrapper, bglb.rootdir)]
            extension.include_dirs = [numpy.get_include(), str(source)]
            extension.library_dirs = [str(package)]
            extension.libraries = ["superlumt_solve"]
            extension.extra_compile_args = ["-std=c++17", "-fvisibility=hidden"]
            extension.runtime_library_dirs = ["@loader_path" if sys.platform == "darwin"
                                               else "$ORIGIN"]
            if sys.platform == "linux":
                extension.extra_link_args = [
                    "-Wl,--version-script=" + str(source / "python_exports.txt")]
            # Use the selected compilers for both CMake and the SWIG extension,
            # following Petra-M--DS's BuildExt.
            if bglb.cc:
                os.environ["CC"] = bglb.cc
            if bglb.cxx:
                os.environ["CXX"] = bglb.cxx
            super().run()
        except Exception as error:
            raise RuntimeError("SuperLU_MT build failed. To omit the native component, "
                               "use -Cwith-superlu-mt=No. Original error: " + str(error)) from error


def long_description():
    rootdir = os.path.abspath(os.path.dirname(__file__))    
    with open(os.path.join(rootdir, 'README.md'), encoding='utf-8') as f:
        return f.read()

def run_setup():
    configure_build(bglb)
    print_config(bglb)
    extensions = []
    if bglb.with_superlu_mt:
        extensions.append(Extension(
            "petram.ext.superlu_mt._superlumt_solve",
            ["superlumt_solve/superlumt_solve.i"], language="c++"))
    setup(
       ext_modules = extensions,
       cmdclass = {"build_ext": BuildExt},
       long_description = long_description(),
       long_description_content_type = "text/markdown",)

def main():
    run_setup()


if __name__ == '__main__':
    main()

