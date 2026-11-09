Install instructions

## Basic installation.
Basic installation can be done by 
```shell
  pip install piScope
  pip install mfem
  pip install petram
```
or
```shell
  pip install petram        # install mfem, Petra-M together
  pip install petram[gui]   # install mfem, piScope and Petra-M together
```
You can also download the file from this repository and install it separately.
```shell
  pip install ".[gui]"
```

For solving a FEM problem using MPI, it is necessary to install PyMFEM
with MPI option. Consult PyMFEM repository for its pip install options.

## source build

Source builds enable the native SuperLU_DIST interface by default. They build
private static PIC SuperLU_DIST, single-threaded OpenBLAS, and METIS/GKlib.
SuperLU_DIST and METIS use matching 32-bit sparse indices. The build requires 
C/C++ compilers with C++17 support, Make, Git, and MPI development headers 
and compiler wrappers.

Install mpi4py against the intended MPI, or use an existing compatible
installation. For an explicit source build of mpi4py:

```shell
MPICC=/path/to/mpi/bin/mpicc python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py
```

Build Petra-M from this checkout using the same MPI installation. Set MPICC and MPICXX
in the environment. Or, the build uses mpicc and mpicxx found on PATH:
```shell
python -m pip install .
```
Alternatively, the compiler wrappers can be explicitly set:
```shell
python -m pip install . \
  -C"MPICC=/path/to/mpi/bin/mpicc" \
  -C"MPICXX=/path/to/mpi/bin/mpicxx"
```

For a native opt-out installation, use:

```shell
python -m pip install . -Cwith-superlu_dist=No
```

## OpenCASCADE, python-occ, and gmesh
Petra-M relys on these librarys in its CAD and meshing interface. They need
to be installed separately, and available from the following repository.
Notice that we are using specific version of these modules, which may change
in future

 - gmsh  (4.10.5)  https://pypi.org/project/gmsh/
 - python-occ (7.9.0) https://github.com/tpaviot/pythonocc-core

## Additional modules
Petra-M features a modular code structure, enabling users to develop specialized 
physics equations and interfaces, or solvers by installing additional modules in 
future. An example of this capability is located within the Petra-M-RF repository 
under piScope. Public repositories can be installed via the package submenu.


