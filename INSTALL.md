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

## OpenCASCADE, python-occ, and gmesh
Petra-M relys on these librarys in its CAD and meshing interface. They need
to be installed separately, and available from the following repository.
Notice that we are using specific version of these modules, which may change
in future

 - gmsh  (4.10.5)  https://pypi.org/project/gmsh/
 - python-occ (7.9.0) https://github.com/tpaviot/pythonocc-core

## Additional modules
### RF module (Petra-M--RF)
 Physics module for frequency domain Maxwell problem in 1D/2D/3D
 geometry and plasma wave interfaces is distributed separately.

### Direct solver interface (Petra-M--DS)
 Interface to direct solver is developed in Petra-M--DS. Current
 version supports MUMPS, and a work to add STRUMPACK interface
 is in progress.


