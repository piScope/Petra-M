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
Petra-M features a modular code structure, enabling users to incorporate specialized 
physics equations and interfaces, or solvers  by installing additional modules in 
future. An example of this capability is located within the Petra-M-RF repository 
under piScope. Public repositories can be installed via the package submenu


