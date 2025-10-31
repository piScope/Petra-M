<img align="right" width="200" height="200" src="https://github.com/piScope/PetraM_Base/blob/master/resources/app_logo.png?raw=true">

## Petra-M (FEM application based on MFEM)

Petra-M (Physics Equation Translator for MFEM) is a physics layer built
on the top of PyMFEM, a python wrapper for Modular FEM library
(MFEM: http://mfem.org). 

Petra-M includes
 - Physics modeling GUI interface on piScope.
 - Weakform module to define PDE using MFEM integrators.
 - Distance module to define PDE for measuring distance
 - (optional) Geometry editor module using OpenCascade and python-occ
 - (optional) Mesh generation using Gmsh

Additinal Petra-M submodules are available from different repository.
 - PetraM-RF : 3D frequency domain Maxwell equation
 - PetraM-DS : interface to use direct solvers (in progress)


### Licence
Petra-M project is released under the GPL v3 license.
See files COPYRIGHT and LICENSE file for full details.

### Optional dependency
 - gmsh  (4.10.5)  https://pypi.org/project/gmsh/
 - python-occ (7.9.0) https://github.com/tpaviot/pythonocc-core

### Reference
  S. Shiraiwa, et al. EPJ Web of Conferences 157, 03048 (2017)




