<img align="right" width="200" height="200" src="https://github.com/piScope/Petra-M/blob/master/resources/app_logo.png?raw=true">

## Petra-M

Petra-M (Physics Equation Translator for MFEM) is a FEM analysis GUI built
on the top of PyMFEM and piScope.

Petra-M includes
 - Weakform module to define PDE using MFEM integrators.
 - Distance module to define PDE for measuring distance
 - Interface for various FEM analysis solver 
 - (optional) Geometry editor module using OpenCascade and python-occ
 - (optional) Mesh generation using Gmsh

### Install
```shell
  pip install petram
```

### Licence
Petra-M project is released under the GPL v3 license.
See files COPYRIGHT and LICENSE file for full details.

### Optional dependency
 - gmsh  (4.10.5)  https://pypi.org/project/gmsh/
 - python-occ (7.9.0) https://github.com/tpaviot/pythonocc-core

### Reference
  S. Shiraiwa, et al. EPJ Web of Conferences 157, 03048 (2017)
