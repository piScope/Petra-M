## Petra-M

Petra-M (Physics Equation Translator for MFEM) is a physics layer built
on the top of PyMFEM, a python wrapper for Modular FEM library
(MFEM: http://mfem.org). A goal of Petra-M is to provide a
tool to define PDE problem using a regular physics
terminology. Petra-M tools to fill a linear system for
PDE, which could be solved by a solver.

 - HypreParCSR matrix utility. 
 - NASTRAN file converter to MFEM mesh format.
 - Physics modeling interface on piScope 

### Build/Install
   These steps are common to all PetraM module installation

   1) decide install destination and set enviromental variables, PetraM
   
      example)
         - export PetraM=/usr/local/PetraM  (public place)
         - export PetraM=/Users/shiraiwa/sandbox_root/PetraM (user directory)

   2) add PYTHONPATH
      - if you dont have, make python package directory 
           mkdir -p $PetraM/lib/python2.7/site-packages
      - set PYTHONPATH
           export PYTHONPATH=$PetraM/lib/python2.7/site-packages:$PYTHONPATH

   4) prepare Makefile
       use a file in Makefile_templates.
       Makefile_default may work well.

   5) build
      - make 

   6) install
      - make install
