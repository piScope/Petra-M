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
   1) decide install destination and set enviromental variables, PetraM
   
      example)
          export PetraM=/usr/local/PetraM  (public place)
          export PetraM=/Users/shiraiwa/sandbox_root/PetraM (user directory)

   2) add PYTHONPATH
          export PYTHONPATH=$PetraM/lib/python2.7/site-packages:$PYTHONPATH

   3) build
      make 

   4) install
      make install
