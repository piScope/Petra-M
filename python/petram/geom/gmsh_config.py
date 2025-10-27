gmsh_init = False

try:
   import gmsh
   has_gmsh = True   
   if not gmsh_init:   
       gmsh.initialize()
   gmsh_init = True

except ImportError:
   has_gmsh = False
   import warnings
   warnings.warn("gmsh not found", RuntimeWarning)
   #assert False, "gmsh api is not found"


