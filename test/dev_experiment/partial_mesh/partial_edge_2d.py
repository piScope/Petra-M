'''

   python partial_edge.py 
   python partial_edge.py  cone.mesh
   python partial_edge.py -p 


'''
import sys
import petram

use_parallel = False
filename = 'circle.mesh'
for param in sys.argv[1:]:
    if param == '-p':
        use_parallel = True
    else:
        filename = param


from petram.helper.load_mfem import load
mfem, MPI = load(use_parallel) 

print("use_parallel", use_parallel)    
from petram.mesh.partial_mesh import edge
from petram.mesh.mesh_utils import  get_extended_connectivity


mesh0 = mfem.Mesh(filename)

if use_parallel:
    mesh = mfem.ParMesh(MPI.COMM_WORLD, mesh0)
else:
    mesh = mesh0
    
get_extended_connectivity(mesh)

out_file = filename.split('.')[0]+"_edge.mesh"
edge(mesh, [1,2,3,4,5], filename=out_file, precision=8)
