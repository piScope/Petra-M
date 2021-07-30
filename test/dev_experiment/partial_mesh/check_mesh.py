import mfem.ser as mfem

from petram.mesh.partial_mesh import *
from petram.mesh.mesh_utils import  get_extended_connectivity

mesh = mfem.Mesh('cone_edge.mesh')

get_extended_connectivity(mesh)

print(mesh.extended_connectivity)

from petram.mesh.read_mfemmesh1 import extract_refined_mesh_data1

print("edge", mesh.GetNEdges())
print(extract_refined_mesh_data1(mesh, 3))
