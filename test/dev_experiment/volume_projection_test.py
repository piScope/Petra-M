'''
 this test program map DoF from solmesh_1.mesh to solmesh_0.mesh.
 It maps the DoF in domain 1, ignororing the rest of domains

 test DoF H1 
 mpirun -np 4 python ~/python_lib/python_lib/experiment/volume_projection_test.py parallel

 test DoF Mapping of Nedelec (order=3)
 mpirun -np 4 python ~/python_lib/python_lib/experiment/volume_projection_test.py order=3 parallel nedelec
'''
 
import sys

import numpy as np
import petram.mfem_config

if 'parallel' in sys.argv: 
    petram.mfem_config.use_parallel = True
else:
    petram.mfem_config.use_parallel = False

order=1   
for x in sys.argv:
   if x.startswith('order'): order = int(x.split('=')[-1])


if petram.mfem_config.use_parallel:
   import mfem.par as mfem
   from mfem.common.mpi_debug import nicePrint, niceCall

   from mpi4py import MPI                               
   comm  = MPI.COMM_WORLD
   myid = MPI.COMM_WORLD.rank
   nump = MPI.COMM_WORLD.size
   smyid = '.'+'{:0>6d}'.format(myid)
   from mfem.common.mpi_debug import nicePrint
else:
   import mfem.ser as mfem
   smyid = ''
   from petram.helper.dummy_mpi import nicePrint

class coeff(mfem.PyCoefficient):
      def EvalValue(self, x):
          return  x[0]
class vcoeff(mfem.VectorPyCoefficient):
      def EvalValue(self, x):
          return  x[0], 0, 0

def test_nd():
    sdim = 3
    fec = mfem.ND_FECollection(order, sdim)
    
    if petram.mfem_config.use_parallel:
       mesh1 = mfem.Mesh('solmesh_1', 1, 1)       
       mesh1 = mfem.ParMesh(comm, mesh1)
       mesh1.ReorientTetMesh()
       sdim = mesh1.SpaceDimension()

       fes1 = mfem.ParFiniteElementSpace(mesh1, fec, 1)
       solr = mfem.ParGridFunction(fes1)
       
       mesh0 = mfem.Mesh('solmesh_0', 1, 1)
       mesh0 = mfem.ParMesh(comm, mesh0)       
       mesh0.ReorientTetMesh()
       fes2 = mfem.ParFiniteElementSpace(mesh0, fec, 1)
       
    else:
       mesh1 = mfem.Mesh('solmesh_1', 1, 1)
       mesh1.ReorientTetMesh()
       sdim = mesh1.SpaceDimension()

       fes1 = mfem.FiniteElementSpace(mesh1, fec, 1)
       solr = mfem.GridFunction(fes1)
       
       mesh0 = mfem.Mesh('solmesh_0'+smyid, 1, 1)
       mesh0.ReorientTetMesh()
       fes2 = mfem.FiniteElementSpace(mesh0, fec, 1)       

    u0 = vcoeff(3)       
    solr.ProjectCoefficient(u0)
    
    idx1 = [1]
    from petram.helper.dof_map import projection_matrix as pm
    # matrix to transfer unknown from trail to test


    M, row, col = pm(idx1, idx1, fes2, [], fes2=fes1,
                     mode='volume', tol=1e-4, filldiag=False)

    if petram.mfem_config.use_parallel:
        # in parallel, need a projection to TrueV space.
        from mfem.common.chypre import MfemVec2PyVec
        R = fes1.GetRestrictionMatrix()
        X = mfem.HypreParVector(fes1)
        X.SetSize(fes1.TrueVSize())
        R.Mult(solr, X)
        
        vec = MfemVec2PyVec(X)
        nicePrint(M.shape, ' solr ', solr.Size(), " X ", X.Size())
        v = M.dot(vec)[0]
        
        gf = mfem.ParGridFunction(fes2)        
        nicePrint("gf ", gf.GetDataArray().shape)
        P = fes2.GetProlongationMatrix()
        nicePrint(P.Height(), " ", P.Width())
        P.Mult(v, gf)
    else:        
        gf = mfem.GridFunction(fes2)
        print M.shape ,gf.Size(), solr.Size()                    
        v = mfem.Vector(M.dot(solr.GetDataArray()))
        gf.Assign(v)        


    gf.SaveToFile('mapped_nd'+smyid, 8)
    mesh0.PrintToFile('mapped_mesh'+smyid, 8)    

def test_h1():
    sdim = 3
    fec = mfem.H1_FECollection(order, sdim)
    
    if petram.mfem_config.use_parallel:
       mesh1 = mfem.Mesh('solmesh_1', 1, 1)       
       mesh1 = mfem.ParMesh(comm, mesh1)
       mesh1.ReorientTetMesh()
       sdim = mesh1.SpaceDimension()

       fes1 = mfem.ParFiniteElementSpace(mesh1, fec, 1)
       solr = mfem.ParGridFunction(fes1)
       
       mesh0 = mfem.Mesh('solmesh_0', 1, 1)
       mesh0 = mfem.ParMesh(comm, mesh0)       
       mesh0.ReorientTetMesh()
       fes2 = mfem.ParFiniteElementSpace(mesh0, fec, 1)
       
    else:
       mesh1 = mfem.Mesh('solmesh_1', 1, 1)
       mesh1.ReorientTetMesh()
       sdim = mesh1.SpaceDimension()

       fes1 = mfem.FiniteElementSpace(mesh1, fec, 1)
       solr = mfem.GridFunction(fes1)
       
       mesh0 = mfem.Mesh('solmesh_0'+smyid, 1, 1)
       mesh0.ReorientTetMesh()
       fes2 = mfem.FiniteElementSpace(mesh0, fec, 1)       

    u0 = coeff()       
    solr.ProjectCoefficient(u0)
    
    idx1 = [1]
    from petram.helper.dof_map import projection_matrix as pm
    # matrix to transfer unknown from trail to test


    M, row, col = pm(idx1, idx1, fes2, [], fes2=fes1,
                     mode='volume', tol=1e-4, filldiag=False)

    if petram.mfem_config.use_parallel:
        # in parallel, need a projection to TrueV space.
        from mfem.common.chypre import MfemVec2PyVec
        R = fes1.GetRestrictionMatrix()
        X = mfem.HypreParVector(fes1)
        X.SetSize(fes1.TrueVSize())
        R.Mult(solr, X)
        
        vec = MfemVec2PyVec(X)
        nicePrint(M.shape, ' solr ', solr.Size(), " X ", X.Size())
        v = M.dot(vec)[0]
        
        gf = mfem.ParGridFunction(fes2)        
        nicePrint("gf ", gf.GetDataArray().shape)
        P = fes2.GetProlongationMatrix()
        nicePrint(P.Height(), " ", P.Width())
        P.Mult(v, gf)
    else:        
        gf = mfem.GridFunction(fes2)
        print M.shape ,gf.Size(), solr.Size()                    
        v = mfem.Vector(M.dot(solr.GetDataArray()))
        gf.Assign(v)        


    gf.SaveToFile('mapped_h1'+smyid, 8)
    mesh0.PrintToFile('mapped_mesh'+smyid, 8)    
    

if __name__ == '__main__':
    if 'nedelec' in sys.argv:
        test_nd()
    else:
        test_h1()
