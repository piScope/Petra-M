'''

  example of running mumps
   
     example      : the same as c_example in MUMPS
     example_dist : distributed version (ICNTL(18)=3)
'''
from mpi4py import MPI
from mfem_pi.solver.mumps.mumps_solve import DMUMPS, d_to_list, i_array, d_array, JOB_1_2_3
import sys

def example(no_outputs = True):
  comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['py_example_dynamic_child.py'],
                           maxprocs = 2)
  s = DMUMPS(1, 0, comm)

  rhs = [1.0, 4.0]
  s.set_n(2)
  s.set_nz(2)  
  s.set_irn(i_array([1,2]))
  s.set_jcn(i_array([1,2]))
  s.set_a(d_array([1.0, 2.0]))    
  s.set_rhs(d_array(rhs))

  # No outputs
  if no_outputs:
       s.set_icntl(1, -1)
       s.set_icntl(2, -1)
       s.set_icntl(3, -1)
       s.set_icntl(4,  0)

  s.set_job(JOB_1_2_3)
  s.run()
  s.finish()
  a = d_to_list(s.get_rhs(), len(rhs))
  print(a)

def example_dist(no_outputs = False):
  myid      = MPI.COMM_WORLD.rank
  num_procs = MPI.COMM_WORLD.size
  if myid == 0:  print("example 2 (distributed matrix)")      
  n = 2*num_procs;     # size of matrix
  nz = 2*num_procs;    # size of diagnal elements
  irn = [2*myid+1,2*myid+2]
  jcn = [2*myid+1,2*myid+2]
  a   = [2.*myid+1.,2.*myid+2.]
  rhs = [1.0]*2*num_procs
  
  s = DMUMPS()
  s.set_icntl(5,0)
  s.set_icntl(18,3)
  
  if myid ==0: 
     s.set_n(n)
     s.set_rhs(d_array(rhs))
     
  s.set_nz_loc(2)
  s.set_irn_loc(i_array(irn))
  s.set_jcn_loc(i_array(jcn))
  s.set_a_loc(d_array(a))
     
  # No outputs
  if no_outputs:
     s.set_icntl(1, -1)
     s.set_icntl(2, -1)
     s.set_icntl(3, -1)
     s.set_icntl(4,  0)

  s.set_job(JOB_1_2_3)
  s.run()
  s.finish()
  if (myid == 0):
      a = d_to_list(s.get_rhs(), len(rhs))
      print(a)
      

if __name__ == '__main__':

    example()

    example_dist()                
