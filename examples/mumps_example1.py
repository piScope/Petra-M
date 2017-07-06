'''

example script of mumps_solve

>  mpirun -n 5 python2.7 example.py 

running example, answer should be (1,2)
Solution is : (    1.00      2.00)
running distributed matrix example, answer should be like rhs[i] = 1/n
Solution is : (0   1.0000)
Solution is : (1   0.5000)
Solution is : (2   0.3333)
Solution is : (3   0.2500)
Solution is : (4   0.2000)
Solution is : (5   0.1667)
Solution is : (6   0.1429)
Solution is : (7   0.1250)
Solution is : (8   0.1111)
Solution is : (9   0.1000)
'''
from mpi4py import MPI
import mfem_pi.solver.mumps.mumps_solve as mumps_solve

# answer should be (1,2)
myid     = MPI.COMM_WORLD.rank
if myid == 0:
   print('running example, answer should be (1,2)') 
mumps_solve.example(MPI.COMM_WORLD)

if myid == 0:
   print('running distributed matrix example, answer should be like rhs[i] = 1/n') 
mumps_solve.example_dist(MPI.COMM_WORLD)
