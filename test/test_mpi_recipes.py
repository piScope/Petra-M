#
#  this routine test 2 subroutine in mpi_recipes
#
#  mpirun -np 4 python test_mpi_recepies.py
#
from __future__ import print_function

from mpi4py import MPI
import numpy as np

comm     = MPI.COMM_WORLD     
num_proc = MPI.COMM_WORLD.size
myid     = MPI.COMM_WORLD.rank

from petram.helper.mpi_recipes import distribute_vec_from_head
if myid == 0:
    b = np.arange(100)*1j
    b0 = distribute_vec_from_head(b)
else:
    b0 = distribute_vec_from_head(None)

print(myid, str(list(b0)))

size = 15

from scipy.sparse import lil_matrix

M = lil_matrix((size, size), dtype = complex)

from random import seed, random
seed(myid)

if myid == 0:
    for i in range(int(size**2/30)):
        r = int(random()*size)
        c = int(random()*size)
        M[r, c] = random() + 1j*random()
    print(M)
    
A = M.tocoo()    
from petram.helper.mpi_recipes import distribute_global_coo

A_local = distribute_global_coo(A)

for i in range(num_proc):
    if i == myid:
        print(myid)
        print(A_local.shape, A_local)
    comm.Barrier()

