'''
 testing Alltoallv (variable length vector version of alltoall)

 mpirun -np 3 python mpi_alltoallv.py
'''
from mpi4py import MPI
import numpy as np
from mfem.common.mpi_dtype import  get_mpi_datatype
from mfem.common.mpi_debug import nicePrint, niceCall
from petram.helper.mpi_recipes import alltoall_vector, alltoall_vectorv

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

a_size = 1

if rank == 0:
    data = [[np.arange(x, dtype="float64")*x*y for x in range(size)]
            for y in range(size)]
else:
    data = [[np.ones(2, dtype="float64")*rank for x in range(rank)]
            for y in range(size)]

nicePrint(data)
hoge = alltoall_vectorv(data)
nicePrint(hoge)
hoge = alltoall_vectorv(hoge)
nicePrint(hoge)

