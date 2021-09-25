'''
 testing Alltoall (variable length vector version of alltoall)
'''
from mpi4py import MPI
import numpy as np
from mfem.common.mpi_dtype import  get_mpi_datatype
from mfem.common.mpi_debug import nicePrint, niceCall
from petram.helper.mpi_recipes import alltoall_vector

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

a_size = 1
orgdata = [(rank+1)*np.arange(x+1, dtype=int) for x in range(size)]
senddata = [(rank+1)*np.arange(x+1, dtype=int) for x in range(size)]

print("process %s sending %s  " % (rank,senddata))

sendsize = np.array([len(x.flatten()) for x in senddata], dtype=int)
senddisp = list(np.hstack((0, np.cumsum(sendsize)))[:-1])

if len(sendsize) != size:
    assert False, "senddata size does not match with mpi size"
recvsize = np.empty(size, dtype=int)

disp = list(range(size))
counts = [1]*size
dtype = get_mpi_datatype(sendsize)

s1 = [sendsize, counts, disp, dtype]
r1 = [recvsize, counts, disp, dtype]
comm.Alltoallv(s1, r1)

print("process %s receiving %s  " % (rank, recvsize))

recvsize = list(recvsize)
recvdisp = list(np.hstack((0, np.cumsum(recvsize)))[:-1])
recvdata = np.empty(np.sum(recvsize), dtype=int)
senddata = np.hstack(senddata).flatten()

dtype = get_mpi_datatype(senddata[0])
s1 = [senddata, sendsize, senddisp, dtype]
r1 = [recvdata, recvsize, recvdisp, dtype]
comm.Alltoallv(s1, r1)

hoge = alltoall_vector(orgdata)
nicePrint(hoge)
hoge = alltoall_vector(hoge)
nicePrint(hoge)
nicePrint("process %s sending %s receiving %s " % (rank,senddata,  r1[0]))
