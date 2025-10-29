'''

   MPI helper

   NOTE: do NOT Load mpi4py in global space in this module
'''
from mfem.common.mpi_debug import nicePrint, niceCall


def nicePrint1(*s):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    nproc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank

    if myid == 0:
        print("[ID:0]" + ' '.join([str(ss) for ss in s]))
