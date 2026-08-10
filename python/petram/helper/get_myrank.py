#
#   rutine to check myid in both serial and parallel
#
#   note : do not load mpi4py in global here (!)
#
def get_myrank():
    from petram.mfem_config import use_parallel
    if use_parallel:
        from mpi4py import MPI
        myid = MPI.COMM_WORLD.rank
    else:
        myid = 0
    return myid
