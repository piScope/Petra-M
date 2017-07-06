from mpi4py import MPI
from mumps_solve import DMUMPS, d_to_list, i_array, d_array, JOB_1_2_3

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

s = DMUMPS(1, 0, comm)
s.set_job(JOB_1_2_3)
s.run()
s.finish()

comm.Disconnect()
