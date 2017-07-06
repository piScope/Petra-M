'''
   matrix_file 
   
   a group of helper routine to read and write matrix/vector to file.


'''
import numpy as np
import os
import six

def read_matvec(file, all = False, verbose=False, complex = False, skip = 0):
    ''' 
    read matrix/vector file.  
    If all is on, read all files with same basename ('matrix.0000, matrix.0001...')
    '''
    if not all:  
        files = [file]
    else:
        dir  = os.path.dirname(file)
        base = os.path.basename(file)
        files= []
        for x in os.listdir(dir):
            if x.find(base) != -1:
                 files.append(x)
        files = sorted(files)
        files = [os.path.join(dir, f) for f in files]
        if verbose: six.print_(files)
 
    if len(files) == 0: return

    ret = []
    for file in files:   
       print file         
       fid = open(file, "r")
       xx = [x.strip().split() for x in fid.readlines()]
       xx = xx[skip:]
       if complex:
           xxx = [[np.complex(x) for x in y] for y in xx]
       else:
           xxx = [[np.float(x) for x in y] for y in xx]
       fid.close()
       ret.append(np.array(xxx))
    return np.vstack(ret)

def write_matrix(file, m):
    from petram.mfem_config import use_parallel
    if use_parallel:
       from mpi4py import MPI                               
       num_proc = MPI.COMM_WORLD.size
       myid     = MPI.COMM_WORLD.rank
       smyid = '.'+'{:0>6d}'.format(myid)
    else:
       smyid = ''
    if hasattr(m, 'save_data'):
       m.save_data(file + smyid)
    else:
       raise NotImplemented("write matrix not implemented for" + m.__repr__())


def write_vector(file, bb):
    from petram.mfem_config import use_parallel
    if use_parallel:
       from mpi4py import MPI                               
       num_proc = MPI.COMM_WORLD.size
       myid     = MPI.COMM_WORLD.rank
       smyid = '.'+'{:0>6d}'.format(myid)
    else:
       smyid = ''

    if hasattr(bb, "SaveToFile"):   # GridFunction
        bb.SaveToFile(file+smyid, 8)
    else:
        fid = open(file+smyid, "w")
        for k, x in enumerate(bb):
            fid.write(str(k) + ' ' + str(x) +'\n')
        fid.close()

def write_coo_matrix(file, A):
    from petram.mfem_config import use_parallel
    if use_parallel:
       from mpi4py import MPI                               
       num_proc = MPI.COMM_WORLD.size
       myid     = MPI.COMM_WORLD.rank
       smyid = '.'+'{:0>6d}'.format(myid)
    else:
       smyid = ''

    if (A.dtype == 'complex'):
        is_complex = True
    else:
        is_complex = False

    fid = open(file+smyid, 'w')
    if is_complex:
        for r,c,a in zip(A.row, A.col, A.data):
            txt = [str(int(r)), str(int(c)), "{0:.5g}".format(a.real), "{0:.5g}".format(a.imag)]
            fid.write(' '.join(txt) + "\n")
    else:
        for r,c,a in zip(A.row, A.col, A.data):
            txt = [str(r), str(c), "{0:.5g}".format(a)]
            fid.write(' '.join(txt) + "\n")
    fid.close()
