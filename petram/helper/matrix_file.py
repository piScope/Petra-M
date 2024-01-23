from __future__ import print_function

'''
   matrix_file 
   
   a group of helper routine to read and write matrix/vector to file.


'''
import numpy as np
import os
import six


def read_matvec(file, all=False, verbose=False, complex=False, skip=0):
    '''
    read matrix/vector file.  
    If all is on, read all files with same basename ('matrix.0000, matrix.0001...')
    '''
    if not all:
        files = [file]
    else:
        dir = os.path.dirname(file)
        base = os.path.basename(file)
        files = []
        for x in os.listdir(dir):
            if x.find(base) != -1:
                files.append(x)
        files = sorted(files)
        files = [os.path.join(dir, f) for f in files]
        if verbose:
            six.print_(files)

    if len(files) == 0:
        return

    ret = []
    for file in files:
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


def read_coo_matrix(file, all=False, verbose=False, complex=False, skip=0):
    '''
    read coo matrix file
    If all is on, read all files with same basename ('matrix.0000, matrix.0001...')
    '''
    import scipy.sparse
    from scipy.sparse import coo_matrix

    if not all:
        files = [file]
    else:
        dir = os.path.dirname(file)
        base = os.path.basename(file)
        files = []
        for x in os.listdir(dir):
            if x.find(base) != -1:
                files.append(x)
        files = sorted(files)
        files = [os.path.join(dir, f) for f in files]
        if verbose:
            six.print_(files)

    if len(files) == 0:
        return

    retall = []
    for file in files:
        ret = []
        fid = open(file, "r")
        xx = [x.strip().split() for x in fid.readlines()]
        xx = xx[skip:]
        if complex:
            xxx = [[np.complex(x) for x in y] for y in xx]
        else:
            xxx = [[np.float(x) for x in y] for y in xx]
        fid.close()
        ret.append(np.array(xxx))
        retall.append(np.vstack(ret))

    w = int(np.max([np.max(x[:, 1]) for x in retall]))+1
    h = [int(np.max(x[:, 0]))+1 for x in retall]

    if len(retall[0][0]) == 4:
        mats = [coo_matrix((m[0][:, 2]+m[:, 3]*1j, (m[0][:, 0].astype(int), m[0][:, 1].astype(int))), shape=(hh, w))
                for hh, m in zip(h, retall)]
    elif len(retall[0][0]) == 3:
        mats = [coo_matrix((m[0][:, 2], (m[0][:, 0].astype(int), m[0][:, 1].astype(int))), shape=(hh, w))
                for hh, m in zip(h, retall)]

    mat = scipy.sparse.vstack(mats)

    return mat


def write_matrix(file, m):
    from petram.mfem_config import use_parallel
    if use_parallel:
        from mpi4py import MPI
        num_proc = MPI.COMM_WORLD.size
        myid = MPI.COMM_WORLD.rank
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
        myid = MPI.COMM_WORLD.rank
        smyid = '.'+'{:0>6d}'.format(myid)
    else:
        smyid = ''

    if hasattr(bb, "SaveToFile"):   # GridFunction
        bb.SaveToFile(file+smyid, 8)
    else:
        fid = open(file+smyid, "w")
        if np.iscomplexobj(bb):
            for k, x in enumerate(bb):
                fid.write(str(k) + ' ' + "{0:.8g}".format(x.real) + ' ' +
                          "{0:.8g}".format(x.imag) + '\n')
        else:
            for k, x in enumerate(bb):
                fid.write(str(k) + ' ' + "{0:.8g}".format(x) + '\n')
        fid.close()


def write_coo_matrix(file, A):
    from petram.mfem_config import use_parallel
    if use_parallel:
        from mpi4py import MPI
        num_proc = MPI.COMM_WORLD.size
        myid = MPI.COMM_WORLD.rank
        smyid = '.'+'{:0>6d}'.format(myid)
    else:
        smyid = ''

    if (A.dtype == 'complex'):
        is_complex = True
    else:
        is_complex = False

    fid = open(file+smyid, 'w')
    rc = np.vstack((A.row, A.col)).transpose()
    tmp = sorted([(k, tuple(x)) for k, x in enumerate(rc)], key=lambda x: x[1])
    idx = np.array([x[0] for x in tmp])
    if len(idx) == 0:
        fid.close()
        return

    row = A.row[idx]
    col = A.col[idx]
    data = A.data[idx]

    if is_complex:
        txt = [' '.join([str(int(r)), str(int(c)), "{0:.8g}".format(a.real),
                         "{0:.8g}".format(a.imag)]) for r, c, a in zip(row, col, data)]
        fid.write('\n'.join(txt) + "\n")
    else:
        txt = [' '.join([str(int(r)), str(int(c)), "{0:.8g}".format(a.real),
                         "{0:.8g}".format(a.imag)]) for r, c, a in zip(row, col, data)]
        fid.write('\n'.join(txt) + "\n")
    fid.close()
