import numpy as np

from petram.mfem_config import use_parallel
if use_parallel:
    from mpi4py import MPI                               
    num_proc = MPI.COMM_WORLD.size
    myid     = MPI.COMM_WORLD.rank
    smyid = '.'+'{:0>6d}'.format(myid)
else:
    smyid = ''

def load_probe(name):
    fid = open(name, 'r')
    format = int(fid.readline().split(':')[-1])

    if format == 0:
        value = load_format_0(fid)
    fid.close()
    return value
    
def load_format_0(fid):
    lines = fid.readlines()
    data = [[float(x) for x in l.split(',')] for l in lines]
    data = np.array(data)
    return data
    
class Probe(object):
    def __init__(self, name, idx):
        self.name = name
        self.sig = []
        self.t = []        
        self.idx = idx
        self.finalized = False

    def write_file(self, filename = None):
        if not self.finalized:
            valid = self.finalize()
        if not valid: return
        
        if filename is None:
            filename = 'probe_'+self.name + smyid
            
        fid = open(filename, 'w')
        fid.write("format : 0\n")
        print self.sig.shape, self.time.shape
        for x, t in zip(self.sig, self.time):
           txt = ', '.join([str(xx) for xx in x])
           fid.write(str(t) + ', '+ txt +"\n")
        fid.close()

    def append_sol(self, sol, t):
        self.sig.append(np.atleast_1d(sol[self.idx].toarray().flatten()))
        self.t.append(t)

    def current_value(self, sol):
        return np.atleast_1d(sol[self.idx].toarray().flatten())
        
    def print_signal(self):
        if not self.finalized:
            self.finalize()
        
    def finalize(self):
        if len(self.sig) == 0:
            self.sig = -1
            self.valid = False
        else:
            self.sig = np.vstack(self.sig)
            self.time= np.hstack(self.t)                    
            self.valid = True
        self.finalized = True
        return self.valid
        
