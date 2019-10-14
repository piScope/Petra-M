'''
   Global Named List

      A Named List having a capability to share its data among MPI proc.

      node 1: {1: [1,2,3], 2: [1,2]}
      node 2: {1: [4], 4: [5]}

      d = GNameList(dtype=<type>, gkey=<iter>, freezekey=<bool>)

      d.freezekey
         prohibit to add new key
      d.sharekeys:
         to make sure all node has the same dict keys.
         (note, in MPI communication, member access is done using self.gkey)
      d.allgather:
         {1: [1,2,3,4], 2:[1,2], 4:[5]}
      d.gather(root=<int>, distributed=<bool>):
         gather to particular node
         if distributed: n-th item is gather to n = myid % nprc 

'''
import numpy as np

from petram.mfem_config import use_parallel
if use_parallel:
    try:
        from mpi4py import MPI
        myid = MPI.COMM_WORLD.rank
        #nprc = MPI.COMM_WORLD.size
        #this is because even if nprc > 1, we may want this behave as if nprc = 1
        comm  = MPI.COMM_WORLD
        from mfem.common.mpi_debug import nicePrint, niceCall        
        from petram.helper.mpi_recipes import allgather, allgather_vector, gather_vector
        hasMPI = True
    except ImportError:
        hasMPI = False
else:
    hasMPI = False
    
class GlobalNamedList(dict):
    def __init__(self, *args, **kwargs):
        self.dtype = kwargs.pop('dtype', int)
        self._gkey = kwargs.pop('gkey', ())# global keys (ordered)
        self._freezekey = kwargs.pop('freezekey', False)
        super(GlobalNamedList, self).__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if self._freezekey and key not in self:
            raise KeyError("{} is not a legal key of this StricDict".format(repr(key)))
        dict.__setitem__(self, key, value)

    def __getitem__(self, key):
        if not self._freezekey and not key in self:
            dict.__setitem__(self, key, [])
        return dict.__getitem__(self, key)

    @property
    def globalkeys(self):
        return self._gkey
    @property
    def freezekey(self):
        return self._freezekey
    @freezekey.setter
    def freezekey(self, value):
        self._freezekey = value

    def setlists(self, keys, values):
        '''
        [1,2,4,1], [1,2,3,7] -> [1:[1,7], 2:[2], 4:[1]]
        values are stored as numpy array

        '''
        values = np.array(values, copy=False, dtype=self.dtype)
        k, idx = np.unique(keys, return_inverse=True)
        for i in range(len(k)):
            dict.__setitem__(self, k[i], values[np.where(idx == i)[0]])
            
    def sort(self):
        '''
        sort eacy item
        '''
        for k in self:
            self[k] = np.sort(self[k]).astype(self.dtype, copy=False)
        return self
    
    def unique(self):            
        for k in self:
            self[k] = np.unique(self[k]).astype(self.dtype, copy=False)
        return self
            
    def sharekeys(self):
        keys = list(self.keys())
        if not hasMPI:
            keys = np.unique(keys)
            self._gkey = keys
            return self
        
        keys = sum(allgather(keys), [])
        keys = np.unique(keys)
        freezekey = self._freezekey
        for k in keys:
            if not k in self: self[k] = np.atleast_1d([]).astype(self.dtype)
        self._freezekey = freezekey
        self._gkey = keys
        return self
        
    def allgather(self, overwrite=True):
        if not hasMPI: return
        dest = self if overwrite else GlobalNamedList()       
        for key in self._gkey:
            if key in self:
                data = np.array(self[key], dtype=self.dtype, copy=False)
            else:
                data = np.atleast_1d([]).astype(self.dtype)
            data = allgather_vector(data)
            dest[key] = data
        return dest
                                
    def gather(self, nprc, root=None, distribute = False, overwrite=True):
        # we don't take nprc from MPI.COMM_WROLD (see a comment above)
        if nprc == 1:
            dest = self if overwrite else GlobalNamedList()            
            if not overwrite:
                for key in self:
                    dest[key] = np.array(self[key]).copy()
            return dest
        dest = self if overwrite else GlobalNamedList()
        for j, key in enumerate(self._gkey):
            if key in self:
                data = np.array(self[key], dtype=self.dtype, copy=False)
            else:
                data = np.atleast_1d([]).astype(self.dtype)
            r = 0 if root is None else root
            r = j % nprc if distribute else r
            data = gather_vector(data, root=r)
            if myid == r:
                dest[key] = data
            else:
                if key in dest: del dest[key]
        return dest
        
    def bcast(self, nprc, root = None, distributed = False, overwrite=True):
        # we don't take nprc from MPI.COMM_WROLD (see a comment above)
        if nprc == 1:
            dest = self if overwrite else GlobalNamedList()            
            if not overwrite:
                for key in self:
                    dest[key] = np.array(self[key]).copy()
            return dest
        dest = self if overwrite else GlobalNamedList()      
        for j, key in enumerate(self._gkey):
            r = 0 if root is None else root
            r = j % nprc if distributed else r
            
            if myid == r:
                if key in self:
                    data = np.array(self[key], dtype=self.dtype, copy=False)
                else:
                    data = np.atleast_1d([]).astype(self.dtype)
            else:
                data = None
            data = comm.bcast(data, root=r)
            dest[key] = data
        return dest
