'''
Utility class to handle BlockMatrix made from scipy-sparse and
Hypre with the same interface
'''
import numpy as np
import scipy
from scipy.sparse import coo_matrix, spmatrix, lil_matrix, csc_matrix

from petram.mfem_config import use_parallel
import mfem.common.chypre as chypre

if use_parallel:
   from petram.helper.mpi_recipes import *
   from mfem.common.parcsr_extra import *
   import mfem.par as mfem
   default_kind = 'hypre'
else:
   import mfem.ser as mfem
   default_kind = 'scipy'

from petram.solver.solver_utils import make_numpy_coo_matrix
from petram.helper.matrix_file import write_coo_matrix, write_vector

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('BlockMatrix')
format_memory_usage = debug.format_memory_usage

class One(object):
    '''
    An identity matrix (used in P and mimic 1*X = X)
    '''
    def __init__(self, ref):
        self._shape = ref.shape
        self._is_hypre = False
        if hasattr(ref, "GetColPartArray"):
           self._cpart = ref.GetColPartArray()
           self._is_hypre = True
        if hasattr(ref, "GetRowPartArray"):
           self._rpart = ref.GetRowPartArray()
           self._is_hypre = True

    def __repr__(self):
        return "IdentityMatrix"
    @property
    def shape(self): return self._shape

    @property
    def nnz(self):
        return self.shape[0]

    def true_nnz(self):
        return self.shape[0]

    def GetColPartArray(self):
        return self._cpart
     
    def GetRowPartArray(self):
        return self._rpart
     
    def __mul__(self, other):
        return other
    def transpose(self):
        return self
    def dot(self, other):
        return other
    @property
    def isHypre(self):
        return self._is_hypre
     
class ScipyCoo(coo_matrix):
    def true_nnz(self):
        if hasattr(self, "eliminate_zeros"):
            self.eliminate_zeros()
        return self.nnz
   
    def __add__(self, other):
        ret = super(ScipyCoo, self).__add__(other)
        return convert_to_ScipyCoo(ret)
     
    def __sub__(self, other):
        ret = super(ScipyCoo, self).__sub__(other)
        return convert_to_ScipyCoo(ret)
     
    def setDiag(self, idx, value=1.0):
        ret = self.tolil()
        for i in idx:
           ret[i,i] = value
        ret = ret.tocoo()
        self.data = ret.data
        self.row  = ret.row
        self.col  = ret.col
        
    def resetDiagImag(self, idx):
        ret = self.tolil()
        for i in idx:
           ret[i,i] = ret[i,i].real
        ret = ret.tocoo()
        self.data = ret.data
        self.row  = ret.row
        self.col  = ret.col
        

    def resetRow(self, rows):
        ret = self.tolil()
        for r in rows: ret[r, :] = 0.0        
        ret = ret.tocoo()
        self.data = ret.data
        self.row  = ret.row
        self.col  = ret.col
       
    def resetCol(self, cols):
        ret = self.tolil()
        for c in cols: ret[:, c] = 0.0        
        ret = ret.tocoo()
        self.data = ret.data
        self.row  = ret.row
        self.col  = ret.col
     
    def selectRows(self, nonzeros):
        m = self.tocsr()
        ret = (m[nonzeros,:]).tocoo()
        return convert_to_ScipyCoo(ret)
     
    def selectCols(self, nonzeros):
        m = self.tocsc()
        ret = (m[:, nonzeros]).tocoo()
        return convert_to_ScipyCoo(ret)        
     
    def rap(self, P):
        PP = P.conj().transpose()
        return convert_to_ScipyCoo(PP.dot(self.dot(P)))

    def conj(self, inplace = False):
        if inplace:
            np.conj(self.data, out=self.data)
            return self
        else:
            return self.conj()

    def elimination_matrix(self, nonzeros):
        '''
        # P elimination matrix for column vector
        [1 0  0][x1]    [x1]
        [0 0  1][x2]  = [x3]
                [x3]    

        P^t (transpose does reverse operation)
           [1 0][x1]    [x1]
           [0 0][x3]  = [0]
           [0,1]        [x3]

        # P^t does elimination matrix for horizontal vector
                  [1,0]    
        [x1 x2 x3][0,0]  = [x1, x3]
                  [0,1]

        # P does reverse operation for horizontal vector
               [1,0 0]    
        [x1 x3][0,0 1]  = [x1, 0  x3]
        '''
        ret = lil_matrix((len(nonzeros), self.shape[0]))
        for k, z in enumerate(nonzeros):
            ret[k, z] = 1.
        return convert_to_ScipyCoo(ret.tocoo())
     
    def get_global_coo(self):
        '''
        global representation:
           zero on non-root node
        '''
        try:
           from mpi4py import MPI
           myid = MPI.COMM_WORLD.rank
           if myid != 0:
              return coo_matrix(self.shape)
           else:
              return self
        except:
           return self
        
    def get_mfem_sparsemat(self):
        '''
        generate mfem::SparseMatrix using the same data
        '''
        if np.iscomplexobj(self):
            csr_r = np.real(self).tocsr()
            csr_r.eliminate_zeros()
            csr_i = np.imag(self).tocsr()
            csr_i.eliminate_zeros()
            return mfem.SparseMatrix(csr_r), mfem.SparseMatrix(csr_i)
        else:
            csr_r = self.tocsr()
            csr_r.eliminate_zeros()
            csr_i = None
            return mfem.SparseMatrix(csr_r), None
     
    def __repr__(self):
        return "ScipyCoo"+str(self.shape)

    def __str__(self):
        return "ScipyCoo"+str(self.shape) + super(ScipyCoo, self).__str__()
     
    @property     
    def isHypre(self):
        return False


    '''
    two function to provde the same interface as CHypre
    '''
    def GetColPartArray(self):
        return (0, self.shape[1], self.shape[1])
     
    def GetRowPartArray(self):
        return (0, self.shape[0], self.shape[0])
    GetPartitioningArray = GetRowPartArray

    def eliminate_RowCol(self, tdof):
        # tdof is intArray....
        tdof = tdof.ToList()
        lil = self.tolil()
        lil[tdof, :] = 0
        Ae = lil_matrix(self.shape, dtype=self.dtype)
        Ae[:, tdof] = lil[:,tdof]
        lil[tdof, tdof] = 1.
        coo = lil.tocoo()
        self.data = coo.data
        self.row = coo.row
        self.col = coo.col
        return Ae.tocoo()
       
def convert_to_ScipyCoo(mat):
    if isinstance(mat, np.ndarray):
       mat = coo_matrix(mat)
    if isinstance(mat, spmatrix):
       if not isinstance(mat, coo_matrix):
          mat = mat.tocoo()
       mat.__class__ = ScipyCoo
    return mat

class BlockMatrix(object):
    def __init__(self, shape, kind = default_kind):
        '''
        kind : scipy
                  stores scipy sparse or numpy array
               hypre
        '''
        self.block = [[None]*shape[1] for x in range(shape[0])]
        self.kind  = kind
        self.shape = shape
        self.complex = False

    def __getitem__(self, idx):
        try:
            r, c = idx
        except:
            r = idx ; c = 0
        if isinstance(r, slice):
           new_r = range(self.shape[0])[r]
           ret = BlockMatrix((len(new_r), self.shape[1]))
           for i, r in enumerate(new_r):
              for j in range(self.shape[1]):
                 ret[i, j] = self[r, j]
           return  ret
        elif isinstance(c, slice):
           new_c = range(self.shape[1])[c]
           ret = BlockMatrix((self.shape[0], len(new_c)))
           for i in range(self.shape[0]):
               for j, c in enumerate(new_c):
                 ret[i, j] = self[i, c]
           return  ret
        else:
           return self.block[r][c]

    def __setitem__(self, idx, v):
        try:
            r, c = idx
        except:
            r = idx ; c = 0
        if v is not None:
            if isinstance(v, chypre.CHypreMat):
                if v.isComplex(): self.complex = True                                      
            elif isinstance(v, chypre.CHypreVec):
                if v.isComplex(): self.complex = True                       
            elif v is None:
                pass
            else:   
                v = convert_to_ScipyCoo(v)
                if np.iscomplexobj(v): self.complex = True        
        self.block[r][c] = v

    def __add__(self, v):
        if self.shape != v.shape:
            raise ValueError("Block format is inconsistent")
         
        shape = self.shape
        ret = BlockMatrix(shape, kind = self.kind)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if self[i,j] is None and v[i,j] is None:
                    ret[i,j] = None                   
                elif self[i,j] is None:
                    ret[i,j] = v[i,j]
                elif v[i,j] is None:
                    ret[i,j] = self[i,j]
                else:
                    ret[i,j] = self[i,j] + v[i,j]
        return ret
     
    def __sub__(self, v):
        if self.shape != v.shape:
            raise ValueError("Block format is inconsistent")
        shape = self.shape
        ret = BlockMatrix(shape, kind = self.kind)

        for i in range(shape[0]):
           for j in range(shape[1]):
                if self[i,j] is None and v[i,j] is None:
                    ret[i,j] = None                   
                elif self[i,j] is None:
                    ret[i,j] = -v[i,j]
                elif v[i,j] is None:
                    ret[i,j] = self[i,j]
                else:
                    ret[i,j] = self[i,j] - v[i,j]
        return ret

    def __repr__(self):
        txt = ["BlockMatrix"+str(self.shape)]
        for i in range(self.shape[0]):
           txt.append(str(i) +" : "+ "  ".join([self.block[i][j].__repr__()
                           for j in range(self.shape[1])]))
        return "\n".join(txt)+"\n"

    def format_nnz(self):
        txt = []
        for i in range(self.shape[0]):       
           txt.append(str(i) +" : "+ ",  ".join([str(self[i,j].nnz)
                                                     if hasattr(self[i,j], "nnz") else "unknown"
                                                  for j in range(self.shape[1])]))
        return "non-zero elements (nnz)\n" + "\n".join(txt)
        
    def print_nnz(self):
        print(self.format_nnz())

    def format_true_nnz(self):
        txt = []
        for i in range(self.shape[0]):       
           txt.append(str(i) +" : "+ ",  ".join([str(self[i,j].true_nnz())
                                                     if hasattr(self[i,j], "true_nnz") else "unknown"
                                                  for j in range(self.shape[1])]))
        return "non-zero elements (true nnz)\n" + "\n".join(txt)
     
    def print_true_nnz(self):
        print(self.format_ture_nnz())

    def transpose(self):
        ret = BlockMatrix((self.shape[1], self.shape[0]), kind = self.kind)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i, j] is not None:
                    ret[j, i] = self[i, j].transpose()
        return ret
     
    def add_to_element(self, i, j, v):
        if self[i,j] is None: self[i,j] = v
        else:
            self[i,j] = self[i,j] + v
            
    def dot(self, mat):
        if self.shape[1] != mat.shape[0]:
            raise ValueError("Block format is inconsistent")

        shape = (self.shape[0], mat.shape[1])
        ret = BlockMatrix(shape, kind = self.kind)

        for i in range(shape[0]):
           for j in range(shape[1]):
               for k in range(self.shape[1]):
                   if self[i, k] is None: continue
                   elif mat[k, j] is None: continue
                   elif ret[i,j] is None:
                       ret[i,j] = self[i, k].dot(mat[k, j])
                   else:
                       ret[i,j] = ret[i,j] + self[i, k].dot(mat[k, j])
                   #try:
                   #    ret[i,j].shape
                   #except:
                   #    ret[i,j] = coo_matrix([[ret[i,j]]]) 
        return ret

    def eliminate_empty_rowcol(self):
        '''
        collect empty row first. (no global communicaiton this step)

        share empty rows..

        apply it to all node

        '''
        from functools import reduce
        ret = BlockMatrix(self.shape, kind = self.kind)        
        P2  = BlockMatrix(self.shape, kind = self.kind)

        dprint1(self.format_true_nnz())

        for i in range(self.shape[0]):
            nonzeros = []
            mat = None
            for j in range(self.shape[1]):
               if self[i,j] is not None:
                   if isinstance(self[i,j], ScipyCoo):
                       coo = self[i,j]
                       csr = coo.tocsr()
                       num_nonzeros = np.diff(csr.indptr)
                       knonzero = np.where(num_nonzeros != 0)[0]
                       if mat is None: mat = coo
                   elif isinstance(self[i,j], chypre.CHypreMat):
                       coo = self[i,j].get_local_coo()
                       if hasattr(coo, "eliminate_zeros"):
                            coo.eliminate_zeros()
                       csr = coo.tocsr()
                       num_nonzeros = np.diff(csr.indptr)
                       knonzero = np.where(num_nonzeros != 0)[0]
                       if mat is None: mat = self[i,j]
                   elif isinstance(self[i,j], chypre.CHypreVec):
                       if self[i,j].isAllZero():
                           knonzero = []
                       else:
                           knonzero = [0]
                   elif (isinstance(self[i,j], np.ndarray) and
                         self[i,j].ndim == 2):
                       knonzero = [k for k in range(self[i,j].shape[0])
                                   if any(self[i,j][k,:])]
                       self[i,j] = convert_to_ScipyCoo(self[i,j])
                       mat = self[i,j]
                   else:
                       raise ValueError("Unsuported Block Element" +
                                        str(type(self[i,j])))
                   nonzeros.append(knonzero)
               else:
                   nonzeros.append([])
            knonzeros = reduce(np.union1d, nonzeros)
            knonzeros = np.array(knonzeros, dtype = np.int64)
            # share nonzero to eliminate column...
            if self.kind == 'hypre':
                if isinstance(self[i,i], chypre.CHypreMat):
                   gknonzeros = self[i,i].GetRowPartArray()[0] + knonzeros
                else:
                   gknonzeros = knonzeros
                gknonzeros = allgather_vector(gknonzeros)
                gknonzeros = np.unique(gknonzeros)
                dprint2('nnz', coo.nnz, len(knonzero))
            else:
                gknonzeros = knonzeros

            if mat is not None:
                # hight and row partitioning of mat is used
                # to construct P2
                if len(gknonzeros) < self[i,i].shape[0]:
                    P2[i,i] = self[i,i].elimination_matrix(gknonzeros)
                else:
                    P2[i,i] = One(self[i,i])
            # what if common zero rows differs from common zero col?
            for j in range(self.shape[1]):
                if ret[i,j] is not None:
                    ret[i,j] = ret[i,j].selectRows(gknonzeros)
                elif self[i,j] is not None:
                    ret[i,j] = self[i,j].selectRows(gknonzeros)
                else: pass
                if ret[j,i] is not None:
                    ret[j,i] = ret[j,i].selectCols(gknonzeros)
                elif self[j,i] is not None:
                    ret[j,i] = self[j,i].selectCols(gknonzeros)
                else: pass

        return ret, P2

    def reformat_central_mat(self, mat, ksol):
        '''
        reformat central matrix into blockmatrix (columne vector)
        so that matrix can be multiplied from the right of this 

        self is a block diagonal matrix
        '''
        L = []
        idx = 0
        ret = BlockMatrix((self.shape[0], 1), kind = self.kind)
        print "self", self
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
               if self[i,j] is not None:
                  l =  self[i, j].shape[0]
                  break
            print i, j, l
            L.append(l)
            ref = self[i,j]
            if mat is not None:
                v = mat[idx:idx+l, ksol]
            else:
                v = None   # slave node (will recive data)
            idx = idx + l
            print "v.shape", v.shape
            ret.set_element_from_central_mat(v, i, 0, ref)

        return ret

    def set_element_from_central_mat(self, v, i, j, ref):
        ''' 
        set element using vector in root node
        row partitioning is taken from column partitioning
        of ref 
        '''
        if self.kind == 'scipy':
            self[i, j] = v.reshape(-1,1)
        else:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            
            if ref.isHypre:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD

                part = ref.GetColPartArray()
                v = comm.bcast(v)
                start_row = part[0]
                end_row = part[1]

                v = np.ascontiguousarray(v[start_row:end_row])
                if np.iscomplexobj(v):
                    rv = ToHypreParVec(v.real)    
                    iv = ToHypreParVec(v.imag)    
                    self[i,j] = chypre.CHypreVec(rv, iv)
                else:
                    rv = ToHypreParVec(v)    
                    self[i,j] = chypre.CHypreVec(rv, None)
            else:
                #slave node gets the copy
                v = comm.bcast(v)
                self[i, j] = v.reshape(-1,1)

    def get_squaremat_from_right(self, r, c):
        size = self[r, c].shape
        if self.kind == 'scipy':
            return ScipyCoo((size[1],size[1]))
        else:
            # this will return CHypreMat
            return self[r, c].get_squaremat_from_right()
    #
    #  methods for coo format
    #
    def gather_densevec(self):
        '''
        gather vector data to head node as dense data (for rhs)
        '''
        if self.kind == 'scipy':
            if self.complex:
                M = scipy.sparse.bmat(self.block, format='coo',
                                      dtype='complex').toarray()
            else:
                M = scipy.sparse.bmat(self.block, format='coo',
                                      dtype='float').toarray()
            return M
        else: 
            data = []
            for i in range(self.shape[0]):
                if isinstance(self[i,0], chypre.CHypreVec):
                    data.append(self[i,0].GlobalVector())
                elif isinstance(self[i,0], np.ndarray):
                    data.append(self[i,0].flatten())
                elif isinstance(self[i,0], ScipyCoo):
                    data.append(self[i,0].toarray().flatten())
                else:
                    raise ValueError("Unsupported element" + str((i,0)) + 
                                     ":" + str(type(self[i,0])))
            return np.hstack(data).reshape(-1,1)

    def get_global_offsets(self, convert_real = False,
                                 interleave = True):
        '''
        build matrix in coordinate format
        '''
        roffset = []
        coffset = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i, j] is not None:
                   roffset.append(self[i,j].shape[0])
                   break
        for j in range(self.shape[1]):
            for i in range(self.shape[0]):
                if self[i, j] is not None:
                   coffset.append(self[i,j].shape[1])
                   break
        #coffset = [self[0, j].shape[1] for j in range(self.shape[1])]
        if self.complex and convert_real:
            if interleave:
                roffset = np.vstack((roffset, roffset)).flatten()
                coffset = np.vstack((roffset, roffset)).flatten()                          
            else:
                roffset = np.hstack((roffset, roffset))
                coffset = np.hstack((roffset, roffset))                                    

        roffsets = np.hstack([0, np.cumsum(roffset)])
        coffsets = np.hstack([0, np.cumsum(coffset)])
        return roffsets, coffsets

    def get_global_coo(self, dtype = 'float'):
        roffsets, coffsets = self.get_global_offsets()
        col = []
        row = []
        data = []
        glcoo = coo_matrix((roffsets[-1], coffsets[-1]), dtype = dtype)
        dprint1("roffset(get_global_coo)", roffsets)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i,j] is None: continue
                gcoo = self[i,j].get_global_coo()
                row.append(gcoo.row + roffsets[i])                
                col.append(gcoo.col + coffsets[j])
                data.append(gcoo.data)
        glcoo.col = np.hstack(col)
        glcoo.row = np.hstack(row)
        glcoo.data = np.hstack(data)

        return glcoo

    #
    #  methods for distributed csr format
    #
    def get_local_partitioning(self, convert_real = True,
                                     interleave = True):
        '''
        build matrix in coordinate format
        '''
        roffset = np.zeros(self.shape[0], dtype=int)
        coffset = np.zeros(self.shape[1], dtype=int)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i, j] is not None:
                   rp = self[i,j].GetRowPartArray()
                   if (roffset[i] != 0 and
                       roffset[i] != rp[1] - rp[0]):
                        assert False, 'row partitioning is not consistent'
                   roffset[i] = rp[1] - rp[0]
                   if use_parallel and not isinstance(self[i,j], chypre.CHypreMat):
                      from mpi4py import MPI
                      myid = MPI.COMM_WORLD.rank
                      if myid != 0: roffset[i] = 0
        for j in range(self.shape[1]):
            for i in range(self.shape[0]):
                if self[i, j] is not None:
                   cp = self[i,j].GetColPartArray()
                   if (coffset[j] != 0 and
                       coffset[j] != cp[1] - cp[0]):
                        assert False, 'col partitioning is not consistent'
                   coffset[j] = cp[1] - cp[0]
                   if use_parallel and not isinstance(self[i,j], chypre.CHypreMat):
                      if myid != 0: coffset[i] = 0                      
                   
        #coffset = [self[0, j].shape[1] for j in range(self.shape[1])]
        if self.complex and convert_real:
            if interleave:
                roffset = np.repeat(roffset, 2)
                coffset = np.repeat(coffset, 2)
            else:
                roffset = np.hstack((roffset, roffset))
                coffset = np.hstack((roffset, roffset))  

        roffsets = np.hstack([0, np.cumsum(roffset)])
        coffsets = np.hstack([0, np.cumsum(coffset)])
        return roffsets, coffsets

    def get_local_partitioning_v(self, convert_real = True,
                                       interleave = True):
        '''
        build matrix in coordinate format
        '''
        roffset = np.zeros(self.shape[0], dtype=int)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i, j] is not None:
                   rp = self[i,j].GetPartitioningArray()
                   if (roffset[i] != 0 and
                       roffset[i] != rp[1] - rp[0]):
                        assert False, 'row partitioning is not consistent'
                   roffset[i] = rp[1] - rp[0]
        #coffset = [self[0, j].shape[1] for j in range(self.shape[1])]
        if self.complex and convert_real:
            if interleave:
                roffset = np.repeat(roffset, 2)
            else:
                roffset = np.hstack((roffset, roffset))

        roffsets = np.hstack([0, np.cumsum(roffset)])
        return roffsets
     
    def gather_blkvec_interleave(self):
        '''
        Construct MFEM::BlockVector

        This routine ordered unkonws in the following order
           Re FFE1, Im FES1, ReFES2, Im FES2, ...
        If self.complex is False, it assembles a nomal block
           vector
        This routine is used together with get_global_blkmat_interleave(self):
        '''
        
        roffsets = self.get_local_partitioning_v(convert_real=True,
                                                 interleave=True)
        dprint1("roffsets(vector)", roffsets)
        offset = mfem.intArray(list(roffsets))
        
        vec = mfem.BlockVector(offset)
        vec._offsets = offset # in order to keep it from freed
        
        data = []
        ii = 0; jj = 0

        # Here I don't like that I am copying the data between two vectors..
        # But, avoiding this takes the large rearangement of program flow...
        for i in range(self.shape[0]):        
            if self[i,0] is not None:
                if isinstance(self[i,0], chypre.CHypreVec):
                    vec.GetBlock(ii).Assign(self[i,0][0].GetDataArray())
                    if self.complex:
                        vec.GetBlock(ii+1).Assign(self[i,0][1].GetDataArray())
                elif isinstance(self[i,0], ScipyCoo):
                    arr =np.atleast_1d( self[i,0].toarray().squeeze())
                    vec.GetBlock(ii).Assign(np.real(arr))
                    if self.complex:
                        vec.GetBlock(ii+1).Assign(np.imag(arr))
                else:
                    assert False, "not implemented, "+ str(type(self[i,0]))
                
            ii = ii + 2 if self.complex else ii+1

        return vec
       
    def get_global_blkmat_interleave(self):
        '''
        This routine ordered unkonws in the following order
           Re FFE1, Im FES1, ReFES2, Im FES2, ...
        If self.complex is False, it assembles a nomal block
        matrix FES1, FES2...
        '''
        roffsets, coffsets = self.get_local_partitioning(convert_real=True,
                                                         interleave=True)
        dprint1("offsets", roffsets, coffsets)
        ro = mfem.intArray(list(roffsets))
        co = mfem.intArray(list(coffsets))        
        glcsr = mfem.BlockOperator(ro, co)

        ii = 0

        for i in range(self.shape[0]):
            jj = 0
            for j in range(self.shape[1]):
                if self[i,j] is not None:
                    if use_parallel:
                        if isinstance(self[i,j], chypre.CHypreMat):
                            gcsr =  self[i,j]
                            cp = self[i,j].GetColPartArray()
                            rp = self[i,j].GetRowPartArray()
                            s = self[i,j].shape
                            #if (cp == rp).all() and s[0] == s[1]:
                            if gcsr[0] is not None:
                                  csr = ToScipyCoo(gcsr[0]).tocsr()
                                  gcsr[0] = ToHypreParCSR(csr, col_starts =cp)
                            if gcsr[1] is not None:
                                  csr = ToScipyCoo(gcsr[1]).tocsr()
                                  print('csr shape', csr.shape)
                                  gcsr[1] = ToHypreParCSR(csr, col_starts =cp)
                                  gcsrm   = ToHypreParCSR(-csr, col_starts =cp)
                            dprint1(i, j, s, rp, cp)
                        else:
                            assert False, "unsupported block element "+str(type(self[i,j]))
                    else:
                        if isinstance(self[i,j], ScipyCoo):
                            gcsr = self[i,j].get_mfem_sparsemat()
                            if gcsr[1] is not None:
                               gcsrm = mfem.SparseMatrix(gcsr[1])
                               gcsrm *= -1.
                        else:
                            assert False, "unsupported block element "+type(self[i,j])
                    glcsr.SetBlock(ii, jj, gcsr[0])
                    if self.complex:
                        glcsr.SetBlock(ii+1, jj+1, gcsr[0])
                    if gcsr[1] is not None:
                        glcsr.SetBlock(ii+1, jj,  gcsr[1])
                        glcsr.SetBlock(ii, jj+1,  gcsrm)
                jj = jj + 2 if self.complex else jj+1
            ii = ii + 2 if self.complex else ii+1

        return glcsr



                
        
          
            


