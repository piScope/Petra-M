'''
  reducer is to eliminate trivial depencencies of 
  blockmatrix

  
  [A  B][x1]  [a1]
  [    ][  ]= [  ] 
  [   C][x2]  [a2]

  
  C x2 = a2 can be solved first.
  
  (1)  When x2 is auxiliary variable, C is often trivial 
           (diagonal, full but small) to invert.
           Then do this first.

  (2)  When x1 is auxiliary variable and A is trivial to invert.
           We can handle this later after x2 is solved by
             A x1 = a1 - B*x2
           Therefore, we dont solve x1 together with x2.

  step1:
       solve (1) and modifiy RHS
  step2:
       remove the row which can be done later by (2)
  step3:
       solve a submatrix which really requres interative solve
  step4:
       solve (2)


  Note: presentely MPI only
'''
from __future__ import print_function

import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('LS_Reducer')

from petram.mfem_config import use_parallel
if use_parallel:
    from petram.helper.mpi_recipes import allgather_vector
    from mfem.common.mpi_debug import nicePrint
    import mfem.par as mfem
   
    from mpi4py import MPI                                   
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank    
else:
    import mfem.ser as mfem
    num_proc = 1
    myid = 0
    def nicePrint(x):
        print(x)

def get_block(Op, i, j):
    try:
        return Op._linked_op[(i,j)]
    except KeyError:
        return None

def do_easy_invert(M):
    '''
    set diagonal element to value (normally 1)
    '''
    col_starts = M.GetColPartArray()
    num_rows, ilower, iupper, jlower, jupper, irn, jcn, data = M.GetCooDataArray()


    from mpi4py import MPI
    myid     = MPI.COMM_WORLD.rank
     
    m = iupper - ilower + 1
    n = jupper - jlower + 1
    n = M.N()

    from scipy.sparse import coo_matrix, lil_matrix
    for i, j in zip(irn, jcn):
        if i != j:
            return -1, None
    try:
        mat =  coo_matrix((1./data, (irn-ilower, jcn)), shape = (m, n)).tolil()
    except:
       print("wrong input")
       print(num_rows, ilower, iupper, jlower, jupper)
       print(np.min(irn-ilower), np.max(irn-ilower) , np.min(jcn),  np.max(jcn), (m, n))
       raise
    from mfem.common.parcsr_extra import ToHypreParCSR
    return  1, ToHypreParCSR(mat.tocsr(), col_starts = col_starts)
    
class LinearSystemReducer(object):
    def __init__(self, opr, name):
        offset = opr.RowOffsets()
        nb = opr.NumRowBlocks()
        self.nb = nb
        
        size = np.diff(offset.ToList())
        lsize = size
        self.lsize = lsize

        if use_parallel:
            size = allgather_vector(size)
            size = np.sum(size.reshape(-1, nb), axis = 0)

        #nicePrint(size)
        filled = np.zeros((nb, nb), dtype=int)
        nnz    = np.zeros((nb, nb), dtype=int)
        for i in range(nb):
           for j in range(nb):
              m = get_block(opr, i, j)
              if m is not None:
                  filled[i, j] = 1
                  nnz[i, j] = m.NNZ()

        flag = [False]*nb
        for i in range(nb): 
            if filled[i, i] == 1 and nnz[i,i] == size[i]:
                flag[i] = True
        #nicePrint("filling pattern", filled)
        #nicePrint("nnz(global)", nnz)
        #nicePrint("easy to invert ", flag)

        easy_inv = [None]*nb
        for i in range(nb):
            if flag[i]:
                check, mat = do_easy_invert(get_block(opr, i, i))
                easy_inv[i] = mat
        self.easy_inv = easy_inv

        first_step_flag = [False]*nb
        for i in range(nb):
            flag = True
            for j in range(nb):
                if  i != j and filled[i, j] == 1:
                    flag = False
            if easy_inv[i] is None: flag = False
            first_step_flag[i] = flag

        last_step_flag = [False]*nb
        for j in range(nb):
            flag = True
            for i in range(nb):
                if filled[i, j] == 1 and i != j:
                    flag = False
            if easy_inv[j] is None: flag = False                
            last_step_flag[j] = flag

        idx = np.logical_or(first_step_flag, last_step_flag)
        reduced_idx = [k for k, n in enumerate(lsize)
                      if not (first_step_flag[k] or last_step_flag[k])]

        self.idx_step1 = list(np.where(first_step_flag)[0])
        self.idx_step2 = list(reduced_idx)
        self.idx_step3 = list(np.where(last_step_flag)[0])
        self.Aname = [n for k, n in enumerate(name) if k in self.idx_step2]
        
        #dprint1("first step: ", self.idx_step1)
        #dprint1("middle step: ", self.idx_step2)        
        #dprint1("last step: ", self.idx_step3)
        #dprint1("A name: ", self.Aname)
        
        rd_offsets = np.hstack([0, np.cumsum(np.array(lsize)[self.idx_step2])])
        self.A = self.make_sub_matrix(opr, rd_offsets, rd_offsets,
                                      self.idx_step2, self.idx_step2)

       
        not_in_step1 = [k for k in range(nb) if not k in self.idx_step1]
        #dprint1("step1", not_in_step1)
        ro = np.hstack([0, np.cumsum(np.array(lsize)[not_in_step1])])
        co = np.hstack([0, np.cumsum(np.array(lsize)[self.idx_step1])])
        self.A_step1 = self.make_sub_matrix(opr, ro, co, 
                                            not_in_step1, self.idx_step1)

        not_in_step3 = [k for k in range(nb) if not k in self.idx_step3]
        #dprint1("step3", not_in_step3)        
        ro = np.hstack([0, np.cumsum(np.array(lsize)[self.idx_step3])])
        co = np.hstack([0, np.cumsum(np.array(lsize)[not_in_step3])])
        #dprint1("step3", ro, co)                
        self.A_step3 = self.make_sub_matrix(opr, ro, co, 
                                            self.idx_step3, not_in_step3)

        self.Ainv_step1 = self.make_inv_mat(self.idx_step1, easy_inv)
        self.Ainv_step3 = self.make_inv_mat(self.idx_step3, easy_inv)
        
    def make_sub_matrix(self, src_mat, roffset, coffset, row_idx, col_idx):
        ro = mfem.intArray(list(roffset))
        co = mfem.intArray(list(coffset))
        mat = mfem.BlockOperator(ro, co)
        for ii, i in enumerate(row_idx):
           for jj, j in enumerate(col_idx):
               m = get_block(src_mat, i, j)
               if m is None: continue
               mat.SetBlock(ii, jj,  m)
        return mat

    def make_inv_mat(self, idx, inv):
        ro = np.hstack([0, np.cumsum(np.array(self.lsize)[idx])])
        row = mfem.intArray(list(ro))
        col = mfem.intArray(list(ro))
        mat = mfem.BlockOperator(row, col)
        
        for ii, i in enumerate(idx):
               mat.SetBlock(ii, ii,  inv[i])
        return mat

    def make_sub_vec(self, src_vec, idx, inv=False, nocopy=False):
        if inv:
            idx = [k for k in range(self.nb) if not k in idx]
        size = [src_vec.BlockSize(i) for i in idx]
        offset = np.hstack([0, np.cumsum(size, dtype=int)])
        offset = mfem.intArray(list(offset))
        
        sub_vec = mfem.BlockVector(offset)
        sub_vec._offsets = offset # in order to keep it from freed

        for k, i in enumerate(idx):
           if nocopy:
               sub_vec.GetBlock(k).Assign(0.0)
           else:
               sub_vec.GetBlock(k).Assign(src_vec.GetBlock(i).GetDataArray())

        return sub_vec

    def set_solver(self, solver):
        self.solver = solver
        
    def Mult(self, bb, xx, assert_convergence = False):
        bb_1 = self.make_sub_vec(bb, self.idx_step1)
        xx_1 = self.make_sub_vec(xx, self.idx_step1)        
        self.Ainv_step1.Mult(bb_1, xx_1)


        for k, i in enumerate(self.idx_step1):
            xx.GetBlock(i).Assign(xx_1.GetBlock(k).GetDataArray())

        #nicePrint("xx_1", xx_1.GetBlock(0).GetDataArray())
        
        bb_1_2 = self.make_sub_vec(bb, self.idx_step1, inv=True, nocopy=True)

        self.A_step1.Mult(xx_1, bb_1_2)

        for k, i in enumerate(self.idx_step2+self.idx_step3):
            v = bb.GetBlock(i)
            v -= bb_1_2.GetBlock(k)

        bb_2 = self.make_sub_vec(bb, self.idx_step2)
        xx_2 = self.make_sub_vec(xx, self.idx_step2)

        self.solver.Mult(bb_2, xx_2)
        max_iter = self.solver.GetNumIterations();
        tol = self.solver.GetFinalNorm()
        
        dprint1("convergence check (max_iter, tol) ", max_iter, " ", tol)
        if assert_convergence:
            if not self.solver.GetConverged():
                raise debug.ConvergenceError("Convergence error when solving Reduced linear system")

        for k, i in enumerate(self.idx_step2):
            xx.GetBlock(i).Assign(xx_2.GetBlock(k).GetDataArray())


        xx_3_1 = self.make_sub_vec(xx, self.idx_step3, inv=True)
        xx_3_2 = self.make_sub_vec(xx, self.idx_step3)
        bb_3 = self.make_sub_vec(bb, self.idx_step3)
        self.A_step3.Mult(xx_3_1, bb_3)

        for k, i in enumerate(self.idx_step3):
            v = bb.GetBlock(i)
            v -= bb_3.GetBlock(k)
        bb_3 = self.make_sub_vec(bb, self.idx_step3)            
        self.Ainv_step3.Mult(bb_3, xx_3_2)
        
        for k, i in enumerate(self.idx_step3):
            xx.GetBlock(i).Assign(xx_3_2.GetBlock(k).GetDataArray())
        
        
        

        
        
                
        
               


            

