from .solver_model import Solver
import numpy as np
import scipy

from petram.namespace_mixin import NS_mixin
from .solver_model import LinearSolverModel, LinearSolver

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('StrumpackModel')

from petram.mfem_config import use_parallel
if use_parallel:
   from petram.helper.mpi_recipes import *
   from mfem.common.parcsr_extra import *
   import mfem.par as mfem
   default_kind = 'hypre'
   
   from mpi4py import MPI                               
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   smyid = '{:0>6d}'.format(myid)
   from mfem.common.mpi_debug import nicePrint
   
else:
   import mfem.ser as mfem
   default_kind = 'scipy'
   
class Strumpack(LinearSolverModel):
    hide_ns_menu = True
    has_2nd_panel = False
    accept_complex = False
    always_new_panel = False

    def __init__(self,  *args, **kwargs):
        LinearSolverModel.__init__(self, *args, **kwargs)

    def init_solver(self):
        pass
    def panel1_param(self):
        return [
                ["log_level",   self.log_level,  400, {}],            
                ["hss compression",  self.hss,   3, {"text":""}],
                ["hss min front size",  self.hss_front_size, 400, {}],
                ["rctol",  self.rctol, 300, {}],
                ["actol",  self.actol, 300, {}],
                ["mc64job",  self.mc64job, 400, {}],
                ["gmres restrart",  self.gmres_restart, 400, {}],
                ["max iter.",  self.maxiter, 400, {}],            
                ["write matrix",  self.write_mat,   3, {"text":""}],]
    
    def get_panel1_value(self):
        return (long(self.log_level),
                self.hss,
                self.hss_front_size,
                self.rctol,
                self.actol,
                self.mc64job,
                self.gmres_restart,                
                self.maxiter,
                self.write_mat, )
    
    def import_panel1_value(self, v):
        self.log_level = long(v[0])        
        self.hss = v[1]
        self.hss_front_size = v[2]
        self.rctol = v[3]
        self.actol = v[4]
        self.mc64job = v[5]
        self.gmres_restart = long(v[6])
        self.maxiter = long(v[7])
        self.write_mat = bool(v[8])

        
    def attribute_set(self, v):
        v = super(Strumpack, self).attribute_set(v)
        v['log_level'] = 0
        v['write_mat'] = False
        v['hss'] = False
        v['hss_front_size'] = 2500
        v['rctol'] = 0.01
        v['actol'] = 1e-10
        v['maxiter'] = 5000
        v['gmres_restart'] = 30
        v['mc64job'] = 0
        return v
    
    def verify_setting(self):
        if not self.parent.assemble_real:
            for phys in self.get_phys():
                if phys.is_complex():
                    return False, "We support only real version of Strumpack.", "A complex problem must be converted to a real value problem"
        return True, "", ""
    
    def linear_system_type(self, assemble_real, phys_complex):
        #if not phys_complex: return 'block'
        return 'blk_interleave'
        #return None
    def real_to_complex(self, solall, M):
        if use_parallel:
           from mpi4py import MPI
           myid     = MPI.COMM_WORLD.rank


           offset = M.RowOffsets().ToList()
           of = [np.sum(MPI.COMM_WORLD.allgather(np.int32(o))) for o in offset]
           if myid != 0: return           

        else:
           offset = M.RowOffsets()
           of = offset.ToList()
           
        rows = M.NumRowBlocks()
        s = solall.shape
        nb = rows/2
        i = 0
        pt = 0
        result = np.zeros((s[0]/2, s[1]), dtype='complex')
        for j in range(nb):
           l = of[i+1]-of[i]
           result[pt:pt+l,:] = (solall[of[i]:of[i+1],:]
                             +  1j*solall[of[i+1]:of[i+2],:])
           i = i+2
           pt = pt + l

        return result
    
    def allocate_solver(self, is_complex=False, engine=None):
        solver = StrumpackSolver(self, engine, int(self.maxiter),
                                 self.actol, self.rctol, self.mc64job,
                                 self.hss, self.hss_front_size,
                                 self.gmres_restart)
        #solver.AllocSolver(datatype)
        return solver
    
class StrumpackSolver(LinearSolver):
    def __init__(self, gui, engine, maxiter, actol, rctol, mc64job,
                 hss, hss_front_size, gmres_restart):
        self.maxiter = maxiter
        self.actol = actol
        self.rctol = rctol
        self.mc64job = mc64job
        self.hss = hss
        self.hss_front_size = hss_front_size
        self.gmres_restart = gmres_restart
        LinearSolver.__init__(self, gui, engine)

    def SetOperator(self, opr, dist=False, name = None):
        self.Aname = name
        self.A = opr                     
        
    def Mult(self, b, x=None, case_base=0):
        if use_parallel:
            return self.solve_parallel(self.A, b, x)
        else:
            raise AttributeError("Serial MFEM does not support Strumpack")
    @staticmethod    
    def get_block(Op, i, j):
            try:
               return Op._linked_op[(i,j)]
            except KeyError:
               return None
        
    def write_mat(self, A, b, x, suffix=""):
        offset = A.RowOffsets()
        rows = A.NumRowBlocks()
        cols = A.NumColBlocks()
        
        for j in range(cols):
           for i in range(rows):
              m = self.get_block(A, i, j)
              if m is None: continue
              m.Print('matrix_'+str(i)+'_'+str(j))
        for i, bb  in enumerate(b):
           for j in range(rows):
              v = bb.GetBlock(j)
              v.Print('rhs_'+str(i)+'_'+str(j)+suffix)
        if x is not None:
           for j in range(rows):
              xx = x.GetBlock(j)
              xx.Print('x_'+str(i)+'_'+str(j)+suffix)
              
    def make_solver(self, A):
        offset = np.array(A.RowOffsets().ToList(), dtype=int)
        rows = A.NumRowBlocks()
        cols = A.NumColBlocks()
        
        local_size = np.diff(offset)
        x = allgather_vector(local_size)
        global_size = np.sum(x.reshape(num_proc,-1), 0)
        nicePrint(local_size)

        global_offset = np.hstack(([0], np.cumsum(global_size)))
        global_roffset = global_offset + offset
        print global_offset

        new_offset = np.hstack(([0], np.cumsum(x)))[:-1]
#                                np.cumsum(x.reshape(2,-1).transpose().flatten())))
        new_size =   x.reshape(num_proc, -1)
        new_offset = new_offset.reshape(num_proc, -1)
        print new_offset
        
        #index_mapping
        def blk_stm_idx_map(i):
            stm_idx = [new_offset[kk, i]+
                       np.arange(new_size[kk, i], dtype=int)
                       for kk in range(num_proc)]
            return np.hstack(stm_idx)
        
        map = [blk_stm_idx_map(i) for i in range(rows)]
            

        newi = []
        newj = []
        newd = []
        nrows = np.sum(local_size)
        ncols = np.sum(global_size)
        
        for i in range(rows):
            for j in range(cols):
                 m = self.get_block(A, i, j)
                 if m is None: continue
#                      num_rows, ilower, iupper, jlower, jupper, irn, jcn, data = 0, 0, 0, 0, 0, np.array([0,0]), np.array([0,0]), np.array([0,0])
#                 else:
                 num_rows, ilower, iupper, jlower, jupper, irn, jcn, data = m.GetCooDataArray()

                 irn = irn         #+ global_roffset[i]
                 jcn = jcn         #+ global_offset[j]

                 nicePrint(i, j, map[i].shape, map[i])
                 nicePrint(irn)
                 irn2 = map[i][irn]
                 jcn2 = map[j][jcn]

                 newi.append(irn2)
                 newj.append(jcn2)
                 newd.append(data)

        newi = np.hstack(newi)
        newj = np.hstack(newj)
        newd = np.hstack(newd)

        from scipy.sparse import coo_matrix

        nicePrint(new_offset)
        nicePrint((nrows, ncols),)
        nicePrint('newJ', np.min(newj), np.max(newj))
        nicePrint('newI', np.min(newi)-new_offset[myid, 0],
                          np.max(newi)-new_offset[myid, 0])
        mat = coo_matrix((newd,(newi-new_offset[myid, 0], newj)),
                          shape=(nrows, ncols),
                          dtype=newd.dtype).tocsr()
        
        AA = ToHypreParCSR(mat)

        import mfem.par.strumpack as strmpk
        Arow = strmpk.STRUMPACKRowLocMatrix(AA)

        args = []
        if self.hss:
            args.extend(["--sp_enable_hss", 
                         "--hss_verbose", 
                         "--sp_hss_min_sep_size",
                         str(int(self.hss_front_size)),
                         "--hss_rel_tol",
                         str(0.01),
                         "--hss_abs_tol",                         
                         str(1e-4),])
        print self.maxiter
        args.extend(["--sp_maxit", str(int(self.maxiter))])
        args.extend(["--sp_rel_tol", str(self.rctol)])
        args.extend(["--sp_abs_tol", str(self.actol)])        
        args.extend(["--sp_gmres_restart", str(int(self.gmres_restart))])

        strumpack = strmpk.STRUMPACKSolver(args, MPI.COMM_WORLD)
        
        if self.gui.log_level == 0:
            strumpack.SetPrintFactorStatistics(False)
            strumpack.SetPrintSolveStatistics(False)
        elif self.gui.log_level == 1:
            strumpack.SetPrintFactorStatistics(True)
            strumpack.SetPrintSolveStatistics(False)
        else:
            strumpack.SetPrintFactorStatistics(True)
            strumpack.SetPrintSolveStatistics(True)

        strumpack.SetKrylovSolver(strmpk.KrylovSolver_DIRECT);
        strumpack.SetReorderingStrategy(strmpk.ReorderingStrategy_METIS)
        strumpack.SetMC64Job(strmpk.MC64Job_NONE)
        # strumpack.SetSymmetricPattern(True)
        strumpack.SetOperator(Arow)
        strumpack.SetFromCommandLine()

        strumpack._mapper = map
        return strumpack        
              
    def solve_parallel(self, A, b, x=None):
        if self.gui.write_mat:                      
            self. write_mat(A, b, x, "."+smyid)
              
        solver = self.make_solver(A)
        sol = []

        # solve the problem and gather solution to head node...
        # may not be the best approach
        
        from petram.helper.mpi_recipes import gather_vector        
        offset = A.RowOffsets()
        for bb in b:
           rows = MPI.COMM_WORLD.allgather(np.int32(bb.Size()))
           rowstarts = np.hstack((0, np.cumsum(rows)))
           dprint1("rowstarts/offser",rowstarts, offset.ToList())
           if x is None:           
              xx = mfem.BlockVector(offset)
              xx.Assign(0.0)
           else:
              xx = x
              #for j in range(cols):
              #   dprint1(x.GetBlock(j).Size())
              #   dprint1(x.GetBlock(j).GetDataArray())
              #assert False, "must implement this"
           solver.Mult(bb, xx)

           s = []
           for i in range(offset.Size()-1):
               v = xx.GetBlock(i).GetDataArray()
               vv = gather_vector(v)
               if myid == 0:
                   s.append(vv)
               else:
                   pass
           if myid == 0:               
               sol.append(np.hstack(s))
        if myid == 0:                              
            sol = np.transpose(np.vstack(sol))
            return sol
        else:
            return None
    
'''   
from Strumpack.SparseSolver import Sp_attrs

class SpSparse(Solver):
    has_2nd_panel = False
    accept_complex = True    
    def init_solver(self):
        pass
    def panel1_param(self):
        return [
                ["hss compression",  self.hss,   3, {"text":""}],
                ["hss min front size",  self.hss_front_size, 400, {}],
                ["rctol",  self.rctol, 300, {}],
                ["actol",  self.actol, 300, {}],
                ["mc64job",  self.mc64job, 400, {}],
                ["write matrix",  self.write_mat,   3, {"text":""}],]
    
    def get_panel1_value(self):
        return (self.hss,
                self.hss_front_size,
                self.rctol,
                self.actol,
                self.mc64job,
                self.write_mat, )
    
    def import_panel1_value(self, v):
        self.hss = v[0]
        self.hss_front_size = v[1]
        self.rctol = v[2]
        self.actol = v[3]
        self.mc64job = v[4]
        self.write_mat = v[5]        

        
    def attribute_set(self, v):
        v = super(SpSparse, self).attribute_set(v)
        v['write_mat'] = False
        v['hss'] = False
        v['hss_front_size'] = 2500
        v['rctol'] = 0.01
        v['actol'] = 1e-10
        v['mc64job'] = 0
        return v
    
    def linear_system_type(self, assemble_real, phys_real):
        if phys_real: return 'coo'
        assert not assemble_real, "no conversion to real matrix is supported"
        return 'coo'
            
    def solve_central_matrix(self, engine, A, b):
        dprint1("entering solve_central_matrix")
        try:
           from mpi4py import MPI
           myid     = MPI.COMM_WORLD.rank
           nproc    = MPI.COMM_WORLD.size
        except:
           myid == 0
           MPI  = None
           
        if (A.dtype == 'complex'):
            is_complex = True
        else:
            is_complex = False

        print("A matrix type:", A.__class__, A.dtype, A.shape)

        if self.write_mat:
            #tocsr().tocoo() forces the output is row sorted.
            write_coo_matrix('matrix', A.tocsr().tocoo())
            for ib in range(b.shape[1]):
                write_vector('rhs_'+str(ib + engine.case_base), b[:,ib])
            engine.case_base = engine.case_base + len(b)
            
        from Strumpack import Sp, SUCCESS, DIRECT
        
        if myid ==0:
            A = A.tocsr()
            s = Sp()
            s.set_csr_matrix(A)
            s.solver.set_Krylov_solver(DIRECT)
            for attr in Sp_attrs:
                if hasattr(self, attr):
                    value = getattr(self, attr)
                    s.set_param(attr, value)
            returncode, sol = s.solve(b)
            assert returncode == SUCCESS, "StrumpackSparseSolver failed"            
            sol = np.transpose(sol.reshape(-1, len(b)))
            return sol
        
    def solve_distributed_matrix(self, engine, A, b):
        dprint1("entering solve_distributed_matrix")

        from mpi4py import MPI
        myid     = MPI.COMM_WORLD.rank
        nproc    = MPI.COMM_WORLD.size
        
        if (A.dtype == 'complex'):
            is_complex = True
        else:
            is_complex = False

        import gc
        A.eliminate_zeros()
        
        if self.write_mat:
            write_coo_matrix('matrix', A)
            if myid == 0:
                for ib in range(b.shape[1]):
                    write_vector('rhs_'+str(ib + engine.case_base), b[:,ib])
                case_base = engine.case_base + b.shape[1]
            else: case_base = None
            engine.case_base = MPI.COMM_WORLD.bcast(case_base, root=0)
            
        print("A matrix type:", A.__class__, A.dtype, A.shape)            


        dprint1("NNZ local: ", A.nnz)
        nnz_array = np.array(MPI.COMM_WORLD.allgather(A.nnz))
        if myid ==0:
            dprint1("NNZ all: ", nnz_array, np.sum(nnz_array))
            dprint1("RHS DoF: ", b.shape[0])
            dprint1("RHS len: ", b.shape[1])

        from petram.helper.mpi_recipes import distribute_global_coo, distribute_vec_from_head, gather_vector
        
        A_local = distribute_global_coo(A)
        b_local = distribute_vec_from_head(b) 

        #assert False, "Lets' stop here"        
        A_local = A_local.tocsr()
        print("A matrix type:", A_local.__class__, A_local.dtype, A_local.shape,
              A_local.nnz)
        
        from Strumpack import SpMPIDist, SUCCESS, DIRECT
        
        s = SpMPIDist()
        s.set_distributed_csr_matrix(A_local)
        s.solver.set_Krylov_solver(DIRECT)
        for attr in Sp_attrs:
            if hasattr(self, attr):
                value = getattr(self, attr)
                s.set_param(attr, value)
        returncode, sol = s.solve(b_local)
        assert returncode == SUCCESS, "StrumpackSparseSolver failed"            
        sol = gather_vector(sol)
        
        if (myid==0):
            sol = np.transpose(sol.reshape(-1, len(b)))            
            return sol
        else:
            return None
        
    def solve(self, engine, A, b):
        try:
           from mpi4py import MPI
           myid     = MPI.COMM_WORLD.rank
           nproc    = MPI.COMM_WORLD.size
           from petram.helper.mpi_recipes import gather_vector           
        except:
           myid == 0
           MPI  = None


        if engine.is_matrix_distributed:
            # call SpMPIDist
            ret = self.solve_distributed_matrix(engine, A, b)
            return ret
        else:
            # call Sp        
            ret = self.solve_central_matrix(engine, A, b)
            return ret
'''
