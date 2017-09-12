from .solver_model import Solver
import numpy as np
import scipy

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('StrumpackModel')

from petram.helper.matrix_file import write_matrix, write_vector, write_coo_matrix
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
