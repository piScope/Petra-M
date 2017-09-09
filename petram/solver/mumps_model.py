from .solver_model import Solver
import numpy as np
import scipy

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('MUMPSModel')

from petram.helper.matrix_file import write_matrix, write_vector, write_coo_matrix

class MUMPS(Solver):
    has_2nd_panel = False
    accept_complex = True    
    def init_solver(self):
        pass
    def panel1_param(self):
        return [["log_level(0-2)",   self.log_level,  400, {}],
                ["ordering",   self.ordering,  4, {"readonly": True,
                      "choices": ["auto", "Metis", "ParMetis", "PT-Scotch"]}],
                ["out-of-core",  self.out_of_core,  3, {"text":""}],
                ["error analysis",   self.error_ana,  4, {"readonly": True,
                      "choices": ["none", "full stat.", "main stat."]}],
                ["write matrix",  self.write_mat,   3, {"text":""}],
                ["centralize matrix",  self.central_mat,   3, {"text":""}],
                ["use BLR",  self.use_blr,   3, {"text":""}],
                ["BLR drop parameter",  self.blr_drop,   300, {}],]
    
    def get_panel1_value(self):
        return (long(self.log_level), self.ordering, self.out_of_core,
                self.error_ana, self.write_mat, self.central_mat,
                self.use_blr, self.blr_drop)
    
    def import_panel1_value(self, v):
        self.log_level = long(v[0])
        self.ordering = str(v[1])
        self.out_of_core = v[2]
        self.error_ana = v[3]                
        self.write_mat = v[4]
        self.central_mat = v[5]
        self.use_blr = v[6]
        self.blr_drop = v[7]                        

        
    def attribute_set(self, v):
        v = super(MUMPS, self).attribute_set(v)
        
        v['log_level'] = 0
        '''
        1 : Only error messages printed.
        2 : Errors, warnings, and main statistics printed.
        3 : Errors and warnings and terse diagnostics 
            (only first ten entries of arrays) printed.
        4 : Errors, warnings and information on input, output parameters printed
        '''
        v['out_of_core'] = False
        v['write_mat'] = False
        v['central_mat'] = False
        v['ordering'] = 'auto'
        v['error_ana'] = 'none'
        v['use_blr'] = False
        v['blr_drop'] = 0.0
        return v
    
    def linear_system_type(self, assemble_real, phys_real):
        if phys_real: return 'coo'
        if assemble_real: return 'coo_real'
        return 'coo'
            
    def set_ordering_flag(self, s):
        from petram.mfem_config import use_parallel        
        if self.ordering == 'auto':
            pass
        elif self.ordering == 'Metis':
            s.set_icntl(28,  1)
            s.set_icntl(7,  5)                            
        elif self.ordering == 'ParMetis' and use_parallel:
            s.set_icntl(28,  2)
            s.set_icntl(29,  2)
        elif self.ordering == 'ParMetis' and not use_parallel:            
            dprint1("!!! ParMetis ordering is selected. But solver is not in parallel mode. Ignored")            
        elif self.ordering == 'PT-Scotch' and use_parallel:
            s.set_icntl(28,  2)            
            s.set_icntl(29,  1)
        elif self.ordering == 'PT-Scotch' and not use_parallel:
            dprint1("!!! PT-Scotch ordering is selected. But solver is not in parallel mode. Ignored")
        else:
            pass
        #s.set_icntl(28,  2)                
        
    def set_error_analysis(self, s):
        from petram.mfem_config import use_parallel        
        if self.error_ana == 'none':
            s.set_icntl(11,  0)
        elif self.error_ana == 'main stat.':
            s.set_icntl(11,  2)                        
        elif self.error_ana == 'full stat.':
            s.set_icntl(11,  1)
        else:
            pass
        
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
        from petram.ext.mumps.mumps_solve import DMUMPS
        from petram.ext.mumps.mumps_solve import ZMUMPS
        from petram.ext.mumps.mumps_solve import i_array, JOB_1_2_3

        if is_complex:
            from petram.ext.mumps.mumps_solve import z_array as data_array
            from petram.ext.mumps.mumps_solve import z_to_list as to_list
            s = ZMUMPS()
        else:
            from petram.ext.mumps.mumps_solve import d_array as data_array
            from petram.ext.mumps.mumps_solve import d_to_list as to_list
            s = DMUMPS()
            
        A = A.tocoo(False)#.astype('complex')
        #dprint1('NNZ: ', len(engine.M.nonzero()[0]))                
        import petram.ext.mumps.mumps_solve as mumps_solve
        dprint1('!!!these two must be consistent')
        dprint1('sizeof(MUMPS_INT) ' , mumps_solve.SIZEOF_MUMPS_INT())
        #dprint1('index data size ' , type(A.col[0]))
        #dprint1('matrix data type ' , type(A.data[0]))

        dtype_int = 'int'+str(mumps_solve.SIZEOF_MUMPS_INT()*8)

        if self.write_mat:
            #tocsr().tocoo() forces the output is row sorted.
            write_coo_matrix('matrix', A.tocsr().tocoo())
            for ib in range(b.shape[1]):
                write_vector('rhs_'+str(ib + engine.case_base), b[:,ib])
            engine.case_base = engine.case_base + len(b)
        # No outputs
        if myid ==0:
            row = A.row
            col = A.col
            row = row.astype(dtype_int) + 1
            col = col.astype(dtype_int) + 1
            dprint1('index data size ' , type(col[0]))
            dprint1('matrix data type ' , type(A.data[0]))
            
            s.set_n(A.shape[0])
            s.set_lrhs_nrhs(b.shape[0], b.shape[1])
            # this way we keep bstack in memory            
            bstack = np.hstack(np.transpose(b))
            s.set_rhs(data_array(bstack))
            s.set_nz(len(A.data))
            s.set_irn(i_array(row))
            s.set_jcn(i_array(col))            
            s.set_a(data_array(A.data))
        # No outputs
        if self.use_blr:   
            s.set_icntl(35,1)
            s.set_cntl(7, float(self.blr_drop))
        
        if self.log_level == 0:
            s.set_icntl(1, -1)
            s.set_icntl(2, -1)
            s.set_icntl(3, -1)
            s.set_icntl(4,  0)
        elif self.log_level == 1:
            pass
        else:
            s.set_icntl(1,  6)            
            s.set_icntl(2,  6)
            s.set_icntl(3,  6)            
            s.set_icntl(4,  6)
        s.set_icntl(14,  50)
        s.set_icntl(6,  5)    # column permutation
        self.set_ordering_flag(s)

        if MPI is not None: MPI.COMM_WORLD.Barrier()
        dprint1("job1")
        s.set_job(1)
        s.run()

        if MPI is not None: MPI.COMM_WORLD.Barrier()
        dprint1("job2")
        s.set_job(2)
        s.run()

        if MPI is not None: MPI.COMM_WORLD.Barrier()
        dprint1("job3")
        self.set_error_analysis(s)        
        s.set_job(3)
        s.run()

        s.finish()

        if (myid == 0):
            if is_complex:
                sol = s.get_real_rhs()+1j*s.get_imag_rhs()
            else:
                sol = s.get_real_rhs()
            sol = np.transpose(sol.reshape(-1, len(b)))
            #sol = sol.reshape(len(b), -1)
            #sol = sol[:, flag]
            return sol
        
    def solve_distributed_matrix(self, engine, A, b):
        dprint1("entering solve_distributed_matrix")

        from mpi4py import MPI
        myid     = MPI.COMM_WORLD.rank
        nproc    = MPI.COMM_WORLD.size
        
        from petram.ext.mumps.mumps_solve import DMUMPS, d_to_list
        from petram.ext.mumps.mumps_solve import ZMUMPS, z_to_list
        from petram.ext.mumps.mumps_solve import i_array, JOB_1_2_3
        
        if (A.dtype == 'complex'):
            is_complex = True
        else:
            is_complex = False

        if is_complex:
            from petram.ext.mumps.mumps_solve import z_array as data_array
            from petram.ext.mumps.mumps_solve import z_to_list as to_list
            s = ZMUMPS()
        else:
            from petram.ext.mumps.mumps_solve import d_array as data_array
            from petram.ext.mumps.mumps_solve import d_to_list as to_list
            s = DMUMPS()
        import gc

        if self.write_mat:
            write_coo_matrix('matrix', A)
            if myid == 0:
                for ib in range(b.shape[1]):
                    write_vector('rhs_'+str(ib + engine.case_base), b[:,ib])
                case_base = engine.case_base + b.shape[1]
            else: case_base = None
            engine.case_base = MPI.COMM_WORLD.bcast(case_base, root=0)
            
        import petram.ext.mumps.mumps_solve as mumps_solve
        dprint1('!!!these two must be consistent')
        dprint1('sizeof(MUMPS_INT) ' , mumps_solve.SIZEOF_MUMPS_INT())
        #dprint1('index data size ' , type(A.col[0]))
        #dprint1('matrix data type ' , type(A.data[0]))

        s.set_icntl(5,0)
        s.set_icntl(18,3)
        if self.use_blr:   
            s.set_icntl(35,1)
            s.set_cntl(7, float(self.blr_drop))
        
        dprint1("NNZ local: ", A.nnz)
        nnz_array = np.array(MPI.COMM_WORLD.allgather(A.nnz))
        if myid ==0:
            dprint1("NNZ all: ", np.sum(nnz_array))
            dprint1("RHS DoF: ", b.shape[0])
            dprint1("RHS len: ", b.shape[1])
            s.set_n(A.shape[1])
            s.set_lrhs_nrhs(b.shape[0], b.shape[1])
            # this way we keep bstack in memory            
            bstack = np.hstack(np.transpose(b)) 
            s.set_rhs(data_array(bstack))
            
        dtype_int = 'int'+str(mumps_solve.SIZEOF_MUMPS_INT()*8)
        row = A.row
        col = A.col
        row = row.astype(dtype_int) + 1
        col = col.astype(dtype_int) + 1
        dprint1('index data size ' , type(col[0]))
        dprint1('matrix data type ' , type(A.data[0]))
        
            
        s.set_nz_loc(len(A.data))
        s.set_irn_loc(i_array(row))
        s.set_jcn_loc(i_array(col))            
        s.set_a_loc(data_array(A.data))

        # No outputs
        if self.log_level == 0:
            s.set_icntl(1, -1)
            s.set_icntl(2, -1)
            s.set_icntl(3, -1)
            s.set_icntl(4,  0)
        elif self.log_level == 1:
            pass
        else:
            s.set_icntl(1,  6)            
            s.set_icntl(2,  6)
            s.set_icntl(3,  6)            
            s.set_icntl(4,  6)
        s.set_icntl(14,  200)

        s.set_icntl(2, 1)
        self.set_ordering_flag(s)

        MPI.COMM_WORLD.Barrier()
        dprint1("job1")
        s.set_job(1)
        s.run()

        MPI.COMM_WORLD.Barrier()
        dprint1("job2")
        s.set_icntl(24, 1)
        # this seem to get things worse...
        #s.set_cntl(3, 1e-5)
        #s.set_cntl(5, 1e20)        
        s.set_job(2)
        s.run()

        MPI.COMM_WORLD.Barrier()
        dprint1("job3")
        self.set_error_analysis(s)        
        s.set_job(3)
        s.run()
        s.finish()
        
        rsol = None; isol = None; sol_extra = None
        if (myid == 0):
            if is_complex:
                sol = s.get_real_rhs()+1j*s.get_imag_rhs()
            else:
                sol = s.get_real_rhs()
            sol = np.transpose(sol.reshape(-1, len(b)))
            return sol

        
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
            if self.central_mat:
                # in this case, we collect all matrix data to root
                # and re-define A on the root node
                dprint1("centralizing distributed matrix")                               
                shape =A.shape
                col = gather_vector(A.col)
                row = gather_vector(A.row)
                data = gather_vector(A.data)
                if myid == 0:
                    A = scipy.sparse.coo_matrix((data,(row, col)),
                                                  shape = shape,
                                                  dtype = A.dtype)
                dprint1("calling solve_central_matrix")
                return self.solve_central_matrix(engine, A, b)
            else:
                dprint1("calling solve_distributed_matrix")                
                return self.solve_distributed_matrix(engine, A, b)
        else:

            dprint1("calling solve_central_matrix")                            
            ret = self.solve_central_matrix(engine, A, b)
            return ret
