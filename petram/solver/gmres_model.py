from .solver_model import Solver
import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('GMRESModel')

from petram.mfem_config import use_parallel
if use_parallel:
   from petram.helper.mpi_recipes import *
   from mfem.common.parcsr_extra import *
   import mfem.par as mfem
   default_kind = 'hypre'
else:
   import mfem.ser as mfem
   default_kind = 'scipy'

class GMRES(Solver):
    has_2nd_panel = False
    accept_complex = False
    def init_solver(self):
        pass
    
    def panel1_param(self):
        return [["log_level",   self.log_level,  400, {}],
                ["max  iter.",  self.maxiter,  300, {}],
                ["rel. tol",    self.reltol,  300,  {}],
                ["abs. tol.",   self.abstol,  300, {}],
                ["restart(kdim)", self.kdim,     400, {}],
                ["preconditioner", self.preconditioner,     0, {}],]    
    
    def get_panel1_value(self):
        return (long(self.log_level), long(self.maxiter),
                self.reltol, self.abstol, long(self.kdim),
                self.preconditioner)
    
    def import_panel1_value(self, v):
        self.log_level = long(v[0])
        self.maxiter = long(v[1])
        self.reltol = v[2]
        self.abstol = v[3]
        self.kdim = long(v[4])
        self.preconditioner = v[5]
        
    def attribute_set(self, v):
        v = super(GMRES, self).attribute_set(v)
        v['log_level'] = 0
        v['maxiter'] = 200
        v['reltol']  = 1e-7
        v['abstol'] = 1e-7
        v['kdim'] =   50
        v['printit'] = 1
        v['preconditioner'] = 'AMS'
        return v
    
    def verify_setting(self):
        if not self.parent.assemble_real:
            root = self.root()
            phys = root['Phys'][self.parent.phys_model]
            if phys.is_complex:
                return False, "Complex Problem not supported.", "AMS does not support complex problem"
        return True, "", ""

    def linear_system_type(self, assemble_real, phys_complex):
        #if not phys_complex: return 'block'
        return 'blk_interleave'
        #return None


    def solve_parallel(self, engine, A, b):
        from mpi4py import MPI
        myid     = MPI.COMM_WORLD.rank
        nproc    = MPI.COMM_WORLD.size
        from petram.helper.mpi_recipes import gather_vector
        
        def get_block(Op, i, j):
            return Op._linked_op[(i,j)]

                      
        offset = A.RowOffsets()
        rows = A.NumRowBlocks()

        M = mfem.BlockDiagonalPreconditioner(offset)

        #A.GetBlock(0,0).Print()
        #M1 = mfem.DSmoother(get_block(A, 0, 0))
        #M1 = mfem.GSSmoother(get_block(A, 0, 0))
        #M1.iterative_mode = False
        #M.SetDiagonalBlock(0, M1)
        A0 = get_block(A, 0, 0)   
        invA0 = mfem.HypreDiagScale(A0)
        invA0.iterative_mode = False
        M.SetDiagonalBlock(0, invA0)

        if offset.Size() > 2:
            B =  get_block(A, 1, 0)
            MinvBt = get_block(A, 0, 1)
            #Md = mfem.HypreParVector(MPI.COMM_WORLD,
            #                        A0.GetGlobalNumRows(),
            #                        A0.GetRowStarts())
            Md = mfem.Vector()
            A0.GetDiag(Md)
            MinvBt.InvScaleRows(Md)
            S = mfem.ParMult(B, MinvBt)
            invS = mfem.HypreBoomerAMG(S)
            invS.iterative_mode = False
            M.SetDiagonalBlock(1, invS)

        maxiter = int(self.maxiter)
        atol = self.abstol
        rtol = self.reltol
        kdim = int(self.kdim)
        printit = 1

        sol = []

        solver = mfem.GMRESSolver(MPI.COMM_WORLD)
        solver.SetKDim(kdim)
        #solver = mfem.MINRESSolver(MPI.COMM_WORLD)
        solver.SetAbsTol(atol)
        solver.SetRelTol(rtol)
        solver.SetMaxIter(maxiter)
        solver.SetOperator(A)
        solver.SetPreconditioner(M)
        solver.SetPrintLevel(1)

        # solve the problem and gather solution to head node...
        # may not be the best approach

        for bb in b:
           rows = MPI.COMM_WORLD.allgather(np.int32(bb.Size()))
           rowstarts = np.hstack((0, np.cumsum(rows)))
           dprint1(rowstarts)
           x = mfem.BlockVector(offset)
           x.Assign(0.0)
           solver.Mult(bb, x)
           s = []
           for i in range(offset.Size()-1):
               v = x.GetBlock(i).GetDataArray()
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
        
    def solve_serial(self, engine, A, b):

        def get_block(Op, i, j):
            return Op._linked_op[(i,j)]

                      
        offset = A.RowOffsets()
        rows = A.NumRowBlocks()

        M = mfem.BlockDiagonalPreconditioner(offset)

        #M1 = mfem.DSmoother(get_block(A, 0, 0))     
        M1 = mfem.GSSmoother(get_block(A, 0, 0))
        M1.iterative_mode = False
        M.SetDiagonalBlock(0, M1)

        if offset.Size() > 2:
            B =  get_block(A, 1, 0)
            MinvBt = get_block(A, 0, 1)
            Md = mfem.Vector(get_block(A, 0, 0).Height())
            get_block(A, 0, 0).GetDiag(Md)
            for i in range(Md.Size()):
                if Md[i] != 0.:
                    MinvBt.ScaleRow(i, 1/Md[i])
                else:
                    assert False, "diagnal element of matrix is zero"
            S = mfem.Mult(B, MinvBt)
            S.iterative_mode = False
            SS = mfem.DSmoother(S)
            SS.iterative_mode = False            
            M.SetDiagonalBlock(1, SS)


        '''
        int GMRES(const Operator &A, Vector &x, const Vector &b, Solver &M,
          int &max_iter, int m, double &tol, double atol, int printit)
        '''
        maxiter = int(self.maxiter)
        atol = self.abstol
        rtol = self.reltol
        kdim = int(self.kdim)
        printit = 1

        sol = []

        solver = mfem.GMRESSolver()
        solver.SetKDim(kdim)
        #solver = mfem.MINRESSolver()
        solver.SetAbsTol(atol)
        solver.SetRelTol(rtol)
        solver.SetMaxIter(maxiter)

        solver.SetOperator(A)
        solver.SetPreconditioner(M)
        solver.SetPrintLevel(1)

        for bb in b:
           #bb.Print()            
           x = mfem.Vector(bb.Size())
           x.Assign(0.0)
           ##print A, M, b[0], x, printit, maxiter, kdim, tol, atol
           #mfem.GMRES(A, M, bb, x, printit, maxiter, kdim, tol, atol)
           solver.Mult(bb, x)
           sol.append(x.GetDataArray().copy())
        sol = np.transpose(np.vstack(sol))
        return sol
    
    def solve(self, engine, A, b):
        if use_parallel:
            return self.solve_parallel(engine, A, b)
        else:
            return self.solve_serial(engine, A, b)
'''    
    def make_ams_preconditioner(self, engine):
        ## this code fragment should go to AMS preconditioner?
        ## extra for ASM
        d_b = engine.new_lf()        
        d_x = engine.new_gf()   ### solution + essential
        d_a = engine.new_bf()   ### matrix
        engine.assemble_bf(phys, d_a)
        engine.assemble_lf_bf(phys, d_a, real=False)
        
        d_X = mfem.Vector()           
        d_A = engine.new_matrix()
        d_B = mfem.Vector()
        da.FormLinearSystem(engine.ess_tdof_list,
                            d_x, d_b, d_A, d_X, d_B)
        
        d_a.FormLinearSystem(ess_tdof_list, d_x, d_b, self.d_A, d_X, d_B)

        ams1 = mfem.HypreAMS(d_A, engine.fespace)
        return ams1
    
    def solve(self, engine, A, b, flag, initial_x, offsets, isComplex = True):
        #solve matrix using GMRES
        #offset is python list of block offsets in real part section
        #like   [0, r_A.GetNumRows()]
        Pr = self.make_ams_preconditioner(engine)
        if isComplex:
            #here offsets should be doble sized
            offsets2 = offsets + offsets
            offsets2.PartialSum()
            imag_block = len(offset)-1
            Pr = mfem.BlockDiagonalPreconditioner(offsets2)
            Pr.SetDiagonalBlock(0, ams1);
            Pr.SetDiagonalBlock(imag_block, ams1);
        else:
            offsets = mfem.intArray(offsets)
            offsets.PartialSum()              
            Pr = mfem.BlockDiagonalPreconditioner(offsets)
            Pr.SetDiagonalBlock(0, ams1);
            
        
        import time
        stime = time.clock()
        solver = mfem.GMRESSolver(MPI.COMM_WORLD)
        solver.SetAbsTol(self.abstol)
        solver.SetRelTol(self.reltol)
        solver.SetMaxIter(self.maxiter)
        solver.SetKDim(self.kdim)
        solver.SetOperator(A)
        solver.SetPreconditioner(Pr)
        solver.SetPrintLevel(2)

        print 'start solving'
        solver.Mult(b, x)

        
        data = x.GetBlock(0).GetDataArray()
        rsol = self.gather_vector(data, MPI.DOUBLE)
        if myid != 0: rsol = None#; isol = None; sol_extra = None
        sol = self.scatter_vector(rsol, MPI.DOUBLE)

        if isComplex:
            data = x.GetBlock(imag_block).GetDataArray()        
            isol = self.gather_vector(data, MPI.DOUBLE)
            if myid != 0: isol = None
            isol = self.scatter_vector(isol, MPI.DOUBLE)
            sol = sol + 1j*isol
        extra = []

        return sol, extra
        
'''    
