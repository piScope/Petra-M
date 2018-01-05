from .solver_model import Solver
import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('MUMPSModel')

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
        return (long(self.log_level), self.maxiter,
                self.reltol, self.abstol, self.kdim,
                self.preconditioner)
    
    def import_panel1_value(self, v):
        self.log_level = long(v[0])
        self.maxiter = v[1]
        self.reltol = v[2]
        self.abstol = v[3]
        self.kdim = v[4]        
        self.preconditioner = v[5]
        
    def attribute_set(self, v):
        v = super(GMRES, self).attribute_set(v)
        v['log_level'] = 0
        v['maxiter'] = 200
        v['reltol']  = 1e-7
        v['abstol'] = 1e-7
        v['kdim'] =   50
        v['preconditioner'] = 'AMS'
        return v
    
    def verify_setting(self):
        if not self.parent.assemble_real:
            root = self.root
            phys = root['Phys'][self.parent.phys_model]
            if phys.is_complex: return False, "Complex Problem not supported.", "AMS does not support complex problem"
        return True, "", ""

    def linear_system_type(self, assemble_real, phys_complex):
        if not phys_complex: return 'block'
        if assemble_real: return 'block_real'
        return None

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
        '''
        solve matrix using GMRES

        offset is python list of block offsets in real part section
        like   [0, r_A.GetNumRows()]

        '''
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
        
    
