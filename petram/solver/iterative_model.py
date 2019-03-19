import numpy as np

from petram.namespace_mixin import NS_mixin
from .solver_model import LinearSolverModel, LinearSolver

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('IterativeSolverModel')

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

from petram.solver.mumps_model import MUMPSPreconditioner   
SparseSmootherCls = {"Jacobi": (mfem.DSmoother, 0),
                     "l1Jacobi": (mfem.DSmoother, 1),
                     "lumpedJacobi": (mfem.DSmoother, 2),
                     "GS": (mfem.GSSmoother, 0),
                     "forwardGS": (mfem.GSSmoother, 1),
                     "backwardGS": (mfem.GSSmoother, 2),
                     "MUMPS": (MUMPSPreconditioner, None),}

class Iterative(LinearSolverModel, NS_mixin): 
    hide_ns_menu = True
    has_2nd_panel = False
    accept_complex = False
    always_new_panel = False

    def __init__(self,  *args, **kwargs):
        LinearSolverModel.__init__(self, *args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)
        
    def init_solver(self):
        pass
    
    def panel1_param(self):
        import wx
        from petram.pi.widget_smoother import WidgetSmoother

        smp1 = [None, None, 99, {"UI":WidgetSmoother, "span":(1,2)}]
        return [[None, 'GMRES', 4, {'style':wx.CB_READONLY,
                                     'choices': ['CG', 'GMRES', 'FGMRES', 'BiCGSTAB',
                                                 'MINRES', 'SLI']}],      
                ["log_level",   self.log_level,  400, {}],
                ["max  iter.",  self.maxiter,  400, {}],
                ["rel. tol",    self.reltol,  300,  {}],
                ["abs. tol.",   self.abstol,  300, {}],
                ["restart(kdim)", self.kdim,     400, {}],
                [None,  [False, [''], [[],]], 27, [{'text':'advanced mode'},
                                               {'elp':[['preconditioner', '', 0, None],]},
                                               {'elp':[smp1,]}],],
                [None,  self.write_mat,   3, {"text":"write matrix"}],
                [None, self.assert_no_convergence,  3, {"text":"check converegence"}],
                [None, self.use_ls_reducer,  3, {"text":"Reduce linear system when possible"}],]     
     
    
    def get_panel1_value(self):
        # this will set _mat_weight
        from petram.solver.solver_model import SolveStep
        p = self.parent
        while not isinstance(p, SolveStep):
            p = p.parent
            if p is None:
                assert False, "Solver is not under SolveStep"
        num_matrix = p.get_num_matrix(self.get_phys())
        
        all_dep_vars = self.root()['Phys'].all_dependent_vars(num_matrix, self.get_phys())
        
        prec = [x for x in  self.preconditioners if x[0] in all_dep_vars]        
        names = [x[0] for x in  prec]
        for n in all_dep_vars:
           if not n in names:
              prec.append((n, ['None', 'None']))
        self.preconditioners = prec
        
        return (self.solver_type,
                long(self.log_level), long(self.maxiter),
                self.reltol, self.abstol, long(self.kdim),
                [self.adv_mode, [self.adv_prc, ], [self.preconditioners,]],
                self.write_mat, self.assert_no_convergence, self.use_ls_reducer)
    
    def import_panel1_value(self, v):
        self.solver_type = str(v[0])
        self.log_level = long(v[1])
        self.maxiter = long(v[2])
        self.reltol = v[3]
        self.abstol = v[4]
        self.kdim = long(v[5])
        self.preconditioners = v[6][2][0]
        self.write_mat = bool(v[7])
        self.assert_no_convergence = bool(v[8])
        self.use_ls_reducer = bool(v[9])        
        self.adv_mode = v[6][0]
        self.adv_prc = v[6][1][0]      
        
    def attribute_set(self, v):
        v = super(Iterative, self).attribute_set(v)
        v['log_level'] = 0
        v['maxiter'] = 200
        v['reltol']  = 1e-7
        v['abstol'] = 1e-7
        v['kdim'] =   50
        v['printit'] = 1
        v['preconditioner'] = ''
        v['preconditioners'] = []        
        v['write_mat'] = False
        v['solver_type'] = 'GMRES'
        v['assert_no_convergence'] = True
        v['use_ls_reducer'] = False
        v['adv_mode'] = False
        v['adv_prc'] = ''
        return v
    
    def verify_setting(self):
        if not self.parent.assemble_real:
            for phys in self.get_phys():
                if phys.is_complex():
                    return False, "Iterative does not support complex.", "A complex problem must be converted to a real value problem"
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

    def allocate_solver(self, datatype='D', engine=None):
        solver = IterativeSolver(self, engine, int(self.maxiter),
                             self.abstol, self.reltol, int(self.kdim))
        #solver.AllocSolver(datatype)
        return solver
     
    def get_possible_child(self):
        '''
        Preconditioners....
        '''
        choice = []
        try:
            from petram.solver.mumps_model import MUMPS
            choice.append(MUMPS)
        except ImportError:
            pass
        return choice
     

class IterativeSolver(LinearSolver):
    is_iterative = True
    def __init__(self, gui, engine, maxiter, abstol, reltol, kdim):
        self.maxiter = maxiter
        self.abstol = abstol
        self.reltol = reltol
        self.kdim = kdim
        LinearSolver.__init__(self, gui, engine)

    def SetOperator(self, opr, dist=False, name = None):
        self.Aname = name
        self.A = opr
        
        from petram.solver.linearsystem_reducer import LinearSystemReducer
        if use_parallel:
            if self.gui.use_ls_reducer:
                self.reducer = LinearSystemReducer(opr, name)
                self.M_reduced = self.make_preconditioner(self.reducer.A,
                                                          name = self.reducer.Aname,
                                                          parallel=True)
                solver  = self.make_solver(self.reducer.A,
                                           self.M_reduced,
                                           use_mpi=True)
                self.reducer.set_solver(solver)
            else:
                self.M = self.make_preconditioner(self.A, parallel=True)
                self.solver = self.make_solver(self.A, self.M, use_mpi=True)
        else:
            if self.gui.use_ls_reducer:
                dprint1("Linear system reducer is not implemented in serial")
            self.M = self.make_preconditioner(self.A)
            self.solver = self.make_solver(self.A, self.M)

            self.reducer = None
                             
    def Mult(self, b, x=None, case_base=0):
        if use_parallel:
            return self.solve_parallel(self.A, b, x)
        else:
            return self.solve_serial(self.A, b, x)

    def make_solver(self, A, M, use_mpi=False):
        maxiter = int(self.maxiter)
        atol = self.abstol
        rtol = self.reltol
        kdim = int(self.kdim)
        printit = 1
       
        args = (MPI.COMM_WORLD,) if use_mpi else ()

        cls = getattr(mfem, self.gui.solver_type+'Solver')

        solver = cls(*args)
        if self.gui.solver_type == 'GMRES':
            solver.SetKDim(kdim)
        if self.gui.solver_type == 'FGMRES':
            solver.SetKDim(kdim)

        solver.SetPreconditioner(M)
        solver.SetOperator(A)
        
        solver.SetAbsTol(atol)
        solver.SetRelTol(rtol)
        solver.SetMaxIter(maxiter)
        solver.SetPrintLevel(self.gui.log_level)
        
        return solver

    def make_preconditioner(self, A, name = None, parallel=False):
        name = self.Aname if name is None else name
        
        if self.gui.adv_mode:
            expr = self.gui.adv_prc
            gen = eval(expr, self.gui._global_ns)
            gen.set_param(A, self.engine, self.gui)
            M = gen()
        else:
            prcs_gui = dict(self.gui.preconditioners)
            assert not self.gui.parent.is_complex(), "can not solve complex"
            if self.gui.parent.is_converted_from_complex():
                name = sum([[n, n] for n in name], [])

            import petram.helper.preconditioners as prcs

            g = prcs.DiagonalPrcGen(opr=A, engine=self.engine, gui=self.gui)
            M = g()

            for k, n in enumerate(name):
                prctxt = prcs_gui[n][1] if parallel else prcs_gui[n][0]
                if prctxt == "None": continue
                if prctxt.find("(") == -1: prctxt=prctxt+"()"
                prcargs = "(".join(prctxt.split("(")[-1:])
                
                nn = prctxt.split("(")[0]
                dprint1(nn)
                try:
                    blkgen = getattr(prcs, nn)
                except:
                    if nn in self.gui._global_ns:
                        blkgen = self.gui._global_ns[nn]
                    else:
                        raise

                blkgen.set_param(g, n)
                blk = eval("blkgen("+prcargs)

                M.SetDiagonalBlock(k, blk)
                
        return M

    def write_mat(self, A, b, x, suffix=""):
        def get_block(Op, i, j):
            try:
               return Op._linked_op[(i,j)]
            except KeyError:
               return None

        offset = A.RowOffsets()
        rows = A.NumRowBlocks()
        cols = A.NumColBlocks()
        
        for i in range(cols):
           for j in range(rows):
              m = get_block(A, i, j)
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
        
    def call_mult(self, solver, bb, xx):
        solver.Mult(bb, xx)
        max_iter = solver.GetNumIterations();
        tol = solver.GetFinalNorm()**2
        
        dprint1("convergence check (max_iter, tol) ", max_iter, " ", tol)
        if self.gui.assert_no_convergence:
            if not solver.GetConverged():
               self.gui.set_solve_error((True, "No Convergence: " + self.gui.name()))
               assert False, "No convergence"
     
    def solve_parallel(self, A, b, x=None):
        if self.gui.write_mat:                      
            self. write_mat(A, b, x, "."+smyid)
            
        sol = []

        # solve the problem and gather solution to head node...
        # may not be the best approach
        
        from petram.helper.mpi_recipes import gather_vector        
        offset = A.RowOffsets()
        for bb in b:
           rows = MPI.COMM_WORLD.allgather(np.int32(bb.Size()))
           #rowstarts = np.hstack((0, np.cumsum(rows)))
           dprint1("row offset", offset.ToList())
           if x is None:           
              xx = mfem.BlockVector(offset)
              xx.Assign(0.0)
           else:
              xx = x

           if self.gui.use_ls_reducer:
               try:
                   self.reducer.Mult(bb, xx, self.gui.assert_no_convergence)
               except debug.ConvergenceError:
                   self.gui.set_solve_error((True, "No Convergence: " + self.gui.name()))
                   assert False, "No convergence"                   
           else:
               self.call_mult(self.solver, bb, xx)

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
        
    def solve_serial(self, A, b, x=None):
        if self.gui.write_mat:                      
            self. write_mat(A, b, x)

        M = self.M
        solver = self.solver
            
        sol = []
        
        for bb in b:
           if x is None:           
              xx = mfem.Vector(bb.Size())
              xx.Assign(0.0)
           else:
              xx = x
              #for j in range(cols):
              #   print x.GetBlock(j).Size()
              #   print x.GetBlock(j).GetDataArray()                 
              #assert False, "must implement this"
           self.call_mult(solver, bb, xx)              

           sol.append(xx.GetDataArray().copy())
        sol = np.transpose(np.vstack(sol))
        return sol
        

        '''           
        prcs = dict(self.gui.preconditioners)
        name = self.Aname
        assert not self.gui.parent.is_complex(), "can not solve complex"
        if self.gui.parent.is_converted_from_complex():
           name = sum([[n, n] for n in name], [])


        for k, n in enumerate(name):
           
           prc = prcs[n][0]
           if prc == "None": continue
           name = "".join([tmp for tmp in prc if not tmp.isdigit()])
           A0 = get_block(A, k, k)
           cls = SparseSmootherCls[name][0]
           arg = SparseSmootherCls[name][1]
           if name == 'MUMPS':
               invA0 = cls(A0, gui=self.gui[prc], engine=self.engine)

           elif name.startswith('schur'):
               args = name.split("(")[-1].split(")")[0].split(",")
               dprint1("setting up schur for ", args)
               if len(args) > 1:
                   assert False, "not yet supported"
               for arg in args:
                    r1 = self.engine.dep_var_offset(arg.strip())
                    c1 = self.engine.r_dep_var_offset(arg.strip())                    
                    B  =  get_block(A, k, c1)
                    Bt =  get_block(A, r1, k)
                    B0 = get_block(A, r1, c1)
                    Md = mfem.Vector(M0.Height())
                    B0.GetDiag(Md)                    
                    for i in range(Md.Size()):
                        if Md[i] != 0.:
                            Bt.ScaleRow(i, 1/Md[i])
                        else:
                            assert False, "diagnal element of matrix is zero"
                    
                    S = mfem.Mult(B, Bt)
                    invA0 = mfem.DSmoother(S)
                    invA0.iterative_mode = False
               
           else:
               invA0 = cls(A0, arg)
           invA0.iterative_mode = False
           M.SetDiagonalBlock(k, invA0)
        '''
        '''
        We should support Shur complement type preconditioner
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

        '''
        int GMRES(const Operator &A, Vector &x, const Vector &b, Solver &M,
          int &max_iter, int m, double &tol, double atol, int printit)
        '''
    

         
        
