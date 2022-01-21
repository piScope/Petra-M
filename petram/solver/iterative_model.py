from petram.solver.mumps_model import MUMPSPreconditioner
from petram.mfem_config import use_parallel
import numpy as np

from petram.debug import flush_stdout
from petram.namespace_mixin import NS_mixin
from .solver_model import LinearSolverModel, LinearSolver

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('IterativeSolverModel')

if use_parallel:
    from petram.helper.mpi_recipes import *
    from mfem.common.parcsr_extra import *
    import mfem.par as mfem
    default_kind = 'hypre'

    from mpi4py import MPI
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank
    smyid = '{:0>6d}'.format(myid)
    from mfem.common.mpi_debug import nicePrint

else:
    import mfem.ser as mfem
    default_kind = 'scipy'

SparseSmootherCls = {"Jacobi": (mfem.DSmoother, 0),
                     "l1Jacobi": (mfem.DSmoother, 1),
                     "lumpedJacobi": (mfem.DSmoother, 2),
                     "GS": (mfem.GSSmoother, 0),
                     "forwardGS": (mfem.GSSmoother, 1),
                     "backwardGS": (mfem.GSSmoother, 2),
                     "MUMPS": (MUMPSPreconditioner, None), }

choices_a = ['CG', 'GMRES', 'FGMRES', 'BiCGSTAB',
             'MINRES', 'SLI', 'Nested FGMRES']
choices_b = ['CG', 'GMRES', 'FGMRES', 'BiCGSTAB', 'MINRES', 'MUMPS']

single1_elp = [["log_level", -1, 400, {}],
               ["max  iter.", 200, 400, {}],
               ["rel. tol", 1e-7, 300, {}],
               ["abs. tol.", 1e-7, 300, {}], ]
single2_elp = [["log_level", -1, 400, {}],
               ["max  iter.", 200, 400, {}],
               ["rel. tol", 1e-7, 300, {}],
               ["abs. tol.", 1e-7, 300, {}],
               ["restart(kdim)", 50, 400, {}]]

nsingle1_elp = [["log_level", -1, 400, {}],
                ["max  iter.", 200, 400, {}], ]
nsingle2_elp = [["log_level", -1, 400, {}],
                ["max  iter.", 200, 400, {}],
                ["restart(kdim)", 50, 400, {}], ]
nmumps_elp = [['MUMPS Config', '', 0, {}], ]
nested_elp = [["log_level", -1, 400, {}],
              ["max  iter.", 200, 400, {}],
              ["rel. tol", 1e-7, 300, {}],
              ["abs. tol.", 1e-7, 300, {}],
              ["restart(kdim)", 50, 400, {}],
              [None, None, 34, [{'text': "Inner Solver",
                                 'choices': choices_b},
                                {'elp': nsingle1_elp},  # CG
                                {'elp': nsingle2_elp},  # GMRES
                                {'elp': nsingle2_elp},  # FGMRES
                                {'elp': nsingle1_elp},  # BiCGSTAB
                                {'elp': nsingle1_elp},  # MINRES
                                {'elp': nmumps_elp}, ]]  # MUMPS
              , ]


class Iterative(LinearSolverModel, NS_mixin):
    hide_ns_menu = True
    has_2nd_panel = False
    accept_complex = False
    always_new_panel = False

    def __init__(self, *args, **kwargs):
        LinearSolverModel.__init__(self, *args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)

    def init_solver(self):
        pass

    def panel1_param(self):
        import wx
        from petram.pi.widget_smoother import WidgetSmoother

        smp1 = [None, None, 99, {"UI": WidgetSmoother, "span": (1, 2)}]

        mm = [[None, self.use_block_symmetric, 3,
               {"text": "block symmetric format"}], ]

        return [[None, None, 34, [{'text': "Solver", 'choices': choices_a},
                                  {'elp': single1_elp},  # CG
                                  {'elp': single2_elp},  # GMRES
                                  {'elp': single2_elp},  # FGMRES
                                  {'elp': single1_elp},  # BiCGSTAB
                                  {'elp': single1_elp},  # MINRES
                                  {'elp': single1_elp},  # SLI
                                  {'elp': nested_elp},  # Nested FGMRES
                                  ], ],
                [None, [False, [''], [[], ]], 27, [{'text': 'advanced mode'},
                                                   {'elp': [
                                                       ['preconditioner', '', 0, None], ]},
                                                   {'elp': [smp1, ]}], ],
                [None, self.write_mat, 3, {"text": "write matrix"}],
                [None, self.assert_no_convergence, 3,
                    {"text": "check converegence"}],
                [None, self.use_ls_reducer, 3, {
                    "text": "Reduce linear system when possible"}],
                [None, (self.merge_real_imag, (self.use_block_symmetric,)),
                 27, ({"text": "Use ComplexOperator"}, {"elp": mm},)], ]

    def get_panel1_value(self):
        # this will set _mat_weight
        from petram.solver.solver_model import SolveStep
        p = self.parent
        while not isinstance(p, SolveStep):
            p = p.parent
            if p is None:
                assert False, "Solver is not under SolveStep"
        num_matrix = p.get_num_matrix(self.get_phys())

        all_dep_vars = self.root()['Phys'].all_dependent_vars(num_matrix,
                                                              self.get_phys(),
                                                              self.get_phys_range())

        prec = [x for x in self.preconditioners if x[0] in all_dep_vars]
        names = [x[0] for x in prec]
        for n in all_dep_vars:
            if not n in names:
                prec.append((n, ['None', 'None']))
        self.preconditioners = prec

        single1 = [int(self.log_level), int(self.maxiter),
                   self.reltol, self.abstol]
        single2 = [int(self.log_level), int(self.maxiter),
                   self.reltol, self.abstol, int(self.kdim)]
        single1in = [int(self.log_level_in), int(self.maxiter_in), ]
        single2in = [int(self.log_level_in), int(self.maxiter_in),
                     int(self.kdim_in)]
        mumpsin = [self.mumps_in]
        nested = [int(self.log_level), int(self.maxiter),
                  self.reltol, self.abstol, int(self.kdim),
                  [self.solver_type_in, single1in, single2in, single2in,
                   single1in, single1in, mumpsin], ]
        value = ([self.solver_type, single1, single2, single2,
                  single1, single1, single1,
                  nested],
                 (self.adv_mode, [self.adv_prc, ], [self.preconditioners, ]),
                 self.write_mat, self.assert_no_convergence,
                 self.use_ls_reducer,
                 (self.merge_real_imag, [self.use_block_symmetric, ]),)

        return value

    def import_panel1_value(self, v):
        self.solver_type = str(v[0][0])
        idx = choices_a.index(self.solver_type)
        vv = v[0][idx + 1]
        self.log_level = int(vv[0])
        self.maxiter = int(vv[1])
        self.reltol = vv[2]
        self.abstol = vv[3]

        if len(vv) > 4:
            self.kdim = int(vv[4])

        if self.solver_type.startswith('Nested'):
            self.solver_type_in = vv[5][0]
            idx = choices_b.index(self.solver_type_in)
            vv = vv[5][idx + 1]
            if self.solver_type_in != 'MUMPS':
                self.log_level_in = int(vv[0])
                self.maxiter_in = int(vv[1])
                if len(vv) > 2:
                    self.kdim_in = int(vv[2])
            else:
                self.mumps_in = vv[0]

        self.preconditioners = v[1][2][0]
        self.write_mat = bool(v[2])
        self.assert_no_convergence = bool(v[3])
        self.use_ls_reducer = bool(v[4])
        self.adv_mode = v[1][0]
        self.adv_prc = v[1][1][0]
        self.merge_real_imag = bool(v[5][0])
        self.use_block_symmetric = bool(v[5][1][0])

    def attribute_set(self, v):
        v = super(Iterative, self).attribute_set(v)
        v['solver_type'] = 'GMRES'
        v['log_level'] = 0
        v['maxiter'] = 200
        v['reltol'] = 1e-7
        v['abstol'] = 1e-7
        v['kdim'] = 50

        v['solver_type_in'] = 'GMRES'
        v['log_level_in'] = 0
        v['maxiter_in'] = 200
        v['reltol_in'] = 1e-7
        v['abstol_in'] = 1e-7
        v['kdim_in'] = 50
        v['mumps_in'] = ''

        v['printit'] = 1
        v['preconditioner'] = ''
        v['preconditioners'] = []
        v['write_mat'] = False
        v['assert_no_convergence'] = True
        v['use_ls_reducer'] = False
        v['adv_mode'] = False
        v['adv_prc'] = ''
        v['merge_real_imag'] = False
        v['use_block_symmetric'] = False
        return v

    def verify_setting(self):
        if not self.parent.assemble_real:
            for phys in self.get_phys():
                if phys.is_complex() and not self.merge_real_imag:
                    return False, "Iterative does not support complex.", "A complex problem must be converted to a real value problem"
        return True, "", ""

    def does_linearsolver_choose_linearsystem_type(self):
        return True

    def linear_system_type(self, assemble_real, phys_real):
        if phys_real:
            if assemble_real:
                dprint1("Use assemble-real is only for complex value problem !!!!")
                return 'blk_interleave'
            else:
                return 'blk_interleave'

        # below phys is complex

        # merge_real_imag -> complex operator
        if self.merge_real_imag and self.use_block_symmetric:
            return 'blk_merged_s'
        elif self.merge_real_imag and not self.use_block_symmetric:
            return 'blk_merged'
        elif assemble_real:
            return 'blk_interleave'
        else:
            assert False, "complex problem must assembled using complex operator or expand as real value problem"
        # return None

    def real_to_complex(self, solall, M):
        if self.merge_real_imag:
            return self.real_to_complex_merged(solall, M)
        else:
            return self.real_to_complex_interleaved(solall, M)

    def real_to_complex_interleaved(self, solall, M):
        if use_parallel:
            from mpi4py import MPI
            myid = MPI.COMM_WORLD.rank

            offset = M.RowOffsets().ToList()
            of = [np.sum(MPI.COMM_WORLD.allgather(np.int32(o)))
                  for o in offset]
            if myid != 0:
                return

        else:
            offset = M.RowOffsets()
            of = offset.ToList()

        rows = M.NumRowBlocks()
        s = solall.shape
        nb = rows // 2
        i = 0
        pt = 0
        result = np.zeros((s[0] // 2, s[1]), dtype='complex')
        for j in range(nb):
            l = of[i + 1] - of[i]
            result[pt:pt + l, :] = (solall[of[i]:of[i + 1], :]
                                    + 1j * solall[of[i + 1]:of[i + 2], :])
            i = i + 2
            pt = pt + l

        return result

    def real_to_complex_merged(self, solall, M):
        if use_parallel:
            from mpi4py import MPI
            myid = MPI.COMM_WORLD.rank

            offset = M.RowOffsets().ToList()
            of = [np.sum(MPI.COMM_WORLD.allgather(np.int32(o)))
                  for o in offset]
            if myid != 0:
                return

        else:
            offset = M.RowOffsets()
            of = offset.ToList()
        dprint1(of)
        rows = M.NumRowBlocks()
        s = solall.shape
        i = 0
        pt = 0
        result = np.zeros((s[0] // 2, s[1]), dtype='complex')
        for i in range(rows):
            l = of[i + 1] - of[i]
            w = int(l // 2)
            result[pt:pt + w, :] = (solall[of[i]:of[i] + w, :]
                                    + 1j * solall[(of[i] + w):of[i + 1], :])
            pt = pt + w
        return result

    def allocate_solver(self, is_complex=False, engine=None):
        solver = IterativeSolver(self, engine, int(self.maxiter),
                                 self.abstol, self.reltol, int(self.kdim))
        # solver.AllocSolver(datatype)
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

    def SetOperator(self, opr, dist=False, name=None):
        self.Aname = name
        self.A = opr

        from petram.solver.linearsystem_reducer import LinearSystemReducer
        if use_parallel:
            if self.gui.use_ls_reducer:
                self.reducer = LinearSystemReducer(opr, name)
                self.M_reduced = self.make_preconditioner(self.reducer.A,
                                                          name=self.reducer.Aname,
                                                          parallel=True)
                solver = self.make_solver(self.reducer.A,
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

        if self.gui.solver_type.startswith("Nested"):
            solver_type = self.gui.solver_type.split(" ")[-1]
            nested = True
        else:
            solver_type = self.gui.solver_type
            nested = False
        cls = getattr(mfem, solver_type + 'Solver')

        solver = cls(*args)
        if solver_type in ['GMRES', 'FGMRES']:
            solver.SetKDim(kdim)

        if nested:
            inner_solver_type = self.gui.solver_type_in
            if inner_solver_type != "MUMPS":
                cls = getattr(mfem, inner_solver_type + 'Solver')
                inner_solver = cls(*args)
                inner_solver.SetAbsTol(0.0)
                inner_solver.SetRelTol(0.0)
                inner_solver.SetMaxIter(self.gui.maxiter_in)
                inner_solver.SetPrintLevel(self.gui.log_level_in)
                if inner_solver_type in ['GMRES', 'FGMRES']:
                    inner_solver.SetKDim(int(self.gui.kdim_in))
                inner_solver.iterative_mode = False
                inner_solver.SetOperator(A)
                inner_solver.SetPreconditioner(M)
                # return inner_solver
                prc = inner_solver
            else:
                from petram.solver.mumps_model import MUMPSBlockPreconditioner
                prc = MUMPSBlockPreconditioner(A, gui=self.gui[self.gui.mumps_in],
                                               engine=self.engine)

        else:
            prc = M
        solver._prc = prc
        solver.SetPreconditioner(prc)
        solver.SetOperator(A)

        solver.SetAbsTol(atol)
        solver.SetRelTol(rtol)
        solver.SetMaxIter(maxiter)
        solver.SetPrintLevel(self.gui.log_level)

        return solver

    def make_preconditioner(self, A, name=None, parallel=False):
        name = self.Aname if name is None else name

        if self.gui.adv_mode:
            expr = self.gui.adv_prc
            gen = eval(expr, self.gui._global_ns)
            gen.set_param(A, name, self.engine, self.gui)
            M = gen()

        else:
            prcs_gui = dict(self.gui.preconditioners)
            #assert not self.gui.parent.is_complex(), "can not solve complex"
            if self.gui.parent.is_converted_from_complex() and not self.gui.merge_real_imag:
                name = sum([[n, n] for n in name], [])

            import petram.helper.preconditioners as prcs

            g = prcs.DiagonalPrcGen(
                opr=A, engine=self.engine, gui=self.gui, name=name)
            M = g()

            pc_block = {}

            for k, n in enumerate(name):
                prctxt = prcs_gui[n][1] if parallel else prcs_gui[n][0]
                if prctxt == "None":
                    continue
                if prctxt.find("(") == -1:
                    prctxt = prctxt + "()"
                prcargs = "(".join(prctxt.split("(")[-1:])

                nn = prctxt.split("(")[0]

                if not n in pc_block:
                    # make a new one
                    dprint1(nn)
                    try:
                        blkgen = getattr(prcs, nn)
                    except BaseException:
                        if nn in self.gui._global_ns:
                            blkgen = self.gui._global_ns[nn]
                        else:
                            raise

                    blkgen.set_param(g, n)
                    blk = eval("blkgen(" + prcargs)

                    M.SetDiagonalBlock(k, blk)
                    pc_block[n] = blk
                else:
                    M.SetDiagonalBlock(k, pc_block[n])
        return M

    def write_mat(self, A, b, x, suffix=""):
        def get_block(Op, i, j):
            try:
                return Op._linked_op[(i, j)]
            except KeyError:
                return None

        offset = A.RowOffsets()
        rows = A.NumRowBlocks()
        cols = A.NumColBlocks()

        for i in range(cols):
            for j in range(rows):
                m = get_block(A, i, j)
                if m is None:
                    continue
                m.Print('matrix_' + str(i) + '_' + str(j))
        for i, bb in enumerate(b):
            for j in range(rows):
                v = bb.GetBlock(j)
                v.Print('rhs_' + str(i) + '_' + str(j) + suffix)
                #np.save('rhs_' + str(i) + '_' + str(j) + suffix, v.GetDataArray())
        if x is not None:
            for j in range(rows):
                xx = x.GetBlock(j)
                xx.Print('x_' + str(i) + '_' + str(j) + suffix)

    @flush_stdout
    def call_mult(self, solver, bb, xx):
        #print(np.sum(bb.GetDataArray()), np.sum(xx.GetDataArray()))
        solver.Mult(bb, xx)
        max_iter = solver.GetNumIterations()
        tol = solver.GetFinalNorm()

        dprint1("convergence check (max_iter, tol) ", max_iter, " ", tol)
        if self.gui.assert_no_convergence:
            if not solver.GetConverged():
                self.gui.set_solve_error(
                    (True, "No Convergence: " + self.gui.name()))
                assert False, "No convergence"

    def solve_parallel(self, A, b, x=None):
        if self.gui.write_mat:
            self. write_mat(A, b, x, "." + smyid)

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
                    self.gui.set_solve_error(
                        (True, "No Convergence: " + self.gui.name()))
                    assert False, "No convergence"
            else:
                self.call_mult(self.solver, bb, xx)

            s = []
            for i in range(offset.Size() - 1):
                v = xx.GetBlock(i).GetDataArray()
                if self.gui.merge_real_imag:
                    w = int(len(v) // 2)
                    vv1 = gather_vector(v[:w])
                    vv2 = gather_vector(v[w:])
                    vv = np.hstack((vv1, vv2))
                else:
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

        #M = self.M
        solver = self.solver

        sol = []

        for bb in b:
            if x is None:
                xx = mfem.Vector(bb.Size())
                xx.Assign(0.0)
            else:
                xx = x
                # for j in range(cols):
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
