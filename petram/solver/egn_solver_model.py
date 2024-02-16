from petram.solver.krylov import KrylovModel
from petram.solver.mumps_model import MUMPSMFEMSolverModel
from petram.solver.strumpack_model import Strumpack
from petram.mfem_config import use_parallel
from petram.solver.mumps_model import MUMPSPreconditioner
import os
import numpy as np

from petram.namespace_mixin import NS_mixin
from petram.solver.solver_model import (LinearSolverModel,
                                        LinearSolver)
from petram.solver.std_solver_model import (StdSolver,
                                            StandardSolver)
from petram.solver.iterative_model import (Iterative,
                                           IterativeSolver)
from petram.solver.solver_model import Solver, SolverInstance

import petram.debug as debug

dprint1, dprint2, dprint3 = debug.init_dprints('EgnSolver')
rprint = debug.regular_print('EgnSolver')


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


class EgnLinearSolver:
    pass


class EgnMUMPS(MUMPSMFEMSolverModel, EgnLinearSolver):
    def __init__(self, *args, **kwargs):
        MUMPSMFEMSolverModel.__init__(self, *args, **kwargs)
        EgnLinearSolver.__init__(self)

    @classmethod
    def fancy_menu_name(self):
        return 'MUMPS'

    @classmethod
    def fancy_tree_name(self):
        return 'MUMPS'

    def get_info_str(self):
        return 'Direct'


class EgnStrumpack(Strumpack, EgnLinearSolver):
    def __init__(self, *args, **kwargs):
        Strumpack.__init__(self, *args, **kwargs)
        EgnLinearSolver.__init__(self)

        self._case_dirs = None

    @classmethod
    def fancy_menu_name(self):
        return 'STRUMPACK'

    @classmethod
    def fancy_tree_name(self):
        return 'STRUMPACK'

    def get_info_str(self):
        return 'Direct'


class EgnIterative(KrylovModel, EgnLinearSolver):

    def __init__(self, *args, **kwargs):
        KrylovModel.__init__(self, *args, **kwargs)
        EgnLinearSolver.__init__(self)

    @classmethod
    def fancy_menu_name(self):
        return 'Kryrov'

    @classmethod
    def fancy_tree_name(self):
        return 'Kryrov'

    def get_info_str(self):
        return 'Iterative'

    def prepare_solver(self, opr, engine):
        solver = self.do_prepare_solver(opr, engine)
        solver.iterative_mode = True

        return solver


class EgnSolver(StdSolver):
    def __init__(self, *args, **kwargs):
        StdSolver.__init__(self, *args, **kwargs)
        self._case_dirs = None

    @classmethod
    def fancy_menu_name(self):
        return 'Eigenmode'

    @classmethod
    def fancy_tree_name(self):
        return 'Eigenmode'

    def attribute_set(self, v):
        super(EgnSolver, self).attribute_set(v)
        v['num_modes'] = 5
        v['merge_real_imag'] = True
        v['use_block_symmetric'] = False
        v['assemble_real'] = True

        return v

    def panel1_param(self):
        panels = super(EgnSolver, self).panel1_param()
        panels = [panels[0], ["num  modes", 10, 400, {}], ] + panels[1:]

        mm = [[None, self.use_block_symmetric, 3,
               {"text": "block symmetric format"}], ]

        p2 = [[None, (self.merge_real_imag, (self.use_block_symmetric,)),
               27, ({"text": "Use ComplexOperator"}, {"elp": mm},)], ]
        panels.extend(p2)

        return panels

    def get_panel1_value(self):
        value = list(super(EgnSolver, self).get_panel1_value())
        value = [value[0], self.num_modes] + value[1:]
        value.append((self.merge_real_imag, [self.use_block_symmetric, ]))
        return value

    def import_panel1_value(self, v):
        vv = v[:-1]
        vvv = [vv[0]] + vv[2:]
        super(EgnSolver, self).import_panel1_value(vvv)
        self.num_modes = vv[1]
        self.merge_real_imag = bool(v[-1][0])
        self.use_block_symmetric = bool(v[-1][1][0])

    def allocate_solver_instance(self, engine):
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = EgnInstance(self, engine)
        return instance

    def does_solver_choose_linearsystem_type(self):
        return True

    def get_linearsystem_type_from_solvermodel(self):
        assemble_real = self.assemble_real
        phys_real = self.get_solve_root().is_allphys_real()

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

    def verify_setting(self):
        '''
        check if solver is given
        '''
        for x in self.iter_enabled():
            if isinstance(x, EigenValueSolver):
                isvalid = True
                txt = ''
                txt_long = ''
                break
        else:
            isValid = False
            txt = 'No EigenValue solver'
            txt_long = 'Model must define one active eigenvalue solver'
        return isvalid, txt, txt_long

    def get_matrix_weight(self, timestep_config):
        wt = [1 if x else 0 for x in timestep_config]
        return wt

    def get_possible_child(self):
        return (HypreLOBPCG,
                HypreAME,
                EgnMUMPS,
                EgnStrumpack,
                EgnIterative)

    def get_possible_child_menu(self):
        choice = [("EigenSolver",  HypreLOBPCG),
                  ("!", HypreAME), ]
        return choice

    def go_case_dir(self, engine, ksol):
        '''
        make case directory and create symlinks
        '''
        od = os.getcwd()

        nsfiles = [n for n in os.listdir() if n.endswith('_ns.py')
                   or n.endswith('_ns.dat')]

        path = os.path.join(od, 'case_' + str(ksol))
        if not os.path.exists(path):
            engine.mkdir(path)
        os.chdir(path)
        engine.cleancwd()

        files = ['model.pmfm'] + nsfiles
        for n in files:
            engine.symlink(os.path.join('../', n), n)

        if self._case_dirs is None:
            self._case_dirs = []
        self._case_dirs.append(path)

        return od

    def call_save_solutions(self, instance):
        pass

    @debug.use_profiler
    def run(self, engine, is_first=True, return_instance=False):
        dprint1("Entering EigenSolver: ", self.fullpath())
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = self.allocate_solver_instance(engine)
        instance.set_blk_mask()
        if return_instance:
            return instance

        if self.init_only:
            engine.sol = engine.assembled_blocks[1][0]
            instance.sol = engine.sol
        else:
            if is_first:
                instance.assemble()
                is_first = False
            instance.solve()

        from petram.sol.probe import Probe
        evprobe = Probe("EigenValue_"+self.name(), -1)

        for ksol in range(self.num_modes):
            instance.configure_probes("")

            if ksol == 0:
                instance.save_solution(mesh_only=True,
                                       save_parmesh=self.save_parmesh)
                save_mesh_linkdir = os.getcwd()

            eigenvalue = instance.extract_sol(ksol)
            evprobe.append_value(eigenvalue, t=ksol)

            od = self.go_case_dir(engine, ksol)

            instance.save_solution(ksol=ksol,
                                   skip_mesh=False,
                                   mesh_only=False,
                                   save_parmesh=self.save_parmesh,
                                   save_mesh_linkdir=save_mesh_linkdir)

            engine.sol = instance.sol
            instance.save_probe()

            os.chdir(od)

        evprobe.write_file()

        dprint1(debug.format_memory_usage())
        return is_first


class EigenValueSolver(LinearSolverModel, NS_mixin):
    hide_ns_menu = True
    has_2nd_panel = False
    accept_complex = False
    always_new_panel = False

    def __init__(self, *args, **kwargs):
        LinearSolverModel.__init__(self, *args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)

    def panel1_param(self):
        return [["log_level", -1, 400, {}],
                ["max  iter.", 200, 400, {}],
                ["rel tol.", 1e-7, 300, {}],
                ["abs tol.", 1e-7, 300, {}],
                [None, self.write_mat, 3, {"text": "write matrix"}],
                [None, self.assert_no_convergence, 3,
                 {"text": "check converegence"}], ]

    def get_panel1_value(self):
        single1 = (int(self.log_level),
                   int(self.maxiter),
                   self.reltol,
                   self.abstol,
                   self.write_mat, self.assert_no_convergence,)

    def import_panel1_value(self, v):
        self.log_level = int(v[0])
        self.maxiter = int(v[1])
        self.reltol = v[2]
        self.abstol = v[3]
        self.write_mat = bool(v[4])
        self.assert_no_convergence = bool(v[5])

    def attribute_set(self, v):
        v = super(EigenValueSolver, self).attribute_set(v)
        v['solver_type'] = ''
        v['log_level'] = 0
        v['maxiter'] = 200
        v['reltol'] = 1e-5
        v['abstol'] = 1e-7
        v['printit'] = 1
        v['write_mat'] = False
        v['assert_no_convergence'] = True
        return v

    def get_info_str(self):
        return 'EigenSolver'

    def get_possible_child(self):
        return EgnMUMPS, EgnStrumpack, EgnIterative

    def get_possible_child_menu(self):
        choice = [("LinearSolver", EgnMUMPS),
                  ("", EgnStrumpack),
                  ("!", EgnIterative)]
        return choice

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

    def does_linearsolver_choose_linearsystem_type(self):
        return False

    def supported_linear_system_type(self):
        return ["blk_interleave",
                "blk_merged_s",
                "blk_merged", ]


class HypreAME(EigenValueSolver):
    @classmethod
    def fancy_menu_name(self):
        return 'AME'

    @classmethod
    def fancy_tree_name(self):
        return 'AME'

    def attribute_set(self, v):
        v = super(HypreAME, self).attribute_set(v)
        v['solver_type'] = 'HypreAME'
        return v

    def allocate_solver(self):
        return mfem.HypreAME(MPI.COMM_WORLD)

    def configure_solver(self, solver):
        solver.SetMaxIter(self.maxiter)
        solver.SetTol(self.abstol)
        solver.SetRelTol(self.reltol)
        solver.SetPrintLevel(self.log_level)


class HypreLOBPCG(EigenValueSolver):
    def attribute_set(self, v):
        v = super(HypreLOBPCG, self).attribute_set(v)
        v['solver_type'] = 'HypreLOBPCG'
        return v

    @classmethod
    def fancy_menu_name(self):
        return 'LOBPCG'

    @classmethod
    def fancy_tree_name(self):
        return 'LOBPCG'

    def allocate_solver(self):
        return mfem.HypreLOBPCG(MPI.COMM_WORLD)

    def configure_solver(self, solver):
        solver.SetMaxIter(self.maxiter)
        solver.SetTol(self.abstol)
        solver.SetRelTol(self.reltol)
        solver.SetPrintLevel(self.log_level)
        solver.SetRandomSeed(775)


class EgnInstance(SolverInstance):
    def __init__(self, gui, engine):
        SolverInstance.__init__(self, gui, engine)
        self.assembled = False
        self.linearsolver = None
        self.solver = None
        self._Amat = None

    @property
    def blocks(self):
        return self.engine.assembled_blocks

    def compute_A(self, M, B, X, mask_M, mask_B):
        '''
        M[0] x = B

        return A and isAnew
        '''
        return M[0], np.any(mask_M[0])

    def compute_rhs(self, M, B, X):
        '''
        M[0] x = B
        '''
        return B

    def assemble(self, inplace=True, update=False):
        engine = self.engine
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()

        # use get_phys to apply essential to all phys in solvestep
        dprint1("Asembling system matrix",
                [x.name() for x in phys_target],
                [x.name() for x in phys_range])

        if not update:
            engine.run_verify_setting(phys_target, self.gui)
        else:
            engine.set_update_flag('TimeDependent')

        M_updated = engine.run_assemble_mat(
            phys_target, phys_range, update=update)
        B_updated = engine.run_assemble_b(phys_target, update=update)

        engine.run_apply_essential(phys_target, phys_range, update=update)
        engine.run_fill_X_block(update=update)

        _blocks, M_changed = self.engine.run_assemble_blocks_egn(inplace=inplace,
                                                                 update=update,)
        # AA, BB, X, B, M, names = blocks
        self.assembled = True
        return M_changed

    def get_eigensolver_model(self):
        for x in self.gui.iter_enabled():
            if isinstance(x, EigenValueSolver):
                return x
        return None

    def allocate_solver(self, Amat, Bmat):
        gui = self.get_eigensolver_model()
        if gui is not None:
            solver = gui.allocate_solver()
        else:
            assert False, "EigenValue solver is not defined"

        solver.SetOperator(Amat)
        solver.SetMassMatrix(Bmat)
        solver.SetNumModes(self.gui.num_modes)
        gui.configure_solver(solver)

        for x in gui.iter_enabled():
            if isinstance(x, EgnLinearSolver):
                prec = x.prepare_solver(Amat, self.engine)
                solver.SetPreconditioner(prec)
                solver.SetPrecondUsageMode(1)
                solver._prec = prec
                break

        return solver

    def solve(self):
        engine = self.engine

        # if not self.assembled:
        #    assert False, "assmeble must have been called"

        AA, BB, X, B, M, depvars = self.blocks
        mask = self.blk_mask
        engine.copy_block_mask(mask)

        depvars = [x for i, x in enumerate(depvars) if mask[0][i]]


        if not use_parallel:
            from scipy.sparse.linalg import eigs, eigsh
            from scipy.sparse import coo_matrix
            evals, evecs = eigs(coo_matrix(AA[0, 0]),
                                k=self.gui.num_modes,
                                M=coo_matrix(BB[0, 0]),
                                #sigma=-499000,
                                sigma=15000.0,
                                #sigma=0.0,
                                which='LR')
            # evals, evecs = eigsh(coo_matrix(AA[0,0]),
            #                     k=self.gui.num_modes,
            #                     M=coo_matrix(BB[0,0]),
            #                     sigma=-497000,
            #                     which='LA')

            print(evals)
            print(np.max(np.abs(evecs.imag)))
            self._evals = evals
            self._evecs = evecs
            return True

        Amat = engine.finalize_matrix(AA, mask, not self.phys_real,
                                      format=self.ls_type)
        Bmat = engine.finalize_matrix(BB, mask, not self.phys_real,
                                      format=self.ls_type)

        print(type(Amat))
        print(type(Bmat))

        self.solver = self.allocate_solver(Amat, Bmat)

        egn_solver_model = self.get_eigensolver_model()
        if egn_solver_model.write_mat:
            from petram.solver.solver_utils import write_blockoperator
            write_blockoperator(A=Amat, suffix=smyid, mat_base="Amat_")
            write_blockoperator(A=Bmat, suffix=smyid, mat_base="Bmat_")

        self.solver.Solve()
        eigenvalues = mfem.doubleArray()
        self.solver.GetEigenvalues(eigenvalues)

        self._eigenvalues = eigenvalues.ToList()
        dprint1("Eigen values : " + ', '.join([str(x) for x in self._eigenvalues]))

        self._Amat = Amat
        return True

    def extract_sol(self, ksol):
        if use_parallel:
            return self.extract_sol_par(ksol)
        else:
            return self.extract_sol_ser(ksol)

    def extract_sol_ser(self, ksol):
        from petram.helper.block_matrix import ScipyCoo
        solall = np.atleast_2d(self._evecs[:, ksol]).transpose().real

        AA, _BB, X, _B, _M, _depvars = self.blocks

        do_real_to_complex = self.gui.is_converted_from_complex()

        AA.reformat_central_mat(solall, 0, X[0], self.blk_mask)
        self.sol = X[0]
        return self._evals[ksol]

    def extract_sol_par(self, ksol):
        eigenvalue = self._eigenvalues[ksol]

        vecs = self.solver.GetEigenvector(ksol).GetDataArray()
        solall = np.transpose(np.vstack([vecs]))

        is_sol_central = (num_proc == 1)
        do_real_to_complex = self.gui.is_converted_from_complex()
        AA, _BB, X, _B, _M, _depvars = self.blocks

        if is_sol_central:
            if do_real_to_complex:
                egn_solver_model = self.get_eigensolver_model()
                solall = egn_solver_model.real_to_complex(solall, self._Amat)
            AA.reformat_central_mat(solall, 0, X[0], self.blk_mask)
        else:
            if not self.phys_real and self.gui.assemble_real:
                assert False, "this operation is not permitted"
            AA.reformat_distributed_mat(solall, 0, X[0], self.blk_mask)

        self.sol = X[0]

        return eigenvalue
