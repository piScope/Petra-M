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

from petram.solver.mumps_model import MUMPSPreconditioner
from petram.mfem_config import use_parallel

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

from petram.solver.strumpack_model import Strumpack
from petram.solver.mumps_model import MUMPSMFEMSolverModel
from petram.solver.krylov import KrylovModel
     
class EgnLinearSolver:
    grid_level = 0

    def __init__(self):
        self.is_preconditioner = True
    
class EgnMUMPS(MUMPSMFEMSolverModel, EgnLinearSolver):
    def __init__(self, *args, **kwargs):
        MUMPSMFEMSolverModel.__init__(self, *args, **kwargs)
        EgnLinearSolver.__init__(self)
        
    @classmethod
    def fancy_menu_name(self):
        return 'MUMPS'

    @classmethod
    def fancy_tree_name(self):
        return 'Direct'

    def get_info_str(self):
        return 'MUMPS'


class EgnStrumpack(Strumpack, EgnLinearSolver):
    def __init__(self, *args, **kwargs):
        Strumpack.__init__(self, *args, **kwargs)
        EgnLinearSolver.__init__(self)
    
    @classmethod
    def fancy_menu_name(self):
        return 'STRUMPACK'

    @classmethod
    def fancy_tree_name(self):
        return 'Direct'

    def get_info_str(self):
        return 'STRUMPACK'

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

        if self.is_preconditioner:
            solver.iterative_mode = False
        else:
            solver.iterative_mode = True

        return solver
    
class EgnSolver(StdSolver):
    @classmethod
    def fancy_menu_name(self):
        return 'Eigenmode'

    @classmethod
    def fancy_tree_name(self):
        return 'Eigenmode'

    def attribute_set(self, v):
        super(EgnSolver, self).attribute_set(v)
        v['merge_real_imag'] = True
        v['use_block_symmetric'] = False
        v['assemble_real'] = True

        return v

    def panel1_param(self):
        panels = super(EgnSolver, self).panel1_param()

        mm = [[None, self.use_block_symmetric, 3,
               {"text": "block symmetric format"}], ]

        p2 = [[None, (self.merge_real_imag, (self.use_block_symmetric,)),
               27, ({"text": "Use ComplexOperator"}, {"elp": mm},)], ]
        panels.extend(p2)

        return panels

    def get_panel1_value(self):
        value = list(super(EgnSolver, self).get_panel1_value())
        value.append((self.merge_real_imag, [self.use_block_symmetric, ]))
        return value

    def import_panel1_value(self, v):
        super(EgnSolver, self).import_panel1_value(v[:-1])
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
        has to have one coarse solver
        '''
        isvalid = True
        txt = ''
        txt_long = ''
        return isvalid, txt, txt_long
    
    def get_possible_child(self):
        return (HypreLOBPCG,
                HypreAME,
                EgnMUMPS,
                EgnStrumpack,
                EgnIterative)

    def get_possible_child_menu(self):
        choice = [("EigenSolver",  HypreLOBPCG),
                  ("!", HypreAME),]
        return choice    

    @debug.use_profiler
    def run(self, engine, is_first=True, return_instance=False):
        if not use_parallel:
            assert False, "Eigen solver works only with MPI"
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

        instance.save_solution(ksol=0,
                               skip_mesh=False,
                               mesh_only=False,
                               save_parmesh=self.save_parmesh)
        engine.sol = instance.sol

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
                ["rel. tol", 1e-7, 300, {}],
                ["abs. tol.", 1e-7, 300, {}],
                [None, self.write_mat, 3, {"text": "write matrix"}],
                [None, self.assert_no_convergence, 3,
                 {"text": "check converegence"}], ]
    
    def get_panel1_value(self):    
        single1 = (int(self.log_level), int(self.maxiter),
                   self.reltol, self.abstol,
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
        v['reltol'] = 1e-7
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
    def attribute_set(self, v):
        v = super(HypreAME, self).attribute_set(v)
        v['solver_type'] = 'HypreAME'
        return v
    
    def allocate_solver(self):
        return mfem.HypreAME(MPI.COMM_WORLD)


class HypreLOBPCG(EigenValueSolver):
    def attribute_set(self, v):
        v = super(HypreLOBPCG, self).attribute_set(v)
        v['solver_type'] = 'HypreLOBPCG'
        return v
    
    def allocate_solver(self):
        return mfem.HypreLOBPCG(MPI.COMM_WORLD)


class EgnInstance(SolverInstance):
    def __init__(self, gui, engine):
        SolverInstance.__init__(self, gui, engine)
        self.assembled = False
        self.linearsolver = None

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

        _blocks, M_changed = self.engine.run_assemble_blocks(self.compute_A,
                                                             self.compute_rhs,
                                                             inplace=inplace,
                                                             update=update,)
        # A, X, RHS, Ae, B, M, names = blocks
        self.assembled = True
        return M_changed


    def allocate_solver(self, AA):
        for x in self.gui.iter_enabled():
            if isinstance(x, EigenValueSolver):
                solver = x.allocate_solver()

        lobpcg.SetOperator(AA)                
        #lobpcg.SetNumModes(nev)
        #lobpcg.SetPreconditioner(precond)
        #lobpcg.SetMaxIter(200)
        #lobpcg.SetTol(1e-8)
        #lobpcg.SetPrecondUsageMode(1)
        #lobpcg.SetPrintLevel(1)
        #lobpcg.SetMassMatrix(M)
        amg = mfem.HypreBoomerAMG(AA)
        amg.SetPrintLevel(0)
        precond = amg

    def solve(self):
        engine = self.engine

        # if not self.assembled:
        #    assert False, "assmeble must have been called"

        A, X, RHS, Ae, B, M, depvars = self.blocks
        mask = self.blk_mask
        engine.copy_block_mask(mask)

        depvars = [x for i, x in enumerate(depvars) if mask[0][i]]

        AA = engine.finalize_matrix(A, mask, not self.phys_real,
                                        format=self.ls_type)

        BB = engine.finalize_rhs([RHS], A, X[0], mask, not self.phys_real,
                                 format=self.ls_type)
        
 
        solver = self.allocate_solver(AA)
        eigenvalues = mfem.doubleArray()
        solver.Solve()
        solver.GetEigenvalues(eigenvalues)
        

        print(type(AA))
        print(BB)

