'''
BlcokPreconditioner Model. 
'''
from petram.solver.mumps_model import MUMPSPreconditioner
from petram.mfem_config import use_parallel
import numpy as np

from petram.debug import flush_stdout
from petram.namespace_mixin import NS_mixin
from .solver_model import LinearSolverModel, LinearSolver

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('BlockSmoother')

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


class BlockSmoother(LinearSolverModel, NS_mixin):
    hide_ns_menu = True
    has_2nd_panel = False
    accept_complex = False
    always_new_panel = False

    def __init__(self, *args, **kwargs):
        LinearSolverModel.__init__(self, *args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)

    @classmethod
    def fancy_menu_name(self):
        return 'BlockPreconditioner'

    @classmethod
    def fancy_tree_name(self):
        return 'BlockPreconditioner'

    def does_linearsolver_choose_linearsystem_type(self):
        return False

    def supported_linear_system_type(self):
        return ["blk_interleave",
                "blk_merged_s",
                "blk_merged", ]

    def real_to_complex(self, solall, M=None):
        raise NotImplementedError(
            "bug. this method sould not be called")


class DiagonalPreconditioner(BlockSmoother):
    @classmethod
    def fancy_menu_name(self):
        return 'DiagonalPreconditioner'

    @classmethod
    def fancy_tree_name(self):
        return 'DiagonalPreconditioner'

    def panel1_param(self):
        import wx
        from petram.pi.widget_smoother import WidgetSmoother

        smp1 = [None, None, 99, {"UI": WidgetSmoother, "span": (1, 2)}]

        return [[None, [False, [''], [[], ]], 27, [{'text': 'advanced mode'},
                                                   {'elp': [
                                                       ['preconditioner', '', 0, None], ]},
                                                   {'elp': [smp1, ]}], ], ]

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

        value = ((self.adv_mode, [self.adv_prc, ], [self.preconditioners, ]),)

        return value

    def import_panel1_value(self, v):
        self.preconditioners = v[0][2][0]
        self.adv_mode = v[0][0]
        self.adv_prc = v[0][1][0]

    def attribute_set(self, v):
        v = super(DiagonalPreconditioner, self).attribute_set(v)
        v['preconditioner'] = ''
        v['preconditioners'] = []
        v['adv_mode'] = False
        v['adv_prc'] = ''
        return v

    def get_possible_child(self):
        from petram.solver.mumps_model import MUMPSPreconditionerModel
        from petram.solver.krylov import KrylovModel, KrylovSmoother
        return KrylovSmoother, MUMPSPreconditionerModel

    def get_possible_child_menu(self):
        from petram.solver.mumps_model import MUMPSPreconditionerModel
        from petram.solver.krylov import KrylovModel, KrylovSmoother
        choice = [("Blocks", KrylovSmoother),
                  ("!", MUMPSPreconditionerModel), ]
        return choice

    def prepare_solver(self, opr, engine):
        def get_operator_block(r, c):
            # if linked_op exists (= op is set from python).
            # try to get it
            # print(self.opr._linked_op)
            if hasattr(opr, "_linked_op"):
                try:
                    return opr._linked_op[(r, c)]
                except KeyError:
                    return None
            else:
                blk = opr.GetBlock(r, c)
                if use_parallel:
                    return mfem.Opr2HypreParMat(blk)
                else:
                    return mfem.Opr2SparseMat(blk)

        names = engine.masked_dep_var_names()

        if self.adv_mode:
            expr = self.adv_prc
            gen = eval(expr, self._global_ns)
            gen.set_param(A, names, engine, self)
            M = gen()

        else:
            prcs_gui = dict(self.preconditioners)

            ls_type = self.get_solve_root().get_linearsystem_type_from_modeltree()
            phys_real = self.get_solve_root().is_allphys_real()

            if ls_type == 'blk_interleave' and not phys_real:
                names = sum([[n, n] for n in names], [])

            import petram.helper.preconditioners as prcs

            g = prcs.DiagonalPrcGen(
                opr=opr, engine=engine, gui=self, name=names)
            M = g()

            pc_block = {}

            for k, n in enumerate(names):
                prctxt = prcs_gui[n][1] if use_parallel else prcs_gui[n][0]
                if prctxt == "None":
                    continue
                if prctxt.find("(") == -1:
                    prctxt = prctxt + "()"
                prcargs = "(".join(prctxt.split("(")[-1:])

                nn = prctxt.split("(")[0]

                if not n in pc_block:
                    # make a new one
                    if not nn in self:
                        try:
                            blkgen = getattr(prcs, nn)
                        except BaseException:
                            if nn in self._global_ns:
                                blkgen = self._global_ns[nn]
                            else:
                                raise

                        blkgen.set_param(g, n)
                        blk = eval("blkgen(" + prcargs)
                    else:
                        rr = engine.masked_dep_var_offset(n)
                        cc = engine.masked_r_dep_var_offset(n)
                        A = get_operator_block(rr, cc)
                        blk = self[nn].prepare_solver(A, engine)
                    M.SetDiagonalBlock(k, blk)
                    pc_block[n] = blk
                else:
                    M.SetDiagonalBlock(k, pc_block[n])
        return M


class DiagonalSmoother(DiagonalPreconditioner):
    @classmethod
    def fancy_menu_name(self):
        return 'DiagonalSmoother'

    @classmethod
    def fancy_tree_name(self):
        return 'DiagonalSmoother'
