from petram.solver.solver_model import SolverInstance
import os
import numpy as np

from petram.model import Model
from petram.solver.solver_model import Solver
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('DistanceSolver')
rprint = debug.regular_print('DistanceSolver')


class DistanceSolver(Solver):
    can_delete = True
    has_2nd_panel = False

    def attribute_set(self, v):
        super(DistanceSolver, self).attribute_set(v)
        v["solver_type"] = "heat flow"
        v["p_lap_p"] = 20
        v["p_lap_iter"] = 50
        v["heat_flow_t"] = 1.
        v["heat_flow_diffuse_iter"] = 1
        v["log_level"] = 1
        return v

    def panel1_param(self):
        return [  # ["Initial value setting",   self.init_setting,  0, {},],
            ["physics model", self.phys_model, 0, {}, ],
            ["method", self.solver_type, 1, {
                "values": ["heat flow", "p-Laplacian"]}],
            ["diffusion t (heat flow)", self.heat_flow_t, 300, {}, ],
            ["diffusion iter. (heat flow)",
             self.heat_flow_diffuse_iter, 400, {}, ],
            ["max power (p_Laplacian)", self.p_lap_p, 400, {}, ],
            ["newton iter(p_Laplacian)", self.p_lap_iter, 400, {}, ],
            ["log_level(0-1)", self.log_level, 400, {}],
            [None, self.init_only, 3, {"text": "initialize solution only"}],
            [None,
             self.clear_wdir, 3, {"text": "clear working directory"}],
            [None,
             self.save_parmesh, 3, {"text": "save parallel mesh"}],
            [None,
             self.use_profiler, 3, {"text": "use profiler"}], ]

    def get_panel1_value(self):
        return (  # self.init_setting,
            self.phys_model,
            self.solver_type,
            self.heat_flow_t,
            self.heat_flow_diffuse_iter,
            self.p_lap_p,
            self.p_lap_iter,
            self.log_level,
            self.init_only,
            self.clear_wdir,
            self.save_parmesh,
            self.use_profiler,)

    def import_panel1_value(self, v):
        #self.init_setting = str(v[0])
        self.phys_model = str(v[0])
        self.solver_type = str(v[1])
        self.heat_flow_t = float(v[2])
        self.heat_flow_diffuse_iter = int(v[3])
        self.p_lap_p = int(v[4])
        self.p_lap_iter = int(v[5])
        self.log_level = int(v[6])
        self.init_only = v[7]
        self.clear_wdir = v[8]
        self.save_parmesh = v[9]
        self.use_profiler = v[10]

    def get_editor_menus(self):
        return []

    def get_possible_child(self):
        return []

    def allocate_solver_instance(self, engine):
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = DistanceSolverInstance(self, engine)
        return instance

    def get_matrix_weight(self, timestep_config):  # , timestep_weight):
        if timestep_config[0]:
            return [1, 0, 0]
        else:
            return [0, 0, 0]

    def get_custom_init(self):
        from petram.init_model import CustomInitSetting

        phys = self.parent.get_phys()
        init = CustomInitSetting(phys, value=[1.0, ])
        return [init, ]

    @debug.use_profiler
    def run(self, engine, is_first=True, return_instance=False):
        dprint1("Entering run (is_first=", is_first, ")", self.fullpath())
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = DistanceSolverInstance(self, engine)

        instance.set_blk_mask()
        if return_instance:
            return instance

        if self.init_only:
            pass
        else:
            instance.solve()

        instance.save_solution(ksol=0,
                               skip_mesh=False,
                               mesh_only=False,
                               save_parmesh=self.save_parmesh)
        #engine.sol = instance.sol
        # instance.save_probe()
        is_first = False

        dprint1(debug.format_memory_usage())
        return is_first

    def does_solver_choose_linearsystem_type(self):
        return True

    def get_linearsystem_type_from_solvermodel(self):
        return 'blk_interleave'


class DistanceSolverInstance(SolverInstance):
    def __init__(self, gui, engine):
        SolverInstance.__init__(self, gui, engine)
        self.assembled = False
        self.linearsolver = None

    @property
    def blocks(self):
        return self.engine.assembled_blocks

    def set_linearsolver_model(self):
        # use its own solver from MFEM
        pass

    def compute_A(self, M, B, X, mask_M, mask_B):
        '''
        M[0] x = B

        return A and isAnew
        '''
        assert False, "not supposed to come here"
        # return M[0], True

    def compute_rhs(self, M, B, X):
        assert False, "not supposed to come here"

    def solve(self, update_operator=True):
        engine = self.engine
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()

        engine.access_idx = 0
        name0 = phys_target[0].dep_vars[0]
        r_x = engine.r_x[engine.r_ifes(name0)]
        if len(phys_target[0].dep_vars) > 1:
            name1 = phys_target[0].dep_vars[1]
            dr_x = engine.r_x[engine.r_ifes(name1)]
            do_vector = True
        else:
            do_vector = False

        import mfem.par as mfem

        pfes_s = r_x.ParFESpace()
        pmesh = pfes_s.GetParMesh()

        filt_gf = mfem.ParGridFunction(pfes_s)

        # run heat solver
        if self.gui.solver_type == "heat flow":
            t_param = self.gui.heat_flow_t
            dx = mfem.dist_solver.AvgElementSize(pmesh)

            ds = mfem.dist_solver.HeatDistanceSolver(t_param * dx * dx)
            ds.smooth_steps = 0
            ds.vis_glvis = False

            ls_coeff = mfem.GridFunctionCoefficient(r_x)
            filt_gf.ProjectCoefficient(ls_coeff)

            ls_filt_coeff = mfem.GridFunctionCoefficient(filt_gf)

            ds.ComputeScalarDistance(ls_filt_coeff, r_x)
            if do_vector:
                ds.ComputeVectorDistance(ls_filt_coeff, dr_x)
        else:
            p = self.gui.p_lap_p
            newton_iter = self.gui.p_lap_iter
            ds = mfem.dist_solver.PLapDistanceSolver(p, newton_iter)

            ls_coeff = mfem.GridFunctionCoefficient(r_x)
            filt_gf.ProjectCoefficient(ls_coeff)
            ls_filt_coeff = mfem.GridFunctionCoefficient(filt_gf)

        # To DO FIX THIS
        #print(ds.print_level)
        #print(dir(ds.print_level))#.SetPrintLevel(self.gui.log_level)
        #ds.print_level.FirstAndLast().Summary()
        ds.ComputeScalarDistance(ls_filt_coeff, r_x)
        if do_vector:
            ds.ComputeVectorDistance(ls_filt_coeff, dr_x)

        return True
        '''
        filter = mfem.dist_solver.PDEFilter(pmesh, 10 * dx)
        '''
        # run PLapSolver

    def save_solution(self, ksol=0, skip_mesh=False,
                      mesh_only=False, save_parmesh=False):

        engine = self.engine
        phys_target = self.get_phys()

        if mesh_only:
            engine.save_sol_to_file(phys_target,
                                    mesh_only=True,
                                    save_parmesh=save_parmesh)
        else:
            engine.save_sol_to_file(phys_target,
                                    skip_mesh=skip_mesh,
                                    mesh_only=False,
                                    save_parmesh=save_parmesh)
            engine.save_extra_to_file(None)
        #engine.is_initialzied = False
