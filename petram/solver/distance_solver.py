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
        return v

    def panel1_param(self):
        return [  # ["Initial value setting",   self.init_setting,  0, {},],
            ["physics model",   self.phys_model,  0, {}, ],
            [None, self.init_only,  3, {"text": "initialize solution only"}],
            [None,
             self.clear_wdir,  3, {"text": "clear working directory"}],
            [None,
             self.save_parmesh,  3, {"text": "save parallel mesh"}],
            [None,
             self.use_profiler,  3, {"text": "use profiler"}], ]

    def get_panel1_value(self):
        return (  # self.init_setting,
            self.phys_model,
            self.init_only,
            self.clear_wdir,
            self.save_parmesh,
            self.use_profiler,)

    def import_panel1_value(self, v):
        #self.init_setting = str(v[0])
        self.phys_model = str(v[0])
        self.init_only = v[1]
        self.clear_wdir = v[2]
        self.save_parmesh = v[3]
        self.use_profiler = v[4]

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
        return init

    @debug.use_profiler
    def run(self, engine, is_first=True, return_instance=False):
        dprint1("Entering run (is_first=", is_first, ")", self.fullpath())
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = DistanceSolverInstance(self, engine)

        instance.set_blk_mask()
        if return_instance:
            return instance

        '''
        instance.configure_probes(self.probe)
        instance.assemble()                    

        if self.init_only:
            engine.sol = engine.assembled_blocks[1][0]
            instance.sol = engine.sol
        else:
            if is_first:

                
            
        '''
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
        return M[0], True

    def compute_rhs(self, M, B, X):
        '''
        M[0] x = B
        '''
        return B

    def assemble(self, inplace=True):
        engine = self.engine
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()

        engine.access_idx = 0
        name = phys_target[0].dep_vars[0]
        ifes = engine.r_ifes(name)
        r_x = engin.r_x[ifes]

        # use get_phys to apply essential to all phys in solvestep
        dprint1("Asembling system matrix",
                [x.name() for x in phys_target],
                [x.name() for x in phys_range])

        engine.run_verify_setting(phys_target, self.gui)
        engine.run_assemble_mat(phys_target, phys_range)
        engine.run_assemble_b(phys_target)
        engine.run_fill_X_block()

        self.engine.run_assemble_blocks(self.compute_A,
                                        self.compute_rhs,
                                        inplace=inplace)
        #A, X, RHS, Ae, B, M, names = blocks
        self.assembled = True

    def assemble_rhs(self):
        engine = self.engine
        phys_target = self.get_phys()
        engine.run_assemble_b(phys_target)
        B = self.engine.run_update_B_blocks()
        self.blocks[4] = B
        self.assembled = True

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
        dx = mfem.dist_solver.AvgElementSize(pmesh)

        filt_gf = mfem.ParGridFunction(pfes_s)

        # run heat solver
        '''
        t_param = 1.0
        ds = mfem.dist_solver.HeatDistanceSolver(t_param * dx * dx)
        ds.mooth_steps = 0
        ds.vis_glvis = False

        
        ls_coeff = mfem.GridFunctionCoefficient(r_x)        
        filt_gf.ProjectCoefficient(ls_coeff)
        
        ls_filt_coeff = mfem.GridFunctionCoefficient(filt_gf)
        
        ds.ComputeScalarDistance(ls_filt_coeff, r_x)
        if do_vector:
            ds.ComputeVectorDistance(ls_filt_coeff, dr_x)        

        return
        '''
        '''
        filter = mfem.dist_solver.PDEFilter(pmesh, 10 * dx)
        '''
        # run PLapSolver
        p = 20
        newton_iter = 50
        ds = mfem.dist_solver.PLapDistanceSolver(
            p, newton_iter, rtol=1e-10, atol=1e-14)
        ds.print_level = 1
        ls_coeff = mfem.GridFunctionCoefficient(r_x)
        filt_gf.ProjectCoefficient(ls_coeff)
        ls_filt_coeff = mfem.GridFunctionCoefficient(filt_gf)

        ds.ComputeScalarDistance(ls_filt_coeff, r_x)
        if do_vector:
            ds.ComputeVectorDistance(ls_filt_coeff, dr_x)

        return True

        '''
        elif solver_type == 1:
            p = 10
            newton_iter = 50
            ds = mfem.dist_solver.PLapDistanceSolver(p, newton_iter)
            dist_solver = ds

        else:
             assert False, "Wrong solver option."

        dist_solver.print_level = 1

        # Smooth-out Gibbs oscillations from the input level set. The smoothing
        # parameter here is specified to be mesh dependent with length scale dx.
        filt_gf = mfem.ParGridFunction(pfes_s)
        filter = mfem.dist_solver.PDEFilter(pmesh, 10 * dx)

        ls_coeff = mfem.GridFunctionCoefficient(r_x)
        if problem != 0:
            filter.Filter(ls_coeff, filt_gf)
            t = r_x.GetDataArray()
            t[:] = filt_gf.GetDataArray()
        else:
           filt_gf.ProjectCoefficient(ls_coeff)

        ls_filt_coeff = mfem.GridFunctionCoefficient(filt_gf)
        $dist_solver.ComputeScalarDistance(ls_filt_coeff, r_x)
        print(r_x)
        '''

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
