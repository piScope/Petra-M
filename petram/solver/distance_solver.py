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
        return [#["Initial value setting",   self.init_setting,  0, {},],
                ["physics model",   self.phys_model,  0, {},],
                [None, self.init_only,  3, {"text":"initialize solution only"}], 
                [None,
                 self.clear_wdir,  3, {"text":"clear working directory"}],
                [None,
                 self.save_parmesh,  3, {"text":"save parallel mesh"}],
                [None,
                 self.use_profiler,  3, {"text":"use profiler"}],]

    def get_panel1_value(self):
        return (#self.init_setting,
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
    
    def get_matrix_weight(self, timestep_config):#, timestep_weight):
        if timestep_config[0]:
            return [1, 0, 0]
        else:
            return [0, 0, 0]
    
    @debug.use_profiler
    def run(self, engine, is_first = True, return_instance=False):
        dprint1("Entering run (is_first=", is_first, ")", self.fullpath())
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = DistanceSolverInstance(self, engine)

        instance.set_blk_mask()
        if return_instance: return instance

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
        #instance.save_probe()
        is_first=False
        
        dprint1(debug.format_memory_usage())
        return is_first


from petram.solver.solver_model import SolverInstance

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
        phys_range  = self.get_phys_range()

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

    def solve(self, update_operator = True):
        engine = self.engine
        phys_target = self.get_phys()
        phys_range  = self.get_phys_range()

        engine.access_idx = 0
        name = phys_target[0].dep_vars[0]
        ifes = engine.r_ifes(name)
        r_x = engine.r_x[ifes]

        import mfem.par as mfem
        
        pfes_s = r_x.ParFESpace()
        pmesh = pfes_s.GetParMesh()
        dx = mfem.dist_solver.AvgElementSize(pmesh)        


        problem = 0
        solver_type = 0
        t_param = 1.0
        if solver_type == 0:
            ds = mfem.dist_solver.HeatDistanceSolver(t_param * dx * dx)
            if problem == 0:
                 ds.transform = False
            ds.mooth_steps = 0
            ds.vis_glvis = False
            dist_solver = ds
        elif solver_type == 1:
            p = 10
            newton_iter = 50
            ds = mfem.dist_solver.PLapDistanceSolver(p, newton_iter)
            dist_solver = ds
        else:
             assert False, "Wrong solver option."

        dist_solver.print_level = 1
                        
        distance_s = r_x

        # Smooth-out Gibbs oscillations from the input level set. The smoothing
        # parameter here is specified to be mesh dependent with length scale dx.
        filt_gf = mfem.ParGridFunction(pfes_s)
        filter = mfem.dist_solver.PDEFilter(pmesh, 1.0 * dx)

        ls_coeff = mfem.GridFunctionCoefficient(r_x)
        #if problem != 0:
        filter.Filter(ls_coeff, filt_gf)
        #else:
        #    filt_gf.ProjectCoefficient(ls_coeff)

        ls_filt_coeff = mfem.GridFunctionCoefficient(filt_gf)
        dist_solver.ComputeScalarDistance(ls_filt_coeff, distance_s)
        print(r_x)
        return
        

        #if not self.assembled:
        #    assert False, "assmeble must have been called"
            
        A, X, RHS, Ae, B, M, depvars = self.blocks
        mask = self.blk_mask
        engine.copy_block_mask(mask)        

        depvars = [x for i, x in enumerate(depvars) if mask[0][i]]

        print(A, X, RHS, Ae, B, M)
        '''
        if self.linearsolver is None:
            linearsolver = self.allocate_linearsolver(self.gui.is_complex(), self. engine)
            self.linearsolver = linearsolver
        else:
            linearsolver = self.linearsolver

        if update_operator:            
            linearsolver.SetOperator(AA,
                                 dist = engine.is_matrix_distributed,
                                 name = depvars)
        '''
        XX = engine.finalize_x(X[0], RHS, mask, not self.phys_real,
                                   format = self.ls_type)
        solall = linearsolver.Mult(BB, x=XX, case_base=0)
        
        #linearsolver.SetOperator(AA, dist = engine.is_matrix_distributed)
        #solall = linearsolver.Mult(BB, case_base=0)
            
        if not self.phys_real and self.gui.assemble_real:
            solall = self.linearsolver_model.real_to_complex(solall, AA)

        A.reformat_central_mat(solall, 0, X[0], mask)
        self.sol = X[0]

        # store probe signal (use t=0.0 in std_solver)
        for p in self.probe:
            p.append_sol(X[0])

        return True

    def save_solution(self, ksol = 0, skip_mesh = False, 
                      mesh_only = False, save_parmesh=False):

        engine = self.engine
        phys_target = self.get_phys()

        if mesh_only:
            engine.save_sol_to_file(phys_target,
                                     mesh_only = True,
                                     save_parmesh = save_parmesh)
        else:
            engine.save_sol_to_file(phys_target, 
                                skip_mesh = skip_mesh,
                                mesh_only = False,
                                save_parmesh = save_parmesh)
            engine.save_extra_to_file(None)
        #engine.is_initialzied = False
        
    
