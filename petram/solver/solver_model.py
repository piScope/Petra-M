import numpy as np
from petram.model import Model
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Solver')

'''

    Solver : Model Tree Object for solvers such as TimeDependent Solver

    SolverInstance: an actual solver logic comes here
       SolverInstance : base class for standard solver
       TimeDependentSolverInstance : an actual solver logic comes here


'''
class SolverBase(Model):
    can_rename = True
    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys', self)
        
    def set_solve_error(self, value):
        self.get_solve_root()._solve_error = value
        
    def get_solve_root(self):
        obj = self
        solver_root = self.root()['Solver']

        while (not isinstance(obj, SolveStep) and
               obj is not solver_root):
            obj = obj.parent
        return obj
    
    def eval_text_in_global(self, value, ll = None):
        if not isinstance(value, str): return value
        ll = {} if ll is None else ll 
        gg = self.root()['General']._global_ns.copy()
        return eval(value, gg, ll)

class SolveStep(SolverBase):
    has_2nd_panel = False    
    def attribute_set(self, v):
        v['phys_model']   = ''
        v['init_setting']   = ''
        v['postprocess_sol']   = ''
        v['dwc_name'] = ''
        v['use_dwc_pp']   = False
        v['dwc_pp_arg']   = ''
        v['use_geom_gen'] = False
        v['use_mesh_gen'] = False       
        
        super(SolveStep, self).attribute_set(v)
        return v
    
    def panel1_param(self):
        ret = [["dwc",   self.dwc_name,   0, {}],
               ["args.",   self.dwc_pp_arg,   0, {}]]
        value = [self.dwc_name, self.dwc_pp_arg]
        return [["Initial value setting",   self.init_setting,   0, {},],
                ["Postporcess solution",    self.postprocess_sol,   0, {},],
                ["trial phys.",self.phys_model, 0, {},],
                [None,  self.use_geom_gen,  3, {"text":"run geometry generator"}],
                [None,  self.use_mesh_gen,  3, {"text":"run mesh generator"}],
                [None, [False, value], 27, [{'text':'Use DWC (postprocess)'},
                                              {'elp': ret}]],]

#                ["initialize solution only",
#                 self.init_only,  3, {"text":""}], ]
               

    def get_panel1_value(self):
        return (self.init_setting, self.postprocess_sol, self.phys_model,
                self.use_geom_gen,
                self.use_mesh_gen,
                [self.use_dwc_pp, [self.dwc_name, self.dwc_pp_arg,]])

    def import_panel1_value(self, v):
        self.init_setting    = v[0]
        self.postprocess_sol = v[1]        
        self.phys_model   = v[2]
        self.use_geom_gen = v[3]
        self.use_mesh_gen = v[4]
        if self.use_geom_gen:
            self.use_mesh_gen = True
        self.use_dwc_pp   = v[5][0]
        self.dwc_name   = v[5][1][0]
        self.dwc_pp_arg   = v[5][1][1]        
        
#        self.init_only    = v[2]
        
    def get_possible_child(self):
        #from solver.solinit_model import SolInit
        from petram.solver.std_solver_model import StdSolver
        from petram.solver.solver_controls import DWCCall
        from petram.solver.timedomain_solver_model import TimeDomain
        from petram.solver.set_var import SetVar
  
        try:
            from petram.solver.std_meshadapt_solver_model import StdMeshAdaptSolver
            return [StdSolver, StdMeshAdaptSolver, TimeDomain, DWCCall, SetVar]
        except:
            return [StdSolver, TimeDomain, DWCCall, SetVar]

    
    def get_phys(self):
        #
        #  phys for rhs and rows of M
        #
        phys_root = self.root()['Phys']
        ret = []        
        for k in self.keys():
            if not self[k].enabled: continue
            for x in self[k].get_target_phys():
                if not x in ret: ret.append(x)
            for s in self[k].get_child_solver():
                for x in s.get_target_phys():
                    if not x in ret: ret.append(x)
        return ret
        
    def get_phys_range(self):
        #
        #  phys for X and col of M
        #
        phys_root = self.root()['Phys']
        phys_test = self.get_phys()
        for n in self.phys_model.split(','):
            n = n.strip()
            p =  phys_root.get(n, None)
            if p is None: continue
            if not p in phys_test: phys_test.append(p)
        return phys_test
        '''
        if self.phys_model.strip() ==  '':
            return self.get_phys()
        else:

            names = [n.strip() for n in names if n.strip() != '']        
            return [phys_root[n] for n in names]
        '''
    def get_target_phys(self):
        return []
    
    def get_active_solvers(self):
        return [x for x in self.iter_enabled()]
    
    '''
    def get_num_matrix(self, phys_target):
        num = []
        for k in self.keys():
            mm = self[k]
            if not mm.enabled: continue
            num.append(self.root()['Phys'].get_num_matrix(mm.get_matrix_weight,
                                           phys_target))
        num_matrix = max(num)
        dprint1("number of matrix", num_matrix)            
        return num_matrix
    '''
    def get_num_matrix(self, phys_target):

        num = []
        num_matrix = 0
        active_solves = [self[k] for k in self if self[k].enabled]        
        ###    
        for phys in phys_target:
            for mm in phys.walk():
                if not mm.enabled: continue

                ww = [False]*10
                for s in active_solves:
                    w = s.get_matrix_weight(mm.timestep_config)
                    for i, v in enumerate(w):
                        ww[i] = (ww[i] or v)
                ww = [bool(x) for x in ww]
                
                mm.set_matrix_weight(ww)
                wt = np.array(ww)
                tmp = int(np.max((wt != 0)*(np.arange(len(wt))+1)))
                num_matrix = max(tmp, num_matrix)
            
        dprint1("number of matrix", num_matrix)            
        return num_matrix
    
    def get_matrix_weight(self, timestep_config):
        raise NotImplementedError(
             "you must specify this method in subclass")
    
    def prepare_form_sol_variables(self, engine):
        solvers = self.get_active_solvers()
        
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()
        
        num_matrix= self.get_num_matrix(phys_target)
        
        engine.set_formblocks(phys_target, phys_range, num_matrix)
        
        for p in phys_range:
            engine.run_mesh_extension(p)
            
        engine.run_alloc_sol(phys_range)
#        engine.run_fill_X_block()

    @property
    def solve_error(self):
        if hasattr(self, "_solve_error"):
            return self._solve_error
        return (False, "")

    def get_init_setting(self):
        names = self.init_setting.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['InitialValue'][n] for n in names]
    
    def get_pp_setting(self):
        names = self.postprocess_sol.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['PostProcess'][n] for n in names]

    def init(self, engine):
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()        
        
        inits = self.get_init_setting()
        engine.run_apply_init(phys_range, inits=inits)
        '''
        if len(inits) == 0:
            # in this case alloate all fespace and initialize all
            # to zero
            engine.run_apply_init(phys_range, 0)
        else:
            for init in inits:
                init.run(engine)
        '''        
        # use get_phys to apply essential to all phys in solvestep
        engine.run_apply_essential(phys_target, phys_range)
        engine.run_fill_X_block()

    def call_run_geom_gen(self, engine):
        name = self.root()['General'].geom_gen
        gen = self.root()['Geometry'][name] 
        engine.run_geom_gen(gen)
        
    def call_run_mesh_gen(self, engine):
        name = self.root()['General'].mesh_gen
        gen = self.root()['Mesh'][name] 
        engine.run_mesh_gen(gen)

    def check_and_run_geom_mesh_gens(self, engine):
        flag = False
        if self.use_mesh_gen:
            if self.use_geom_gen:
                self.call_run_geom_gen(engine)
            self.call_run_mesh_gen(engine)
            flag = True
        return flag
        
    def run(self, engine, is_first = True):
        dprint1("!!!!! Entering SolveStep :" + self.name() + " !!!!!")
        solvers = self.get_active_solvers()

        is_new_mesh = self.check_and_run_geom_mesh_gens(engine)
        if is_first or is_new_mesh:        
            engine.preprocess_modeldata()
        
        # initialize and assemble her        
        # in run method..
        #   std solver : make sub block matrix and solve
        #   time-domain solver : do step
        self.prepare_form_sol_variables(engine)
        self.init(engine)
        
        is_first = True
        for solver in solvers:
             is_first = solver.run(engine, is_first=is_first)
             engine.add_FESvariable_to_NS(self.get_phys()) 
             engine.store_x()
             if self.solve_error[0]:
                 dprint1("SolveStep failed " + self.name() + ":"  + self.solve_error[1])
                 break
             
        postprocess = self.get_pp_setting()
        engine.run_postprocess(postprocess, name = self.name())
        
        if self.use_dwc_pp:
            engine.call_dwc(self.get_phys_range(),
                            method="postprocess",
                            callername=self.name(),
                            dwcname=self.dwc_name,
                            args=self.dwc_pp_arg)

        
class Solver(SolverBase):
    def attribute_set(self, v):
        v['clear_wdir'] = False
        v['init_only'] = False   
        v['assemble_real'] = False
        v['save_parmesh'] = False        
        v['phys_model']   = ''
        #v['init_setting']   = ''
        v['use_profiler'] = False
        v['probe'] = ''
        super(Solver, self).attribute_set(v)
        return v
    
    def get_phys(self):
        return self.parent.get_phys()
    
    def get_phys_range(self):
        return self.parent.get_phys_range()

    def get_target_phys(self):
        names = self.phys_model.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['Phys'][n] for n in names]
    
    def get_child_solver(self):
        return []
    
    def is_complex(self):
        phys = self.get_phys()
        is_complex = any([p.is_complex() for p in phys])
        if self.assemble_real: return False        
        #if is_complex: return True
        return is_complex
    
    def is_converted_from_complex(self):
        phys = self.get_phys()
        is_complex = any([p.is_complex() for p in phys])
        if is_complex and self.assemble_real: return True
        return False
        
    def get_init_setting(self):
        raise NotImplementedError(
             "bug should not need this method")
        '''
        names = self.init_setting.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['InitialValue'][n] for n in names]
        '''
    def get_active_solver(self, mm = None):
        for x in self.iter_enabled():
            if isinstance(x, LinearSolverModel):
               return x

    def get_num_matrix(self, phys_target=None):    
        raise NotImplementedError(
             "bug should not need this method")

    '''
    @property
    def has_num_matrix(self):
        return True
    
    def get_num_matrix(self, phys_target=None):
        solver_root = self.get_solve_root()
        num = []
        for k in solver_root.keys():
            mm = solver_root[k]
            if not mm.enabled: continue
            if not mm.has_num_matrix: continue
            num.append(self.root()['Phys'].get_num_matrix(mm.get_matrix_weight,
                                           phys_target))
        return max(num)
    '''

    def get_matrix_weight(self, *args, **kwargs):
        raise NotImplementedError(
             "you must specify this method in subclass")
    
    def run(self, engine, is_first = True):        
        raise NotImplementedError(
             "you must specify this method in subclass")
    

class SolverInstance(object):
    '''
    Solver instance is where the logic of solving linear system
    (time stepping, adaptation, non-linear...) is written.

    It is not a model object. SolverModel will generate this
    instance to do the actual solve step.
    '''
    def __init__(self, gui, engine):
        self.gui = gui
        self.engine = engine
        self.sol = None
        self.linearsolver_model= None # LinearSolverModel
        self.linearsolver = None      # Actual LinearSolver        
        self.probe = []
        self.linearsolver_model = None
        self.phys_real = True
        self.ls_type = ''
        
        if not gui.init_only:
             self.set_linearsolver_model()
        
    def get_phys(self):
        return self.gui.get_phys()

    def get_target_phys(self):
        return self.gui.get_target_phys()
    
    def get_phys_range(self):
        return self.gui.get_phys_range()
    
    @property
    def blocks(self):
        return self.engine.assembled_blocks
        
    def get_init_setting(self):

        names = self.gui.init_setting.split(',')
        names = [n.strip() for n in names if n.strip() != '']
        
        root = self.engine.model        
        return [root['InitialValue'][n] for n in names]

    def set_blk_mask(self):
        # mask defines which FESspace will be solved by
        # a linear solver.
        all_phys = self.get_phys()        
        phys_target = self.get_target_phys()
        mask1 = self.engine.get_block_mask(all_phys, phys_target)

        all_phys = self.get_phys_range()                
        mask2 = self.engine.get_block_mask(all_phys, phys_target, use_range=True)
        
        self.blk_mask = (mask1, mask2)
        self.engine._matrix_blk_mask = self.blk_mask
        
    def save_solution(self, ksol = 0, skip_mesh = False, 
                      mesh_only = False, save_parmesh=False):

        engine = self.engine
        phys_target = self.get_phys()

        if mesh_only:
            engine.save_sol_to_file(phys_target,
                                     mesh_only = True,
                                     save_parmesh = save_parmesh)
        else:
            sol, sol_extra = engine.split_sol_array(self.sol)
            engine.recover_sol(sol)
            extra_data = engine.process_extra(sol_extra)


            engine.save_sol_to_file(phys_target, 
                                skip_mesh = skip_mesh,
                                mesh_only = False,
                                save_parmesh = save_parmesh)
            engine.save_extra_to_file(extra_data)
        #engine.is_initialzied = False
        
        
    def save_probe(self):
        for p in self.probe:
            p.write_file()
        
    def set_linearsolver_model(self):
        solver = self.gui.get_active_solver()
        if solver is None:
             assert False, "Linear solver is not chosen"
        phys_target = self.get_phys()
        
        self.linearsolver_model = solver
        self.phys_real = all([not p.is_complex() for p in phys_target])        
        self.ls_type = solver.linear_system_type(self.gui.assemble_real,
                                                 self.phys_real)

    def configure_probes(self, probe_txt):
        from petram.sol.probe import Probe

        all_phys = self.get_phys()
        txt = [phys.collect_probes() for phys in all_phys]
        txt = [probe_txt]+txt
        probe_txt = ','.join([t for t in txt if len(t) > 0])
                                 
        dprint1("configure probes: "+probe_txt)
        if probe_txt.strip() != '':
            probe_names = [x.strip() for x in probe_txt.split(',')]
            probe_idx =  [self.engine.dep_var_offset(n) for n in probe_names]
            for n, i in zip(probe_names, probe_idx):
                self.probe.append(Probe(n, i))

    def allocate_linearsolver(self, is_complex, engine):
        if self.linearsolver_model.accept_complex:
            linearsolver  = self.linearsolver_model.allocate_solver(is_complex, engine)            
        else:
            linearsolver  = self.linearsolver_model.allocate_solver(False, engine)                        

        return linearsolver
        
    def solve(self):
        raise NotImplementedError(
             "you must specify this method in subclass")

    def compute_rhs(self, M, B, X):
        raise NotImplementedError(
             "you must specify this method in subclass")

    def compute_A(self, M, B, X, mask_M, mask_B):
        # must return A and isAnew        
        raise NotImplementedError(
             "you must specify this method in subclass")
    
class TimeDependentSolverInstance(SolverInstance):
    def __init__(self, gui, engine):
        self.st = 0.0
        self.et = 1.0
        self.checkpoint = [0, 0.5, 1.0]
        self._icheckpoint = 0        
        self._time = 0.0
        self.child_instance = []
        SolverInstance.__init__(self, gui, engine)

    @property
    def time(self):
        return self._time
    
    @time.setter
    def time(self, value):
        self._time = value
        self.engine.model['General']._global_ns['t']=value
        
    @property
    def icheckpoint(self):
        return self._icheckpoint
    
    @icheckpoint.setter
    def icheckpoint(self, value):
        self._icheckpoint = value
        
    def set_start(self, st):
        self.st = st
        self.time = st
        
    def set_end(self, et):
        self.et = et

    @property
    def timestep(self):
        return self._time_step
    
    def set_timestep(self, time_step):
        self._time_step = time_step
        
    def set_checkpoint(self, checkpoint):
        self.checkpoint = checkpoint

    def add_child_instance(self, instance):
        self.child_instance.append(instance)
        
    def solve(self):
        assert False, "time dependent solver does not have solve method. call step"
        
    def step(self):
        raise NotImplementedError(
             "you must specify this method in subclass")

'''

    LinearSolverModel : Model Tree Object for linear solver
    LinearSolver : an interface to actual solver


'''    
class LinearSolverModel(SolverBase):
    is_iterative = True    
    def get_phys(self):
        return self.parent.get_phys()
    
    def get_phys_range(self):
        return self.parent.get_phys_range()

    def linear_system_type(self, assemble_real, phys_real):
        '''
        ls_type: coo  (matrix in coo format : DMUMP or ZMUMPS)
                 coo_real  (matrix in coo format converted from complex 
                            matrix : DMUMPS)
                 # below is a plan...
                 blk (matrix made mfem:block operator)
                 blk_real (matrix made mfem:block operator for complex
                             problem)
                          (unknowns are in the order of  R_fes1, R_fes2,... I_fes1, Ifes2...)
                 blk_interleave (unknowns are in the order of  R_fes1, I_fes1, R_fes2, I_fes2,...)
                 None(not supported)
        '''
        raise NotImplementedError(
             "you must specify this method in subclass")
    
    def allocate_solver(self, is_complex = False, engine=None):
        raise NotImplementedError(
             "you must specify this method in subclass")

    def real_to_complex(self, solall, M=None):
        # method called when real value solver is used for complex value problem
        raise NotImplementedError(
             "you must specify this method in subclass")
        
class LinearSolver(object):
    '''
    LinearSolver is an interface to linear solvers such as MUMPS.
    It is generated by SolverInstan.
    '''
    is_iterative = True        
    def __init__(self, gui, engine):
        self.gui = gui
        self.engine = engine

    
    def SetOperator(self, opr, dist=False, name=None):
        # opr : operator (matrix)
        # dist: disributed matrix or not
        # name: name of variables in block operator
        raise NotImplementedError(
             "you must specify this method in subclass")
    def Mult(self, b, case_base=0):
        raise NotImplementedError(
             "you must specify this method in subclass")

    
