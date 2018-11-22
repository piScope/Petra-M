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
        v['use_dwc_pp']   = False
        v['dwc_pp_arg']   = ''                        
        super(SolveStep, self).attribute_set(v)
        return v
    
    def panel1_param(self):
        ret = ["args.",   self.dwc_pp_arg,   0, {},]
        value = self.dwc_pp_arg
        return [["Initial value setting",   self.init_setting,   0, {},],
                ["addtional physics for range (blank: range = test)",   self.phys_model, 0, {},],
                [None, [False, [value]], 27, [{'text':'Use DWC (postprocess)'},
                                              {'elp': [ret]}]],]

#                ["initialize solution only",
#                 self.init_only,  3, {"text":""}], ]
               

    def get_panel1_value(self):
        return (self.init_setting, self.phys_model,
                [self.use_dwc_pp, [self.dwc_pp_arg,]])
    
    def import_panel1_value(self, v):
        self.init_setting = v[0]        
        self.phys_model   = v[1]
        self.use_dwc_pp   = v[2][0]
        self.dwc_pp_arg   = v[2][1][0]         
#        self.init_only    = v[2]
        
    def get_possible_child(self):
        #from solver.solinit_model import SolInit
        from petram.solver.std_solver_model import StdSolver
        from petram.solver.timedomain_solver_model import TimeDomain
        from petram.solver.parametric import Parametric
        
        return [StdSolver, TimeDomain, Parametric]
    
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
        ret = []
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
    
    def get_num_matrix(self, phys_target):
        num = []
        for k in self.keys():
            mm = self[k]
            if not mm.enabled: continue
            num.append(self.root()['Phys'].get_num_matrix(mm.get_matrix_weight,
                                           phys_target))
        return max(num)
    
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

    def run(self, engine, is_first = True):
        dprint1("Entering SolveStep :" + self.name())
        solvers = self.get_active_solvers()

        # initialize and assemble here

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

        if self.use_dwc_pp:
            engine.call_dwc(self.get_phys_range(),
                            method="postprocess",
                            callername = self.name(),
                            args = self.dwc_pp_arg)
        
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

    def get_matrix_weight(self):
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
        dprint1("configure probes: "+probe_txt)
        if probe_txt.strip() != '':
            probe_names = [x.strip() for x in probe_txt.split(',')]
            probe_idx =  [self.engine.dep_var_offset(n) for n in probe_names]
            for n, i in zip(probe_names, probe_idx):
                self.probe.append(Probe(n, i))

    def allocate_linearsolver(self, is_complex, engine):
        if self.ls_type.startswith('coo'):
            datatype = 'Z' if (is_complex) else 'D'
        else:
            datatype = 'D'
            
        linearsolver  = self.linearsolver_model.allocate_solver(datatype, engine)
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
    def allocate_solver(self, datatype='D', engine=None):
        # datatype = S, D, C, Z
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

    
