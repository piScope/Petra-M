from abc import ABC, abstractmethod
import numpy as np
import warnings
import os
from petram.model import Model
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Solver')

'''

    Solver configurations


    (GUI)

    SolveStep:   This level defines the block matrix assembled at one time
    Solve:       Solver solves a problem (linear or non-lienar or time-dependent or parametric),
                 using blocks defined in SolveStep

    LinterSolverModel:
                 LinearSolver solves a linear system.
                 Preconditioner is a child LinearSolver of LinearSolver


    (SolverInstance)
n
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
        '''
        return a model directly under Solver section such as SolveStep
        '''
        obj = self
        solver_root = self.root()['Solver']

        while (not isinstance(obj, SolveStep) and
               obj is not solver_root):
            obj = obj.parent
        return obj

    def eval_text_in_global(self, value, ll=None):
        if not isinstance(value, str):
            return value
        ll = {} if ll is None else ll
        gg = self.root()['General']._global_ns.copy()
        return eval(value, gg, ll)


class SolveStep(SolverBase):
    has_2nd_panel = False

    #
    # GUI and object parameters
    #
    def attribute_set(self, v):
        v['phys_model'] = ''
        v['init_setting'] = ''
        v['postprocess_sol'] = ''
        v['dwc_name'] = ''
        v['use_dwc_pp'] = False
        v['dwc_pp_arg'] = ''
        v['use_geom_gen'] = False
        v['use_mesh_gen'] = False
        v['use_profiler'] = False

        super(SolveStep, self).attribute_set(v)
        return v

    def panel1_param(self):
        ret = [["dwc",   self.dwc_name,   0, {}],
               ["args.",   self.dwc_pp_arg,   0, {}]]
        value = [self.dwc_name, self.dwc_pp_arg]
        return [["Initial value setting",   self.init_setting,   0, {}, ],
                ["Postprocess solution",    self.postprocess_sol,   0, {}, ],
                ["trial phys.", self.phys_model, 0, {}, ],
                [None,  self.use_geom_gen,  3, {
                    "text": "run geometry generator"}],
                [None,  self.use_mesh_gen,  3, {"text": "run mesh generator"}],
                [None,  self.use_profiler,  3, {"text": "use profiler"}],
                [None, [False, value], 27, [{'text': 'use DWC (postprocess)'},
                                            {'elp': ret}]], ]

#                ["initialize solution only",
#                 self.init_only,  3, {"text":""}], ]

    def get_panel1_value(self):
        return (self.init_setting, self.postprocess_sol, self.phys_model,
                self.use_geom_gen,
                self.use_mesh_gen,
                self.use_profiler,
                [self.use_dwc_pp, [self.dwc_name, self.dwc_pp_arg, ]])

    def import_panel1_value(self, v):
        self.init_setting = v[0]
        self.postprocess_sol = v[1]
        self.phys_model = v[2]
        self.use_geom_gen = v[3]
        self.use_mesh_gen = v[4]
        if self.use_geom_gen:
            self.use_mesh_gen = True
        self.use_profiler = bool(v[5])
        self.use_dwc_pp = v[6][0]
        self.dwc_name = v[6][1][0]
        self.dwc_pp_arg = v[6][1][1]

#        self.init_only    = v[2]

    def get_possible_child(self):
        #from solver.solinit_model import SolInit
        from petram.solver.std_solver_model import StdSolver
        from petram.solver.nl_solver_model import NLSolver
        from petram.solver.mg_solver_model import MGSolver
        from petram.solver.ml_solver_model import MultiLvlStationarySolver
        from petram.solver.solver_controls import DWCCall, ForLoop
        from petram.solver.timedomain_solver_model import TimeDomain
        from petram.solver.set_var import SetVar
        from petram.solver.distance_solver import DistanceSolver

        try:
            from petram.solver.std_meshadapt_solver_model import StdMeshAdaptSolver
            return [MultiLvlStationarySolver,
                    TimeDomain,
                    DistanceSolver,
                    StdSolver,
                    StdMeshAdaptSolver,
                    NLSolver,
                    # MGSolver,
                    ForLoop,
                    DWCCall, SetVar]
        except:
            return [MultiLvlStationarySolver,
                    TimeDomain,
                    DistanceSolver,
                    # MGSolver,
                    StdSolver,
                    NLSolver,
                    ForLoop,
                    DWCCall, SetVar]

    def get_possible_child_menu(self):
        #from solver.solinit_model import SolInit
        from petram.solver.std_solver_model import StdSolver
        from petram.solver.nl_solver_model import NLSolver
        from petram.solver.mg_solver_model import MGSolver
        from petram.solver.ml_solver_model import MultiLvlStationarySolver
        from petram.solver.solver_controls import DWCCall, InnerForLoop
        from petram.solver.timedomain_solver_model import TimeDomain
        from petram.solver.set_var import SetVar
        from petram.solver.distance_solver import DistanceSolver

        try:
            from petram.solver.std_meshadapt_solver_model import StdMeshAdaptSolver
            return [("", StdSolver),
                    ("", MultiLvlStationarySolver),
                    ("", NLSolver),
                    ("", TimeDomain),
                    ("extra", DistanceSolver),
                    ("", StdMeshAdaptSolver),
                    ("", InnerForLoop),
                    ("", DWCCall),
                    ("!", SetVar)]
        except:
            return [("", StdSolver),
                    ("", MultiLvlStationarySolver),
                    ("", NLSolver),
                    ("", TimeDomain),
                    ("extra", DistanceSolver),
                    ("", InnerForLoop),
                    ("", DWCCall),
                    ("!", SetVar)]

    @property
    def solve_error(self):
        if hasattr(self, "_solve_error"):
            return self._solve_error
        return (False, "")

    #
    # Data access
    #
    def get_phys(self):
        #
        #  phys for rhs and rows of M
        #
        phys_root = self.root()['Phys']
        ret = []
        for k in self.keys():
            if not self[k].enabled:
                continue
            for x in self[k].get_target_phys():
                if not x in ret:
                    ret.append(x)
            for s in self[k].get_child_solver():
                for x in s.get_target_phys():
                    if not x in ret:
                        ret.append(x)
        return ret

    def get_phys_range(self):
        #
        #  phys for X and col of M
        #
        phys_root = self.root()['Phys']
        phys_test = self.get_phys()
        for n in self.phys_model.split(','):
            n = n.strip()
            p = phys_root.get(n, None)
            if p is None:
                continue
            if not p in phys_test:
                phys_test.append(p)
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

    def get_num_matrix(self, phys_target, set_active_matrix=False, engine=None):
        from petram.engine import max_matrix_num

        num = []
        num_matrix = 0
        active_solves = [self[k] for k in self if self[k].enabled]
        ###

        all_weights = []
        for phys in phys_target:
            for mm in phys.walk():
                if not mm.enabled:
                    continue

                ww = [False]*max_matrix_num
                for s in active_solves:
                    w = s.get_matrix_weight(mm.timestep_config)
                    for i, v in enumerate(w):
                        ww[i] = (ww[i] or v)
                ww = [bool(x) for x in ww]

                mm.set_matrix_weight(ww)
                wt = np.array(ww)
                tmp = int(np.max((wt != 0)*(np.arange(len(wt))+1)))
                num_matrix = max(tmp, num_matrix)

                all_weights.append(mm.get_matrix_weight())

        if set_active_matrix:
            flag = np.sum(all_weights, 0).astype(bool)
            engine.set_active_matrix(flag)
            dprint1("active_matrix flag", flag)
        return num_matrix

    def get_matrix_weight(self, timestep_config):
        raise NotImplementedError(
            "you must specify this method in subclass")

    def is_allphys_real(self):
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()

        phys_real = all([not p.is_complex()
                         for p in phys_target + phys_range])
        return phys_real
    #
    #   data access for init, postprocess, mesh, and goemetry
    #

    def get_init_setting(self):
        names = self.init_setting.split(',')
        names = [n.strip() for n in names if n.strip() != '']
        return [self.root()['InitialValue'][n] for n in names]

    def get_pp_setting(self):
        names = self.postprocess_sol.split(',')
        names = [n.strip() for n in names if n.strip() != '']
        return [self.root()['PostProcess'][n] for n in names]

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
    #
    #  verify
    #

    def verify_setting(self):
        try:
            self.get_linearsystem_type_from_modeltree()

        except AssertionError as msg:
            return False, "Can not select linear system type consistently", str(msg)

        return True, "", ""

    def get_linearsystem_type_from_modeltree(self):
        '''
        find appropriate linear system type from model tree. this supports
        two ways to choose linear system type.
           1) LinearSolverModel has an interface to choose the type (old way)
           2) SolverModel has an interface to choose the type (new way)
        '''
        ls_selected = None
        ls_candidates = None

        def make_assertion(cond, message):
            if not cond:
                dprint1("Error: selected ls", ls_selected)
                dprint1("Error: candidate ls", ls_candidates)
                assert cond, message

        def collect_assemble_real(top):
            tmp = []
            for x in top.iter_enabled():
                if hasattr(x, 'assemble_real'):
                    tmp.append(x.assemble_real)
            if True in tmp and False in tmp:
                assert False, "Assemble real is not selected consistently"
            return (True in tmp)
        for x in self.walk():
            if not x.is_enabled():
                continue
            if isinstance(x, LinearSolverModel):
                if x.does_linearsolver_choose_linearsystem_type():
                    assemble_real = collect_assemble_real(self)
                    phys_real = self.is_allphys_real()

                    tmp = x.linear_system_type(assemble_real, phys_real)

                    if ls_selected is None:
                        ls_selected = tmp
                    elif ls_selected == tmp:
                        pass
                    else:
                        make_assertion(
                            False, "Can not select linear system type consistently.(A)")
                else:
                    tmp = x.supported_linear_system_type()
                    if tmp == 'ANY':
                        continue
                    if ls_candidates is None:
                        ls_candidates = set(tmp)
                    else:
                        ls_candidates = ls_candidates & set(tmp)

            if isinstance(x, Solver):
                if x.does_solver_choose_linearsystem_type():
                    tmp = x.get_linearsystem_type_from_solvermodel()
                    if ls_selected is None:
                        ls_selected = tmp
                    elif ls_selected == tmp:
                        pass
                    else:
                        make_assertion(
                            False, "Can not select linear system type consistently. (B)")
                else:
                    pass

        if ls_candidates is not None:
            make_assertion(ls_selected in ls_candidates,
                           "Can not select linear system type consistently. (C)")

        if ls_selected is None:
            make_assertion(
                False, "Model tree does not choose linear system type")
        return ls_selected

    #
    # Preparation for assembly
    #
    def prepare_form_sol_variables(self, engine, n_levels=1):
        solvers = self.get_active_solvers()

        phys_target = self.get_phys()
        phys_range = self.get_phys_range()

        num_matrix = self.get_num_matrix(phys_target,
                                         set_active_matrix=True,
                                         engine=engine)

        engine.set_formblocks(phys_target, phys_range, num_matrix)

        for p in phys_range:
            engine.run_mesh_extension(p)

        engine.run_alloc_sol(phys_range)

#        engine.run_fill_X_block()

    def init(self, engine):
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()

        inits = self.get_init_setting()

        solvers = self.get_active_solvers()
        for solver in solvers:
            inits.extend(solver.get_custom_init())

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

    @debug.use_profiler
    def run(self, engine, is_first=True):
        dprint1("!!!!! Entering SolveStep " + self.name() + " !!!!!")

        wc = self.root()["General"].warning_control
        warnings.simplefilter(wc)
        dprint1("Settiing warning mode :", wc)

        solvers = self.get_active_solvers()

        is_new_mesh = self.check_and_run_geom_mesh_gens(engine)
        if is_first or is_new_mesh:
            engine.preprocess_modeldata()

        # initialize and assemble
        # in run method..
        #   std solver : make sub block matrix and solve
        #   time-domain solver : do step

        # prepare all MG refinement levels here

        flag = True
        lvl = 0
        enabled_flag = engine.model.gather_enebled_flags(engine.model['Phys'])

        while flag:
            engine.level_idx = lvl
            self.prepare_form_sol_variables(engine)
            self.init(engine)

            lvl = lvl + 1
            flag = any([s.create_refined_levels(engine, lvl)
                        for s in solvers])

        engine.model.apply_enebled_flags(engine.model['Phys'], enabled_flag)

        is_first = True
        for solver in solvers:
            is_first = solver.run(engine, is_first=is_first)
            engine.add_FESvariable_to_NS(self.get_phys())
            engine.store_x()
            if self.solve_error[0]:
                dprint1("SolveStep failed " + self.name() +
                        ":" + self.solve_error[1])
                break

        for solver in solvers:
            solver.free_instance()

        postprocess = self.get_pp_setting()
        engine.run_postprocess(postprocess, name=self.name())

        if self.use_dwc_pp:
            engine.call_dwc(self.get_phys_range(),
                            method="postprocess",
                            callername=self.name(),
                            dwcname=self.dwc_name,
                            args=self.dwc_pp_arg)

        warnings.resetwarnings()
        if "PYTHONWARNINGS" in os.environ:
            wc = os.environ["PYTHONWARNINGS"]
            warnings.simplefilter(wc)
        else:
            wc = "Default"
        dprint1("Resettiing warning mode :", wc)
        dprint1("Exiting SolveStep " + self.name())
        return False


class Solver(SolverBase):
    def attribute_set(self, v):
        v['clear_wdir'] = False
        v['init_only'] = False
        v['assemble_real'] = False
        v['save_parmesh'] = False
        v['phys_model'] = ''
        #v['init_setting']   = ''
        v['use_profiler'] = False
        v['probe'] = ''
        v['skip_solve'] = False
        v['load_sol'] = False
        v['sol_file'] = ''
        super(Solver, self).attribute_set(v)
        return v

    @property
    def solve_error(self):
        if hasattr(self, "_solve_error"):
            return self._solve_error
        return (False, "")

    @property
    def instance(self):
        if hasattr(self, "_instance"):
            return self._instance
        return None

    @instance.setter
    def instance(self, value):
        self._instance = value

    def free_instance(self):
        self._instance = None

    def get_phys(self):
        my_solve_step = self.get_solve_root()
        return my_solve_step.get_phys()

    def get_phys_range(self):
        my_solve_step = self.get_solve_root()
        return my_solve_step.get_phys_range()

    def get_target_phys(self):
        names = self.phys_model.split(',')
        names = [n.strip() for n in names if n.strip() != '']
        return [self.root()['Phys'][n] for n in names]

    def get_child_solver(self):
        return []

    def get_custom_init(self):
        return []

    def is_complex(self):
        phys = self.get_phys()
        is_complex = any([p.is_complex() for p in phys])
        if self.assemble_real:
            return False
        # if is_complex: return True
        return is_complex

    def is_converted_from_complex(self):
        phys = self.get_phys()
        is_complex = any([p.is_complex() for p in phys])
        if is_complex and self.assemble_real:
            return True
        return False

    def get_init_setting(self):
        raise NotImplementedError(
            "bug should not need this method")
        '''
        names = self.init_setting.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['InitialValue'][n] for n in names]
        '''

    def get_active_solver(self, mm=None, cls=None):
        if cls is None:
            cls = LinearSolverModel
        for x in self.iter_enabled():
            if isinstance(x, cls):
                return x

    def get_active_solvers(self, mm=None, cls=None):
        if cls is None:
            cls = LinearSolverModel

        solvers = []
        for x in self.iter_enabled():
            if isinstance(x, cls):
                solvers.append(x)
        return solvers

    def get_num_matrix(self, phys_target=None):
        raise NotImplementedError(
            "bug should not need this method")

    def get_num_levels(self):
        return 1

    def create_refined_levels(self, engine, lvl):
        '''
        create refined levels and return True if it is created.
        default False (no refined level)
        '''
        return False

    def does_solver_choose_linearsystem_type(self):
        return False

    def get_linearsystem_type_from_solvermodel(self):
        raise NotImplementedError(
            "bug should not need this method")

    @abstractmethod
    def get_matrix_weight(self, *args, **kwargs):
        raise NotImplementedError(
            "bug should not need this method")

    @abstractmethod
    def run(self, engine, is_first=True):
        ...


class SolverInstance(ABC):
    '''
    Solver instance is where the logic of solving a PDF usng
    linearlized matrices (time stepping, adaptation, non-linear...) 
    is written.

    It is not a model object. SolverModel will generate this
    instance to do the actual solve step.
    '''

    def __init__(self, gui, engine):
        self.gui = gui
        self.engine = engine
        self.sol = None
        self.linearsolver_model = None  # LinearSolverModel
        self.linearsolver = None      # Actual LinearSolver
        self.probe = []
        self.linearsolver_model = None

        self._ls_type = self.gui.get_solve_root().get_linearsystem_type_from_modeltree()
        self._phys_real = self.gui.get_solve_root().is_allphys_real()

        if not gui.init_only:
            self.set_linearsolver_model()

    @property
    def ls_type(self):
        return self._ls_type

    @property
    def phys_real(self):
        return self._phys_real

    @ls_type.setter
    def ls_type(self, _v):
        warnings.warn(
            "Setting ls_type does not have any effect.", RuntimeWarning)

    @phys_real.setter
    def phys_real(self, _v):
        warnings.warn(
            "Setting phys_real does not have any effect.", RuntimeWarning)

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
        mask2 = self.engine.get_block_mask(
            all_phys, phys_target, use_range=True)

        self.blk_mask = (mask1, mask2)
        self.engine._matrix_blk_mask = self.blk_mask

    def recover_solution(self, ksol=0):
        '''
        bring linear algebra level solution to gridfunction.
        called when we need a solution vector in gridfucntion vector.
        '''
        engine = self.engine
        phys_target = self.get_phys()
        sol, sol_extra = engine.split_sol_array(self.sol)
        engine.recover_sol(sol)
        extra_data = engine.process_extra(sol_extra)

    def save_solution(self, ksol=0, skip_mesh=False,
                      mesh_only=False, save_parmesh=False,
                      save_mesh_linkdir=None):

        engine = self.engine
        phys_target = self.get_phys()

        if mesh_only:
            engine.save_sol_to_file(phys_target,
                                    mesh_only=True,
                                    save_parmesh=save_parmesh)
        else:
            sol, sol_extra = engine.split_sol_array(self.sol)
            engine.recover_sol(sol)
            extra_data = engine.process_extra(sol_extra)

            engine.save_sol_to_file(phys_target,
                                    skip_mesh=skip_mesh,
                                    mesh_only=False,
                                    save_parmesh=save_parmesh,
                                    save_mesh_linkdir=save_mesh_linkdir)
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

    def configure_probes(self, probe_txt):
        from petram.sol.probe import Probe

        all_phys = self.get_phys()
        txt = [phys.collect_probes() for phys in all_phys]
        txt = [probe_txt]+txt
        probe_txt = ','.join([t for t in txt if len(t) > 0])

        dprint1("configure probes: "+probe_txt)
        if probe_txt.strip() != '':
            probe_names = [x.strip() for x in probe_txt.split(',')]
            probe_idx = [self.engine.dep_var_offset(n) for n in probe_names]
            for n, i in zip(probe_names, probe_idx):
                self.probe.append(Probe(n, i))

    def allocate_linearsolver(self, is_complex, engine, solver_model=None):
        if solver_model is None:
            solver_model = self.linearsolver_model
        if solver_model.accept_complex:
            linearsolver = self.linearsolver_model.allocate_solver(
                is_complex, engine)
        else:
            linearsolver = solver_model.allocate_solver(
                False, engine)

        return linearsolver

    @abstractmethod
    def solve(self):
        ...

    @abstractmethod
    def compute_rhs(self, M, B, X):
        ...

    @abstractmethod
    def compute_A(self, M, B, X, mask_M, mask_B):
        ...


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
        self.engine.model['General']._global_ns['t'] = value

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

    @abstractmethod
    def step(self):
        ...


'''
    LinearSolverModel : Model Tree Object for linear solver
    LinearSolver : an interface to actual solver
'''


class LinearSolverModel(SolverBase):
    '''
    Model tree object for a linear solver
    '''
    is_iterative = True

    def get_phys(self):
        return self.parent.get_phys()

    def get_phys_range(self):
        return self.parent.get_phys_range()

    def allocate_solver(self, is_complex=False, engine=None):
        '''
        this method create LinearSolverInstance

        LinearSolverInstance is intermediate obect to prepare LinearSolver
        '''
        raise NotImplementedError(
            "bug. this method sould not be called")

    def prepare_solver(self):
        '''
        this method create LinearSolver. This should return MFEM LinearOperator
        '''
        raise NotImplementedError(
            "bug. this method sould not be called")

    def prepare_solver_with_multtranspose(self):
        '''
        this method create LinearSolver. This should return MFEM LinearOperator
        '''
        raise NotImplementedError(
            "bug. this method sould not be called")

    @abstractmethod
    def real_to_complex(self, solall, M=None):
        ...

    @abstractmethod
    def does_linearsolver_choose_linearsystem_type(self):
        '''
        determins how linearsolvermodel informs the type of linear system to assemble.

        if True: 
            linear_system_type should be implemented.
        if False:
            supported_linear_system_type
        '''

    def linear_system_type(self, assemble_real, phys_real):
        '''
        ls_type: coo  (matrix in coo format : DMUMP or ZMUMPS)
                 coo_real  (matrix in coo format converted from complex 
                            matrix : DMUMPS)
                 blk_interleave (R_fes1, I_fes1, R_fes2, I_fes2,..., I is skipped if real)
                 blk_merged_s (Block operator using ComplexOperator, block_symmetric format)
                 blk_merged   (Block operator using ComplexOperator, asymetric format)

        return None if the model does not specify the linear system type.
        '''
        raise NotImplementedError(
            "bug. this method sould not be called")

    def supported_linear_system_type(self):
        raise NotImplementedError(
            "bug. this method sould not be called")


class LinearSolver(ABC):
    '''
    LinearSolver is an interface to linear solvers such as MUMPS.
    '''
    is_iterative = True

    def __init__(self, gui, engine):
        self.gui = gui
        self.engine = engine
        self._skip_solve = False

    @abstractmethod
    def SetOperator(self, opr, dist=False, name=None):
        ...

    @abstractmethod
    def Mult(self, b, case_base=0):
        ...

    @property
    def skip_solve(self):
        return self._skip_solve

    @skip_solve.setter
    def skip_solve(self, val):
        self._skip_solve = val


def convert_realblocks_to_complex(solall, M, merge_real_imag):
    if merge_real_imag:
        return real_to_complex_merged(solall, M)
    else:
        return real_to_complex_interleaved(solall, M)


def real_to_complex_interleaved(solall, M):
    from petram.mfem_config import use_parallel
    if use_parallel:
        from mpi4py import MPI
        myid = MPI.COMM_WORLD.rank

        of = M.RowOffsets().ToList()
        # of = [np.sum(MPI.COMM_WORLD.allgather(np.int32(o)))
        #      for o in offset]
        # if myid != 0:
        #    return

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


def real_to_complex_merged(solall, M):
    from petram.mfem_config import use_parallel
    if use_parallel:
        from mpi4py import MPI
        myid = MPI.COMM_WORLD.rank

        of = M.RowOffsets().ToList()
        # of = [np.sum(MPI.COMM_WORLD.allgather(np.int32(o)))
        #      for o in offset]
        # if myid != 0:
        #    return

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
