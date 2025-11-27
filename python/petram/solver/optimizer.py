#
#  Optimizer
#
#     parameter optimizer find a set of global_ns parameters
#     which minimizes a user defined (in global_ns.py) cost
#     funciton.
#

from petram.mfem_config import use_parallel
import os
import traceback
import numpy as np

from petram.model import Model
from petram.solver.solver_model import Solver, SolveStep
from petram.namespace_mixin import NS_mixin
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Optimizer')
format_memory_usage = debug.format_memory_usage

if use_parallel:
    import mfem.par as mfem
    from mfem.common.mpi_debug import nicePrint
    from mpi4py import MPI
    myid = MPI.COMM_WORLD.rank
else:
    import mfem.ser as mfem
    myid = 0
    nicePrint = dprint1


class Optimizer(SolveStep, NS_mixin):
    '''
    parametric optimizer of model
    '''
    can_delete = True
    has_2nd_panel = False

    def __init__(self, *args, **kwargs):
        SolveStep.__init__(self, *args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)

        self.case_dirs = None

    def init_solver(self):
        pass

    def panel1_param(self):
        v = self.get_panel1_value()
        return [["Initial value setting",   self.init_setting,  0, {}, ],
                ["Postprocess solution",    self.postprocess_sol,   0, {}, ],
                ["trial phys. ",   self.phys_model, 0, {}, ],
                self.make_param_panel('minimizer',  v[2]),
                ["inner solver", '', 2, None],
                [None,  True,  3, {"text": "save separate mesh"}],
                [None, False, 3, {"text": "clear working dir."}],
                [None, False, 3, {"text": "keep all cases"}],
                [None,  self.use_geom_gen,  3, {
                    "text": "run geometry generator"}],
                [None,  self.use_mesh_gen,  3, {"text": "run mesh generator"}],
                [None,  self.use_profiler,  3, {"text": "use profiler"}],
                ]

    def get_panel1_value(self):
        return (self.init_setting,
                self.postprocess_sol,
                self.phys_model,
                str(self.minimizer),
                self.get_inner_solver_names(),
                self.save_separate_mesh,
                self.clear_wdir,
                self.keep_cases,
                self.use_geom_gen,
                self.use_mesh_gen,
                self.use_profiler,)

    def import_panel1_value(self, v):
        self.init_setting = str(v[0])
        self.postprocess_sol = v[1]
        self.phys_model = str(v[2])
        self.minimizer = v[-8]
        self.save_separate_mesh = v[-6]
        self.clear_wdir = v[-5]
        self.keep_cases = v[-4]
        self.use_geom_gen = v[-3]
        self.use_mesh_gen = v[-2]
        if self.use_geom_gen:
            self.use_mesh_gen = True
        if self.use_mesh_gen:
            self.assembly_method = 0
        self.use_profiler = bool(v[-1])

    def get_inner_solver_names(self):
        names = [s.name() for s in self.get_active_solvers()]
        return ', '.join(names)

    '''
    def get_inner_solvers(self):
        return [self[k] for k in self if self[k].enabled]
    '''

    def attribute_set(self, v):
        v = super(Optimizer, self).attribute_set(v)
        v['minimizer'] = 'Minimizer(cost, (0.15, 0.45), "a", [1,3], "b",[4, 5], tol=1e-2, maxiter=10, verbose=True)'
        v['save_separate_mesh'] = False
        v['clear_wdir'] = True
        v['keep_cases'] = False

        return v

    def get_possible_child(self):
        # from solver.solinit_model import SolInit
        from petram.solver.std_solver_model import StdSolver
        from petram.solver.nl_solver_model import NLSolver
        from petram.solver.ml_solver_model import MultiLvlStationarySolver
        from petram.solver.solver_controls import DWCCall, ForLoop
        from petram.solver.set_var import SetVar
        from petram.solver.solver_model import SolveStep
        from petram.solver.parametric import Parametric

        try:
            from petram.solver.std_meshadapt_solver_model import StdMeshAdaptSolver
            return [SolveStep,
                    Parametric,
                    MultiLvlStationarySolver,
                    StdSolver,
                    StdMeshAdaptSolver,
                    NLSolver,
                    DWCCall, ForLoop, SetVar]
        except:
            return [MultiLvlStationarySolver,
                    StdSolver,
                    NLSolver,
                    DWCCall, ForLoop, SetVar]

    def get_possible_child_menu(self):
        # from solver.solinit_model import SolInit
        from petram.solver.std_solver_model import StdSolver
        from petram.solver.nl_solver_model import NLSolver
        from petram.solver.ml_solver_model import MultiLvlStationarySolver
        from petram.solver.solver_controls import DWCCall, ForLoop
        from petram.solver.set_var import SetVar
        from petram.solver.solver_model import SolveStep
        from petram.solver.parametric import Parametric

        try:
            from petram.solver.std_meshadapt_solver_model import StdMeshAdaptSolver
            return [("SolveSteps", SolveStep),
                    ("!", Parametric),
                    ("Solvers", StdSolver),
                    ("", MultiLvlStationarySolver),
                    ("!", NLSolver),
                    ("extra", ForLoop),
                    ("", StdMeshAdaptSolver),
                    ("", DWCCall),
                    ("!", SetVar)]
        except:
            return [("SolveSteps", SolveStep),
                    ("!", Parametric),
                    ("Solvers", StdSolver),
                    ("", MultiLvlStationarySolver),
                    ("!", NLSolver),
                    ("extra", ForLoop),
                    ("", DWCCall),
                    ("!", SetVar)]

    def verify_setting(self):
        if (len(self.get_active_solversteps()) > 0 and
                len(self.get_active_solvers()) > 0):
            assert False, "Optimizer Child Needs to be either Solver or SolveStep."

        if (len(self.get_active_solversteps()) == 0 and
                len(self.get_active_solvers()) == 0):
            assert False, "Optimizer has no active Solver nor SolveStep."

        if len(self.get_active_solversteps()) > 0:
            return True, "", ""

        return SolveStep.verify_setting(self)

    def get_minimizer(self, nosave=False):
        if not self.enabled:
            return
        try:
            minimizer = self.eval_param_expr(str(self.minimizer),
                                             'minimizer')[0]
            minimizer.set_model(self.root())
        except:
            traceback.print_exc()
            return

        return minimizer

    def get_probes(self):
        probes = super(Optimizer, self).get_probes()
        minimizer = self.get_minimizer(nosave=True)
        if minimizer is not None:
            probes.extend([self.name()+"_"+x for x in minimizer.get_params()])
            probes.append(self.name()+"_costs")
        return probes

    def get_default_ns(self):
        from petram.solver.minimizer import Minimizer
        return {'Minimizer': Minimizer}

    def get_default_weak_ns(self):
        from petram.solver.minimizer import default_cost
        return {'cost': default_cost}

    def go_case_dir(self, engine, ksol, mkdir):
        '''
        make case directory and create symlinks
        '''

        od = os.getcwd()

        nsfiles = [n for n in os.listdir() if n.endswith('_ns.py')
                   or n.endswith('_ns.dat')]

        path = os.path.join(od, 'case_' + str(ksol))
        if mkdir:
            engine.mkdir(path)
            os.chdir(path)
            engine.cleancwd()
        else:
            os.chdir(path)
        files = ['model.pmfm'] + nsfiles
        for n in files:
            if not os.path.exists(n):
                engine.symlink(os.path.join('../', n), n)
        self.case_dirs.append(path)
        return od

    def call_minimizer(self, minimizer, engine):
        a_ssteps = self.get_active_solversteps()
        a_solvers = self.get_active_solvers()

        def run_solvers(kcase, engine, solvers=a_solvers):
            is_first = kcase == 0
            postprocess = self.get_pp_setting()

            if self.keep_cases:
                od = self.go_case_dir(engine, kcase, True)

            engine.record_environment()
            engine.build_ns()

            is_new_mesh = self.check_and_run_geom_mesh_gens(engine)
            if is_new_mesh or is_first:
                self.check_and_end_geom_mesh_gens(engine)

            if is_new_mesh or is_first:
                engine.preprocess_modeldata()
            else:
                engine.save_processed_model()

            self.prepare_form_sol_variables(engine)

            self.init(engine)

            is_first0 = True
            for ksolver, s in enumerate(solvers):
                is_first0 = s.run(engine, is_first=is_first0)
                engine.add_FESvariable_to_NS(self.get_phys())
                engine.store_x()
                if self.solve_error[0]:
                    dprint1("Optimizer solver failed " + self.name() + ":" +
                            self.solve_error[1])

            engine.run_postprocess(postprocess, name=self.name())

            from petram.sol.probe import collect_probesignals
            prbs = collect_probesignals(os.getcwd())

            if self.keep_cases:
                src = os.path.join(os.getcwd(), 'model_proc.pmfm')
                os.chdir(od)
                dst = os.path.join(os.getcwd(), 'model_proc.pmfm')
                if is_first and myid == 0:
                    os.symlink(src, dst)
            return prbs

        def run_solvesteps(kcase, engine, solvesteps=a_ssteps):
            if self.keep_cases:
                od = self.go_case_dir(engine, kcase, True)

            is_first = True
            for s in solvesteps:
                s.run(engine, is_first=is_first)
                is_first = False

            from petram.sol.probe import collect_probesignals
            prbs = collect_probesignals(os.getcwd())

            if self.keep_cases:
                src = os.path.join(os.getcwd(), 'model_proc.pmfm')
                os.chdir(od)
                dst = os.path.join(os.getcwd(), 'model_proc.pmfm')
                if is_first and myid == 0:
                    os.symlink(src, dst)

            return prbs

        if len(a_solvers) > 0:
            minimizer.generate_cost_function(engine, run_solvers)
        else:
            minimizer.generate_cost_function(engine, run_solvesteps)
        minimizer.run()

    def save_probe_signals(self, minimizer):
        from petram.sol.probe import Probe

        xvalues = minimizer.costobj.xvalues
        costs = minimizer.costobj.costs

        probes = []

        time = np.atleast_2d(np.arange(len(costs), dtype=float)).transpose()

        for x in minimizer.get_params():
            n = self.name()+"_"+x
            probe = Probe(n, xnames=["iter_count"])
            probe.sig = xvalues[x]
            probe.t = time
            probes.append(probe)

        probe = Probe(self.name()+"_costs", xnames=["iter_count"])
        probe.sig = costs
        probe.t = time
        probes.append(probe)

        for p in probes:
            p.write_file(nosmyid=True)

    def write_optimizer_data(self, minimizer):
        #
        #  called at the end of parametric scan to save parameters
        #  to "cases.(solvername).txt". This is used in subdir
        #  menus in dialog
        #
        xvalues = minimizer.costobj.xvalues

        fid = open("cases."+self.name()+".txt", "w")
        k = 0
        for items in zip(*xvalues.values()):
            txt = str(k) + " : " + str(items)
            fid.write(txt+"\n")
            k = k + 1
        fid.close()

    @debug.use_profiler
    def run(self, engine, is_first=True):
        #
        # is_first is not used
        #
        dprint1("Entering Optimizer")
        if self.clear_wdir:
            engine.remove_solfiles()
        engine.remove_case_dirs()

        minimizer = self.get_minimizer()
        if minimizer is None:
            return

        self.case_dirs = []

        self.call_minimizer(minimizer, engine)

        if myid == 0:
            self.save_probe_signals(minimizer)
            if self.keep_cases:
                 self.write_optimizer_data(minimizer)
