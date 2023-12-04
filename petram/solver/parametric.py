import os
import traceback
import gc

from petram.model import Model
from petram.solver.solver_model import Solver, SolveStep
from petram.namespace_mixin import NS_mixin
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Parametric')
format_memory_usage = debug.format_memory_usage

assembly_methods = {'Full assemble': 0,
                    'Reuse matrix': 1}

from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
    from mfem.common.mpi_debug import nicePrint
    from mpi4py import MPI
else:
    import mfem.ser as mfem
    nicePrint = dprint1

class Parametric(SolveStep, NS_mixin):
    '''
    parametric sweep of some model paramter
    and run solver
    '''
    can_delete = True
    has_2nd_panel = False

    def __init__(self, *args, **kwargs):
        SolveStep.__init__(self, *args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)

    def init_solver(self):
        pass

    def panel1_param(self):
        v = self.get_panel1_value()
        return [["Initial value setting",   self.init_setting,  0, {}, ],
                ["Postprocess solution",    self.postprocess_sol,   0, {}, ],
                ["trial phys. ",   self.phys_model, 0, {}, ],
                ["assembly method",  'Full assemble',  4, {"readonly": True,
                                                           "choices": list(assembly_methods)}],
                self.make_param_panel('scanner',  v[2]),
                ["inner solver", '', 2, None],
                [None,  True,  3, {"text": "save separate mesh"}],
                [None, False, 3, {"text": "clear working dir."}],
                [None,  self.use_geom_gen,  3, {
                    "text": "run geometry generator"}],
                [None,  self.use_mesh_gen,  3, {"text": "run mesh generator"}],
                [None,  self.use_profiler,  3, {"text": "use profiler"}],
                ]

    def get_panel1_value(self):
        txt = list(assembly_methods)[0]
        for k, n in assembly_methods.items():
            if n == self.assembly_method:
                txt = k

        return (self.init_setting,
                self.postprocess_sol,
                self.phys_model,
                str(txt),
                str(self.scanner),
                self.get_inner_solver_names(),
                self.save_separate_mesh,
                self.clear_wdir,
                self.use_geom_gen,
                self.use_mesh_gen,
                self.use_profiler,)

    def import_panel1_value(self, v):
        self.init_setting = str(v[0])
        self.postprocess_sol = v[1]
        self.phys_model = str(v[2])
        self.assembly_method = assembly_methods[v[-8]]
        self.scanner = v[-7]
        self.save_separate_mesh = v[-5]
        self.clear_wdir = v[-4]
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
        v = super(Parametric, self).attribute_set(v)
        v['assembly_method'] = 0
        v['scanner'] = 'Scan("a", [1,2,3])'
        v['save_separate_mesh'] = False
        v['clear_wdir'] = True

        return v

    def get_possible_child(self):
        #from solver.solinit_model import SolInit
        from petram.solver.std_solver_model import StdSolver
        from petram.solver.nl_solver_model import NLSolver
        from petram.solver.ml_solver_model import MultiLvlStationarySolver
        from petram.solver.solver_controls import DWCCall, ForLoop
        from petram.solver.set_var import SetVar

        try:
            from petram.solver.std_meshadapt_solver_model import StdMeshAdaptSolver
            return [MultiLvlStationarySolver,
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
        #from solver.solinit_model import SolInit
        from petram.solver.std_solver_model import StdSolver
        from petram.solver.nl_solver_model import NLSolver
        from petram.solver.ml_solver_model import MultiLvlStationarySolver
        from petram.solver.solver_controls import DWCCall, ForLoop
        from petram.solver.set_var import SetVar

        try:
            from petram.solver.std_meshadapt_solver_model import StdMeshAdaptSolver
            return [("", StdSolver),
                    ("", MultiLvlStationarySolver),
                    ("", NLSolver),
                    ("extra", ForLoop),
                    ("", StdMeshAdaptSolver),
                    ("", DWCCall),
                    ("!", SetVar)]
        except:
            return [("", StdSolver),
                    ("", MultiLvlStationarySolver),
                    ("", NLSolver),
                    ("extra", ForLoop),
                    ("", DWCCall),
                    ("!", SetVar)]

    def get_scanner(self, nosave=False):
        try:
            scanner = self.eval_param_expr(str(self.scanner),
                                           'scanner')[0]
            scanner.set_data_from_model(self.root())
        except:
            traceback.print_exc()
            return

        if not nosave:
            scanner.save_scanner_data(self)

        return scanner

    def get_default_ns(self):
        from petram.solver.parametric_scanner import Scan
        return {'Scan': Scan}

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
            engine.symlink(os.path.join('../', n), n)
        self.case_dirs.append(path)
        return od

    def _run_full_assembly(self, engine, solvers, scanner, is_first=True):

        postprocess = self.get_pp_setting()
        for kcase, case in enumerate(scanner):
            is_first0 = True

            od = self.go_case_dir(engine, kcase, True)

            is_new_mesh = self.check_and_run_geom_mesh_gens(engine)

            if is_new_mesh or is_first:
                engine.preprocess_modeldata()
                is_first = False

            self.prepare_form_sol_variables(engine)

            self.init(engine)

            for ksolver, s in enumerate(solvers):
                is_first0 = s.run(engine, is_first=is_first0)
                engine.add_FESvariable_to_NS(self.get_phys())
                engine.store_x()
                if self.solve_error[0]:
                    dprint1("Parametric failed " + self.name() + ":" +
                            self.solve_error[1])

            engine.run_postprocess(postprocess, name=self.name())
            os.chdir(od)

    def _run_rhs_assembly(self, engine, solvers, scanner, is_first=True):
        self.prepare_form_sol_variables(engine)
#        self.init(engine, skip_essential=True)

        l_scan = len(scanner)

        all_phys = self.get_phys()
        phys_target = self.get_target_phys()
        postprocess = self.get_pp_setting()

        linearsolver = None
        for ksolver, s in enumerate(solvers):
            RHS_ALL = []
            instance = s.allocate_solver_instance(engine)

            phys_target = self.get_phys()
            phys_range = self.get_phys_range()

            for kcase, case in enumerate(scanner):

                if kcase == 0:
                    if ksolver == 0:
                        self.init(engine)
                    instance.set_blk_mask()
                    instance.assemble(inplace=False)
                else:
                    engine.set_update_flag('ParametricRHS')

                    done = []
                    for phys in phys_target + phys_range:
                        if phys not in done:
                            engine.run_update_param(phys)
                            done.append(phys)

                    engine.run_apply_essential(phys_target,
                                               phys_range,
                                               update=True)
                    engine.run_fill_X_block(update=True)
                    engine.run_assemble_extra_rhs(phys_target, phys_range,
                                                  update=True)
                    engine.run_assemble_b(phys_target, update=True)
                    engine.run_assemble_blocks(instance.compute_A,
                                               instance.compute_rhs,
                                               inplace=False,
                                               update=True)

                A, X, RHS, Ae, B, M, depvars = instance.blocks
                mask = instance.blk_mask
                depvars = [x for i, x in enumerate(depvars)
                           if mask[0][i]]
                if kcase == 0:
                    ls_type = instance.ls_type
                    phys_real = not s.is_complex()
                    AA = engine.finalize_matrix(A, mask, not phys_real,
                                                format=ls_type)

                RHS_ALL.append(RHS)

                if kcase == l_scan-1:
                    BB = engine.finalize_rhs(RHS_ALL, A, X[0], mask,
                                             not phys_real,
                                             format=ls_type)

                    if linearsolver is None:
                        linearsolver = instance.allocate_linearsolver(s.is_complex(),
                                                                      engine)
                    linearsolver.SetOperator(AA,
                                             dist=engine.is_matrix_distributed,
                                             name=depvars)

                    XX = None
                    solall = linearsolver.Mult(BB, x=XX, case_base=0)

                    is_sol_central = (True if not use_parallel else
                                      any(MPI.COMM_WORLD.allgather(solall is None)))

                    if not phys_real and s.assemble_real:
                        if not is_sol_central:
                            assert False, "this operation is not permitted"
                        oprt = linearsolver.oprt
                        solall = instance.linearsolver_model.real_to_complex(solall,
                                                                             oprt)

                    for ksol in range(l_scan):
                        instance.configure_probes('')
                        if ksol == 0:
                            instance.save_solution(mesh_only=True,
                                                   save_parmesh=s.save_parmesh)
                            save_mesh_linkdir = None

                        if is_sol_central:
                            A.reformat_central_mat(solall, ksol, X[0], mask)
                        else:
                            A.reformat_distributed_mat(solall, ksol, X[0], mask)

                        instance.sol = X[0]
                        for p in instance.probe:
                            p.append_sol(X[0])

                        od = self.go_case_dir(engine,
                                              ksol,
                                              ksolver == 0)

                        instance.save_solution(ksol=ksol,
                                               skip_mesh=False,
                                               mesh_only=False,
                                               save_parmesh=s.save_parmesh,
                                               save_mesh_linkdir=save_mesh_linkdir)

                        if save_mesh_linkdir is None:
                            save_mesh_linkdir = os.getcwd()

                        engine.sol = instance.sol
                        instance.save_probe()

                        engine.add_FESvariable_to_NS(self.get_phys())
                        engine.store_x()

                        engine.run_postprocess(postprocess, name=self.name())
                        os.chdir(od)

    def collect_probe_signals(self, dirs, scanner):
        from petram.sol.probe import list_probes, load_probe,  Probe
        params = scanner.list_data()

        od = os.getcwd()

        filenames, probenames = list_probes(dirs[0])

        names = scanner.names
        probes = [Probe(n, xnames=names) for n in probenames]

        for param, dirname in zip(params, dirs):
            os.chdir(dirname)
            for f, p in zip(filenames, probes):
                xdata, ydata = load_probe(f)
                p.append_value(ydata, param)

        os.chdir(od)
        for p in probes:
            p.write_file()
            # else:
            #    dprint1("skipping summarizing probe data for ", p.name)

    def set_scanner_physmodel(self, scanner):
        solvers = self.get_active_solvers()
        phys_models = []
        for s in solvers:
            for p in s.get_phys():
                if not p in phys_models:
                    phys_models.append(p)
        scanner.set_phys_models(phys_models)
        return solvers

    @debug.use_profiler
    def run(self, engine, is_first=True):
        #
        # is_first is not used
        #
        dprint1("Parametric Scan (assemly_methd=", self.assembly_method, ")")
        if self.clear_wdir:
            engine.remove_solfiles()

        engine.remove_case_dirs()

        scanner = self.get_scanner()
        if scanner is None:
            return

        solvers = self.set_scanner_physmodel(scanner)

        self.case_dirs = []
        if self.assembly_method == 0:
            self._run_full_assembly(
                engine, solvers, scanner, is_first=is_first)
        else:
            is_new_mesh = self.check_and_run_geom_mesh_gens(engine)
            if is_first or is_new_mesh:
                engine.preprocess_modeldata()
            self._run_rhs_assembly(engine, solvers, scanner, is_first=is_first)

        self.collect_probe_signals(self.case_dirs, scanner)
