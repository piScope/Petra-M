from petram.mfem_config import use_parallel
import os
import traceback
import gc
import numpy as np

from petram.model import Model
from petram.solver.solver_controls import SolveControl
from petram.namespace_mixin import NS_mixin
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Superposition')
format_memory_usage = debug.format_memory_usage

if use_parallel:
    import mfem.par as mfem
    from mfem.common.mpi_debug import nicePrint
else:
    import mfem.ser as mfem
    nicePrint = print


class Superposition(SolveControl, NS_mixin):
    can_delete = True
    has_2nd_panel = False

    def __init__(self, *args, **kwargs):
        SolveControl.__init__(self, *args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)

    def init_solver(self):
        pass

    def attribute_set(self, v):
        super(Superposition, self).attribute_set(v)
        v["sol_directory"] = ""
        v["sol_weight_txt"] = ""
        v["phys_model"] = ""
        v['save_parmesh'] = False
        v['save_sersol'] = False
        return v

    def panel1_param(self):
        return [
            ["Weight",   self.phys_model,  0, {}, ],
            ['Sol(default="")',   self.sol_directory,  0, {}, ],
            [None,
             self.save_parmesh,  3, {"text": "save parallel mesh"}],
            [None,  self.save_sersol,  3, {
                "text": "save serialized solution (for MPI run)"}], ]

    def get_panel1_value(self):
        return [self.sol_weight_txt,
                self.sol_directory,
                self.save_parmesh,
                self.save_sersol, ]

    def import_panel1_value(self, v):
        self.sol_weight_txt = v[0]
        self.sol_directory = v[1]
        self.save_parmesh = bool(v[2])
        self.save_sersol = bool(v[3])

    def get_target_phys(self):
        return []

    def get_child_solver(self):
        return []

    def get_matrix_weight(self, timestep_config):  # , timestep_weight):
        return [0, 0, 0]

    def get_custom_init(self):
        ret = []
        return ret

    def get_num_levels(self):
        return 1

    def create_refined_levels(self, engine, lvl):
        return False

    def free_instance(self):
        return

    def eval_weight(self):
        txt = self.sol_weight_txt
        g = self._global_ns.copy()

        try:
            # (test)
            # txt0 = "scanner.Smat"
            # xx = eval(txt0, g, self._local_ns)
            # nicePrint("Smat", xx)

            xx = eval(txt, g, self._local_ns)
            val = np.array(xx)

            return val
        except BaseException:
            import traceback
            traceback.print_exc()
            return None

    def run(self, engine, is_first=True):
        dprint1("!!!!! Entering Superposition " + self.name() + " !!!!!")

        weight = self.eval_weight()
        if weight is None:
            assert False, "Failed to evaulate weight"
        dprint1(weight)

        cwd = os.getcwd()

        soldir = os.path.abspath(self.sol_directory)
        os.chdir(soldir)
        files = os.listdir(soldir)
        cases = [(int(f.split("_")[1]), f) for f in files
                 if f.startswith('case') and os.path.isdir(f)]
        cases = [x[1] for x in sorted(cases)]

        phys_target = self.get_phys()
        self.access_idx = 0

        for phys in phys_target:
            emesh_idx = phys.emesh_idx
            emesh = engine.get_emesh(emesh_idx)
            for name in phys.dep_vars:
                fnamer, fnamei = engine.solfile_name(name, emesh_idx)
                suffix = engine.solfile_suffix()

                fnamer = fnamer+suffix
                fnamei = fnamei+suffix

                data = None
                for ii, case in enumerate(cases):
                    tmp = None
                    f = os.path.join(soldir, case, fnamer)
                    if os.path.exists(f):
                        tmp = mfem.GridFunction(emesh, f).GetDataArray()
                    f = os.path.join(soldir, case, fnamei)
                    if os.path.exists(f):
                        tmp = tmp + 1j * \
                            mfem.GridFunction(emesh, f).GetDataArray()

                    if data is None:
                        data = tmp*weight[ii]
                    else:
                        data = data + tmp*weight[ii]

                ifes = engine.r_ifes(name)
                r_x = engine.r_x[ifes]
                i_x = engine.i_x[ifes]

                r_x.GetDataArray()[:] = np.real(data)
                if i_x is not None:
                    i_x.GetDataArray()[:] = np.imag(data)

        for ii, case in enumerate(cases):
            check, val = engine.load_extra_from_file(case)
            print(val)
            if val is None:
                sol_extra = None
                break
            if ii == 0:
                sol_extra = val
                for x in sol_extra:
                    for y in sol_extra[x]:
                        sol_extra[x][y] = val[x][y]*weight[ii]

            else:
                for x in sol_extra:
                    for y in sol_extra[x]:
                        sol_extra[x][y] = (sol_extra[x][y] +
                                           val[x][y]*weight[ii])
        os.chdir(cwd)
        engine.save_sol_to_file(phys_target,
                                skip_mesh=False,
                                mesh_only=False,
                                save_parmesh=self.save_parmesh,
                                save_sersol=self.save_sersol)

        if sol_extra is not None:
            engine.save_extra_to_file(sol_extra)
