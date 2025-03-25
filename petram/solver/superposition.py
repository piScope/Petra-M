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
    from mfem.common.mpi_debug import nicePrint
else:
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
        v["sol_weight_txt"] = ""
        v["phys_model"] = ""
        return v

    def panel1_param(self):
        return [
            ["physics model",   self.phys_model,  0, {}, ],
            ["Weight",   self.phys_model,  0, {}, ], ]

    def get_panel1_value(self):
        return (self.phys_model,
                self.sol_weight_txt)

    def import_panel1_value(self, v):
        self.init_setting = v[0]
        self.sol_weight_txt = v[1]

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

        files = os.listdir(os.getcwd())
        cases = [(int(f.split("_")[1]), f) for f in files
                 if f.startswith('case') and os.path.isdir(f)]
        cases = [x[1] for x in sorted(cases)]
        print(cases)
        phys_target = self.get_phys()
        self.access_idx = 0
        for phys in phys_target:
            emesh_idx = phys.emesh_idx
            for name in phys.dep_vars:
                fnamer, fnamei = engine.solfile_name(name, emesh_idx)
                suffix = engine.solfile_suffix()

                fnamer = fnamer+suffix
                fnamei = fnamei+suffix
                nicePrint(fnamer, fnamei)

                ifes = engine.r_ifes(name)
                r_x = engine.r_x[ifes]
                i_x = engine.i_x[ifes]

                print(r_x)
