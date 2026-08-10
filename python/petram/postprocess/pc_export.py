import numpy as np
import traceback

from petram.postprocess.dxp_model import (DataExportBase,
                                          MeshWrap,
                                          call_pointcloud_eval)
from petram.phys.vtable import VtableElement, Vtable, Vtable_mixin



import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('PointCloudExport')


data = [("pc_x", VtableElement("pc_x", type='any',
                               guilabel="x",
                               default="0",
                               no_func=True,
                               tip="x coordinates",)),
        ("pc_y", VtableElement("pc_y", type='any',
                               guilabel="y",
                               default="0",
                               no_func=True,
                               tip="y coordinates",)),
        ("pc_z", VtableElement("pc_z", type='any',
                               guilabel="z",
                               default="0",
                               no_func=True,
                               tip="z coordinates",)),
        ]

class PointCloud(DataExportBase, Vtable_mixin):
    has_2nd_panel = True
    vt_coeff = Vtable(data)

    def attribute_set(self, v):
        v = super(DataExportBase, self).attribute_set(v)
        self.vt_coeff.attribute_set(v)
        v["export_expr"] = ""
        v['sel_index'] = ['all']
        v['sel_index_txt'] = 'all'
        return v

    def panel1_param(self):
        panels = [['Expression', '', 0, {}],]
        pnls2 = self.vt_coeff.panel_param(self)
        panels.extend(pnls2)
        return panels

    def get_panel1_value(self):
        values = [self.export_expr,]
        val2 = self.vt_coeff.get_panel_value(self)
        values.extend(val2)
        return values

    def import_panel1_value(self, v):
        self.export_expr = v[0]
        self.vt_coeff.import_panel_value(self, v[1:])

    def import_panel2_value(self, v):
        self.sel_index_txt = str(v[0])
        from petram.model import convert_sel_txt
        try:
            g = self._global_ns
            arr = convert_sel_txt(self.sel_index_txt, g)
            self.sel_index = arr
        except:
            import traceback
            traceback.print_exc()
            assert False, "failed to convert "+self.sel_index_txt

    def run_dataexport(self, engine):
        dprint1("running dataexport: " + self.name())

        self.vt_coeff.preprocess_params(self)
        xx, yy, zz = self.vt_coeff.make_value_or_expression(self)

        xyz_points = np.stack([xx, yy, zz], -1)
        if self.sel_index[0] == 'all':
            attrs = list(range(1, engine.max_attr+1))
        else:
            attrs = self.sel_index[0]

        expr = self.export_expr
        solvars = engine.model._variables
        phys = engine.current_solve_step.get_phys()

        from petram.sol.pointcloud_evaluator import PointcloudEvaluator

        evltr= PointcloudEvaluator(attrs, "xyz", xyz_points)
        evltr.mesh = MeshWrap(engine.emeshes)

        ptx, result, attrs = call_pointcloud_eval(evltr, expr, solvars, phys, engine)

        if result is not None:
            return {"ptx":ptx, "data":result, "attr":attrs}
        else:
            return {}
