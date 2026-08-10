import numpy as np
import traceback

from petram.postprocess.dxp_model import (DataExportBase,
                                          MeshWrap,
                                          call_pointcloud_eval)
from petram.phys.vtable import VtableElement, Vtable, Vtable_mixin
import traceback

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('SliceExport')


data = [("pc_abcd", VtableElement("pc_abcd", type='any',
                                  guilabel="plane (a, b, c, d)",
                                  default="0, 0, 1, 0",
                                  no_func=True,
                                  tip="slice surface (ax + by + cz + d =0)",)),
        ("pc_ax1", VtableElement("pc_ax1", type='any', guilabel="1st axis",
                                 default="1, 0, 0", no_func=True,
                                 tip="1st axis on the cut-plane",)),
        ("pc_res", VtableElement("pc_res", type='float',
                                 guilabel="resolution",
                                 default=0.01,
                                 tip="cut-plane mesh size",)),
        ]


class Slice(DataExportBase, Vtable_mixin):
    has_2nd_panel = True
    vt_coeff = Vtable(data)

    def attribute_set(self, v):
        v = super(Slice, self).attribute_set(v)
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
        engine.show_variables()
        self.vt_coeff.preprocess_params(self)
        abcd, ax1, res = self.vt_coeff.make_value_or_expression(self)
        ax1 = ax1/np.linalg.norm(ax1)

        if self.sel_index[0] == 'all':
            attrs = list(range(1, engine.max_attr+1))
        else:
            attrs = self.sel_index[0]

        expr = self.export_expr
        solvars = engine.model._variables
        phys = engine.current_solve_step.get_phys()

        from petram.helper.cutplane import process_abcd
        abcd_txt = self.pc_abcd_txt
        ns = self._global_ns
        planes = process_abcd(abcd_txt, ns)

        if planes is None:
            dprint1("plane is not defined from given a, b, c, and d")
            return
        planes = [x for x in zip(*planes)]

        # find boundinb box
        mesh = engine.emeshes[0]
        bbox1, bbox2 = mesh.GetBoundingBox()
        if len(bbox1) == 2:
            bbx1 = ((bbox1[0], bbox1[1], 0.0),
                    (bbox1[0], bbox2[1], 0.0),
                    (bbox2[0], bbox1[1], 0.0),
                    (bbox2[0], bbox2[1], 0.0),)
            bbx1 = np.transpose(bbx1)
            sdim = 2
        elif len(bbox1) == 3:
            bbx1 = ((bbox1[0], bbox1[1], bbox1[2]),
                    (bbox1[0], bbox1[1], bbox2[2]),
                    (bbox1[0], bbox2[1], bbox1[2]),
                    (bbox1[0], bbox2[1], bbox2[2]),
                    (bbox2[0], bbox1[1], bbox1[2]),
                    (bbox2[0], bbox1[1], bbox2[2]),
                    (bbox2[0], bbox2[1], bbox1[2]),
                    (bbox2[0], bbox2[1], bbox2[2]), )
            bbx1 = np.transpose(bbx1)
            sdim = 3

        # evaluate over each plane
        from petram.sol.pointcloud_evaluator import PointcloudEvaluator
        from petram.helper.mpi_recipes import gather_masked_array

        values = {}

        for iplane, abcd in enumerate(planes):
            n1 = abcd[0:3]/np.linalg.norm(abcd[0:3])

            origin = -n1*abcd[3]
            e2 = np.cross(n1, ax1)
            e1 = np.cross(e2, n1)

            tmp = e1.dot(bbx1)
            xmin = np.floor(np.min(tmp)/res)*res
            xmax = (np.floor(np.max(tmp)/res)+1)*res

            tmp = e2.dot(bbx1)
            ymin = np.floor(np.min(tmp)/res)*res
            ymax = (np.floor(np.max(tmp)/res)+1)*res

            pc_param = (origin,
                        e1,
                        e2,
                        (xmin, xmax, res),
                        (ymin, ymax, res))

            evltr = PointcloudEvaluator(attrs, "cutplane", pc_param)
            evltr.mesh = MeshWrap(engine.emeshes)

            ptx, result, attrs = call_pointcloud_eval(evltr, expr, solvars,
                                                      phys, engine)

            if result is not None:
                key0 = "plane"+str(iplane+1)+"_"
                values[key0 + "ptx"] = ptx
                values[key0 + "data"] = result
                values[key0 + "attr"] = attrs

        from petram.helper.get_myrank import get_myrank
        myid = get_myrank()

        if myid == 0:
            return values
        return {}
