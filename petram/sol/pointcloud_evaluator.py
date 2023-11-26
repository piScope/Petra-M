import numpy as np
import scipy
import six
import weakref
from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD


from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

from petram.sol.evaluator_agent import EvaluatorAgent
from petram.sol.bdr_nodal_evaluator import process_iverts2nodals
from petram.sol.bdr_nodal_evaluator import eval_at_nodals, get_emesh_idx


class PointcloudEvaluator(EvaluatorAgent):
    def __init__(self, attrs, pc_type=None, pc_param=None):
        '''
           attrs = [1,2,3]
           plane = [ax, ay, az, c]

           cut-plane is defined as
           ax * x + ay * y + ax * z + c = 0
        '''
        super(PointcloudEvaluator, self).__init__()
        self.points = None
        self.subset = None
        self.attrs = set(attrs)
        self.pc_type = pc_type
        self.pc_param = pc_param

    def preprocess_geometry(self, attrs, emesh_idx=0, pc_type=None,
                            pc_param=None):

        from petram.helper.geom import generate_pc_from_cpparam

        self.attrs = attrs
        if pc_param is not None:
            pc_param = self.pc_param
            pc_type = self.pc_type

        if pc_type == 'cutplane':  # cutplane
            param = {"origin": pc_param[0], "e1": pc_param[1], "e2": pc_param[2],
                     "x": pc_param[3], "y": pc_param[4]}
            cp_abc = np.cross(pc_param[1], pc_param[2])
            cp_d = -np.sum(cp_abc*pc_param[0])
            points = generate_pc_from_cpparam(**param)

        elif pc_type == 'line':
            sp = np.array(pc_param[0])
            ep = np.array(pc_param[1])
            num = pc_param[2]

            ii = np.linspace(0, 1., num)
            points = np.vstack([sp * (1-i) + ep * i for i in ii])

        elif pc_type == 'xyz':
            points = pc_param

        mesh = self.mesh()[emesh_idx]
        sdim = mesh.SpaceDimension()

        if points.shape[-1] > sdim:
            points = points[..., :sdim]

        if np.prod(points.shape) == 0:
            assert False, "PointCloud: Number of points = 0"
            return

        self.ans_shape = points.shape
        self.ans_points = points
        self.points = points.reshape(-1, points.shape[-1])

        v = mfem.Vector()
        mesh.GetVertices(v)
        vv = v.GetDataArray()

        vv = vv.reshape(sdim, -1)
        max_mesh_ptx = np.max(vv, 1)
        min_mesh_ptx = np.min(vv, 1)

        max_ptx = np.max(self.points, 0)
        min_ptx = np.min(self.points, 0)

        out_of_range = False

        for i in range(len(max_mesh_ptx)):
            if max_mesh_ptx[i] < min_ptx[i]:
                out_of_range = True
            if min_mesh_ptx[i] > max_ptx[i]:
                out_of_range = True

        self.subset = None

        if pc_type == "cutplane" and sdim == 3:
            # in 3D, we try to cut down the number of point query to FindPoints
            param = vv[0, :]*cp_abc[0] + vv[1, :] * \
                cp_abc[1] + vv[2, :]*cp_abc[2] + cp_d
            if np.max(param)*np.min(param) == 0:
                out_of_range = True

            else:
                x1 = np.sum((vv.transpose() - pc_param[0])*pc_param[1], -1)
                xxx = np.min(x1), np.max(x1)
                y1 = np.sum((vv.transpose() - pc_param[0])*pc_param[2], -1)
                yyy = np.min(y1), np.max(y1)

                xmin, xmax, xsize = pc_param[3]
                ymin, ymax, ysize = pc_param[4]
                xxx = [int((xxx[0]-xmin)//xsize), int((xxx[1]-xmin)//xsize)]
                yyy = [int((yyy[0]-ymin)//ysize), int((yyy[1]-ymin)//ysize)]

                ss = self.ans_points.shape

                if xxx[0] < 0:
                    xxx[0] = 0
                if yyy[0] < 0:
                    yyy[0] = 0
                if xxx[1] >= ss[1]:
                    xxx[1] = ss[1]-1
                if yyy[1] >= ss[0]:
                    yyy[1] = ss[0]-1
                if xxx[0] > 0:
                    xxx[0] = xxx[0] - 1
                if yyy[0] > 0:
                    yyy[0] = yyy[0] - 1
                if xxx[1] < self.ans_points.shape[1]-1:
                    xxx[1] = xxx[1] + 1
                if yyy[1] < self.ans_points.shape[0]-1:
                    yyy[1] = yyy[1] + 1

                subset = np.zeros(ss[:-1], dtype=bool)
                subset[yyy[0]:yyy[1], xxx[0]:xxx[1]] = True

                ptx = self.ans_points[yyy[0]:yyy[1], :, :]
                ptx = ptx[:, xxx[0]:xxx[1], :]
                self.points = ptx.reshape(-1, self.ans_points.shape[-1])
                self.subset = subset

        if out_of_range:
            counts = 0
            elem_ids = np.zeros(len(self.points), dtype=int)-1
            int_points = [None]*len(self.points)
            print("skipping mesh")
        else:
            print("Chekcing " + str(len(self.points)) + " points")
            counts, elem_ids, int_points = mesh.FindPoints(
                self.points, warn=False)
            print("FindPoints found " + str(counts) + " points")
        attrs = [mesh.GetAttribute(id) if id != -1 else -1 for id in elem_ids]
        attrs = np.array([i if i in self.attrs else -1 for i in attrs])

        elem_ids = [-1 if a == -1 else eid for a, eid in zip(attrs, elem_ids)]
        counts = np.sum(np.array(elem_ids) != -1)

        self.elem_ids = elem_ids
        self.masked_attrs = attrs

        self.int_points = int_points
        self.counts = counts

        idx = np.where(attrs != -1)[0]
        self.locs = self.points[idx]

        self.valid_idx = idx
        self.emesh_idx = emesh_idx
        self.knowns = WKD()

    def eval_at_points(self, expr, solvars, phys):
        from petram.helper.variables import (Variable,
                                             var_g,
                                             NativeCoefficientGenBase,
                                             CoefficientVariable,
                                             NumbaCoefficientVariable)

        variables = []
        code = compile(expr, '<string>', 'eval')
        names = code.co_names

        g = {}
        # print solvars.keys()
        for key in phys._global_ns.keys():
            g[key] = phys._global_ns[key]
        for key in solvars.keys():
            g[key] = solvars[key]

        ll_name = []
        ll_value = []
        var_g2 = var_g.copy()

        new_names = []
        name_translation = {}

        all_names = list(names[:])

        def get_names(names):
            for n in names:
                if (n in g and isinstance(g[n], Variable)):
                    new_names = g[n].get_names()
                    for x in new_names:
                        all_names.append(x)
                    get_names(new_names)
        get_names(names)

        for n in all_names:
            if (n in g and isinstance(g[n], NativeCoefficientGenBase)):
                g[n+"_coeff"] = CoefficientVariable(g[n], g)
                new_names.append(n+"_coeff")
                name_translation[n+"_coeff"] = n

            elif (n in g and isinstance(g[n], NumbaCoefficientVariable)):
                ind_vars = [xx.strip() for xx in phys.ind_vars.split(',')]
                g[n].set_coeff(ind_vars, g)
                new_names.append(n)
                name_translation[n] = n

            elif (n in g and isinstance(g[n], Variable)):
                new_names.extend(g[n].dependency)
                new_names.append(n)
                name_translation[n] = n

            elif n in g:
                new_names.append(n)
                name_translation[n] = n

        for n in new_names:
            if (n in g and isinstance(g[n], Variable)):
                if not g[n] in self.knowns:
                    self.knowns[g[n]] = g[n].point_values(counts=self.counts,
                                                          locs=self.locs,
                                                          attrs=self.masked_attrs,
                                                          elem_ids=self.elem_ids,
                                                          mesh=self.mesh()[
                                                              self.emesh_idx],
                                                          int_points=self.int_points,
                                                          g=g,
                                                          knowns=self.knowns)

                #ll[n] = self.knowns[g[n]]
                ll_name.append(name_translation[n])
                ll_value.append(self.knowns[g[n]])
            elif (n in g):
                var_g2[n] = g[n]

        if len(ll_value) > 0:
            val = np.array([eval(code, var_g2, dict(zip(ll_name, v)))
                            for v in zip(*ll_value)])
        else:
            # if expr does not involve Varialbe, evaluate code once
            # and generate an array
            val = np.array([eval(code, var_g2)]*len(self.locs))

        return val

    def eval(self, expr, solvars, phys):
        from petram.sol.bdr_nodal_evaluator import get_emesh_idx

        emesh_idx = get_emesh_idx(self, expr, solvars, phys)
        if len(emesh_idx) > 1:
            assert False, "expression involves multiple mesh (emesh length != 1)"

        if len(emesh_idx) == 1:
            if self.emesh_idx != emesh_idx[0]:
                self.preprocess_geometry(self.attrs, emesh_idx=emesh_idx[0],
                                         pc_type=self.pc_type,
                                         pc_param=self.pc_param)

        if self.counts == 0:
            return None, None, None

        val = self.eval_at_points(expr, solvars, phys)

        if val is None:
            return None, None, None

        shape = self.ans_shape[:-1]

        if self.subset is None:
            attrs = self.masked_attrs.reshape(shape)
        else:
            attrs = np.zeros(self.ans_shape[:-1], dtype=int)-1
            attrs[self.subset] = self.masked_attrs.reshape(
                self.points.shape[:-1])

        return self.ans_points, val, attrs
