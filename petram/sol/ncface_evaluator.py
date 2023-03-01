'''
   NCFaceEvaluator:
      not-continous face evaluator
'''
from petram.sol.bdr_nodal_evaluator import get_emesh_idx
from petram.sol.evaluator_agent import EvaluatorAgent
import numpy as np
import weakref
import six

from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD


from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
    from mfem.par import GlobGeometryRefiner as GR
else:
    import mfem.ser as mfem
    from mfem.ser import GlobGeometryRefiner as GR

Geom = mfem.Geometry()


def eval_on_faces(obj, expr, solvars, phys):
    '''
    evaluate nodal valus based on preproceessed 
    geometry data

    to be done : obj should be replaced by a dictionary
    '''
    from petram.helper.variables import (Variable,
                                         var_g,
                                         NativeCoefficientGenBase,
                                         CoefficientVariable,
                                         NumbaCoefficientVariable,)

    if len(obj.ifaces) == 0:
        return None
    variables = []

    code = compile(expr, '<string>', 'eval')
    names = code.co_names

    g = {}

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
            if not g[n] in obj.knowns:
                obj.knowns[g[n]] = (
                    g[n].ncface_values(ifaces=obj.ifaces,
                                       irs=obj.irs,
                                       gtypes=obj.gtypes,
                                       locs=obj.ptx,
                                       attr1=obj.elattr1,
                                       attr2=obj.elattr2,
                                       g=g, knowns=obj.knowns,
                                       mesh=obj.mesh()[obj.emesh_idx]))

            ll_name.append(name_translation[n])
            ll_value.append(obj.knowns[g[n]])
        elif (n in g):
            var_g2[n] = g[n]

    if len(ll_value) > 0:
        val = np.array([eval(code, var_g2, dict(zip(ll_name, v)))
                        for v in zip(*ll_value)])
    else:
        # if expr does not involve Varialbe, evaluate code once
        # and generate an array
        val = np.array([eval(code, var_g2)]*len(obj.ptx))

    return val

class NCFaceEvaluator(EvaluatorAgent):
    def __init__(self, battrs, **kwargs):
        super(NCFaceEvaluator, self).__init__()
        self.battrs = battrs
        self.refine = -1
        self.decimate = kwargs.pop("decimate", 1)

    def preprocess_geometry(self, battrs, emesh_idx=0, decimate=1):
        # we will ignore deciamte for a moment

        mesh = self.mesh()[emesh_idx]
        self.battrs = battrs
        self.knowns = WKD()
        self.iverts = []
        self.ifaces = []

        if mesh.Dimension() == 3:
            def f1(ibele):
                iface, o = mesh.GetBdrElementFace(ibele)
                e1 = mesh.GetFaceElementTransformations(iface).Elem1No
                return mesh.GetAttribute(e1)

            def f2(ibele):
                iface, o = mesh.GetBdrElementFace(ibele)
                e2 = mesh.GetFaceElementTransformations(iface).Elem2No
                if e2 >= 0:
                    return mesh.GetAttribute(e2)
                else:
                    return -1

            getface = mesh.GetBdrElementFace
            gettrans = mesh.GetBdrElementTransformation
            getarray = mesh.GetBdrArray
            getelement = mesh.GetBdrElement
            getbasegeom = mesh.GetBdrElementBaseGeometry
            getvertices = mesh.GetBdrElementVertices
            getattr1 = f1
            getattr2 = f2

        elif mesh.Dimension() == 2:
            def getface(x): return (x, 1)
            gettrans = mesh.GetElementTransformation
            getarray = mesh.GetDomainArray
            getelement = mesh.GetElement
            getbasegeom = mesh.GetElementBaseGeometry
            getvertices = mesh.GetElementVertices
            getattr1 = mesh.GetAttribute
            def getattr2(x): return -1
        else:
            assert False, "NCFace Evaluator is not supported for this dimension"

        x = [getarray(battr) for battr in battrs]
        if np.sum([len(xx) for xx in x]) == 0:
            return

        ibdrs = np.hstack(x).astype(int).flatten()
        self.ibeles = np.array(ibdrs)

        ptx = []
        data = []
        ridx = []
        ifaces = []
        self.gtypes = np.zeros(len(self.ibeles), dtype=int)
        self.elattr1 = np.zeros(len(self.ibeles), dtype=int)
        self.elattr2 = np.zeros(len(self.ibeles), dtype=int)

        self.irs = {}

        gtype_st = -1
        nele = 0

        p = mfem.DenseMatrix()

        for k, i in enumerate(self.ibeles):
            verts = getvertices(i)
            gtype = getbasegeom(i)
            iface, ort = getface(i)
            #Trs = mesh.GetFaceElementTransformations(iface)

            if gtype != gtype_st:
                RefG = GR.Refine(gtype, self.refine)
                ir = RefG.RefPts
                npt = ir.GetNPoints()
                ele0 = np.array(RefG.RefGeoms.ToList()).reshape(-1, len(verts))
                gtype_st = gtype
                self.irs[gtype] = ir

                # we handle quad as two triangles to handle mixed element case
                ele = []
                for eee in ele0:
                    if len(eee) == 3:
                        ele.append(eee)
                    elif len(eee) == 4:
                        x = eee[:-1]
                        y = np.array([eee[0], eee[2], eee[3]])
                        ele.extend([x, y])
                ele = np.array(ele)

            if mesh.Dimension() == 3:
                eir = mfem.IntegrationRule(ir.GetNPoints())
                fi = mesh.GetBdrFace(i)
                Transf = mesh.GetFaceElementTransformations(fi)
                Transf.Loc1.Transform(ir, eir)
                Transf.Elem1.Transform(eir, p)
                pt = p.GetDataArray().copy().transpose()
            elif mesh.Dimension() == 2:
                T = gettrans(i)
                T.Transform(ir, p)
                pt = p.GetDataArray().copy().transpose()
                #pt = np.vstack([T.Transform(ir.IntPoint(j)) for j in range(npt)])

            ptx.append(pt)

            ridx.append(ele + nele)
            nele = nele + ir.GetNPoints()
            ifaces.append(iface)
            self.gtypes[k] = gtype

            self.elattr1[k] = getattr1(i)
            self.elattr2[k] = getattr2(i)

        self.ptx = np.vstack(ptx)
        self.ridx = np.vstack(ridx)
        self.ifaces = np.hstack(ifaces)

        self.emesh_idx = emesh_idx

    def eval(self, expr, solvars, phys, **kwargs):
        refine = kwargs.pop("refine", 1)
        emesh_idx = get_emesh_idx(self, expr, solvars, phys)

        if len(emesh_idx) > 1:
            assert False, "expression involves multiple mesh (emesh length != 1)"
        if len(emesh_idx) < 1:
            # use default mesh in this case. (for non mesh dependent expression)
            emesh_idx = [0]
        #    assert False, "expression is not defined on any mesh"

        if (refine != self.refine or self.emesh_idx != emesh_idx[0]):
            self.refine = refine
            self.preprocess_geometry(self.battrs, emesh_idx=emesh_idx[0])
        val = eval_on_faces(self, expr, solvars, phys)
        if val is None:
            return None, None, None

        edge_only = kwargs.pop('edge_only', False)
        export_type = kwargs.pop('export_type', 1)

        # print self.ptx.shape, val.shape, self.ridx.shape
        if export_type == 2:
            return self.ptx, val, None
        if not edge_only:
            return self.ptx, val, self.ridx
        else:
            assert False, "NCFace does not support edge_only"
