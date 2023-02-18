'''
   IntegralEvaluator:
      a thing to evaluate integral on a boundary/domain
'''
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
    
from petram.sol.evaluator_agent import EvaluatorAgent
Geom = mfem.Geometry()

def do_integration(expr, solvars, phys, mesh, kind, attrs,
                   order, num):
    from petram.helper.variables import (Variable,
                                         var_g,
                                         NativeCoefficientGenBase,
                                         CoefficientVariable)
    from petram.phys.coefficient import SCoeff

    code = compile(expr, '<string>', 'eval')
    names = code.co_names

    g = {}
    #print solvars.keys()
    for key in phys._global_ns.keys():
       g[key] = phys._global_ns[key]
    for key in solvars.keys():
       g[key] = solvars[key]

    l = var_g.copy()

    ind_vars = ','.join(phys.get_independent_variables())

    if kind == 'Domain':
        size = max(max(mesh.attributes.ToList()), max(attrs))
    else:
        size = max(max(mesh.bdr_attributes.ToList()), max(attrs))
        
    arr = [0]*(size)
    for k in attrs:
        arr[k-1] = 1
    flag = mfem.intArray(arr)

    s = SCoeff(expr, ind_vars, l, g, return_complex=False)

    ## note L2 does not work for boundary....:D
    if kind == 'Domain':
        fec = mfem.L2_FECollection(order, mesh.Dimension())
    else:
        fec = mfem.H1_FECollection(order, mesh.Dimension())

    fes = mfem.FiniteElementSpace(mesh, fec)
    one = mfem.ConstantCoefficient(1)

    gf = mfem.GridFunction(fes)
    gf.Assign(0.0)

    if kind == 'Domain':
        gf.ProjectCoefficient(mfem.RestrictedCoefficient(s, flag))
    else:
        gf.ProjectBdrCoefficient(mfem.RestrictedCoefficient(s, flag), flag)

    b = mfem.LinearForm(fes)
    one = mfem.ConstantCoefficient(1)
    if kind == 'Domain':
        itg = mfem.DomainLFIntegrator(one)
        b.AddDomainIntegrator(itg)
    else:
        itg = mfem.BoundaryLFIntegrator(one)
        b.AddBoundaryIntegrator(itg)

    b.Assemble()

    from petram.engine import SerialEngine
    en = SerialEngine()
    ans = mfem.InnerProduct(en.x2X(gf), en.b2B(b))
    
    if not np.isfinite(ans):
        print("not finite", ans, arr)
        print(size, mesh.bdr_attributes.ToList())
        from mfem.common.chypre import LF2PyVec, PyVec2PyMat, Array2PyVec, IdentityPyMat
        #print(list(gf.GetDataArray()))
        print(len(gf.GetDataArray()), np.sum(gf.GetDataArray()))
        print(np.sum(list(b.GetDataArray())))
    return ans

class IntegralEvaluator(EvaluatorAgent):
    def __init__(self, battrs, decimate=1):
        super(IntegralEvaluator, self).__init__()
        self.battrs = battrs
        self.decimate = decimate

    def eval_integral(self, expr, solvars, phys,
                      kind='domain', attrs='all', order=2, num=-1):

        from .bdr_nodal_evaluator import get_emesh_idx

        emesh_idx = get_emesh_idx(self, expr, solvars, phys)

        if len(emesh_idx) > 1:
            assert False, "expression involves multiple mesh (emesh length != 1)"

        mesh = self.mesh()[emesh_idx[0]]
        itg = do_integration(expr, solvars, phys, mesh, kind, attrs, order, num)

        return itg
