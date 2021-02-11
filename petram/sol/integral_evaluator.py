'''
   IntegralEvaluator:
      a thing to evaluate integral on a boundary/domain
'''
import numpy as np
import parser
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

    st = parser.expr(expr)
    code= st.compile('<string>')
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
        size = max(mesh.attributes.ToList())
    else:
        size = max(mesh.bdr_attributes.ToList())
    arr = [0]*size

    for k in attrs: arr[k-1] = 1
    flag = mfem.intArray(arr)

    s = SCoeff(expr, ind_vars, l, g, return_complex=False)

    ## note L2 does not work for boundary....:D
    if kind == 'Domain':
        fec = mfem.L2_FECollection(order, mesh.Dimension())
    else:
        fec = mfem.H1_FECollection(order, mesh.Dimension())

    fes = mfem.FiniteElementSpace(mesh, fec)
    gf = mfem.GridFunction(fes)

    if kind == 'Domain':
        gf.ProjectCoefficient(mfem.RestrictedCoefficient(s, flag))
    else:
        gf.ProjectBdrCoefficient(s, flag)

    b = mfem.LinearForm(fes)
    one = mfem.ConstantCoefficient(1)
    if kind == 'Domain':
        itg = mfem.DomainLFIntegrator(one)
        b.AddDomainIntegrator(itg)
    else:
        itg = mfem.BoundaryLFIntegrator(one)
        b.AddBoundaryIntegrator(itg, flag)

    b.Assemble()
    ans = mfem.InnerProduct(gf, b)

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
