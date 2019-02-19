'''

  evaulate derivative of Nedelec element using
  DiscreteLinearOperator

'''
from petram.mfem_config import use_parallel
import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('eval_deriv')

if use_parallel:
   import mfem.par as mfem
   FiniteElementSpace = mfem.ParFiniteElementSpace
   DiscreteLinearOperator = mfem.ParDiscreteLinearOperator
   GridFunction = mfem.ParGridFunction
   getFESpace = 'ParFESpace'
   getMesh = 'GetParMesh'   
else:
   import mfem.ser as mfem
   FiniteElementSpace = mfem.FiniteElementSpace
   DiscreteLinearOperator = mfem.DiscreteLinearOperator
   GridFunction = mfem.GridFunction
   getFESpace = 'FESpace'   
   getMesh = 'GetMesh'


def eval_curl(gfr, gfi = None):
    '''
    evaluate curl 
    '''
    fes = getattr(gfr, getFESpace)()
    ordering = fes.GetOrdering()
    mesh = getattr(fes, getMesh)()
    vdim = 1
    sdim = mesh.SpaceDimension()
    p = fes.GetOrder(0)
    rt_coll = mfem.RT_FECollection(p-1, sdim)

    rts = FiniteElementSpace(mesh,  rt_coll, vdim, ordering)
    
    curl = DiscreteLinearOperator(fes, rts)
    itp = mfem.CurlInterpolator()
    curl.AddDomainInterpolator(itp)
    curl.Assemble();
    curl.Finalize();

    br   = GridFunction(rts)    
    curl.Mult(gfr,  br)
    if gfi is not None:
       bi = GridFunction(rts)           
       curl.Mult(gfi,  bi)
    else:
       bi = None
    ### needs to return rts to prevent rts to be collected.
    return br, bi, (rt_coll, rts)
 
def eval_grad(gfr, gfi = None):
    '''
    evaluate gradient
    '''
    fes = getattr(gfr, getFESpace)()
    ordering = fes.GetOrdering()
    mesh = getattr(fes, getMesh)()
    vdim = 1
    sdim = mesh.SpaceDimension()
    p = fes.GetOrder(0)
    nd_coll = mfem.ND_FECollection(p, sdim)

    nds = FiniteElementSpace(mesh,  nd_coll, vdim, ordering)
    
    grad = DiscreteLinearOperator(fes, nds)
    itp = mfem.GradientInterpolator()
    grad.AddDomainInterpolator(itp)
    grad.Assemble();
    grad.Finalize();

    br   = GridFunction(rts)    
    grad.Mult(gfr,  br)
    if gfi is not None:
       bi = GridFunction(rts)           
       grad.Mult(gfi,  bi)
    else:
       bi = None
    ### needs to return rts to prevent rts to be collected.
    return br, bi, (rt_coll, nds)
 
def eval_div(gfr, gfi = None):
    '''
    evaluate divergence
    '''
    fes = getattr(gfr, getFESpace)()
    ordering = fes.GetOrdering()
    mesh = getattr(fes, getMesh)()
    vdim = 1
    sdim = mesh.SpaceDimension()
    p = fes.GetOrder(0)
    l2_coll = mfem.L2_FECollection(p, 1)

    l2s = FiniteElementSpace(mesh,  l2_coll, vdim, ordering)
    
    div = DiscreteLinearOperator(fes, l2s)
    itp = mfem.DivergenceInterpolator()
    div.AddDomainInterpolator(itp)
    div.Assemble();
    div.Finalize();

    br   = GridFunction(rts)    
    div.Mult(gfr,  br)
    if gfi is not None:
       bi = GridFunction(rts)           
       div.Mult(gfi,  bi)
    else:
       bi = None
    ### needs to return rts to prevent rts to be collected.
    return br, bi, (rt_coll, l2s)


