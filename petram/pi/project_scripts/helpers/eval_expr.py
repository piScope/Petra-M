def eval_expr(expr, battrs, phys = None):
    '''
    expr = expression to evaluate
    battrs = list of boundary attribute
    '''
    from petram.model import Bdry
    from petram.utils import eval_expression
    m0 = model.param.getvar('mfem_model')
    if phys is None: 
        phys = m0['Phys'][m0['Phys'].keys()[0]]
    else:
        phys = m0['Phys'][phys]

    ret = {}
    engine.assign_sel_index(phys)
    for m in phys.walk():
       if isinstance(m, Bdry):
          for battr in battrs:
             #if battr in m._sel_index and not battr in ret:
              ns = m._global_ns
              ind_vars = m.get_independent_variables()
              ret[battr] = eval_expression(expr, mesh, battr, ind_vars, ns)
    return ret
ans(eval_expr(*args, **kwargs))
