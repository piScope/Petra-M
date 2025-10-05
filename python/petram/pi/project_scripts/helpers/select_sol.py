def select_sol(solname = ''):
    if not model.solutions.has_child(solname): return

    sol = model.solutions.get_child(name = solname)
    model.param.setvar(sol, '='+sol.get_full_path())

    model.scripts.helpers.read_sol()
    return sol
ans(select_sol(*args, **kwargs))
