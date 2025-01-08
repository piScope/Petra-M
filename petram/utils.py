'''
   utility routines
'''
import os
import numpy as np
import resource
from petram.mfem_config import use_parallel
import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Utils')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


def file_write(fid, *args):
    txt = ' '.join([str(x) for x in args])
    print(txt)
    fid.write(txt + "\n")


def print_mem(myid=0):
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("memory usage (" + str(myid) + "): " + str(mem))


def set_array_attribute(v, base, suffix, values):
    for s, vv in zip(suffix, values):
        v[base + s] = vv
    return v


def txt2indexlist(txt):
    try:
        return [int(x) for x in txt.split(',')]
    except:
        raise ValueError("can not convert text to index list")


def eval_expression(expr, mesh, battr, ind_vars, ns, use_dom=False):
    '''
    example:
        expr = 'x', or 'sin(x)'
        ibdry = 3 (boundary attribute number)
        ind_vars ['x', 'y', 'z']
    '''
    def _do_eval(code, verts):
        l = {n: verts[k] for k, n in enumerate(ind_vars)}
        return eval(code, ns, l)

    if use_dom:
        get_array = mesh.GetDomainArray
        get_element = mesh.GetElement
    else:
        get_array = mesh.GetBdrArray
        get_element = mesh.GetBdrElement

    ibdr = get_array(battr)

    code = compile(expr, '<string>', 'eval')

    iverts = np.stack([get_element(i).GetVerticesArray() for i in ibdr])
    locs = np.stack([np.stack([mesh.GetVertexArray(k) for k in ivert])
                     for ivert in iverts])
    data = np.stack([np.stack([_do_eval(code, mesh.GetVertexArray(k))
                               for k in ivert])
                     for ivert in iverts])
    return locs, data


def eval_expr(model, engine, expr, battrs, phys=None):
    '''
    expr = expression to evaluate
    battrs = list of boundary attribute
    '''
    from petram.model import Bdry

    if phys is None:
        phys = model['Phys'][list(model['Phys'])[0]]
    else:
        phys = model['Phys'][phys]

    ret = {}
    mesh = engine.get_mesh()
    engine.assign_sel_index(phys)

    use_dom = (phys.dim == 2 or phys.dim == 1)
    ns = phys._global_ns
    battrs = list(np.atleast_1d(eval(battrs, ns, {})).flatten().astype(int))

    for m in phys.walk():
        for battr in battrs:
            ind_vars = m.get_independent_variables()
            ret[battr] = eval_expression(
                expr, mesh, battr, ind_vars, ns, use_dom=use_dom)
    return ret


def eval_sol(sol, battrs, dim=0):
    mesh = sol.FESpace().GetMesh()
    ibdr = np.hstack([mesh.GetBdrArray(battr) for battr in battrs])
    if len(ibdr) == 0:
        return None, None
    iverts = np.stack([mesh.GetBdrElement(i).GetVerticesArray() for i in ibdr])
    locs = np.stack([np.stack([mesh.GetVertexArray(k) for k in ivert])
                     for ivert in iverts])

    data = sol.GetNodalValues(dim)
    data = data[iverts.flatten()].reshape(iverts.shape)

    return locs, data


def eval_loc(sol, battrs):
    mesh = sol.FESpace().GetMesh()
    ibdr = np.hstack([mesh.GetBdrArray(battr) for battr in battrs])
    if len(ibdr) == 0:
        return None, None
    iverts = np.stack([mesh.GetBdrElement(i).GetVerticesArray() for i in ibdr])
    locs = np.stack([np.stack([mesh.GetVertexArray(k) for k in ivert])
                     for ivert in iverts])
    return locs


def get_pkg_datafile(pkg, *path):
    '''
    return package data

    ex) get_pkg_datafile(petram.geom, 'icon', 'line.png')
    '''
    file = getattr(pkg, '__file__')
    root = os.path.abspath(os.path.dirname(file))
    return os.path.join(os.path.dirname(root), 'data', *path)


def get_evn_petram_root():
    petram = os.getenv("PetraM")
    return petram


def get_evn_twopiroot():
    petram = os.getenv("TwoPiRoot")
    return petram


def check_cluster_access():
    petram = get_evn_twopiroot()

    if petram is None:
        from ifigure.ifigure_config import rcdir as petram

    return os.path.exists(os.path.join(petram, "etc", "cluster_access"))


def check_addon_access():
    petram = get_evn_twopiroot()

    if petram is None:
        from ifigure.ifigure_config import rcdir as petram        

    if os.path.exists(os.path.join(petram, "etc", "addon_access")):
        return "any"
    return "none"


''''
 pv (paired value) handling (used in processing GUI data)
'''


def pv_get_gui_value(mm, paired_var):
    '''
    return value for GUI panel
    '''
    mfem_physroot = mm.get_root_phys().parent
    paired_var = mm.paired_var
    if paired_var is not None:
        try:
            var_s = mfem_physroot[paired_var[0]].dep_vars
            name1 = var_s[paired_var[1]]
            model1 = paired_var[0]
        except BaseException:
            paired_var = None

    if paired_var is None:
        names = mm.get_root_phys().dep_vars
        if len(names) > 0:
            name1 = names[0]
        else:
            name1 = ""
        model1 = mm.get_root_phys().name()
    var = name1 + " ("+model1 + ")"
    return var, paired_var


def pv_from_gui_value(mm, value):
    '''
    return paired_var from GUI input
    '''
    mfem_physroot = mm.get_root_phys().parent
    names, pnames, pindex = mfem_physroot.dependent_values(include_disabled=True)

    if len(value) == 0:
        # v[0] could be '' if object is based to a tree.
        idx = 0
    else:
        tmp = str(value).split("(")[0].strip()
        if tmp in names:
            idx = names.index(tmp)
        else:
            idx = 0

    if len(pnames) == 0:
        return None
    paired_var = (pnames[idx], pindex[idx])
    return paired_var


def pv_panel_param(mm, label):
    '''
    panel value for paired_var
    '''
    from wx import CB_READONLY
    mfem_physroot = mm.get_root_phys().parent
    names, pnames, _pindex = mfem_physroot.dependent_values(include_disabled=True)
    names = [n+" ("+p + ")" for n, p in zip(names, pnames)]

    ll1 = [label, "S", 4,
           {"style": CB_READONLY, "choices": names}]
    return ll1


''''
 pm (paired model) handling (used in processing GUI data)
'''


def pm_get_gui_value(mm, paired_model):
    '''
    return value for GUI panel
    '''
    mfem_physroot = mm.get_root_phys().parent

    paired_model = mm.paired_model
    if paired_model not in mfem_physroot:
        model1 = mm.get_root_phys().name()
        paired_model = None
    else:
        model1 = paired_model
    return model1, paired_model


def pm_from_gui_value(mm, value):
    '''
    return paired_model from GUI input
    '''
    mfem_physroot = mm.get_root_phys().parent
    names = [x.name() for x in mfem_physroot.get_children()]

    if len(names) == 0:
        return None

    if len(value) == 0:
        # v[0] could be '' if object is based to a tree.
        idx = 0
    else:
        if value in names:
            idx = names.index(value)
        else:
            idx = 0

    paired_model = names[idx]
    return paired_model


def pm_panel_param(mm, label):
    '''
    panel value for paired_var
    '''
    from wx import CB_READONLY
    mfem_physroot = mm.get_root_phys().parent
    names = [x.name() for x in mfem_physroot.get_children()]

    ll1 = [label, "S", 4,
           {"style": CB_READONLY, "choices": names}]
    return ll1
