import shutil
import os
import ifigure.widgets.dialog as dialog

def import_ns(model, path = None):
    from ifigure.mto.py_code import PyData
    from ifigure.mto.py_script import PyScript        

    proj = model.get_root_parent()
    if path is None:
        path = dialog.read(wildcard='piScope subtree(pfs)|*.pfs|Python Script (py)|*.py')
        if path == '': return
    if path.endswith(".pfs"):
        if not proj.has_child(name = 'mfem_tmp'):
            proj.add_folder('mfem_tmp')

        child = proj.mfem_tmp.load_subtree(path)

        nsname = '_'.join(child.get_child(idx=0).name.split('_')[:-1])
        nsname0 = nsname
        k = 0
        while model.namespaces.has_child(nsname + '_ns'):
           k = k + 1
           nsname = nsname0 + str(k)

        ns    = child.get_child(name = nsname0+'_ns')
        data  = child.get_child(name = nsname0 + '_data')

        if ns is None: return
        ns.duplicate(model.namespaces, new_name = nsname+'_ns')
        if data is not None:
            data.duplicate(model.datasets, new_name = nsname+'_data')

        proj.mfem_tmp.destroy()
    elif path.endswith('.py'):

        shutil.copy(os.path.join(path), model.namespaces.owndir())
        name = os.path.basename(path)
        sc = model.namespaces.add_childobject(PyScript, name[:-3])
        sc.load_script(os.path.join(model.namespaces.owndir(), name))
        obj =model.datasets.add_data(name[:-6]+'_data')
    else:
        raise AssertionError("Unsupported file")
        
def export_ns(model, name, path = None):
    proj = model.get_root_parent()

    ns    = model.namespaces.get_child(name = name + '_ns')
    data  = model.datasets.get_child(name = name + '_data')

    if ns is None: return
    if path is None:
        path = dialog.write(defaultfile=name+'_mfem_ns.pfs',
                            wildcard='*.pfs')
        if path == '': return
    if not proj.has_child(name = 'mfem_tmp'):
        proj.add_folder('mfem_tmp')
    tmp = proj.mfem_tmp.add_folder(name)
    ns.duplicate(tmp)
    if data is not None:
        data.duplicate(tmp)

    tmp.save_subtree(path)
    proj.mfem_tmp.destroy()
