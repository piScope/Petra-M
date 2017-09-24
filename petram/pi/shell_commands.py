#
#  petra 
#
#   piScope command to start PetraM
#
import os

def petra(reload_scripts = False):
    '''
    setup PetraM simulation enveroment
    '''
    from __main__ import ifig_app    
    proj = ifig_app.proj
    if proj.setting.parameters.hasvar('PetraM'):
        model = proj.setting.parameters.eval('PetraM')
    else:
        try:
            model = proj.model1.mfem
            scripts = model.scripts
            from ifigure.mto.hg_support import has_repo
            if has_repo(scripts):
                scripts.onHGturnoff(evt=None, confirm = False)
                model.param.setvar('remote', None)
            reload_scripts = True
        except:
            model = load_petra_model(proj)
    if reload_scripts:
        scripts = model.scripts
        for name, child in scripts.get_children():
            child.destroy()
        scripts.clean_owndir()
        import_project_scripts(scripts)

    if model is not None:
        model.scripts.helpers.open_gui()
        proj.setting.parameters.setvar('PetraM', '='+model.get_full_path())

def load_petra_model(proj):


    model_root = proj.onAddModel()
    model = model_root.add_model('mfem')
    model.onAddNewNamespace(e = None)

    model.param.setvar('nproc', 2)
    model.add_folder('namespaces')
    model.add_folder('datasets')
    model.add_folder('solutions')
    scripts = model.add_folder('scripts')
    import_project_scripts(scripts)

    scripts.helpers.reset_model()
    model.set_guiscript('.scripts.helpers.open_gui')
    model.scripts.helpers.create_ns('global')

    param = model.param
    param.setvar('mfem', None)
    param.setvar('sol', None)
    param.setvar('remote', None)

    return model

def import_project_scripts(scripts):
    import petram.pi.project_scripts

    path =os.path.dirname(petram.pi.project_scripts.__file__)
    scripts.load_script_folder(path, skip_underscore=True)


