#
#  petra
#
#   piScope command to start PetraM
#
import os


class PetraMHelper(object):
    '''
    configurator/helper
    '''

    def __init__(self, p=None):
        object.__init__(self)
        self.properties = {'refine': 5}

        if p is not None:
            for k in p:
                if k in self.properties:
                    self.properties[k] = p[k]
        for k in self.properties:
            self.set(k, self.properties[k])

    def set(self, name, value):
        if name == 'refine':
            import petram.mesh.refined_mfem_geom
            petram.mesh.refined_mfem_geom.default_refine = value

    def set_debug(self, name, value=True):
        import petram.debug
        assert hasattr(petram.debug, "debug_"+name), "debug_" + \
            name+" is not valid debug option."
        setattr(petram.debug, "debug_"+name, value)

    def show(self):
        for k in self.properties:
            print(k + ' : ' + str(self.properties[k]))

    def versions(self):
        import mfem
        import wx
        import matplotlib
        import OCC

        print("WX:", wx.__version__)
        print("Matplotlib:", matplotlib.__version__)
        print("OCC:", OCC.VERSION)
        print("MFEM:", mfem.__version__)


def petram(reload_scripts=False, addon="none", cluster=False):
    '''
    setup PetraM simulation enveroment

       cluster : turn-on cluster access menu, temporariliy.
       reload_scripts : reload project sciripts
       addon : turn-on addon access, temporariliy.
    '''
    import petram.mfem_model as mm
    if not mm.has_cluster_access and cluster is True:
        mm.has_cluster_access = True
    if mm.has_addon_access == "none":
        mm.has_addon_access = addon

    import wx
    ifig_app = wx.GetApp().TopWindow
    proj = ifig_app.proj
    if proj.setting.parameters.hasvar('PetraM'):
        model = proj.setting.parameters.eval('PetraM')
    else:
        try:
            model = proj.model1.mfem
            scripts = model.scripts
            from ifigure.mto.hg_support import has_repo
            if has_repo(scripts):
                scripts.onHGturnoff(evt=None, confirm=False)
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
        if not model.has_child('mfembook'):
            book = model.add_book('mfembook')
            ipage = book.add_page()
            book.get_page(ipage).add_axes()
            book.set_keep_data_in_tree(True)

        from petram.mfem_viewer import MFEMViewer
        viewer = model.mfembook.Open(MFEMViewer)
        viewer.isec(0)
        viewer.threed('on')
        viewer.view('noclip')
        proj.setting.parameters.setvar('PetraM', '='+model.get_full_path())

        import wx
        wx.CallAfter(model.mfembook.find_bookviewer().Raise)

    return PetraMHelper()


def load_petra_model(proj):

    model_root = proj.onAddModel()
    model = model_root.add_model('mfem')
    model.onAddNewNamespace(e=None)

    model.param.setvar('nproc', 2)
    model.param.setvar('openmp_num_threads', 'auto')
    model.param.setvar('openblas_num_threads', 1)
    model.add_folder('namespaces')
    model.add_folder('datasets')
    model.add_folder('solutions')
    scripts = model.add_folder('scripts')
    import_project_scripts(scripts)

    scripts.helpers.reset_model()
    # model.set_guiscript('.scripts.helpers.open_gui')
    model.scripts.helpers.create_ns('global')

    param = model.param
    param.setvar('mfem', None)
    param.setvar('sol', None)
    param.setvar('remote', None)

    return model


def import_project_scripts(scripts):
    import petram.pi.project_scripts

    path = os.path.dirname(petram.pi.project_scripts.__file__)
    scripts.load_script_folder(path, skip_underscore=True)
