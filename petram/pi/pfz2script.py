import os
from os.path import expanduser, abspath
import shutil
import wx
import time

from ifigure.utils.mp_tarzip import MPTarzip
'''
  usage

  from petram.pi.pfz2script import generate, generate_all

  path_src = '~/piscope_projects/PetraM_test/piscope_projects'
  path_dst = '~/src/TwoPiTest/PetraM_test'
  generate(path_src, path_dst, 'em3d_TEwg.pfz')
'''

def generate(path_src, path_dst, f, create_new = True, reload_scripts=False):
    #  script generater 
    #    create_new : start a new project and load the project file
    #
    path_src = abspath(expanduser(path_src))
    path_dst = abspath(expanduser(path_dst))
    
    app = wx.GetApp().TopWindow
    if create_new: app.onNew(e=None, use_dialog=False)
    app.onOpen(path=os.path.join(path_src, f))
    proj = app.proj
    model = proj.setting.parameters.eval("PetraM")

    from petram.pi.shell_commands import import_project_scripts
    if reload_scripts:
        scripts = model.scripts
        for name, child in scripts.get_children():
            child.destroy()
        scripts.clean_owndir()
        import_project_scripts(scripts)
        app.onSave(None)
        while not MPTarzip().isReady():
            time.sleep(3)
            
    
    m = proj.model1.mfem.param.eval("mfem_model")
    m.set_root_path(model.owndir())
    
    for od in m.walk():
        if hasattr(od, 'use_relative_path'):
            od.use_relative_path()
            
    dst = os.path.join(path_dst, f.split(".")[0])
    if not os.path.exists(dst):
        os.mkdir(dst)
    m.generate_script(dir=dst)

    for od in m.walk():
        if hasattr(od, 'use_relative_path'):
            od.restore_fullpath()
            mesh_path = od.get_real_path()
            mesh_path2 = os.path.join(dst, os.path.basename(od.path))
            if os.path.exists(mesh_path2): continue
            if os.path.exists(mesh_path): shutil.copy(mesh_path, mesh_path2)

def generate_all(path_src, path_dst):
    '''
     process all pfz file in a directoy

    '''
    path_src = abspath(expanduser(path_src))    
    files = os.listdir(path_src)
    create_new = True
    for f in files:
        if not f.endswith('.pfz'): continue
        print("working on ", f)
        generate(path_src, path_dst, f, create_new=create_new)
        create_new = False
