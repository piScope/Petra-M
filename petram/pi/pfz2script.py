import os
from os.path import expanduser, abspath
import shutil
import wx

'''
  usage

  from petram.pi.pfz2script import generate, generate_all

  path_src = '~/piscope_projects/PetraM_test/piscope_projects'
  path_dst = '~/src/TwoPiTest/PetraM_test'
  generate(path_src, path_dst, 'em3d_TEwg.pfz')
'''

def generate(path_src, path_dst, f, create_new = True):
    path_src = abspath(expanduser(path_src))
    path_dst = abspath(expanduser(path_dst))
    
    app = wx.GetApp().TopWindow
    if create_new: app.onNew(e=None, use_dialog=False)
    app.onOpen(path=os.path.join(path_src, f))
    proj = app.proj
    model = proj.setting.parameters.eval("PetraM")
    
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
            shutil.copy(mesh_path, mesh_path2)

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
