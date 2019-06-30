import os
import shutil
import petram.helper.pickle_wrapper as pickle

from os.path import expanduser

def save_test(model):
    from ifigure.widgets.dialog import writedir
    path = writedir(message='Enter Save dirctory')
    if path == '': return
    path = expanduser(path)

    for od in model.walk():
         if hasattr(od, 'use_relative_path'):
             mpath = od.get_real_path()             
             od.use_relative_path()
             shutil.copyfile(mpath,
                             os.path.join(path, os.path.basename(mpath)))

    try:
        pickle.dump(model, open(os.path.join(path, 'model.pmfm'),
                                'wb'))        
        model.generate_script(dir = path, 
                          parallel = False,
                          filename = 'model.py')

        model.generate_script(dir = path, 
                          parallel = True,
                          filename = 'modelp.py')
    except:
        import traceback
        traceback.print_exc()

    for od in model.walk():
       if hasattr(od, 'use_relative_path'):
            od.restore_fullpath()
    
