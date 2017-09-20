#     **  Template for a new script  **

#   Following variabs/functions can be used
#    obj : script object
#    top. proj : = obj.get_root_parent()
#    wdir : proj.getvar('wdir')
#    model: model object containing this script
#    param : model param
#    app : proj.app (ifigure application)
#    stop() : exit from script 

import numpy as np
import os

def launch_remote_job(TD_sol = None, retrieve_files = True):
    TD_mfem = model
    remote = model.param.eval('remote')
    if TD_sol is None: return
    if remote is None: return
    
    from petram.remote.client_script import submit_job, retrieve_files

    submit_job(model)
    if ret:
        if retrieve_files:
            retrieve_files(model)
            model.scripts.helpers.read_sol()
        globals()['default_sol_path'] = os.path.dirname(TD_sol.owndir())
        return True
    return False
ans(launch_remote_job(*args, **kwargs))
