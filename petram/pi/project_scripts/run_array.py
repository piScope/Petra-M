from petram.pi.run_petram import run_parallel

import os

values = kwargs.pop("values")
folder = kwargs.pop("folder")
viewer = kwargs.pop("viewer")

odir = os.getcwd()

for array_id in values:
    path = os.path.join(folder.owndir(), 'case_'+str(array_id))
    os.mkdir(path)
    kwargs["path"] = path
    kwargs["array_id"] = array_id
    kwargs["array_len"] = len(values)

    run_parallel(model, *args, **kwargs)
    os.chdir(odir)

if viewer.plotsoldlg is not None:
    import wx
    wx.CallAfter(viewer.plotsoldlg.update_sollist_local1)
    wx.CallAfter(viewer.plotsoldlg.update_sollist_local2)
    wx.CallAfter(viewer.plotsoldlg.load_sol_if_needed)
ans()
