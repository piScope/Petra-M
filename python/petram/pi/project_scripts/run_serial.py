from petram.pi.run_petram import run_serial

viewer = kwargs.pop("viewer", None)
if viewer is None:
    viewer = model.mfembook.find_bookviewer()

ret = run_serial(model, *args, **kwargs)

if viewer is not None and viewer.plotsoldlg is not None:
    import wx
    wx.CallAfter(viewer.plotsoldlg.update_sollist_local1)
    wx.CallAfter(viewer.plotsoldlg.update_sollist_local2)
    wx.CallAfter(viewer.plotsoldlg.load_sol_if_needed)

ans(ret)
