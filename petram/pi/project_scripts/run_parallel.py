from petram.pi.run_petram import run_parallel

viewer = kwargs.pop("viewer")

ret = run_parallel(model, *args, **kwargs)

if viewer.plotsoldlg is not None:
    import wx
    wx.CallAfter(viewer.plotsoldlg.update_sollist_local1)
    wx.CallAfter(viewer.plotsoldlg.update_sollist_local2)
    wx.CallAfter(viewer.plotsoldlg.load_sol_if_needed)

ans(ret)

