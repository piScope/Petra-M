import wx

def progressbar(parent, message, title, count, can_abort=True):
    style = wx.PD_AUTO_HIDE|wx.PD_APP_MODAL
    style = style|wx.PD_CAN_ABORT
    dlg = wx.GenericProgressDialog(title,
                                   message,
                                   count,
                                   parent,
                                   style=style)
    def close_dlg(evt, dlg=dlg):
        dlg.Destroy()
    dlg.Bind(wx.EVT_CLOSE, close_dlg)
    return dlg
