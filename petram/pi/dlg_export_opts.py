import os
import wx
from ifigure.utils.edit_list import EditListPanel, EDITLIST_CHANGED

default_panel_value = (False, False)


class dlg_export_opts(wx.Dialog):

    def __init__(self, parent, value, support_integ=True):

        wx.Dialog.__init__(self, parent, wx.ID_ANY, "Export options",
                           style=wx.STAY_ON_TOP | wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        self.SetSizer(vbox)
        vbox.Add(hbox2, 1, wx.EXPAND | wx.ALL, 1)

        ll = [[None, False, 3, {"text": "compute integration"}],
              [None, False, 3, {"text": "repeat under all subdirectories"}]]
        if not support_integ:
            ll = ll[1:]

        self.elp = EditListPanel(self, ll)

        hbox2.Add(self.elp, 1, wx.EXPAND | wx.RIGHT | wx.LEFT, 1)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        button = wx.Button(self, wx.ID_ANY, "Cancel")
        button2 = wx.Button(self, wx.ID_ANY, "Apply")

        hbox.Add(button, 0, wx.EXPAND)
        hbox.AddStretchSpacer()
        hbox.Add(button2, 0, wx.EXPAND)
        vbox.Add(hbox, 0, wx.EXPAND | wx.ALL, 5)

        button.Bind(wx.EVT_BUTTON, self.onCancel)
        button2.Bind(wx.EVT_BUTTON, self.onApply)

        self.elp.SetValue(value)

        self.Layout()
        self.Fit()
        self.CenterOnParent()

        self.Show()

    def onCancel(self, evt=None):
        self.value = None
        self.EndModal(wx.ID_CANCEL)
        evt.Skip()

    def onApply(self, evt):
        self.value = self.elp.GetValue()
        self.EndModal(wx.ID_OK)
        evt.Skip()


def ask_export_opts(win, value=None, support_integ=True):
    if value is None:
        value = default_panel_value
        if not support_integ:
            value = value[1:]

    dlg = dlg_export_opts(win, value, support_integ)

    outvalue = None
    try:
        if dlg.ShowModal() == wx.ID_OK:
            outvalue = dlg.value
        else:
            pass
    finally:
        dlg.Destroy()
    return outvalue
