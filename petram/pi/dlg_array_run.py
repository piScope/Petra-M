import os
import wx
from ifigure.utils.edit_list import EditListPanel, EDITLIST_CHANGED

default_panel_value = "1-5"


def process_txt(txt):
    x = [ss.split('-') for ss in txt.split(',')]
    x = [list(range(int(i[0]), int(i[1])+1)) if len(i)
         == 2 else [int(i[0])] for i in x]
    return sum(x, [])


def check_array_expr(value, _param, _ctrl):
    try:
        process_txt(value)
        return True
    except:
        import petram.debug
        import traceback
        if petram.debug.debug_default_level > 2:
            traceback.print_exc()
        return False


class dlg_array_opts(wx.Dialog):

    def __init__(self, parent, value):

        wx.Dialog.__init__(self, parent, wx.ID_ANY, "ArrayRun options",
                           style=wx.STAY_ON_TOP | wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        self.SetSizer(vbox)
        vbox.Add(hbox2, 1, wx.EXPAND | wx.ALL, 1)

        ll = [["ArrayID", False, 0, {'validator': check_array_expr,
                                     'validator_param': {}}], ]

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

        self.elp.SetValue([value, ])

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


def ask_array_opts(win, value=None):
    if value is None:
        value = default_panel_value

    dlg = dlg_array_opts(win, value)

    outvalue = None
    try:
        if dlg.ShowModal() == wx.ID_OK:
            outvalue = dlg.value
        else:
            pass
    finally:
        dlg.Destroy()

    return outvalue[0], process_txt(outvalue[0])
