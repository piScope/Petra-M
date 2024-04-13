import os
import sys
import time
import wx
import wx.adv
import wx.propgrid as wxpg

from ifigure.utils.edit_list import DialogEditList
from ifigure.ifigure_config import icondir


class ValueObject:
    def __init__(self):
        pass


class MemoDialog(wx.Dialog):
    """
    Dialog for multi-line text editing.
    """

    def __init__(self, parent=None, title="", text="", pos=None, size=(500, 500)):
        wx.Dialog.__init__(self, parent, -1, title,
                           style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        topsizer = wx.BoxSizer(wx.VERTICAL)

        tc = wx.TextCtrl(self, 11, text, style=wx.TE_MULTILINE)
        self.tc = tc
        topsizer.Add(tc, 1, wx.EXPAND | wx.ALL, 8)

        rowsizer = wx.BoxSizer(wx.HORIZONTAL)
        rowsizer.Add(wx.Button(self, wx.ID_OK, 'Ok'),
                     0, wx.ALIGN_CENTRE_VERTICAL, 8)
        rowsizer.Add((0, 0), 1, wx.ALIGN_CENTRE_VERTICAL, 8)
        rowsizer.Add(wx.Button(self, wx.ID_CANCEL, 'Cancel'),
                     0, wx.ALIGN_CENTRE_VERTICAL, 8)
        topsizer.Add(rowsizer, 0, wx.EXPAND | wx.ALL, 8)

        self.SetSizer(topsizer)
        topsizer.Layout()

        self.SetSize(size)
        if not pos:
            self.CenterOnScreen()
        else:
            self.Move(pos)


class WidgetParameters(wx.Panel):
    def __init__(self, parent, id, *args, **kwargs):

        wx.Panel.__init__(self, parent, id)
        self.prop_names = []
        setting = kwargs.pop("setting", {})

        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2 = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer1)
        sizer1.Add(sizer2, 1, wx.EXPAND)
        self.pg = wxpg.PropertyGridManager(self,
                                           style=wxpg.PG_SPLITTER_AUTO_CENTER |
                                           wxpg.PGMAN_DEFAULT_STYLE)

        self.pg.AddPage("Parameters")
        self.pg.SetColumnCount(2)
        self.pg.SetColumnTitle(0, "Name")
        self.pg.SetColumnTitle(1, "Value")

        #self.pg.SetPageSplitterPosition(0, 0.5)
        #self.pg.SetPageSplitterPosition(1, 0.5)

        self.pg.ShowHeader(True)

        self.pg.Bind(wxpg.EVT_PG_CHANGED, self.OnPropGridChange)

        sizer2.Add(self.pg, 1, wx.EXPAND | wx.ALL, 1)

        rowsizer = wx.BoxSizer(wx.HORIZONTAL)
        but = wx.Button(self, -1, "Edit...")
        but.Bind(wx.EVT_BUTTON, self.OnSetPropertyValues)
        rowsizer.Add(but, 1)
        sizer2.Add(rowsizer, 0)

        # self.SetValue(value)

    def OnPropGridChange(self, evt):
        p = evt.GetProperty()
        index = self.pg.GetSelectedPage()
        name = p.GetName()
        newvalue = p.GetValueAsString()

        for n, v in self._value:
            if n == name:
                v[index] = str(newvalue)
        # evt.Skip()
        self.GetParent().send_event(self, evt)
        # wx.CallAfter(self.GetTopLevelParent().Show)

    def GetValue(self):
        try:
            d = self.pg.GetPropertyValues(inc_attributes=True)

            ss = []
            for k, v in d.items():
                v = str(v)
                ss.append(k + " = " + v)
        except:
            import traceback
            traceback.print_exc()
        lines = "\n".join(ss)

        return self._eval_lines(lines)

    def _eval_lines(self, lines):
        ll = {}
        g = {}

        exec(lines, g, ll)
        return ll

    def SetValue(self, value):
        ''' 
        Dictionary {"name": value}
        '''
        page = self.pg.GetPage(0)
        self.pg.ClearPage(0)

        for x in value:
            page.Append(wxpg.StringProperty(x, wxpg.PG_LABEL,
                                            str(value[x])))

    def OnSetPropertyValues(self, event):
        try:
            d = self.pg.GetPropertyValues(inc_attributes=True)

            ss = []
            for k, v in d.items():
                v = str(v)
                ss.append(k + " = " + v)

            with MemoDialog(self,
                            "Enter Content for Object Used in SetPropertyValues",
                            '\n'.join(ss)) as dlg:  # default_object_content1

                if dlg.ShowModal() == wx.ID_OK:
                    lines = dlg.tc.GetValue()
                    ll = self._eval_lines(lines)

                    self.SetValue(ll)

        except:
            import traceback
            traceback.print_exc()


'''
ll = [[None, None, 99, {"UI":WidgetParameters, "span":(1,2)}],]
ret= DialogEditList(ll)
print(ret)
'''
