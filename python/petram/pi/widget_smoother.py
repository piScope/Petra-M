import os

import wx
import wx.adv
import wx.propgrid as wxpg

from ifigure.utils.edit_list import DialogEditList
from ifigure.ifigure_config import icondir

bmp1=wx.Bitmap(os.path.join(icondir, '16x16', 'smode.png'))
bmp2=wx.Bitmap(os.path.join(icondir, '16x16', 'pmode.png'))

HypreSmoother = ["None", "Jacobi", "l1Jacobi", "l1GS", "l1GStr", 
                "lumpedJacobi", "GS", "Chebyshev", "Taubin",
                 "FIR", "MUMPS"]
SparseSmoother = ["None", "Jacobi", "l1Jacobi", "lumpedJacobi",
                  "GS", "forwardGS", "backwrdGS", "MUMPS"]

class SmootherChoiceProperty(wxpg.StringProperty):
    def __init__(self, label, name=wxpg.PG_LABEL, value='', 
                 choices = None):
        wxpg.StringProperty.__init__(self, label, name, value)
        if choices is None: choices = ["empty choices"]
        # Prepare choices
        dialog_choices = []
        for x in choices:
            dialog_choices.append(x)
        self.dialog_choices = dialog_choices

    def DoGetEditorClass(self):
        return wxpg.PropertyGridInterface.GetEditorByName("TextCtrlAndButton")

    def GetEditorDialog(self):
        # Set what happens on button click
        return SingleChoiceDialogAdapter(self.dialog_choices)

class SingleChoiceDialogAdapter(wxpg.PGEditorDialogAdapter):
    """ This demonstrates use of wxpg.PGEditorDialogAdapter.
    """
    def __init__(self, choices):
        wxpg.PGEditorDialogAdapter.__init__(self)
        self.choices = choices

    def DoShowDialog(self, propGrid, property):
        s = wx.GetSingleChoice("Choose Smoother", "Smoother...", self.choices)

        if s:
            self.SetValue(s)
            return True

        return False;

class WidgetSmoother(wx.Panel):
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
                              #wxpg.PG_AUTO_SORT|
                              #wxpg.PG_DESCRIPTION |
                              wxpg.PG_TOOLBAR)
        self.pg.AddPage("Serial", bmp=bmp1)
        self.pg.SelectPage(0)
        self.pg.SetColumnCount(2)
        self.pg.SetColumnTitle(0, "FES var.")
        self.pg.Append(SmootherChoiceProperty("E1", choices=SparseSmoother))

        self.pg.AddPage( "Parallel", bmp=bmp2)
        self.pg.SelectPage(1)
        self.pg.SetColumnCount(2)

        self.pg.SetColumnTitle(0, "FES var.")

        self.pg.Append(SmootherChoiceProperty("E1", choices = HypreSmoother))

        #self.pg.SetPageSplitterPosition(0, 0.5) 
        #self.pg.SetPageSplitterPosition(1, 0.5) 

        self.pg.ShowHeader(True)
        sizer2.Add(self.pg, 1, wx.EXPAND|wx.ALL, 1)

        self.pg.Bind( wxpg.EVT_PG_PAGE_CHANGED, self.OnPropGridPageChange )
        self.pg.Bind( wxpg.EVT_PG_CHANGED, self.OnPropGridChange )
        
        value = [('E1', ['GS', 'GS'],),
                 ('V1', ['GS', 'Jacobi'])]

        self.SetValue(value)

    def set_column_title(self):
        index = self.pg.GetSelectedPage()
        if index == 0:
            self.pg.SetColumnTitle(1, "Serial Preconditioner")
        else:
            self.pg.SetColumnTitle(1, "Parallel Preconditioner")

    def OnPropGridPageChange(self, evt):
        self.set_column_title()
        
    def OnPropGridChange(self, evt):
        p = evt.GetProperty()
        index = self.pg.GetSelectedPage()
        name = p.GetName()
        newvalue = p.GetValueAsString()

        for n, v in self._value:
            if n == name: v[index] = str(newvalue)
        #evt.Skip()
        self.GetParent().send_event(self, evt)        
        #wx.CallAfter(self.GetTopLevelParent().Show)
        
    def GetValue(self):
        return self._value

    def SetValue(self, value):
        selp = self.pg.GetSelectedPage()        
        self.pg.ClearPage(0)
        self.pg.SelectPage(0)
        self.prop_names = []
        for name, v in value:
            pgp = self.pg.Append(SmootherChoiceProperty(name, 
                                                  choices = SparseSmoother))   
            pgp.SetValue(v[0])
            self.prop_names.append(name)

        self.pg.ClearPage(1)
        self.pg.SelectPage(1)
        for name, v in value:
            pgp = self.pg.Append(SmootherChoiceProperty(name, 
                                                  choices = HypreSmoother))
            pgp.SetValue(v[1])

        self._value = value
        self.pg.SelectPage(selp)
        self.set_column_title()        
'''
from petram.pi.widget_smoother import WidgetSmoother
ll = [[None, None, 99, {"UI":WidgetSmoother, "span":(1,2)}],]
ret= DialogEditList(ll)
print ret
'''
