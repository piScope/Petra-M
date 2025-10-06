from __future__ import print_function
import wx
import traceback
import numpy as np

import ifigure.widgets.dialog as dialog
from ifigure.utils.edit_list import EditListPanel
from ifigure.utils.edit_list import EDITLIST_CHANGED
from ifigure.utils.edit_list import EDITLIST_CHANGING
from ifigure.utils.edit_list import EDITLIST_SETFOCUS

from petram.pi.simple_frame_plus import SimpleFramePlus

class DlgPlotExpr(SimpleFramePlus):
    def __init__(self, parent, id = wx.ID_ANY, title = 'Plot Expression', **kwargs):
        
        iattr = kwargs.pop("iattr", [])
        
        style = (wx.CAPTION|
                 wx.CLOSE_BOX|
                 wx.MINIMIZE_BOX| 
                 wx.RESIZE_BORDER|
                 wx.FRAME_FLOAT_ON_PARENT)
        #        wx.FRAME_TOOL_WINDOW : this styles may not work on Windows/Mac
        super(DlgPlotExpr, self).__init__(parent, id, title, style=style, **kwargs)
        
        self.parent = parent
        choices = list(parent.model.param.getvar('mfem_model')['Phys'])
        if len(choices) == 0:
            choices = ['No Physics is defined!']

        ll = [['Expression', '', 0, {}],
              ['Index', ','.join(iattr), 0, {}],
              [None, 'Index: Domain (1D/2D) and Boundary (3D)', 2, {}],              
              ['NameSpace', choices[0], 4, {'style':wx.CB_READONLY,
                                            'choices': choices}],
              [None, ' '*50, 2, {}],]
        self.elp = EditListPanel(self, ll)
        button=wx.Button(self, wx.ID_ANY, "Apply")

        box = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(box)        
        box.Add(self.elp, 1, wx.EXPAND|wx.ALL, 1)

        hbox = wx.BoxSizer(wx.HORIZONTAL)        
        hbox.AddStretchSpacer()        
        hbox.Add(button, 0, wx.ALL, 5)
        box.Add(hbox, 0, wx.EXPAND|wx.ALL, 1)

        button.Bind(wx.EVT_BUTTON, self.onApply)
        self.Show()
        self.Layout()
        self.Fit()
        

    def onApply(self, evt):
        value = self.elp.GetValue()
        self._doPlotExpr(value)
        evt.Skip()

    def _doPlotExpr(self, value):
        model = self.parent.model.param.getvar('mfem_model')
        if model is None: return
        
        expr = str(value[0])
        phys = str(value[3])
        use_2dplot = model['Phys'][phys].dim == 1

        from petram.utils import eval_expr
        try:
           engine = self.parent.engine        
           d = eval_expr(model, engine, expr, value[1], phys = phys)
        except:
           dialog.showtraceback(parent = self,
                                txt='Failed to evauate expression',
                                title='Error',
                                traceback=traceback.format_exc())
           return

        
        from ifigure.interactive import figure
        v = figure()
        v.update(False)
        v.suptitle(expr)
        if use_2dplot:
            for k in d.keys():
                v.plot(d[k][0][:, 0, 0].flatten(),
                       d[k][1][:, 0].flatten())                
            v.update(True)            
        else:
            from petram.pi.dlg_plot_sol import setup_figure
            setup_figure(v, self.GetParent())                    
            for k in d.keys():
                v.solid(d[k][0], cz=True, cdata= d[k][1])
            v.update(True)
            v.view('noclip')
            v.lighting(light = 0.5)
        

