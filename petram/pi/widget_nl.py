import wx
import wx.grid as gr


class NonlinearTermPanel0(wx.Panel):
    def __init__(self, parent, id, names = None):
        wx.Panel.__init__(self, parent, id)

        self.names = [] if names is None else names
        self.grids = {}
        self.btns = {}
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)
        for n in names:
            self.add_panel(n, sizer)

    def add_panel(self, name, top_sizer):
        sizer = wx.BoxSizer(wx.VERTICAL)
        top_sizer.Add(sizer, 0, wx.EXPAND)        
        h = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(h, 0, wx.EXPAND)
        t=wx.StaticText(self, wx.ID_ANY, "  " + name + "  " )
        bp = wx.Button(self, wx.ID_ANY, '+', style=wx.BU_EXACTFIT)
        bm = wx.Button(self, wx.ID_ANY, '-', style=wx.BU_EXACTFIT)

        h.Add(t, 0)
        h.AddStretchSpacer()        
        h.Add(bp, 0)
        h.Add(bm, 0)
        def hit_plus(evt, name=name):
            self.hit_plus(evt, name)
        def hit_minus(evt, name=name):
            self.hit_minus(evt, name)
        self.Bind(wx.EVT_BUTTON, hit_plus, bp)
        self.Bind(wx.EVT_BUTTON, hit_minus, bm)

        grid = gr.Grid(self, -1)
        grid.CreateGrid(3, 2)
        grid.HideRowLabels()
        grid.SetColLabelValue(0, 'variable')
        grid.SetColLabelValue(1, 'expression')        
        sizer.Add(grid, 1, wx.EXPAND|wx.ALL, 5)

        def onSize(evt, grid = grid):
            from ifigure.utils.wx3to4 import grid_ClientSizeTuple
            width, height = grid_ClientSizeTuple(grid)
            s = width - grid.GetColSize(0)
            if s < 1: return
            grid.SetColSize(1, s)
        
        grid.Bind(wx.EVT_SIZE, onSize)
        
        self.grids[name] = grid
        self.btns[name] = (bp, bm)
        
    def hit_plus(self, evt, name):
        grid = self.grids[name]
        grid.AppendRows(1)
        self.Layout()
        self.send_event(evt)        
        
    def hit_minus(self, evt, name):
        grid = self.grids[name]
        cursor = grid.GetGridCursorRow()
        grid.DeleteRows(cursor)
        self.Layout()
        self.send_event(evt)

    def SetValue(self, value):
        for name in self.names:
            grid = self.grids[name]
            for k in range(grid.GetNumberRows()):
                grid.SetCellValue(k, 0, '')
                grid.SetCellValue(k, 1, '')                
            k = 0
            if not name in value: continue
            for c1, c2 in value[name]:
                grid.SetCellValue(k, 0, c1)
                grid.SetCellValue(k, 1, c2)                
                k = k + 1

    def GetValue(self):
        value = {}
        for name in self.names:
            grid = self.grids[name]
            v = []
            for k in range(grid.GetNumberRows()):
                a = grid.GetCellValue(k, 0)
                b = grid.GetCellValue(k, 1)
                if a.strip() == '': continue
                v.append((a, b))
            value[name] = v
        return value

    def Enable(self, value):
        for name in self.names:
            grid = self.grids[name]
            grid.Enable(value)
            self.btns[name][0].Enable(value)
            self.btns[name][1].Enable(value)
            
    def send_event(self, evt):
        evt.SetEventObject(self)
        if hasattr(self.GetParent(), 'send_event'):
            self.GetParent().send_event(self, evt)
        
class NonlinearTermPanel(wx.Panel):
    def __init__(self, parent, id, setting = None):
        setting = {} if setting is None else setting
        names = setting.pop('names', [])
        wx.Panel.__init__(self, parent, id)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)
        
        self.bt_use_nl = wx.CheckBox(self, wx.ID_ANY, 'Use NL')
        self.nl_terms =  NonlinearTermPanel0(self, wx.ID_ANY, names = names)
        
        sizer.Add(self.bt_use_nl, 0, wx.EXPAND)
        sizer.Add(self.nl_terms, 1 , wx.EXPAND|wx.ALL, 3)
        
        self.bt_use_nl.Bind(wx.EVT_CHECKBOX, self.onHitToggle)
        self.Bind(gr.EVT_GRID_CELL_CHANGED, self.onCellChanged)        
        self.Layout()
        
    def SetValue(self, value):
        self.bt_use_nl.SetValue(value[0])
        self.nl_terms.Enable(True)
        self.nl_terms.SetValue(value[1])
        if value[0]:
            self.nl_terms.Enable(True)
        else:
            self.nl_terms.Enable(False)            

    def GetValue(self):
        return [self.bt_use_nl.GetValue(),
                self.nl_terms.GetValue()]

    def onHitToggle(self, evt):
        tg =self.bt_use_nl.GetValue()
        if tg:
            self.nl_terms.Enable(True)
        else:
            self.nl_terms.Enable(False)
        self.send_event(self, evt)
        
    def onCellChanged(self, evt):
        self.send_event(self, evt)

    def send_event(self, obj, evt):
        #print("sending event", self.GetValue())
        evt.SetEventObject(self)
        if hasattr(self.GetParent(), 'send_event'):
            self.GetParent().send_event(self, evt)

class Example(wx.Frame):
    def __init__(self, parent=None, title='test'):
        super(Example, self).__init__(parent, title=title)
        w = NonlinearTermPanel(self, wx.ID_ANY, setting = {'names':['k', 'alpha']})
                 
        self.SetSizer(wx.BoxSizer(wx.VERTICAL))
        self.GetSizer().Add(w, 1, wx.EXPAND, 0)

        value = [True, {'k':[('x', 'a*3'), ('y', 'b+b')], 'alpha':[]}]
        w.SetValue(value)

        self.Show()

if __name__=='__main__':
   app = wx.App(False)
   e = Example()
   app.MainLoop()                 
