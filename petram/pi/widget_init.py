import wx
import wx.grid as gr
from ifigure.utils.edit_list import TextCtrlCopyPaste        
from petram.helper.init_helper import *

class InitSettingPanel(wx.Panel):
    def __init__(self, parent, id, setting = None):
        wx.Panel.__init__(self, parent, id)

        sizer  = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)
        
        sizer2  = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(sizer2, 1, wx.EXPAND|wx.ALL, 5)


        rb1 = wx.RadioButton(self, wx.ID_ANY, "Zero",
                              style=wx.RB_GROUP)
        rb2 = wx.RadioButton(self, wx.ID_ANY, "Use Value")
        rb3 = wx.RadioButton(self, wx.ID_ANY, "Use init panel value")
        rb4 = wx.RadioButton(self, wx.ID_ANY, "From File")
        rb5 = wx.RadioButton(self, wx.ID_ANY, "No initializetion")        

        self.rbs = [rb1, rb2, rb3, rb4, rb5]

        self.st1 = wx.StaticText(self, wx.ID_ANY, '     value: ')
        self.tc1 = TextCtrlCopyPaste(self, wx.ID_ANY, '0.0', 
                                validator = validator )

        self.st4 = wx.StaticText(self, wx.ID_ANY, '     path:  ')
        self.tc4 = TextCtrlCopyPaste(self, wx.ID_ANY, '')
        self.bt4 = wx.Button(self, label='Browse...', style=wx.BU_EXACTFIT)

        sizer2.Add(rb1, 0, wx.EXPAND)
        sizer2.Add(rb2, 0, wx.EXPAND)
        sizer3  = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(self.st1, 0, wx.ALL, 2)
        sizer3.Add(self.tc1, 1, wx.EXPAND|wx.ALL, 2)        
        sizer2.Add(sizer3, 0, wx.EXPAND)        
        sizer2.Add(rb3, 0, wx.EXPAND)
        sizer2.Add(rb4, 0, wx.EXPAND)
        sizer3  = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(self.st4, 0, wx.ALL, 2)
        sizer3.Add(self.tc4, 1, wx.EXPAND|wx.ALL, 2)
        sizer3.Add(self.bt4, 0, wx.ALL, 2)
        sizer2.Add(sizer3, 0, wx.EXPAND)                
        sizer2.Add(rb5, 0, wx.EXPAND)
        
        self.bt4.Bind(wx.EVT_BUTTON, self.onBrowse)
        for rb in self.rbs:
            self.Bind(wx.EVT_RADIOBUTTON, self.onHit, rb)
            
        self.rbs[0].SetValue(True)
        self.adjust_enables(0)
        self.Layout()

    def SetValue(self, value):
        for sel, rb in enumerate(self.rbs):
            if sel == value[0]:
                rb.SetValue(True)
            else:
                rb.SetValue(False)
        self.tc1.SetValue(value[1])
        self.tc4.SetValue(value[2])
        self.adjust_enables(value[0])

    def GetValue(self):
        for sel, rb in enumerate(self.rbs):
            if rb.GetValue(): break
        val = self.tc1.GetValue()
        path = self.tc4.GetValue()
        if not validator(val, None, None): val = '0.0'

        return sel, val, path

    def Enable(self, value):
        pass
    
    def onBrowse(self, evt):
        diag = wx.DirDialog(self)
        ret = diag.ShowModal()
        if ret == wx.ID_OK:
            path = diag.GetPath()
            self.tc4.SetValue(path)
        diag.Destroy()
        evt.Skip()

    def adjust_enables(self, selection):
        if selection == 3:
           self.st4.Enable(True)            
           self.tc4.Enable(True)
           self.bt4.Enable(True)           
        else:
           self.st4.Enable(False)                        
           self.tc4.Enable(False)
           self.bt4.Enable(False)           
        if selection == 1:
           self.st1.Enable(True)
           self.tc1.Enable(True)
        else:
           self.st1.Enable(False)            
           self.tc1.Enable(False)
           
    def onHit(self, evt):
        w = evt.GetEventObject()
        sel = self.rbs.index(w)
        self.adjust_enables(sel)
        self.send_event(self, evt)
        evt.Skip()
            
    def send_event(self, obj, evt):
        evt.SetEventObject(self)
        if hasattr(self.GetParent(), 'send_event'):
            self.GetParent().send_event(self, evt)
        
class Example(wx.Frame):
    def __init__(self, parent=None, title='test'):
        super(Example, self).__init__(parent, title=title)
        self.w = InitSettingPanel(self, wx.ID_ANY)
        self.SetSizer(wx.BoxSizer(wx.VERTICAL))
        self.GetSizer().Add(self.w, 1, wx.EXPAND, 0)

        #value = [True, {'k':[('x', 'a*3'), ('y', 'b+b')], 'alpha':[]}]
        self.w.SetValue((1, '500', '/home/shiraiwa'))

        self.Show()
        self.Bind(wx.EVT_CLOSE, self.onClose)
        
    def onClose(self, evt):
        print self.w.GetValue()   
        evt.Skip()

if __name__=='__main__':
   app = wx.App(False)
   e = Example()
   app.MainLoop()                 
