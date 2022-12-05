import os
import wx

img_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'icon')
        

class WidgetForms(wx.Panel):
    def __init__(self, parent, id, *args, **kwargs):
        wx.Panel.__init__(self, parent, id)
        
        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2 = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer1)
        sizer1.Add(sizer2, 1, wx.EXPAND)

        bcb = wx.adv.BitmapComboBox(self, pos=(25,25), size=(200,-1))

        self.bcb = bcb
        
        setting = kwargs.pop("setting")
        self.choices_cb = setting['choices_cb']

        choices = setting['choices']
        self._value = choices[0]
        self.SetChoices(choices, index=0)
        
        self.Bind(wx.EVT_COMBOBOX, self.onHit, self.bcb)
        self.Bind(wx.EVT_COMBOBOX_DROPDOWN, self.onDropDown, self.bcb)
            
        sizer2.Add(self.bcb, 1, wx.EXPAND|wx.ALL, 1)        

    def onHit(self, evt):
        self.GetParent().send_event(self, evt)

    def GetValue(self):
        return self._value

    def SetValue(self, value):
        self._value = value

    def onDropDown(self, evt):
        sel = self.GetValue()
        self._current_value = sel

        ch = self.choices_cb()
        if sel in ch:
            idx = ch.index(sel)
        else:
            idx = 0
        self.SetChoices(ch, index=idx)
        
    def SetChoices(self, choices, index=-1):

        self.bcb.Clear()        
        sel = self.GetValue()
        
        for x in choices:
            name = os.path.join(img_path, 'form_' + x + '.png')
            img = wx.Image(name, wx.BITMAP_TYPE_PNG)
            size = img.GetSize()
            #print(size)
            fac = 33/size[1]
            img.Rescale(int(size[0]*fac), int(size[1]*fac))
            bmp = img.ConvertToBitmap()
            self.bcb.Append(x, bmp, x)
        
        if index != -1:
            self.bcb.SetSelection(index)
        else:
            if sel in choices:
                index = choices.index(sel)
                self.bcb.SetSelection(index)
        
