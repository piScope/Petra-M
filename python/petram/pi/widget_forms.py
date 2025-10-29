import os
import wx
from PIL import Image

import logging
logging.getLogger('PIL').setLevel(logging.WARNING)

img_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'icon')
bmp_data = {}


class WidgetForms(wx.Panel):
    def __init__(self, parent, id, *args, **kwargs):
        wx.Panel.__init__(self, parent, id)

        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2 = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer1)
        sizer1.Add(sizer2, 1, wx.EXPAND)

        bcb = wx.adv.BitmapComboBox(self, pos=(25, 25), size=(200, -1))

        self.bcb = bcb

        setting = kwargs.pop("setting")
        self.choices_cb = setting['choices_cb']

        choices = ["no integrator available"]
        self._current_choices = ["no integrator available"]
        self._value = "no integrator available"

        self.SetChoices(choices, index=0)

        self.Bind(wx.EVT_COMBOBOX, self.onHit, self.bcb)
        self.Bind(wx.EVT_COMBOBOX_DROPDOWN, self.onDropDown, self.bcb)

        sizer2.Add(self.bcb, 1, wx.EXPAND | wx.ALL, 1)

    def onHit(self, evt):
        sel = self.bcb.GetValue()
        self._value = sel
        self.GetParent().send_event(self, evt)

    def GetValue(self):
        sel = self.bcb.GetValue()
        self._value = sel
        return sel

    def SetValue(self, value):
        self._value = value
        ch = self.choices_cb()
        self.SetChoices(ch)

    def onDropDown(self, evt):
        sel = self._value

        ch = self.choices_cb()
        if sel in ch:
            idx = ch.index(sel)
        else:
            idx = 0
        self.SetChoices(ch, index=idx)

    def SetChoices(self, choices, index=-1):
        sel = self._value

        if len(choices) == 0:
            choices = ["no integrator available"]
            sel = "no integrator available"
            self._value = "no integrator available"

        self.bcb.Clear()

        for x in choices:
            if x == "no integrator available":
                x = "none"

            if not x in bmp_data:
                name = os.path.join(img_path, 'form_' + x + '.png')
                img = Image.open(name)
                size = img.size
                fac = 33/size[1]
                img1 = img.resize((int(size[0]*fac), int(size[1]*fac)),
                                  resample=Image.LANCZOS,)

                image = wx.Image(*img1.size)
                image.SetData(img1.convert("RGB").tobytes())

                bmp = wx.Bitmap(image)
                bmp_data[x] = bmp
            else:
                bmp = bmp_data[x]

            if x == "none":
                self.bcb.Append("no integrator available", bmp, "none")
            else:
                self.bcb.Append(x, bmp, x)

        self._current_choices = choices

        if index != -1:
            self.bcb.SetSelection(index)
        else:
            if sel in choices:
                index = choices.index(sel)
            elif sel == "no integrator available":
                index = 0
            else:
                index = 0
            self.bcb.SetSelection(index)
