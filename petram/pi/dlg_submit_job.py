import wx
from ifigure.utils.edit_list import EditListPanel

ll = [["Number of Nodes", 1, 400, 2], 
      ["Number of Cores", 16, 400, 2], 
      [None,   False,  3, {"text":"Retrieve Data"}],]     

class dlg_jobsubmission(wx.Dialog):

    def __init__(self, parent, id = wx.ID_ANY, title = ''):

        wx.Dialog.__init__(self, parent, wx.ID_ANY, title,
                    style=wx.STAY_ON_TOP|wx.DEFAULT_DIALOG_STYLE)

        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        self.SetSizer(vbox)
        vbox.Add(hbox2, 1, wx.EXPAND|wx.ALL,1)      

        self.elp=EditListPanel(self, ll)
        hbox2.Add(self.elp, 1, wx.EXPAND|wx.ALIGN_CENTER|wx.RIGHT|wx.LEFT, 1)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        button=wx.Button(self, wx.ID_ANY, "Cancel")
        button2=wx.Button(self, wx.ID_ANY, "Submit")

        hbox.Add(button, 0, wx.EXPAND)
        hbox.AddStretchSpacer()
        hbox.Add(button2, 0, wx.EXPAND)
        vbox.Add(hbox, 0, wx.EXPAND|wx.ALL,5)

        button.Bind(wx.EVT_BUTTON, self.onCancel)
        button2.Bind(wx.EVT_BUTTON, self.onSubmit)

#        self.panel.Layout()
        size= self.GetSize()
        self.SetSizeHints(minH=-1, minW=size.GetWidth())
        self.Show()
        self.Layout()
        self.Fit()
        self.CenterOnScreen()
        #wx.CallAfter(self.Fit)
        self.value = self.elp.GetValue()     
    def onCancel(self, evt):
        self.value = self.elp.GetValue()     
        self.EndModal(wx.ID_CANCEL)
        
    def onSubmit(self, evt):
        self.value = self.elp.GetValue()     
        self.EndModal(wx.ID_OK)
 
def get_job_submisson_setting(parent, servername = ''):
    dlg = dlg_jobsubmission(parent, title='Submit to '+servername)
    value = {}
    try:
        if dlg.ShowModal() == wx.ID_OK:
            value["num_nodes"] = dlg.value[0]
            value["num_cores"] = dlg.value[1]
            value["retrieve_files"] = dlg.value[2]
        else:
            pass
    finally:
        dlg.Destroy()
    return value 
        

 




