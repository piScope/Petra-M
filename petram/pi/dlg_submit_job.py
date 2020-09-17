import os
import wx
from ifigure.utils.edit_list import EditListPanel

setting1 = {"style": wx.CB_READONLY, "choices": ("None", "Begin", "End")}

def elp_setting(log_keywords):
    ll = [["Num. of Nodes", 1, 400, {}],
          ["Num. of Cores(total)", 16, 400, {}],
          ["Num. of OMP threads", 4, 400, {}],
          ["Wall clock", "00:15:00", 0, {}],
          ["Queue", "Debug", 0, {}],
          ["Remote Dir.", "", 0, {}],
          ["Note", None, 235, {'nlines': 5}],
          ["Keywords", None, 36, {'col': 3, 'labels': list(log_keywords)}],
          ["Notification",  "None", 4, setting1],
          [None,   False,  3, {"text":"Skip sending mesh file"}],]
    return ll


values = ['1', '1', '1', '00:10:00', 'debug', '', 
          '', '', "None", False, False, ]
keys = ['num_nodes', 'num_cores', 'num_openmp', 'walltime',
        'queue', 'rwdir', 
        'log_txt', 'log_keywords', 'notification', 'skip_mesh',
        'retrieve_files',]

def_queues = {'type':'SLURM',
             'queus':[{'name': 'debug',
                       'maxnode': 1}]}

def get_defaults():
    return values[:], keys[:]

class dlg_jobsubmission(wx.Dialog):
    def __init__(self, parent, id=wx.ID_ANY, title='',
                       value=None, queues=None):

        if queues is None:
            queues = def_queues

        wx.Dialog.__init__(self, parent, wx.ID_ANY, title,
                           style=wx.STAY_ON_TOP|wx.DEFAULT_DIALOG_STYLE|wx.RESIZE_BORDER)

        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        self.SetSizer(vbox)
        vbox.Add(hbox2, 1, wx.EXPAND|wx.ALL,1)      

        q_names = [x['name'] for x in queues['queues']]
        dp_setting = {"style":wx.CB_DROPDOWN, "choices": q_names}

        if "keywords" in queues:
            log_keywords = queues["keywords"]
        else:
            log_keywords = ["production", "debug"]
        ll = elp_setting(log_keywords)
        ll[4] = ["Queue", q_names[0], 4, dp_setting]
        self.elp=EditListPanel(self, ll)

        hbox2.Add(self.elp, 1, wx.EXPAND|wx.RIGHT|wx.LEFT, 1)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        button=wx.Button(self, wx.ID_ANY, "Cancel")
        button2=wx.Button(self, wx.ID_ANY, "Submit")

        hbox.Add(button, 0, wx.EXPAND)
        hbox.AddStretchSpacer()
        hbox.Add(button2, 0, wx.EXPAND)
        vbox.Add(hbox, 0, wx.EXPAND|wx.ALL,5)

        button.Bind(wx.EVT_BUTTON, self.onCancel)
        button2.Bind(wx.EVT_BUTTON, self.onSubmit)
        if value is not None:
            v, names = get_defaults()
            for k, n in enumerate(names):
                if n in value: v[k] = value[n]
            if not value[4] in q_names:
                value[4] = q_names[0]
                
            value7 = [False]*len(log_keywords)
            for name, v in value[7]:
                if name in log_keywords:
                    value7[log_keywords.index(name)]=v
            value[7] = value7

            self.elp.SetValue(value)

        #self.SetSizeHints(minH=-1, minW=size.GetWidth())
        self.SetSizeHints(minH=-1, minW=300)
        self.Show()
        self.Layout()
        self.Fit()
        size= self.GetSize()
        width = max(len(value[5])*12, 550)
        width = min(width, 1200)
        self.SetSize((width, size.GetHeight()))
        self.CenterOnScreen()
        #wx.CallAfter(self.Fit)
        self.value = self.elp.GetValue()
        
    def onCancel(self, evt):
        self.value = self.elp.GetValue()     
        self.EndModal(wx.ID_CANCEL)
        
    def onSubmit(self, evt):
        self.value = self.elp.GetValue()

        if (self.value[6].strip()=='' or 
            not any([x[1] for x in self.value[7]])):
            
            from ifigure.widgets.dialog import message
            message(self, title="Error", message="Enter job description and select at least one keyword")
            return
        
        self.EndModal(wx.ID_OK)
 
def get_job_submisson_setting(parent, servername = '', value = None,
                              queues = None):
    
    from petram.remote.client_script import base_remote_path

    if value[5] == '':
        from petram.remote.client_script import wdir_from_datetime
        value[5] = wdir_from_datetime()
    else:
        value[5] = os.path.basename(value[5])
    
    dlg = dlg_jobsubmission(parent, title='Submit to '+servername, value=value,
                            queues = queues)
    value = {}
    try:
        if dlg.ShowModal() == wx.ID_OK:
            value["num_nodes"] = dlg.value[0]
            value["num_cores"] = dlg.value[1]
            value["num_openmp"] = dlg.value[2]
            value["walltime"] = str(dlg.value[3])
            value["queue"] = str(dlg.value[4])
            value["retrieve_files"] = False
            value["rwdir"] = os.path.join(base_remote_path, dlg.value[5])
            value["log_txt"] = dlg.value[6]
            value["log_keywords"] = dlg.value[7]
            value["notification"] = dlg.value[8]            
            value["skip_mesh"] = dlg.value[9]
        else:
            pass
    finally:
        dlg.Destroy()
    return value 
        

 




