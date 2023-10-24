import os
import wx
from ifigure.utils.edit_list import EditListPanel, EDITLIST_CHANGED

setting1 = {"style": wx.CB_READONLY, "choices": (
    "None", "Begin", "End", "Fail", "All")}


def elp_setting(log_keywords):
    ll = [["Num. of Nodes", 1, 400, {}],

          ["Num. of Cores(total)", 16, 400, {}],
          ["Num. of OMP threads", 4, 400, {}],
          ["Wall clock", "00:15:00", 0, {}],
          ["Queue(Repo.)", "Debug", 0, {}],
          ["PetraM Ver.", "19700521", 0, {}],
          ["Remote Dir.", "", 0, {}],
          ["Note", None, 2235, {'nlines': 3}],
          ["Keywords", None, 36, {'col': 3, 'labels': list(log_keywords)}],
          ["Notification", "None", 4, setting1],
#          ["-> Adv. options", None, None, {"no_tlw_resize": True}],
          ["-> Adv. options", None, None, {"tlb_resize_samewidth": True}],
#          ["-> Adv. options", None, None, {"no_tlw_resize": False}],
          ["Run ", None, 2235, {'nlines': 3}],
          ["Env. ", None, 2235, {'nlines': 2}],
          ["<-"],
          [None, False, 3, {"text": "Skip sending mesh file"}],
          ]
    return ll


values = ['1', '1', '1', '00:10:00', 'regular(PROJ_19700521)', '', '',
          '', '', "None", '', '', False, False, ]

keys = ['num_nodes', 'num_cores', 'num_openmp', 'walltime',
        'queue', 'petramver', 'rwdir',
        'log_txt', 'log_keywords', 'notification', 'adv_opts', 'env_opts', 'skip_mesh',
        'retrieve_files']

def_queues = {'type': 'SLURM',
              'queus': [{'name': 'debug',
                         'maxnode': 1}]}


def get_defaults():
    return values[:], keys[:]


class dlg_jobsubmission(wx.Dialog):
    def __init__(self, parent, id=wx.ID_ANY, title='',
                 value=None, queues=None):

        if queues is None:
            queues = def_queues

        wx.Dialog.__init__(self, parent, wx.ID_ANY, title,
                           style=wx.STAY_ON_TOP | wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        self.SetSizer(vbox)
        vbox.Add(hbox2, 1, wx.EXPAND | wx.ALL, 1)

        q_names = [x['name'] for x in queues['queues']]
        dp_setting = {"style": wx.CB_DROPDOWN, "choices": q_names}

        v_names = list(queues['versions'])
        dv_setting = {"style": wx.CB_DROPDOWN, "choices": v_names}
        if "keywords" in queues:
            log_keywords = queues["keywords"]
        else:
            log_keywords = ["production", "debug"]
        ll = elp_setting(log_keywords)

        ll[4] = [ll[4][0], q_names[0], 4, dp_setting]
        ll[5] = [ll[5][0], v_names[0], 4, dv_setting]

        self.elp = EditListPanel(self, ll)

        hbox2.Add(self.elp, 1, wx.EXPAND | wx.RIGHT | wx.LEFT, 1)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        button = wx.Button(self, wx.ID_ANY, "Cancel")
        button2 = wx.Button(self, wx.ID_ANY, "Submit")

        hbox.Add(button, 0, wx.EXPAND)
        hbox.AddStretchSpacer()
        hbox.Add(button2, 0, wx.EXPAND)
        vbox.Add(hbox, 0, wx.EXPAND | wx.ALL, 5)

        button.Bind(wx.EVT_BUTTON, self.onCancel)
        button2.Bind(wx.EVT_BUTTON, self.onSubmit)

        #print("value", value)
        petram_version = value[4].split("_")[1][:-1]
        queue = value[4].split("_")[0]+")"
        if value is not None:
            v, names = get_defaults()
            for k, n in enumerate(names):
                if n in value:
                    value[n] = v[k]

            value[4] = queue
            value[5] = petram_version
            if not value[4] in q_names:
                value[4] = q_names[0]
            if not value[5] in v_names:
                value[5] = v_names[0]

            value8 = [False]*len(log_keywords)
            for name, v in value[8]:
                if name in log_keywords:
                    value8[log_keywords.index(name)] = v
            value[8] = value8

            tmp = [y for x, y in queues['versions']
                   [value[5]] if x == 'srun_option']
            value[10] = '\n'.join(tmp)
            tmp = [y for x, y in queues['versions']
                   [value[5]] if x == 'env_option']
            value[11] = '\n'.join(tmp)

            self.elp.SetValue(value)

        #self.SetSizeHints(minH=-1, minW=size.GetWidth())
        self.SetSizeHints(minH=-1, minW=300)

        self.Layout()
        self.Fit()
        size = self.GetSize()
        width = max(len(value[5])*12, 550)
        width = min(width, 1200)
        self.SetSize((width, size.GetHeight()))
        self.CenterOnParent()

        self.Show()
        # wx.CallAfter(self.Fit)

        self.value = self.elp.GetValue()
        self._queues = queues
        self.Bind(EDITLIST_CHANGED, self.onEL_Changed)

        user_notice = "\n".join(queues["notice"])
        from ifigure.widgets.dialog import message

        ret = message(parent=self,
                      message=user_notice,
                      title='Notice',
                      style=2,
                      icon=wx.ICON_EXCLAMATION,
                      center_on_screen=False,
                      center_on_parent=True,
                      labels=["Accept", "Cancel"])

        if ret == "cancel":
            wx.CallAfter(self.onCancel)

    def onCancel(self, evt=None):
        self.value = self.elp.GetValue()
        self.EndModal(wx.ID_CANCEL)

    def onSubmit(self, evt):
        self.value = self.elp.GetValue()

        if (self.value[7].strip() == '' or
                not any([x[1] for x in self.value[8]])):

            from ifigure.widgets.dialog import message
            message(self, title="Error",
                    message="Enter job description and select at least one keyword")
            return
        self.EndModal(wx.ID_OK)
        evt.Skip()

    def onEL_Changed(self, evt):
        # update adv. options if Petra-M
        value = self.elp.GetValue()

        if self.value[5] != value[5]:
            # need to update values for buttons
            value[8] = [y for x, y in value[8]]

            # update adv. options.
            #tmp = [y for x, y in self._queues['versions'][value[5]]]
            #value[10] = '\n'.join(tmp)
            tmp = [y for x, y in self._queues['versions']
                   [value[5]] if x == 'srun_option']
            value[10] = '\n'.join(tmp)
            tmp = [y for x, y in self._queues['versions']
                   [value[5]] if x == 'env_option']
            value[11] = '\n'.join(tmp)

            self.elp.SetValue(value)
            self.value = value

        evt.Skip()


def get_job_submisson_setting(parent, servername='', value=None,
                              queues=None):

    if value[6] == '':
        from petram.remote.client_script import wdir_from_datetime
        value[6] = wdir_from_datetime()
    else:
        value[6] = os.path.basename(value[6])

    dlg = dlg_jobsubmission(parent, title='Submit to '+servername, value=value,
                            queues=queues)

    value = {}
    base_remote_path = queues['scratch']
    try:
        if dlg.ShowModal() == wx.ID_OK:
            value["num_nodes"] = dlg.value[0]
            value["num_cores"] = dlg.value[1]
            value["num_openmp"] = dlg.value[2]
            value["walltime"] = str(dlg.value[3])

            queue_value = str(dlg.value[4])[:-1]+'_'+str(dlg.value[5])+")"
            value["queue"] = queue_value
            value["retrieve_files"] = False
            value["rwdir"] = os.path.join(base_remote_path, dlg.value[6])
            value["log_txt"] = dlg.value[7]
            value["log_keywords"] = dlg.value[8]
            value["notification"] = dlg.value[9]
            value["adv_opts"] = dlg.value[10]
            value["env_opts"] = dlg.value[11]
            value["skip_mesh"] = dlg.value[12]
        else:
            pass
    finally:
        dlg.Destroy()
    return value
