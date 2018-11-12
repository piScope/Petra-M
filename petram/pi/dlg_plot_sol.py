import os
import sys
import wx
import traceback
import numpy as np
import weakref
import subprocess as sp
import cPickle as pk
from collections import defaultdict
from weakref import WeakKeyDictionary as WKD

import ifigure.widgets.dialog as dialog
import ifigure.events
from ifigure.utils.edit_list import EditListPanel
from ifigure.utils.edit_list import EDITLIST_CHANGED
from ifigure.utils.edit_list import EDITLIST_CHANGING
from ifigure.utils.edit_list import EDITLIST_SETFOCUS
from ifigure.widgets.miniframe_with_windowlist import DialogWithWindowList, MiniFrameWithWindowList

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Dlg_plot_sol')

def setup_figure(fig, fig2):
    fig.nsec(1)
    fig.threed('on')
    fig.property(fig.get_axes(0), 'axis', False)
    fig.get_page(0).set_nomargin(True)
    fig.property(fig.get_page(0), 'bgcolor', 'white')
    xlim = fig2.xlim()
    ylim = fig2.ylim()
    zlim = fig2.zlim()
    fig.xlim(xlim)
    fig.ylim(ylim)
    fig.zlim(zlim)

def read_solinfo_remote(server, path):
    path = "/home/shiraiwa/myscratch/mfem_batch/pancake_10_2018v4_small_scale_0_2"    
    txt = "python -c \"from petram.sol.listsoldir import gather_soldirinfo_s;print gather_soldirinfo_s('"+path+"')\""

    command = ["ssh", server, txt]
    p = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT)
    p.wait()
    res = ''.join(p.stdout.readlines()))
    res = cPickle.loads(binascii.a2b_hex(res))    
    return res
    
from functools import wraps
import threading

ThreadEnd = wx.NewEventType()
EVT_THREADEND = wx.PyEventBinder(ThreadEnd, 1)

class _XY(tuple):
    def __call__(self, value):
        return (0, 0, 1., -value)
class _YZ(tuple):
    def __call__(self, value):
        return (1, 0, 0., -value)
class _ZX(tuple):
    def __call__(self, value):
        return (0, 1., 0., -value)

def get_mapper(mesh_in):
    from petram.mesh.mesh_utils import FaceOf, EdgeOf, PointOf
    def mapper1(*args):
        return FaceOf(args, mesh=mesh_in)
    def mapper2(*args):
        return EdgeOf(args, mesh=mesh_in)
    def mapper3(*args):
        return PointOf(args, mesh=mesh_in)
    return mapper1, mapper2, mapper3

def run_in_piScope_thread(func):
    @wraps(func)
    def func2(self, *args, **kwargs):
        title = self.GetTitle()
        self.SetTitle(title + '(*** processing ***)')
        app = wx.GetApp().TopWindow
        petram = app.proj.setting.parameters.eval('PetraM')
        if str(petram._status) != '':
            assert False, "other job is running (thread status error)"
        maxt = app.aconfig.setting['max_thread']
        if len(app.logw.threadlist) < maxt:
             args = (self,) + args
             t = threading.Thread(target=func, args=args, kwargs=kwargs)
             petram._status = 'evaluating sol...'             
             ifigure.events.SendThreadStartEvent(petram,
                                                 w=app,
                                                 thread=t,
                                                 useProcessEvent = True )
    return func2

#class DlgPlotSol(MiniFrameWithWindowList):
class DlgPlotSol(DialogWithWindowList):
    def __init__(self, parent, id = wx.ID_ANY, title = 'Plot Solution'):
        '''
        (use this style if miniframe is used)
        style=(wx.CAPTION|
                       wx.CLOSE_BOX|
                       wx.MINIMIZE_BOX| 
                       wx.RESIZE_BORDER|
                       wx.FRAME_FLOAT_ON_PARENT)
        '''
        style =  wx.CAPTION|wx.CLOSE_BOX#|wx.RESIZE_BORDER        
        from petram.sol.evaluators import def_config
        self.config = def_config
        remote = parent.model.param.eval('remote')
        if remote is not None:
            host =  parent.model.param.eval('host')            
            self.config['cs_soldir'] = remote['rwdir']
            self.config['cs_server'] = host.getvar('server')
            self.config['cs_user'] = host.getvar('user')
        
        
        super(DlgPlotSol, self).__init__(parent, id, title, style=style)

        self.nb =  wx.Notebook(self)
        box = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(box)
        box.Add(self.nb, 1, wx.EXPAND|wx.ALL, 1)
        
        tabs = ['GeomBdr', 'Edge', 'Bdr', 'Bdr(arrow)', 'Slice',
                'Probe', 'Config']
        self.pages = {}
        self.elps = {}
        for t in tabs:
            p = wx.Panel(self.nb) 
            self.nb.AddPage(p, t)
            self.pages[t] = p
            
        self.local_soldir    = None
        self.local_solsubdir = None

        # these are sol info
        self.local_sols = None                
        self.remote_sols = None

        text = 'all'
        mfem_model = parent.model.param.getvar('mfem_model')
        
        if 'GeomBdr' in tabs:
            p = self.pages['GeomBdr']
            vbox = wx.BoxSizer(wx.VERTICAL)
            p.SetSizer(vbox)
            
            choices = mfem_model['Phys'].keys()
            choices = [mfem_model['Phys'][c].fullpath() for c in choices]

            if len(choices)==0: choices = ['no physcs in mode']
            ll = [['x', 'x', 0, {}],
                  ['y', 'y', 0, {}],
                  ['z', 'z', 0, {}],
                  ['Boundary Index', text, 0, {}],
                  ['Physics', choices[0], 4, {'style':wx.CB_READONLY,
                                           'choices': choices}],
                  ['Color', ['blue', 'none'], 506, {}], 
                  [None, True, 3, {"text":'merge solutions'}],
                  [None, True, 3, {"text":'keep surface separated'}],
                  [None, True, 3, {"text":'show edge only'}],]                    

            elp = EditListPanel(p, ll)
            vbox.Add(elp, 1, wx.EXPAND|wx.ALL,1)
            self.elps['GeomBdr'] = elp

            hbox = wx.BoxSizer(wx.HORIZONTAL)
            vbox.Add(hbox, 0, wx.EXPAND|wx.ALL,5)
            ebutton=wx.Button(p, wx.ID_ANY, "Export")
            button=wx.Button(p, wx.ID_ANY, "Apply")
            ebutton.Bind(wx.EVT_BUTTON, self.onExport)                                    
            button.Bind(wx.EVT_BUTTON, self.onApply)
            hbox.Add(ebutton, 0, wx.ALL,1)            
            hbox.AddStretchSpacer()
            hbox.Add(button, 0, wx.ALL,1)

        if 'Point' in tabs:
            p = self.pages['Point']
            vbox = wx.BoxSizer(wx.VERTICAL)
            p.SetSizer(vbox)
            
            choices = mfem_model['Phys'].keys()
            choices = [mfem_model['Phys'][c].fullpath() for c in choices]

            if len(choices)==0: choices = ['no physcs in model']
            ll = [['Expression', '', 0, {}],
                  ['x:', '', 0, {}],
                  ['y:', '', 0, {}],
                  ['z:', '', 0, {}],
                  ['Physics', choices[0], 4, {'style':wx.CB_READONLY,
                                           'choices': choices}],      
                  [None, False, 3, {"text":'dynamic extension'}],]

            elp = EditListPanel(p, ll)
            vbox.Add(elp, 1, wx.EXPAND|wx.ALL,1)
            self.elps['Point'] = elp
            button=wx.Button(p, wx.ID_ANY, "Apply")
            button.Bind(wx.EVT_BUTTON, self.onApply)
            
            hbox = wx.BoxSizer(wx.HORIZONTAL)
            vbox.Add(hbox, 0, wx.EXPAND|wx.ALL,5)     
            hbox.AddStretchSpacer()
            hbox.Add(button, 0, wx.ALL,1)

        if 'Edge' in tabs:
            p = self.pages['Edge']
            vbox = wx.BoxSizer(wx.VERTICAL)
            p.SetSizer(vbox)
            
            choices = mfem_model['Phys'].keys()
            choices = [mfem_model['Phys'][c].fullpath() for c in choices]

            if len(choices)==0: choices = ['no physcs in model']
            ll = [['Expression', '', 0, {}],
                  ['Expression(x)', '', 0, {}],
                  ['Edge ', text, 0, {}],
                  ['Physics', choices[0], 4, {'style':wx.CB_READONLY,
                                           'choices': choices}],      
                  [None, False, 3, {"text":'dynamic extension'}],
                  [None, True, 3, {"text":'merge solutions'}],]

            elp = EditListPanel(p, ll)
            vbox.Add(elp, 1, wx.EXPAND|wx.ALL,1)
            self.elps['Edge'] = elp

            hbox = wx.BoxSizer(wx.HORIZONTAL)
            vbox.Add(hbox, 0, wx.EXPAND|wx.ALL,5)     
            #ibutton=wx.Button(p, wx.ID_ANY, "Integrate")
            ebutton=wx.Button(p, wx.ID_ANY, "Export")                     
            button=wx.Button(p, wx.ID_ANY, "Apply")
            #ibutton.Bind(wx.EVT_BUTTON, self.onInteg)
            ebutton.Bind(wx.EVT_BUTTON, self.onExport)            
            button.Bind(wx.EVT_BUTTON, self.onApply)
            hbox.Add(ebutton, 0, wx.ALL,1)                                  
            #hbox.Add(ibutton, 0, wx.ALL,1)                                  
            hbox.AddStretchSpacer()
            hbox.Add(button, 0, wx.ALL,1)
            #button.Enable(False)
        
        if 'Bdr' in tabs:
            p = self.pages['Bdr']
            vbox = wx.BoxSizer(wx.VERTICAL)
            p.SetSizer(vbox)
            
            choices = mfem_model['Phys'].keys()
            choices = [mfem_model['Phys'][c].fullpath() for c in choices]

            if len(choices)==0: choices = ['no physcs in model']
            
            s4 = {"style":wx.TE_PROCESS_ENTER,
                  "choices":[str(x+1) for x in range(10)]}
            ll = [['Expression', '', 0, {}],
                  ['Offset (x, y, z)', '0, 0, 0', 0, {}],
                  ['Boundary Index', 'all', 4, {'style':wx.CB_DROPDOWN,
                                                'choices': ['all', 'visible', 'hidden']}],      
                  ['Physics', choices[0], 4, {'style':wx.CB_READONLY,
                                           'choices': choices}],      
                  [None, False, 3, {"text":'dynamic extenstion'}],
                  [None, True, 3, {"text":'merge solutions'}],
                  ['Refine', 1, 104, s4],            
                  [None, True, 3, {"text":'averaging'}],]
            elp = EditListPanel(p, ll)
            vbox.Add(elp, 1, wx.EXPAND|wx.ALL,1)
            self.elps['Bdr'] = elp

            hbox = wx.BoxSizer(wx.HORIZONTAL)
            vbox.Add(hbox, 0, wx.EXPAND|wx.ALL,5)     
            ibutton=wx.Button(p, wx.ID_ANY, "Integrate")
            ebutton=wx.Button(p, wx.ID_ANY, "Export")                     
            button=wx.Button(p, wx.ID_ANY, "Apply")
            ibutton.Bind(wx.EVT_BUTTON, self.onInteg)
            ebutton.Bind(wx.EVT_BUTTON, self.onExport)            
            button.Bind(wx.EVT_BUTTON, self.onApply)
            hbox.Add(ebutton, 0, wx.ALL,1)                                  
            hbox.Add(ibutton, 0, wx.ALL,1)                                  
            hbox.AddStretchSpacer()
            hbox.Add(button, 0, wx.ALL,1)
            
        if 'Bdr(arrow)' in tabs:
            p = self.pages['Bdr(arrow)']
            vbox = wx.BoxSizer(wx.VERTICAL)
            p.SetSizer(vbox)
            
            choices = mfem_model['Phys'].keys()
            choices = [mfem_model['Phys'][c].fullpath() for c in choices]

            if len(choices)==0: choices = ['no physcs in model']
            ll = [['Expression(u)', '', 0, {}],
                  ['Expression(v)', '', 0, {}],
                  ['Expression(w)', '', 0, {}],
                  ['Boundary Index', text, 0, {}],
                  ['Physics', choices[0], 4, {'style':wx.CB_READONLY,
                                           'choices': choices}],
                  [None, False, 3, {"text":'dynamic extension (does not work)'}],
                  [None, True, 3, {"text":'merge solutions'}],
                  ['Arrow count', 300, 400, None],]


            elp = EditListPanel(p, ll)
            vbox.Add(elp, 1, wx.EXPAND|wx.ALL,1)
            self.elps['Bdr(arrow)'] = elp

            hbox = wx.BoxSizer(wx.HORIZONTAL)
            vbox.Add(hbox, 0, wx.EXPAND|wx.ALL,5)
            ebutton=wx.Button(p, wx.ID_ANY, "Export")                                 
            button=wx.Button(p, wx.ID_ANY, "Apply")
            ebutton.Bind(wx.EVT_BUTTON, self.onExport)            
            button.Bind(wx.EVT_BUTTON, self.onApply)
            hbox.Add(ebutton, 0, wx.ALL,1)                                              
            hbox.AddStretchSpacer()
            hbox.Add(button, 0, wx.ALL,1)

        if 'Slice' in tabs:
            p = self.pages['Slice']
            vbox = wx.BoxSizer(wx.VERTICAL)
            p.SetSizer(vbox)
            
            choices = mfem_model['Phys'].keys()
            choices = [mfem_model['Phys'][c].fullpath() for c in choices]

            if len(choices)==0: choices = ['no physcs in model']
            ll = [['Expression', '', 0, {}],
                  ['Plane', '1.0, 0, 0, 0', 0, {}],
                  ['Domain Index', text, 0, {}],
                  ['Physics', choices[0], 4, {'style':wx.CB_READONLY,
                                           'choices': choices}],      
                  [None, False, 3, {"text":'dynamic extension'}],
                  [None, True, 3, {"text":'merge solutions'}],]

            elp = EditListPanel(p, ll)
            vbox.Add(elp, 1, wx.EXPAND|wx.ALL,1)
            self.elps['Slice'] = elp

            hbox = wx.BoxSizer(wx.HORIZONTAL)
            vbox.Add(hbox, 0, wx.EXPAND|wx.ALL,5)
            ebutton=wx.Button(p, wx.ID_ANY, "Export")                                 
            button=wx.Button(p, wx.ID_ANY, "Apply")
            ebutton.Bind(wx.EVT_BUTTON, self.onExport)            
            button.Bind(wx.EVT_BUTTON, self.onApply)
            hbox.Add(ebutton, 0, wx.ALL,1)                                              
            hbox.AddStretchSpacer()
            hbox.Add(button, 0, wx.ALL,1)
            
        if 'Probe' in tabs:
            p = self.pages['Probe']
            vbox = wx.BoxSizer(wx.VERTICAL)
            p.SetSizer(vbox)
            
            choices = mfem_model['Phys'].keys()
            choices = [mfem_model['Phys'][c].fullpath() for c in choices]
            if len(choices)==0: choices = ['no physcs in model']

            ll = [['Expression', '', 0, {}],
                  ['Physics', choices[0], 4, {'style':wx.CB_READONLY,
                                           'choices': choices}], ]

            elp = EditListPanel(p, ll)
            vbox.Add(elp, 1, wx.EXPAND|wx.ALL,1)
            self.elps['Probe'] = elp

            hbox = wx.BoxSizer(wx.HORIZONTAL)
            vbox.Add(hbox, 0, wx.EXPAND|wx.ALL,5)
            button=wx.Button(p, wx.ID_ANY, "Apply")
            button.Bind(wx.EVT_BUTTON, self.onApply)
            hbox.AddStretchSpacer()
            hbox.Add(button, 0, wx.ALL,1)
            
        if 'Config' in tabs:
            p = self.pages['Config']
            vbox = wx.BoxSizer(wx.VERTICAL)
            p.SetSizer(vbox)
            
            elp1 = [["Sol", "sol", 4, {"style":wx.CB_READONLY,
                                       "choices": ["sol", ], 
                                       "UpdateUI": self.OnUpdateUI_local}],
                    ["Sub dir.", "None", 4, {"style":wx.CB_READONLY,
                                       "choices": ["", ]}, ],
                    [None, None, 141, {"alignright": True,
                                       "func": self.OnLoadLocalSol,
                                       "noexpand": True,
                                       "label": "Reload chocies"}],]
            elp2 = [["Number of workers", self.config['mp_worker'], 400,],
                    ["Sol", "sol", 4, {"style":wx.CB_READONLY,
                                       "choices": ["sol", ], }],
                    ["Sub dir.", "None", 4, {"style":wx.CB_READONLY,
                                       "choices": ["", ]}, ],
                    [None, None, 141, {"alignright": True,
                                       "func": self.OnLoadLocalSol,
                                       "noexpand": True,
                                       "label": "Reload chocies"}],]
            elp3 = [["Server", self.config['cs_server'], 0,],
                    ["Number of workers", self.config['cs_worker'], 400,],
                    ["Sol dir.", self.config['cs_soldir'], 0,], 
                    ["Sub dir.", "None", 4, {"style":wx.CB_READONLY,
                                       "choices": ["", ]}, ],
                    [None, None, 141, {"alignright": True,
                                       "func": self.OnUpdateUI_remote,
                                       "noexpand": True,
                                       "label": "Reload choices"}],]
            
            ll = [[None, None, 34, ({'text': "Worker Mode",
                                     'choices': ['Single', 'MP', 'C/S'],
                                     'call_fit': False},
                                    {'elp': elp1},
                                    {'elp': elp2},
                                    {'elp': elp3},),],]
            
            elp = EditListPanel(p, ll)
            print "widgets", elp.widgets
            vbox.Add(elp, 1, wx.EXPAND|wx.ALL,1)
            self.elps['Config'] = elp
            elp.SetValue([['Single', ['', 'sol', "", None],
                                     [2, 'sol', "", None],
                                     [self.config['cs_server'],
                                      self.config['cs_worker'],
                                      self.config['cs_soldir'],
                                      '', None]],])
            
            
        self.Show()
        self.Layout()
        self.SetSize((500, 400))
        self.Bind(EDITLIST_CHANGED, self.onEL_Changed)        
        self.Bind(EDITLIST_CHANGING, self.onEL_Changing)
        self.Bind(EDITLIST_SETFOCUS, self.onEL_SetFocus)
        self.Bind(EVT_THREADEND, self.onThreadEnd)

        self.Bind(wx.EVT_CLOSE, self.onClose)        
        wx.GetApp().add_palette(self)
        wx.CallAfter(self.CentreOnParent)

        self.solvars = WKD()
        self.evaluators = {}
        self.solfiles = {}

        self.Bind(wx.EVT_CHILD_FOCUS, self.OnChildFocus)

    def update_subdir_local(self, path, ss1):
        single_cb2 = self.elps['Config'].widgets[0][0].elps[0].widgets[1][0]
        multi_cb2  = self.elps['Config'].widgets[0][0].elps[1].widgets[2][0]

        from petram.sol.listsoldir import gather_soldirinfo
        info = gather_soldirinfo(path)
        
        dirnames = [""]
        choices = [""]
        solvers = info["checkpoint"].keys()
        for solver in solvers:
            kk = sorted(info["checkpoint"][solver].keys())
            for k in kk:
                dirnames.append(info["checkpoint"][solver][k])
                choices.append(solver+"("+str(k[1]) + ")")
        choices = choices + info["cases"]
        dirnames = dirnames + info["cases"]

        single_cb2.SetChoices(choices)
        multi_cb2.SetChoices(choices)

        if ss1 in dirnames:
            single_cb2.SetSelection(dirnames.index(ss1))
            multi_cb2.SetSelection(dirnames.index(ss1))
        else:
            ss1 = dirnames[0]

        probes = info["probes"] # mapping from probe name to file
        self.local_sols = (path, probes, dict(zip(choices, dirnames)))
        return ss1
        
    def update_sollist_local(self):
        model = self.GetParent().model

        sol_names = [name for name, child in  model.solutions.get_children()]
        sols =  [child for name, child in  model.solutions.get_children()]
        
        single_cb1 = self.elps['Config'].widgets[0][0].elps[0].widgets[0][0]
        multi_cb1  = self.elps['Config'].widgets[0][0].elps[1].widgets[1][0]

        param = model.param        
        solfiles = model.variables.getvar('solfiles')

        if self.local_soldir is not None:
            s1 = self.local_soldir
            ss1 = self.local_solsubdir
        else:
            if model.param.eval('sol') is not None:
                s1 = model.param.eval('sol').owndir()
                ss1 = ""
            else:
                s1 = None
                ss1 = None

        if s1 is None:
           s1 = str(single_cb1.GetString(single_cb1.GetSelection()))
           s2 = str(multi_cb1.GetString(multi_cb1.GetSelection()))
        else:
           s2 = s1
        
        single_cb1.SetChoices(sol_names)#+["Other..."])
        if s1 in sol_names:
            single_cb1.SetSelection(sol_names.index(s1))

        multi_cb1.SetChoices(sol_names)#+["Other..."])                    
        if s2 in sol_names:
            multi_cb1.SetSelection(sol_names.index(s2))            

        s1 = str(single_cb1.GetString(single_cb1.GetSelection()))
        s2 = str(multi_cb1.GetString(multi_cb1.GetSelection()))
                    
        single_cb2 = self.elps['Config'].widgets[0][0].elps[0].widgets[1][0]
        multi_cb2  = self.elps['Config'].widgets[0][0].elps[1].widgets[2][0]

        path = sols[sol_names.index(s1)].owndir()
        self.update_subdir_local(path, ss1)

    def OnLoadLocalSol(self, evt):
        self.update_sollist_local()
        self.load_sol_if_needed()
        '''
        from petram.sol.listsoldir import gather_soldirinfo
        info = gather_soldirinfo(path)
        
        dirnames = [""]
        choices = [""]
        solvers = info["checkpoint"].keys()
        for solver in solvers:
            kk = sorted(info["checkpoint"][solver].keys())
            for k in kk:
                dirnames.append(info["checkpoint"][solver][k])
                choices.append(solver+"("+str(k[1]) + ")")
        choices = choices + info["cases"]
        dirnames = dirnames + info["cases"]

        single_cb2.SetChoices(choices)
        multi_cb2.SetChoices(choices)

        if ss1 in dirnames:
            single_cb2.SetSelection(dirnames.index(ss1))
            multi_cb2.SetSelection(dirnames.index(ss1))        

        probes = info["probes"] # mapping from probe name to file
        self.local_sols = (path, probes, dict(zip(choices, dirnames)))
        '''
        
    def OnUpdateUI_local(self, evt):
        print "single UI update", evt.GetEventObject().GetParent()        
        self.update_sollist_local()

    def OnUpdateUI_remote(self, evt):
        print "CS UI update", evt.GetEventObject().GetParent()
         
    def OnChildFocus(self, evt):
        self.GetParent()._palette_focus = 'plot'        
        evt.Skip()

    def post_threadend(self, func, *args, **kwargs):
        evt = wx.PyCommandEvent(ThreadEnd, wx.ID_ANY)
        evt.pp_method = (func, args, kwargs)
        wx.PostEvent(self, evt)
        
    def set_title_no_status(self):
        title = self.GetTitle()
        self.SetTitle(title.split('(')[0])

    def onThreadEnd(self, evt):
        self.set_title_no_status()
        m = evt.pp_method[0]
        args = evt.pp_method[1]
        kargs = evt.pp_method[2]
        m(*args, **kargs)
        evt.Skip()
        
    def onClose(self, evt):
        wx.GetApp().rm_palette(self)
        self.Destroy()
        evt.Skip()

    def load_sol_if_needed(self):
        from petram.sol.solsets import read_sol, find_solfiles        
        model = self.GetParent().model
        solfiles = model.variables.getvar('solfiles')

        doit = False
        if solfiles is not None:
            cpath = os.path.dirname(solfiles.set[0][0][0])
            npath = os.path.join(self.local_soldir, self.local_solsubdir)

            if npath != cpath: doit =True
        else:
            doit = True
            if self.local_soldir is not None:
                npath = os.path.join(self.local_soldir, self.local_solsubdir)
            else:
                npath = model.param.eval('sol').owndir()
                self.local_soldir = npath
                self.local_solsubdir = ""
        if doit:
            try:
                print "reading sol from ", npath
                solfiles = find_solfiles(path = npath)
                model.variables.setvar('solfiles', solfiles)          
            except:
                import traceback
                traceback.print_exc()
                model.variables.delvar('solfiles')

    def onEL_Changed(self, evt):
        model = self.GetParent().model
        v  = self.elps['Config'].GetValue()
        #print str(v[0][0])
        print v
        if str(v[0][0]) == 'Single':
            if (self.config['use_mp'] or
                self.config['use_cs']):
                self.evaluators = {}                
            self.config['use_mp'] = False
            self.config['use_cs'] = False
            model.variables.setvar('remote_soldir', None)

            #info (path, probes, dirnames)
            sol = model.solutions.get_child(name=str(v[0][1][0]))
            ss1 = self.local_sols[2][str(v[0][1][1])]
            ss1 = self.update_subdir_local(sol.owndir(), ss1)
            self.local_soldir    = sol.owndir()
            self.local_solsubdir = ss1

            self.load_sol_if_needed()
            
        elif str(v[0][0]) == 'MP':
            if not self.config['use_mp']:
                self.evaluators = {}
            if self.config['mp_worker'] != v[0][2][0]:
                self.evaluators = {}
            self.config['mp_worker'] = v[0][2][0]
            self.config['use_mp'] = True
            self.config['use_cs'] = False
            model.variables.setvar('remote_soldir', None)
            
            sol = model.solutions.get_child(name=str(v[0][1][0]))
            ss1 = self.local_sols[2][str(v[0][1][1])]
            ss1 = self.update_subdir_local(sol.owndir(), ss1)
            self.local_soldir    = sol.owndir()
            self.local_solsubdir = ss1

            self.load_sol_if_needed()
            
        elif str(v[0][0]) == 'C/S':
            if not self.config['use_cs']:
                self.evaluators = {}
            if self.config['cs_worker'] != v[0][3][1]:
                self.evaluators = {}
                
            self.config['cs_worker'] = v[0][3][1]
            self.config['cs_server'] = str(v[0][3][0])
            self.config['cs_soldir'] = str(v[0][3][2])                        
            self.config['use_mp'] = False
            self.config['use_cs'] = True
            model.variables.setvar('remote_soldir', self.config['cs_soldir'])
            
        #print('EL changed', self.config)

    def onEL_Changing(self, evt):
        pass
    def onEL_SetFocus(self, evt):
        pass
    
    def onApply(self, evt):
        t = self.get_selected_plotmode()                        
        m = getattr(self, 'onApply'+t)
        m(evt)
    def onInteg(self, evt):
        t = self.get_selected_plotmode()                
        m = getattr(self, 'onInteg'+t)
        m(evt)
    def onExport(self, evt):
        t = self.get_selected_plotmode()        
        m = getattr(self, 'onExport'+t)
        m(evt)
    def onExport2(self, evt):
        t = self.get_selected_plotmode()
        m = getattr(self, 'onExport2'+t)
        m(evt)
        
    def get_selected_plotmode(self, kind = False):
        t = self.nb.GetPageText(self.nb.GetSelection())
        t = t.replace('(','').replace(')','')
        if kind:
            kinds = {'Bdr': 'bdry',
                     'BdrArrorw': 'bdry',
                     'Edge': 'edge',
                     'Slice': 'domain',
                     'Domain': 'domain'}
            i = getattr(self, 'get_attrs_field_'+t)
            value = self.elps[t].GetValue()
            attrs = str(value[i()])
            if attrs.strip().lower() != 'all':            
               attrs = [int(x) for x in attrs.split(',') if x.strip() != '']            
            return kinds[t], attrs
        else: 
            return t
                       
    def add_selection(self, sel):
        t = self.get_selected_plotmode()                
        i = getattr(self, 'get_attrs_field_'+t)
        attrs = self.elps[t].GetValue()[i()]
        if attrs.strip().lower() == 'all': return
        attrs = sorted(set([int(x) for x in attrs.split(',')]+sel))
        self.set_selection(attrs)
        
    def rm_selection(self, sel):
        t = self.get_selected_plotmode()                        
        i = getattr(self, 'get_attrs_field_'+t)
        attrs = self.elps[t].GetValue()[i()]
        if attrs.strip().lower() == 'all': return
        print(attrs, sel)
        attrs = sorted([int(x) for x in attrs.split(',') if not int(x) in sel])
        self.set_selection(attrs)
        
    def set_selection(self, sel):
        t = self.get_selected_plotmode()                               
        i = getattr(self, 'get_attrs_field_'+t)
        txt = ', '.join([str(s) for s in sel])
        v = self.elps[t].GetValue()
        v[i()] = txt
        self.elps[t].SetValue(v)
    
    #    
    #   Edge value ('Edge' tab)
    #
    @run_in_piScope_thread        
    def onApplyEdge(self, evt):
        value = self.elps['Edge'].GetValue()
        expr = str(value[0]).strip()
        expr_x = str(value[1]).strip()
        
        if value[4]:
            from ifigure.widgets.wave_viewer import WaveViewer
            cls = WaveViewer
        else:
            cls = None
            
        data, data_x, battrs = self.eval_edge(mode = 'plot')
        if data is None: return

        self.post_threadend(self.make_plot_edge, data, battrs,
                            data_x = data_x,
                            cls = cls, expr = expr, expr_x = expr_x)
        
    def make_plot_edge(self, data, battrs,
                             data_x = None, cls = None,
                             expr='', expr_x=''):
        from ifigure.interactive import figure
        if data_x is None:
            v = figure(viewer = cls)
            v.update(False)        
            setup_figure(v, self.GetParent())                
            v.suptitle(expr + ':' + str(battrs))
            for verts, cdata, adata in data:
               if cls is None:
                    v.solid(verts, adata, cz=True, cdata= cdata.astype(float),
                            shade='linear')                    
               else:
                    v.solid(verts, adata, cz=True, cdata= cdata,
                            shade='linear')
            v.update(True)
            v.view('noclip')
            v.view('equal')
            v.update(False)                
            ax = self.GetParent().get_axes()
            param = ax.get_axes3d_viewparam(ax._artists[0])
            ax2 = v.get_axes()
            ax2.set_axes3d_viewparam(param, ax2._artists[0])
            v.lighting(light = 0.5)
            v.update(True)
        else:  # make 2D plot
            v = figure();
            for yy, xx in zip(data, data_x):
                y = yy[1].flatten()
                x = xx[1].flatten()
                xidx = np.argsort(x)
                v.plot(x[xidx], y[xidx])

    '''
    This should be changed to perform line integration?
    def onIntegEdge(self, evt):
        value = self.elps['Edge'] .GetValue()
        expr = str(value[0]).strip()

        from petram.sol.evaluators import area_tri
        data, battrs = self.eval_edge(mode = 'integ')
        if data is None: return
        
        integ = 0.0
        for verts, cdata in data:
            area = area_tri(verts)
            integ += np.sum(area * np.mean(cdata, 1))

        print("Area Ingegration")
        print("Expression : " + expr)
        print("Boundary Index :" + str(list(battrs)))
        print("Value : "  + str(integ))
    '''
    def onExportEdge(self, evt):
        from petram.sol.evaluators import area_tri
        data, data_x, battrs = self.eval_edge(mode = 'integ')
        if data is None: return
        
        verts, cdata = data[0]
        data = {'vertices': verts, 'data': cdata}
        self.export_to_piScope_shell(data, 'bdr_data')
                       
    def get_attrs_field_Edge(self):
        return 2
                       
    def eval_edge(self, mode = 'plot'):
        from petram.sol.evaluators import area_tri
        value = self.elps['Edge'] .GetValue()
        
        expr = str(value[0]).strip()
        expr_x = str(value[1]).strip()
        battrs = str(value[2])
        phys_path = value[3]
        if mode == 'plot':
            do_merge1 = value[5]
        else:
            do_merge1 = True

        data, void = self.evaluate_sol_edge(expr, battrs, phys_path,
                                             do_merge1, True)
        if data is None: return None, None, None

        if expr_x != '':
            data_x, void = self.evaluate_sol_edge(expr_x, battrs, phys_path,
                                                    do_merge1, True)
            if data_x is None: return None, None, None
        else:
            data_x = None
        return data, data_x, battrs
        

    #    
    #   Boundary value ('Bdr' tab)
    #
    '''
    ll = [['Expression', '', 0, {}],
                  ['Offset (x, y, z)', '0, 0, 0', 0, {}],                  
                  ['Boundary Index', text, 0, {}],
                  ['Physics', choices[0], 4, {'style':wx.CB_READONLY,
                                           'choices': choices}],      
                  [None, False, 3, {"text":'dynamic extenstion'}],
                  [None, True, 3, {"text":'merge solutions'}],
                  ['Refine', 1, 104, s4],            
                  [None, True, 3, {"text":'averaging'}],]
    '''
    @run_in_piScope_thread    
    def onApplyBdr(self, evt):
        value = self.elps['Bdr'] .GetValue()
        expr = str(value[0]).strip()
        
        if value[4]:
            from ifigure.widgets.wave_viewer import WaveViewer
            cls = WaveViewer
        else:
            cls = None
        refine = int(value[6])
        data, battrs = self.eval_bdr(mode = 'plot', refine=refine)
        if data is None: return
        self.post_threadend(self.make_plot_bdr, data, battrs,
                            cls = cls, expr = expr)
        
    def make_plot_bdr(self, data, battrs, cls = None, expr=''):
        
        from ifigure.interactive import figure
        viewer = figure(viewer = cls)
        viewer.update(False)        
        setup_figure(viewer, self.GetParent())                
        viewer.suptitle(expr + ':' + str(battrs))

        dd = defaultdict(list)
        # regroup to sepparte triangles and quads.
        for k, datasets in enumerate(data):
            v, c, i = datasets #verts, cdata, idata
            idx = i.shape[-1]
            dd[idx].append((k+1, v, c, i))

        for key in dd.keys():
            kk, verts, cdata, idata = zip(*(dd[key]))
            #print([v.shape for v in verts])
            #print([v.shape for v in cdata])
            #print([v.shape for v in idata])                
            offsets = np.hstack((0, np.cumsum([len(c) for c in cdata], dtype=int)))[:-1]
            offsets_idx = np.hstack([np.zeros(len(a), dtype=int)+o
                                 for o, a in zip(offsets, idata)])
            array_idx = np.hstack([np.zeros(len(c), dtype=int)+k
                                  for k, c in zip(kk, cdata)])
            array_idx = array_idx + 1
                                  
            verts = np.vstack(verts)
            cdata = np.hstack(cdata)                
            idata = np.vstack(idata)
            idata = idata + np.atleast_2d(offsets_idx).transpose()


            if cls is None:
               obj = viewer.solid(verts, idata, array_idx=array_idx,
                       cz=True, cdata= cdata.astype(float),
                       shade='linear', )
               obj.set_gl_hl_use_array_idx(True)
            else:
               obj = viewer.solid(verts, idata, array_idx=array_idx,
                       cz=True, cdata= cdata, shade='linear')
               obj.set_gl_hl_use_array_idx(True)
               
        viewer.update(True)
        viewer.view('noclip')
        viewer.view('equal')
        viewer.update(False)                
        ax = self.GetParent().get_axes()
        param = ax.get_axes3d_viewparam(ax._artists[0])
        ax2 = viewer.get_axes()
        ax2.set_axes3d_viewparam(param, ax2._artists[0])
        viewer.lighting(light = 0.5)
        viewer.update(True)

    def onIntegBdr(self, evt):
        value = self.elps['Bdr'] .GetValue()
        expr = str(value[0]).strip()

        from petram.sol.evaluators import area_tri
        data, battrs = self.eval_bdr(mode = 'integ')
        if data is None: return
        
        integ = 0.0
        for verts, cdata, adata in data:
            v = verts[adata]
            c = cdata[adata, ...]
            area = area_tri(v)
            integ += np.sum(area * np.mean(c, 1))

        print("Area Ingegration")
        print("Expression : " + expr)
        print("Boundary Index :" + str(list(battrs)))
        print("Value : "  + str(integ))

    def onExportBdr(self, evt):
        from petram.sol.evaluators import area_tri
        data, battrs = self.eval_bdr(mode = 'integ')
        if data is None: return
        
        verts, cdata, adata = data[0]
        data = {'vertices': verts, 'data': cdata, 'index': adata}
        self.export_to_piScope_shell(data, 'bdr_data')

    def get_attrs_field_Bdr(self):
        return 1
                       
    def eval_bdr(self, mode = 'plot', export_type = 1, refine = 1):
        from petram.sol.evaluators import area_tri
        value = self.elps['Bdr'] .GetValue()
        
        expr = str(value[0]).strip()
        battrs = str(value[2])
        phys_path = value[3]
        if mode == 'plot':
            do_merge1 = value[5]
            do_merge2 = True
        elif mode == 'integ':
            do_merge1 = True
            do_merge2 = False
        else:
            do_merge1 = False
            do_merge2 = False
                  
        average = value[7]
        data, battrs2 = self.evaluate_sol_bdr(expr, battrs, phys_path,
                                             do_merge1, do_merge2,
                                             export_type = export_type,
                                             refine = refine,
                                             average = average)
        uvw = str(value[1]).split(',')
        if len(uvw) == 3:
            for kk, expr in enumerate(uvw):
                try:
                    u = float(expr.strip())
                    isfloat=True
                except:
                    isfloat=False                    
                    u, battrs2 = self.evaluate_sol_bdr(expr.strip(),
                                                       battrs, phys_path,
                                                       do_merge1, do_merge2,
                                                       export_type = export_type,
                                                       refine = refine,
                                                       average = average)
                data = [list(x) for x in data]
                for k, datasets in enumerate(data):
                    if datasets[0].shape[1]==2:
                        datasets[0] = np.hstack((datasets[0],
                                                 np.zeros((datasets[0].shape[0], 1))))
                    elif datasets[0].shape[1]==1:
                        datasets[0] = np.hstack((datasets[0],
                                                 np.zeros((datasets[0].shape[0], 2))))
                        
                    if isfloat:
                        datasets[0][:,kk] += u
                    else:
                        datasets[0][:,kk] += u[k][1]
                
        if data is None: return None, None
        return data, battrs
        

    #    
    #   Geometry Boundary ('GeomBdr' tab)
    #
    def onApplyGeomBdr(self, evt):
        x, y, z = self.eval_geombdr(mode = 'plot')
        
        value = self.elps['GeomBdr'] .GetValue()        
        battrs = str(value[3])
        edge_only = bool(value[8])
        
        c1 = value[5][0]; c2 = value[5][1]
        kwargs = {'facecolor': c1,
                  'edgecolor': c2,}
        if c2 == (0,0,0,0): kwargs['linewidth'] = 0.

        from ifigure.interactive import figure        
        v = figure()
        v.update(False)        
        setup_figure(v, self.GetParent())                
        v.suptitle('Boundary '+ str(battrs))
        for xdata, ydata, zdata in zip(x, y, z):
            verts = np.vstack((xdata[1], ydata[1], zdata[1])).transpose()
            adata = xdata[2]
            v.solid(verts, adata, **kwargs)

        v.update(True)
        v.view('noclip')
        v.view('equal')
        v.update(False)                
        ax = self.GetParent().get_axes()
        param = ax.get_axes3d_viewparam(ax._artists[0])
        ax2 = v.get_axes()
        ax2.set_axes3d_viewparam(param, ax2._artists[0])
        v.lighting(light = 0.5)
        v.update(True)

    def onExportGeomBdr(self, evt):
        from petram.sol.evaluators import area_tri
        x, y, z = self.eval_geombdr(mode = 'integ')
        #if data is None: return
        
        verts = np.dstack((x[0][1], y[0][1], z[0][1]))
        data = {'vertices': verts}
        self.export_to_piScope_shell(data, 'geom_data')
                       
    def get_attrs_field_GeomBdr(self):
        return 3
        
    def eval_geombdr(self, mode = 'plot'):        
        value = self.elps['GeomBdr'] .GetValue()
        cls = None
        expr_x = str(value[0]).strip()
        expr_y = str(value[1]).strip()
        expr_z = str(value[2]).strip()

        battrs = str(value[3])
        phys_path = value[4]
        edge_only = bool(value[8])        
        if mode  == 'plot':
            do_merge1 = value[6]
            do_merge2 = value[7]
        else:
            do_merge1 = True
            do_merge2 = False
        if edge_only:
            do_merge1 = False
            do_merge2 = False
            
        def call_eval_sol_bdr(expr, battrs = battrs, phys_path = phys_path,
                              do_merge1 = do_merge1, do_merge2 = do_merge2,
                              edge_only = edge_only):
            if str(expr).strip() != '':
                v, battrs = self.evaluate_sol_bdr(expr, battrs, phys_path,
                                                  do_merge1, do_merge2,
                                                  edge_only = edge_only)
            else:
                v = None
            return v
        x = call_eval_sol_bdr(expr_x)
        y = call_eval_sol_bdr(expr_y)
        z = call_eval_sol_bdr(expr_z)        
        if x is None and y is None and z is None: return
        basedata = x
        if basedata is None: basedata = y
        if basedata is None: basedata = z

        zerodata = [(None, cdata * 0, adata) for verts, cdata, adata
                    in basedata]
        if x is None: x = zerodata
        if y is None: y = zerodata
        if z is None: z = zerodata
        return x, y, z

    #    
    #   Arrow on Boundary ('Bdr(arrow)' tab)
    #
    def onApplyBdrarrow(self, evt):
        u, v, w, battrs= self.eval_bdrarrow(mode = 'plot')
        
        value = self.elps['Bdr(arrow)'] .GetValue()
        
        expr_u = str(value[0]).strip()
        expr_v = str(value[1]).strip()
        expr_w = str(value[2]).strip()
        if value[5]:
            from ifigure.widgets.wave_viewer import WaveViewer
            cls = WaveViewer
        else:
            cls = None
            
        self.post_threadend(self.make_plot_bdrarrow, u, v, w, battrs, value,
                            expr_u = expr_u,
                            expr_v = expr_v,
                            expr_w = expr_w,                            
                            cls = cls)
        
    def make_plot_bdrarrow(self, u, v, w, battrs, value,
                            expr_u = '', expr_v = '', expr_w = '',
                            cls = None):

        from ifigure.interactive import figure        
        viewer = figure(viewer = cls)
        viewer.update(False)                        
        setup_figure(viewer,  self.GetParent())
        viewer.suptitle('['+ ','.join((expr_u, expr_v, expr_w)) + '] : '+ str(battrs))

        allxyz = np.vstack([udata[0] for udata in u])
        dx = np.max(allxyz[:,0])-np.min(allxyz[:,0])
        if allxyz.shape[1]>1:
            dy = np.max(allxyz[:,1])-np.min(allxyz[:,1])
        else:
            dy = dx*0.
        if allxyz.shape[1]>2:           
            dz = np.max(allxyz[:,2])-np.min(allxyz[:,2])
        else:
            dz = dy*0.
        length = np.max((dx, dy, dz))/20.
        
        for udata, vdata, wdata in zip(u, v, w):
           xyz = udata[0]
               
           u = udata[1]
           v = vdata[1]
           w = wdata[1]

           ll = np.min([xyz.shape[0]-1,int(value[7])])
           idx = np.linspace(0, xyz.shape[0]-1,ll).astype(int)
           
           x = xyz[idx,0]
           if xyz.shape[1]>1:
               y = xyz[idx,1]
           else:
               y = x*0.
           if xyz.shape[1]>2:
               z = xyz[idx,2]
           else:
               z = x*0.
               
           viewer.quiver3d(x, y, z, u[idx], v[idx], w[idx],
                           length = length)

        viewer.update(True)
        viewer.view('noclip')
        viewer.view('equal')
        viewer.update(False)                
        ax = self.GetParent().get_axes()
        param = ax.get_axes3d_viewparam(ax._artists[0])
        ax2 = viewer.get_axes()
        ax2.set_axes3d_viewparam(param, ax2._artists[0])
        viewer.lighting(light = 0.5)
        viewer.update(True)
        
    def onExportBdrarrow(self, evt):
        u, v, w, battrs= self.eval_bdrarrow(mode = 'export')        
        udata = u[0][1]
        vdata = v[0][1]
        wdata = w[0][1]
        verts = v[0][0]
        xyz = np.mean(verts, 1)
        u = np.mean(udata, 1)
        v = np.mean(vdata, 1)
        w = np.mean(wdata, 1)           
        data = {'x': xyz[:,0],
                'y': xyz[:,1],
                'z': xyz[:,2],
                'u': u,
                'v': v,
                'w': w}
        self.export_to_piScope_shell(data, 'arrow_data')
                       
    def get_attrs_field_Bdrarrow(self):
        return 3
    
    def eval_bdrarrow(self, mode = 'plot'):        
        value = self.elps['Bdr(arrow)'] .GetValue()
        cls = None
        expr_u = str(value[0]).strip()
        expr_v = str(value[1]).strip()
        expr_w = str(value[2]).strip()

        battrs = str(value[3])
        phys_path = value[4]
        if mode  == 'plot':
            do_merge1 = value[6]
            do_merge2 = False
        else:
            do_merge1 = True
            do_merge2 = False
            
        def call_eval_sol_bdr(expr, battrs = battrs, phys_path = phys_path,
                              do_merge1 = do_merge1, do_merge2 = do_merge2):
            if str(expr).strip() != '':
                v, battrs = self.evaluate_sol_bdr(expr, battrs, phys_path,
                                              do_merge1, do_merge2)
            else:
                v = None
                battrs = None
            return v, battrs

        u, ubattrs = call_eval_sol_bdr(expr_u)
        v, vbattrs = call_eval_sol_bdr(expr_v)
        w, wbattrs = call_eval_sol_bdr(expr_w)
        if u is None and v is None and w is None: return
                     
        basedata = u; battrs = ubattrs
        if basedata is None:
            basedata = v
            battrs = vbattrs
        if basedata is None:
            basedata = w
            battrs = wbattrs

        zerodata = [(verts, cdata * 0, adata) for verts, cdata, adata in basedata]
        if u is None: u = zerodata
        if v is None: v = zerodata
        if w is None: w = zerodata
        return u, v, w, battrs
    #    
    #   Slice plane ('Slice' tab)
    #
    @run_in_piScope_thread    
    def onApplySlice(self, evt):
        value = self.elps['Slice'] .GetValue()
        expr = str(value[0]).strip()
        
        if value[4]:
            from ifigure.widgets.wave_viewer import WaveViewer
            cls = WaveViewer
        else:
            cls = None
            
        data, battrs = self.eval_slice(mode = 'plot')
        if data is None:
            wx.CallAfter(dialog.message, parent = self,
                         message ='Error in evaluating slice', 
                         title ='Error')
            wx.CallAfter(self.set_title_no_status)        
            return
        self.post_threadend(self.make_plot_slice, data, battrs,
                            cls = cls, expr = expr)
        
    def make_plot_slice(self, data, battrs, cls = None, expr=''):
        from ifigure.interactive import figure
        v = figure(viewer = cls)
        v.update(False)        
        setup_figure(v, self.GetParent())                
        v.suptitle(expr + ':' + str(battrs))
        for verts, cdata, adata in data:
           if cls is None:
                v.solid(verts, adata,  cz=True, cdata= cdata.astype(float),
                        shade='linear')                    
           else:
                v.solid(verts, adata, cz=True, cdata= cdata, shade='linear')
        v.update(True)
        v.view('noclip')
        v.view('equal')
        v.update(False)                
        ax = self.GetParent().get_axes()
        param = ax.get_axes3d_viewparam(ax._artists[0])
        ax2 = v.get_axes()
        ax2.set_axes3d_viewparam(param, ax2._artists[0])
        v.lighting(light = 0.5)
        v.update(True)
                       
    def get_attrs_field_Slice(self):
        return 2
        
    def eval_slice(self, mode = 'plot'):
        from petram.sol.evaluators import area_tri
        value = self.elps['Slice'] .GetValue()
        
        expr = str(value[0]).strip()
        #plane = [float(x) for x in str(value[1]).split(',')]
        plane = str(value[1])
        attrs = str(value[2])
        phys_path = value[3]
        if mode == 'plot':
            do_merge1 = value[5]
            do_merge2 = False
        else:
            do_merge1 = True
            do_merge2 = False
        data, verts = self.evaluate_sol_slice(expr, attrs, plane,  phys_path,
                                              do_merge1, do_merge2)

        if data is None: return None, None
        return data, verts
    
    @run_in_piScope_thread    
    def onApplyProbe(self, evt):
        value = self.elps['Probe'] .GetValue()
        expr = str(value[0]).strip()
        
        xdata, data = self.eval_probe(mode = 'plot')
        if data is None:
            wx.CallAfter(dialog.message, parent = self,
                         message ='Error in evaluating probe', 
                         title ='Error')
            wx.CallAfter(self.set_title_no_status)        
            return
        self.post_threadend(self.make_plot_probe, (xdata, data), expr = expr)
        
    def make_plot_probe(self, data, expr='', cls=None):
        from ifigure.interactive import figure
        v = figure(viewer = cls)
        v.update(False)        
        v.suptitle(expr)
        v.plot(data[0], data[1])
        v.update(True)        
        
    def eval_probe(self, mode = 'plot'):
        value = self.elps['Probe'] .GetValue()
        expr = str(value[0]).strip()
        phys_path = value[1]        
        xdata, data = self.evaluate_sol_probe(expr, phys_path)
        return xdata, data
        
    #
    #   common routines
    #
    def evaluate_sol_edge(self, expr, battrs, phys_path, do_merge1, do_merge2,
                         **kwargs):
        '''
        evaluate sol using boundary evaluator
        '''
        model = self.GetParent().model
        solfiles = self.get_model_soldfiles()        
        mfem_model = model.param.getvar('mfem_model')
        phys_ns = mfem_model[str(phys_path)]._global_ns.copy()
        
        if solfiles is None:
             wx.CallAfter(dialog.showtraceback, parent = self,
                          txt='Solution does not exist',
                          title='Error',
                          traceback='')
             wx.CallAfter(self.set_title_no_status)                     
             return None, None
        mesh = model.variables.getvar('mesh')
        if mesh is None: return
        
        FaceOf, EdgeOf, PointOf = get_mapper(mesh)
        ll = {'FaceOf': FaceOf, 'EdgeOf': EdgeOf, 'PointOf':PointOf}
        
        battrs = str(battrs).strip()
        if battrs.lower() == 'all':
            battrs = ['all']            
        else:
            try:
               battrs = list(np.atleast_1d(eval(battrs, ll, phys_ns)))
            except:
               import traceback
               traceback.print_exc()
               assert False, "invalid selection: " + battrs
        
            '''
            battrs0 = [int(x) for x in battrs.split(',')]
            if mesh.Dimension() == 3:
               s = self.GetParent()._s_v_loop['phys'][0]
               battrs = []
               for i in battrs0:
                   connected_surf = []                
                   for k in s.keys():
                       if i in s[k]: connected_surf.append(k)
                   battrs.append(tuple(connected_surf))
            elif mesh.Dimension() == 2:
               battrs = battrs0               
            else:
               assert False, "edge plot is not supported for 1D"
            '''

        if 'Edge' in self.evaluators:
            try:
                self.evaluators['Edge'].validate_evaluator('EdgeNodal', battrs, solfiles)
            except IOError:
                dprint1("IOError detected setting failed=True")
                self.evaluators['Edge'].failed = True
           
        from petram.sol.evaluators import build_evaluator
        if (not 'Edge' in self.evaluators or
            self.evaluators['Edge'].failed):
            self.evaluators['Edge'] =  build_evaluator(battrs,
                                               mfem_model,
                                               solfiles,
                                               name = 'EdgeNodal',
                                               config = self.config)
            
        self.evaluators['Edge'].validate_evaluator('EdgeNodal', battrs, 
                                                   solfiles)

        try:
            self.evaluators['Edge'].set_phys_path(phys_path)
            return self.evaluators['Edge'].eval(expr, do_merge1, do_merge2,
                                               **kwargs)
        except:
            import traceback
            wx.CallAfter(dialog.showtraceback,parent = self,
                                txt='Failed to evauate expression',
                                title='Error',
                  traceback=''.join(traceback.format_exception_only(
                                    sys.exc_info()[0], sys.exc_info()[1])))
            wx.CallAfter(self.set_title_no_status)
        return None, None
    

    def evaluate_sol_bdr(self, expr, battrs, phys_path, do_merge1, do_merge2,
                         **kwargs):
        '''
        evaluate sol using boundary evaluator
        '''
        model = self.GetParent().model
        solfiles = self.get_model_soldfiles()
        mfem_model = model.param.getvar('mfem_model')
        phys_ns = mfem_model[str(phys_path)]._global_ns.copy()

        if solfiles is None:
             wx.CallAfter(dialog.showtraceback,parent = self,
                                  txt='Solution does not exist',
                                  title='Error',
                                  traceback='')
             wx.CallAfter(self.set_title_no_status)
             return None, None
        mesh = model.variables.getvar('mesh')
        if mesh is None: return
        FaceOf, EdgeOf, PointOf = get_mapper(mesh)
        ll = {'FaceOf': FaceOf, 'EdgeOf': EdgeOf, 'PointOf':PointOf}
        
        if battrs == 'all':
            battrs = mesh.extended_connectivity['surf2line'].keys()
        elif battrs == 'visible':
            m = self.GetParent()
            battrs = []
            for name, child in m.get_axes(0).get_children():
                if name.startswith('face'):
                     battrs.extend(child.shown_component)
            battrs = list(set(battrs))
        elif battrs == 'hidden':
            m = self.GetParent()
            battrs = []
            for name, child in m.get_axes(0).get_children():
                if name.startswith('face'):
                     battrs.extend(child.hidden_component)
            battrs = list(set(battrs))
        else:
            try:
               battrs = list(np.atleast_1d(eval(battrs, ll, phys_ns)))
            except:
               import traceback
               traceback.print_exc()
               assert False, "invalid selection: " + battrs

           #battrs = [x+1 for x in range(mesh.bdr_attributes.Size())]

        average = kwargs.pop('average', True)
                  
        from petram.sol.evaluators import build_evaluator
        if average:
            key, name = 'Bdr', 'BdrNodal'
            kkwargs = {}
        else:
            key, name = 'NCFace', 'NCFace'
            
        if key in self.evaluators:
            try:
                self.evaluators[key].validate_evaluator(name, battrs, solfiles)
            except IOError:
                dprint1("IOError detected setting failed=True")
                self.evaluators[key].failed = True
                
        if (not key in self.evaluators or
            self.evaluators[key].failed):
            self.evaluators[key] =  build_evaluator(battrs,
                                                    mfem_model,
                                                    solfiles,
                                                    name = name,
                                                    config = self.config)
        self.evaluators[key].validate_evaluator(name, battrs, solfiles)
        
        try:
            self.evaluators[key].set_phys_path(phys_path)
            return self.evaluators[key].eval(expr, do_merge1, do_merge2,
                                               **kwargs)
        except:
            import traceback            
            wx.CallAfter(dialog.showtraceback, parent = self,
                                txt='Failed to evauate expression',
                                title='Error',
                  traceback=''.join(traceback.format_exception_only(
                                    sys.exc_info()[0], sys.exc_info()[1])))
            wx.CallAfter(self.set_title_no_status)        
        return None, None

    
    def evaluate_sol_slice(self, expr, attrs, plane, phys_path, do_merge1,
                           do_merge2):
        '''
        evaluate sol using slice evaluator
        '''
        model = self.GetParent().model
        solfiles = self.get_model_soldfiles()
        mfem_model = model.param.getvar('mfem_model')

        phys_ns = mfem_model[str(phys_path)]._global_ns.copy()
        ll = {"YZ": _YZ((1, 0, 0., 0)),
              "XY": _XY((0., 0, 1., 0)),
              "ZX": _ZX((0., 1, 0., 0)),
              "yz": _YZ((1, 0, 0., 0)),
              "xy": _XY((0., 0, 1., 0)),
              "zx": _ZX((0., 1, 0., 0)),}

        try:
            plane = list(eval(plane, ll, phys_ns))
        except:
            traceback.print_exc()
            assert False, "Failed to evaluate plane " + plane
        
        if solfiles is None:
             wx.CallAfter(dialog.showtraceback, parent = self,
                                  txt='Solution does not exist',
                                  title='Error',
                                  traceback='')
             wx.CallAfter(self.set_title_no_status)
             return None, None
        mesh = model.variables.getvar('mesh')
        if mesh is None: return

        if attrs != 'all':
            try:
                attrs = list(np.atleast_1d(eval(attrs, ll, phys_ns)))
            except:
                traceback.print_exc()
                assert False, "Failed to evaluate attrs " + attrs
        else:
            attrs = mesh.extended_connectivity['vol2surf'].keys()            
           #attrs = [x+1 for x in range(mesh.attributes.Size())]
           
        if 'Slice' in self.evaluators:
            try:
                self.evaluators['Slice'].validate_evaluator('Slice', attrs, solfiles, plane=plane)
            except IOError:
                dprint1("IOError detected setting failed=True")
                self.evaluators['Slice'].failed = True
           
        from petram.sol.evaluators import build_evaluator
        if (not 'Slice' in self.evaluators or
            self.evaluators['Slice'].failed):
            self.evaluators['Slice'] =  build_evaluator(attrs, 
                                                        mfem_model,
                                                        solfiles,
                                                        name = 'Slice',
                                                        config = self.config,
                                                        plane = plane)
            
        self.evaluators['Slice'].validate_evaluator('Slice', attrs, 
                                                    solfiles, plane = plane)

        try:
            self.evaluators['Slice'].set_phys_path(phys_path)
            return self.evaluators['Slice'].eval(expr, do_merge1, do_merge2)
        except:
            wx.CallAfter(dialog.showtraceback,
                         parent = self,
                         txt='Failed to evauate expression',
                         title='Error',
                         traceback=''.join(traceback.format_exception_only(
                                    sys.exc_info()[0], sys.exc_info()[1])))
            wx.CallAfter(self.set_title_no_status)        
        return None, None
    
    def evaluate_sol_probe(self, expr, phys_path):
        model = self.GetParent().model
        solfiles = self.get_model_soldfiles()
        mfem_model = model.param.getvar('mfem_model')

        attrs = [1]
        if 'Probe' in self.evaluators:
            try:
                self.evaluators['Probe'].validate_evaluator('Probe', attrs, solfiles)
            except IOError:
                dprint1("IOError detected setting failed=True")
                self.evaluators['Probe'].failed = True
                
        from petram.sol.evaluators import build_evaluator                
        if (not 'Probe' in self.evaluators or
            self.evaluators['Probe'].failed):
            
            self.evaluators['Probe'] =  build_evaluator(attrs, 
                                                        mfem_model,
                                                        solfiles,
                                                        name = 'Probe',
                                                        config = self.config)

            
        try:
            if model.variables.getvar('remote_soldir') is None:
                probes = self.local_sols[0:2]
            else:
                probes = self.remote_sols[0:2]

            self.evaluators['Probe'].set_phys_path(phys_path)
            return self.evaluators['Probe'].eval_probe(expr, probes)
        except:
            wx.CallAfter(dialog.showtraceback,
                         parent = self,
                         txt='Failed to evauate expression (probe)',
                         title='Error',
                         traceback=''.join(traceback.format_exception_only(
                                    sys.exc_info()[0], sys.exc_info()[1])))
            wx.CallAfter(self.set_title_no_status)        
        return None, None

    #
    #   utilites
    #
    def export_to_piScope_shell(self, data, dataname):
        import wx
        import ifigure.widgets.dialog as dialog
        
        app = wx.GetApp().TopWindow
        app.shell.lvar[dataname] = data
        app.shell.SendShellEnterEvent()
        ret=dialog.message(app, dataname + ' is exported', 'Export', 0)
        
    def get_model_soldfiles(self):
        model = self.GetParent().model
        solfiles = model.variables.getvar('solfiles')
        soldir = model.variables.getvar('remote_soldir')

        if not self.config['use_cs']:
            return solfiles
        else:
            return soldir

            
        
        
