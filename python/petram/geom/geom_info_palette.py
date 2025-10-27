from __future__ import print_function

import os
import numpy as np
import wx
from collections import OrderedDict
import traceback
from ifigure.utils.cbook import BuildPopUpMenu
from ifigure.utils.edit_list import EditListPanel, ScrolledEditListPanel
from ifigure.utils.edit_list import EDITLIST_CHANGED,  EDITLIST_CHANGING
from ifigure.utils.edit_list import EDITLIST_SETFOCUS
from ifigure.widgets.miniframe_with_windowlist import MiniFrameWithWindowList
from ifigure.widgets.miniframe_with_windowlist import DialogWithWindowList

from petram.pi.simple_frame_plus import SimpleFramePlus

choices = ['Property', 'Distance', 'Find small face', 'Find short edge', 'Find same...']
choices2 = ['property', 'distance', 'smallface', 'shortedge', 'findsame']

def find_surf(x, s, l):
    lines = s[x]
    points = np.unique([l[ll] for ll in lines])
    return lines, points

class GeomInfoPalette(SimpleFramePlus):
    def __init__(self, parent, wid, title):
        style = (wx.CAPTION|
                 wx.CLOSE_BOX|
                 wx.MINIMIZE_BOX|
                 wx.RESIZE_BORDER|
                 wx.FRAME_FLOAT_ON_PARENT)
        #        wx.FRAME_TOOL_WINDOW) # this does not work on apptainer on Mac/Windows

        super(GeomInfoPalette, self).__init__(parent, wid, title, style=style)

        elp1 = [['entity (p/l/f/v)', '', 0, {},],]
        elp2 = [['point', '', 0, {},],
                ['target (p/l/f)', '', 0, {},],]
        elp3 = [['threshold', 1e-6, 300, {},],]
        elp4 = [['threshold', 1e-4, 300, {},],]
        elp5 = [['entity (l/f)', '', 0, {},],
                ['tolelance', 1e-5, 300, {}],]

        setting = [{'choices':choices,
                    'text': ''},
                   {'elp': elp1},
                   {'elp': elp2},
                   {'elp': elp3},
                   {'elp': elp4},
                   {'elp': elp5},]
        ll = [(None, None, 34, setting), ]

        p = self

        self.elp = EditListPanel(p, ll)

        vbox = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vbox)
        vbox.Add(self.elp, 0, wx.EXPAND|wx.ALL, 1)

        self.txt = wx.TextCtrl(self, wx.ID_ANY,
                               style=wx.TE_MULTILINE|wx.TE_READONLY)
        self.txt.SetValue('')
        txt_w = self.Parent.GetTextExtent('A'* 30)[0]
        txt_h = self.Size[1] * 5
        self.txt.SetMinSize((txt_w, txt_h))

        vbox.Add(self.txt, 1, wx.EXPAND|wx.ALL, 1)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        vbox.Add(hbox, 0, wx.EXPAND|wx.ALL, 5)

        button = wx.Button(p, wx.ID_ANY, "Apply")
        button2 = wx.Button(p, wx.ID_ANY, "Show")
        button.Bind(wx.EVT_BUTTON, self.OnApply)
        button2.Bind(wx.EVT_BUTTON, self.OnShow)
        hbox.Add(button2, 0, wx.ALL, 1)        
        hbox.AddStretchSpacer()
        hbox.Add(button, 0, wx.ALL, 1)

        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(EDITLIST_CHANGED,  self.OnELPChanged)
        wx.GetApp().add_palette(self)

        self.Layout()
        self.Show()
        self.Fit()
        self.button2 = button2
        self.button2.Disable()
        self.data = None

    def get_inspect_param(self):
        value = self.elp.GetValue()[0]

        inspect_type = value[0]
        idx = choices.index(inspect_type)
        inspect_type = choices2[idx]
        params = value[idx+1]

        return inspect_type, params

    def get_viewer_geom(self):
        viewer = self.GetParent()
        model = viewer.model.param.getvar('mfem_model')
        geom = model['Geometry'][viewer.view_mode_group]

        return viewer, geom

    def OnApply(self, evt):
        _viewer, geom = self.get_viewer_geom()

        inspect_type, params = self.get_inspect_param()

        txt, data = geom.inspect_geom(inspect_type, params)

        if data is not None:
            self.button2.Enable()

        self.txt.SetValue(txt)
        self.data = data

        evt.Skip()

    def OnShow(self, evt):
        if self.data is None:
            evt.Skip()
            return

        viewer, geom = self.get_viewer_geom()
        inspect_type, params = self.get_inspect_param()

        l = geom.geom_data[3]
        s = geom.geom_data[4]
        v = geom.geom_data[5]

        def show_solids(solids):
            solids = [int(x) for x in solids]            
            x = [v[xx] for xx in solids]
            faces = np.unique(x)
            show_faces(faces)
            
        def show_faces(faces):
            faces = [int(x) for x in faces]
            xx = [find_surf(x, s, l) for x in faces]
            lines = np.unique(np.hstack([x[0] for x in xx]))
            points = np.unique(np.hstack([x[1] for x in xx]))
            viewer.highlight_face(faces)
            viewer.highlight_edge(lines, unselect=False)
            viewer.highlight_point(points, unselect=False)

        def show_edges(edges):
            edges = [int(x) for x in edges]
            points = np.unique(np.hstack([l[x] for x in edges]))
            viewer.highlight_edge(edges)
            viewer.highlight_point(points, unselect=False)

        viewer.highlight_none()

        if inspect_type == 'smallface':
            show_faces(self.data)

        if inspect_type == 'shortedge':
            show_edges(self.data)

        if inspect_type == 'findsame':
            sid = [int(x) for x in self.data if x.idx == 3]
            if len(sid) > 0:
                viewer.highlight_face(sid)
            eid = [int(x) for x in self.data if x.idx == 1]
            if len(eid) > 0:
                viewer.highlight_edge(eid)

        if inspect_type == 'property':
            p = params[0]
            idx = [int(p[1:])]
            if p.startswith('f'):
                show_faces(idx)
            if p.startswith('l'):
                show_edges(idx)
            if p.startswith('v'):
                show_solids(idx)
            if p.startswith('p'):
                viewer.highlight_point(idx, unselect=False)                

        evt.Skip()

    def OnELPChanged(self, evt):
        self.button2.Disable()
        self.txt.SetValue('')        
        evt.Skip()

    def OnClose(self, evt):
        wx.GetApp().rm_palette(self)
        self.GetParent().geom_info_palette = None
        evt.Skip()
