
'''
   Viewer/Editor of MFEM model

   * this modules is designed to avoid importing mfem directly here

'''
from ifigure.interactive import figure
from ifigure.widgets.book_viewer import BookViewer
from ifigure.utils.cbook import BuildMenu
import ifigure.widgets.dialog as dialog
import numpy as np
import os
import wx
import traceback

try:
    import petram.geom
    hasGeom = True
except ImportError:
    hasGeom = False
    

def setup_figure(fig):
    fig.nsec(1)
    fig.property(fig.get_axes(0), 'axis', False)
    fig.get_page(0).set_nomargin(True)
    fig.property(fig.get_page(0), 'bgcolor', 'white')
ID_SOL_FOLDER = wx.NewId()
    
class MFEMViewer(BookViewer):
    def __init__(self, *args, **kargs):
        kargs['isattachable'] = False
        kargs['isinteractivetarget'] = False        
        BookViewer.__init__(self, *args, **kargs)
        extra_menu = wx.Menu()  
        self.menuBar.Insert(self.menuBar.GetMenuCount()-1, 
                        extra_menu,"MFEM")
        menus = [("+Open Model...", None, None),
                 #("Binary...", self.onOpenPMFEM, None, None),
                 ("Script/Data Files...", self.onOpenModelS, None),
                 ("!", None, None),                 
                 ("+Mesh", None, None), 
                 ("New Mesh File...",  self.onNewMesh, None),
                 
                 ("Reload Mesh",  self.onLoadMesh, None),                  
                 ("!", None, None),
                 ("+Namespace", None, None),
                 ("New...", self.onNewNS, None),
                 ("Load...", self.onLoadNS, None),
                 ("Export...", self.onExportNS, None),
                 ("Rebuild", self.onRebuildNS, None),                 
                 ("!", None, None),                 
                 ("Edit Model...", self.onEditModel, None),
                 ("+Solve", None, None),]
        from petram.helper.driver_path import serial, parallel
        if os.path.exists(serial):
            menus.append(("Serial",    self.onSerDriver, None),)
        if os.path.exists(parallel):
            menus.append(("Parallel",  self.onParDriver, None),)
        menus.extend([("+Server", None, None),
                 ("Setting...", self.onServerSetting, None),
                 ("New WorkDir...", self.onServerNewDir, None),
                 ("Solve...", self.onServerSolve, None),
                 ("Retrieve File", self.onServerRetrieve, None),                 
                 ("!", None, None),
                 ("+Manual check", None, None),
                 ("Preprocess data",  self.onRunPreprocess, None),
                 ("!", None, None),                 
                 ("!", None, None),
                 ("+Plot", None, None),
                 ("GLVIS",    self.onCallGlvis, None),
                 ("Function...",    self.onPlotExpr, None),
                 ("Solution ...",    self.onDlgPlotSol, None),
                 ("!", None, None),
                 ("+Solution", None, None, None, ID_SOL_FOLDER),
                 ("Reload Sol", None, None,), 
                 ("Clear...",    self.onClearSol, None),                 
                 ("!", None, None),                 
                 ("+Export Model...", self.onSaveModel, None),
                 ("Binary...", self.onSaveModel, None),
                 ("Script/Data Files...", self.onSaveModelS, None),
                 ("!", None, None),                                  
                 ("---", None, None),
                 ("Reset Model", self.onResetModel, None),])

        ret = BuildMenu(extra_menu, menus)
        self._solmenu = ret[ID_SOL_FOLDER]
        self._hidemesh = True
        self._sel_mode = ''  # selecting particular geomgetry element
        self.model = self.book.get_parent()
        self.editdlg = None
        self.plotsoldlg = None
        self.plotexprdlg = None        
        self.engine = None
        self.dombdr = None

        from petram.pi.sel_buttons import btask
        self.canvas.install_navibar_palette('petram_palette',
                                             btask,
                                             mode = '3D')
        self.canvas.use_navibar_palette('petram_palette',
                                        mode = '3D')    
        if hasGeom:
            from petram.geom.geom_sel_buttons import btask
            self.canvas.install_navibar_palette('petram_geom',
                                                btask,
                                                mode = '3D')
            
        od = self.model.param.getvar('mfem_model')
        if od is None:
            self.model.scripts.helpers.reset_model()

        self.cla()
        setup_figure(self)
        self.start_engine()
        self.load_mesh()
        self.model.scripts.helpers.rebuild_ns()                
        self.engine.run_config()
        #self.Bind(wx.EVT_ACTIVATE, self.onActivate)

    def onUpdateUI(self, evt):
        if evt.GetId() == ID_SOL_FOLDER:
            m = self._solmenu
            for item in m.GetMenuItems():
                m.DestroyItem(item)
            try:
                dir =self.model.solutions.owndir()

                sol_names = os.listdir(dir)
                sol_names = [n for n in sol_names 
                             if os.path.isdir(os.path.join(dir, n))]

                menus = []
                for n in sol_names:
                    flag = True
                    for nn in os.listdir(os.path.join(dir, n)):
                        if os.path.isdir(os.path.join(dir, n, nn)):
                             flag = False
                             menus.append((os.path.join(n, nn), n +'/'+nn))
                    if flag: menus.append((n, n))
            except:
                evt.Enable(False)
                return
            mm = []
            from petram.sol.solsets import read_solsets, find_solfiles            
            for m0, m2 in menus:
                def handler(evt, dir0 = m0):
                    self.model.scripts.helpers.rebuild_ns()                    
                    self.engine.assign_sel_index()
                    path = os.path.join(dir, dir0)
                    print('loading sol from ' + path)

                    solfiles = find_solfiles(path = path)
                    self.model.variables.setvar('solfiles', solfiles)                    
                    #solsets = read_solsets(path = path)
                    #self.model.variables.setvar('solsets', solsets)
                    evt.Skip()
                def handler2(evt):
                    path = dialog.readdir(message='Select solution directory',)
                    if path == '': return
                    self.model.scripts.helpers.rebuild_ns()            
                    self.engine.assign_sel_index()                    

                    solfiles = find_solfiles(path = path)
                    self.model.variables.setvar('solfiles', solfiles)                    
                    #solsets = read_solsets(path = str(path))
                    #self.model.variables.setvar('solsets', solsets)
                    evt.Skip()
                mm.append((m2, 'Load from ' + m0, handler))
            mm.append(('Other...', 'Load from ohter place (FileDialog will open)',
                       handler2))                
            if len(mm) > 0:
               for a,b,c in mm:
                   mmi = self.add_menu(m, wx.ID_ANY, a, b, c)
            evt.Enable(True)
        else:
            super(MFEMViewer, self).onUpdateUI(evt)

    def start_engine(self):
        self.engine =  self.model.scripts.helpers.start_engine()
        #if self.model.variables.hasvar('engine')
        #from engine import SerialEngine
        #self.engine = SerialEngine()
        #self.model.variables.setvar('engine', self.engine)

    def onOpenPMFEM(self, evt):
        from petram.mesh.mesh_model import MeshFile        
        path = dialog.read(message='Select model file to read', wildcard='*.pmfm')
        if path == '': return

        import cPickle as pickle
        od = pickle.load(open(path, 'rb'))
        self.model.param.setvar('mfem_model', od)
        self.cla()
        self.load_mesh()

        if self.editdlg is not None:
            od = self.model.param.getvar('mfem_model')        
            self.editdlg.set_model(od)        
        self.model.variables.setvar('modelfile_path', path)
        
    def onOpenModelS(self, evt):
        import imp, shutil
        import cPickle as pickle
        from ifigure.mto.py_code import PyData
        from ifigure.mto.py_script import PyScript        
        path = dialog.read(message='Select model file to read', wildcard='*.py')
        try:
            m = imp.load_source('petram.user_model', path)        
            model = m.make_model()
        except:
            dialog.showtraceback(parent = self,
                               txt='Model file load error',
                               title='Error',
                               traceback=traceback.format_exc())
            return
        dir = os.path.dirname(path)
        c = [child for name, child in self.model.datasets.get_children()]
        print c
        for child in c: child.destroy()
        c = [child for name, child in self.model.namespaces.get_children()]
        print c
        for child in c: child.destroy()
            
        for file in os.listdir(dir):
            if file == os.path.basename(path): continue
            if file.endswith('.py'):
                shutil.copy(os.path.join(dir, file), self.model.namespaces.owndir())
                sc = self.model.namespaces.add_childobject(PyScript, file[:-3])
                sc.load_script(os.path.join(self.model.namespaces.owndir(), file))
            if file.endswith('.dat'):
                fid = open(os.path.join(dir, file), 'r')
                data = pickle.load(fid)
                fid.close()
                obj = self.model.datasets.add_childobject(PyData, file[:-6]+'data')
                obj.setvar(data)

        self.model.param.setvar('mfem_model', model)
        self.cla()
        self.load_mesh()

        if self.editdlg is not None:
            od = self.model.param.getvar('mfem_model')        
            self.editdlg.set_model(od)        

        evt.Skip()
        
    def onTD_SelectionInFigure(self, evt = None):
        mesh = self.engine.get_mesh()
        if mesh is None: return
        names = [x().figobj._name for x in self.canvas.selection if x() is not None]
        
        for x in self.canvas.selection:
            print x().figobj.getSelectedIndex()
        
        self._dom_bdr_sel  = (None, None)
        try:
           if mesh.Dimension() == 3:
              bdr_idx = [int(n.split('_')[1]) for n in names]
              dom_idx = []
              for y in bdr_idx:
                  for k, x in enumerate(self.dombdr):
                      if y in x: dom_idx.append(int(k+1))
              if self._sel_mode == 'volume':
                  self.highlight_domain(dom_idx)
                  names = [x().figobj._name for x in self.canvas.selection
                           if x() is not None]
                  bdr_idx = [int(n.split('_')[1]) for n in names]   
           elif mesh.Dimension() == 2:
              bdr_idx = [int(n.split('_')[1]) for n in names if n.startswith('edge')]
              dom_idx = [int(n.split('_')[1]) for n in names if n.startswith('face')]
           elif mesh.Dimension() == 1:               
              bdr_idx = [int(n.split('_')[1]) for n in names if n.startswith('point')]
              dom_idx = [int(n.split('_')[1]) for n in names if n.startswith('edge')]

           if len(dom_idx) > 0:
              text1 = 'domain: '+ ','.join([str(x) for x in dom_idx])
           else:
              text1 = ''
           if len(bdr_idx) > 0:
              text2 = 'boundry: '+ ','.join([str(x) for x in bdr_idx])
           else:
              text2 = ''
              
           self.set_status_text('Selection: ' + text1  + '  ' + text2,
                                timeout = 3000)
              
               
           self._dom_bdr_sel  = (dom_idx, bdr_idx)
        except:
           traceback.print_exc()
        evt.selections = self.canvas.selection
        self.property_editor.onTD_Selection(evt)           

    def onNewMesh(self, evt):
        from ifigure.widgets.dialog import read
        from petram.mesh.mesh_model import MeshFile        
        path = read(message='Select mesh file to read', wildcard='*.mesh')
        if path == '': return
        od = self.model.param.getvar('mfem_model')
        
        data = MeshFile(path=path)
        od['Mesh'].add_itemobj('MeshFile', data)
        self.load_mesh()
        
    def onLoadMesh(self, evt):
        self.load_mesh()
        
    def load_mesh(self):
        if self.engine is None: self.start_engine()
        od = self.model.param.getvar('mfem_model')            
        self.engine.set_model(od)
        try:
            self.engine.run_mesh()
            mesh = self.engine.get_mesh()
            self.model.variables.setvar('mesh', mesh)
        except:
            dialog.showtraceback(parent = self,
                               txt='Mesh load error',
                               title='Error',
                               traceback=traceback.format_exc())       
            return
        from petram.mesh.plot_mesh  import plot_bdrymesh, find_domain_bdr, plot_domainmesh
        self.cls()
        if mesh is not None:
            if mesh.Dimension() == 2:
                lw = 3.0
                dom_check = plot_bdrymesh(mesh = mesh,
                                          viewer = self,
                                          linewidths = lw)
            else:
                lw = 0.0 if self._hidemesh else 1.0
                dom_check = plot_bdrymesh(mesh = mesh,
                                          viewer = self,
                                          linewidths = lw)
                
            self.dombdr = find_domain_bdr(mesh, dom_check)
            if mesh.Dimension() == 2:
                lw = 0.0 if self._hidemesh else 1.0                
                plot_domainmesh(mesh = mesh,
                                viewer = self, 
                                linewidths = lw)               
            
#            plot_domain(mesh = mesh, viewer = self)

    def highlight_domain(self, i):
        '''
        i is 1-based index
        '''
        if self.dombdr is None: return

        try:
          x = len(i)
        except:
          i = list(i)
        self.canvas.unselect_all()
        for ii in i:
            if ii > len(self.dombdr): continue
            for k in self.dombdr[ii-1]:
                self._select_bdry(k-1)            
        self.canvas.refresh_hl()
        
    def highlight_bdry(self, i):
        '''
        i is 1-based index
        '''
        try:
          x = len(i)
        except:
          i = list(i)
        
        self.canvas.unselect_all()
        for ii in i:        
           self._select_bdry(ii-1)
        self.canvas.refresh_hl()
        
    def _select_bdry(self, i):
        '''
        i is 0-based index
        '''
        from petram.mesh.plot_mesh import dim2name_bdry

        key =  dim2name_bdry(self.engine.get_mesh().Dimension())
        ch = self.book.page1.axes1.get_child(name = key + '_'+str(i+1))
        if ch is not None and len(ch._artists) != 0:
            self.canvas.add_selection(ch._artists[0])
        
    def onResetModel(self, evt):
        ans = dialog.message(self,
                             "Do you want to delete all model setting?",
                             style = 2)
        if ans == 'ok':
            self.model.scripts.helpers.reset_model()
            self.model.scripts.helpers.rebuild_ns()        
            self.cla()
            if self.editdlg is not None:
                od = self.model.param.getvar('mfem_model')        
                self.editdlg.set_model(od)        
            
        
    def onEditModel(self, evt):
        from pi.dlg_edit_model import DlgEditModel
        try:
           self.engine.assign_sel_index()
        except:
           traceback.print_exc()

        model = self.model.param.getvar('mfem_model')
        if self.editdlg is None:
            self.model.scripts.helpers.rebuild_ns()                        
            self.editdlg = DlgEditModel(self, wx.ID_ANY, 'Model Tree',
                           model = model)
            self.editdlg.Show()
        self.editdlg.Raise()            


    def onSaveModel(self, evt):
        from ifigure.widgets.dialog import write
        from petram.mesh.mesh_model import MeshFile
        path = write(parent = self,
                     message='Enter model file name',
                     wildcard='*.pmfm')
        if path == '': return
        self.model.scripts.helpers.save_model(path)
        
    def onSaveModelS(self, evt):
        from ifigure.widgets.dialog import writedir        
        path =  writedir(parent = self,
                         message='Directory to write')

        m = self.model.param.getvar('mfem_model')
        try:
            m.generate_script(dir = path)
        except:
           dialog.showtraceback(parent = self,
                                txt='Failed to evauate expression',
                                title='Error',
                                traceback=traceback.format_exc())
    def onRunPreprocess(self, evt):
        try:
            self.run_preprocess()
        except:
           dialog.showtraceback(parent = self,
                                txt='Failed to during pre-processing model data',
                                title='Error',
                                traceback=traceback.format_exc())
    def onSerDriver(self, evt):
        try:
            self.run_preprocess()
        except:
           dialog.showtraceback(parent = self,
                                txt='Failed to during pre-processing model data',
                                title='Error',
                                traceback=traceback.format_exc())
           return
        self.model.scripts.run_serial.RunT()

        
    def onParDriver(self, evt):
        try:
            self.run_preprocess()
        except:
           dialog.showtraceback(parent = self,
                                txt='Failed to during pre-processing model data',
                                title='Error',
                                traceback=traceback.format_exc())
           return
        nproc = self.model.param.getvar('nproc')       
        if nproc is None: nproc = 2
        self.model.scripts.run_parallel.RunT(nproc = nproc)        

    def viewer_canvasmenu(self):
        menus = [("+MFEM", None, None), 
                 ("+Hide...",  self.onHideBdry, None),
                 ("Boundaries",  self.onHideBdry, None), 
                 ("Domains",  self.onHideDom, None),
                 ('!', None, None),
                 ("Show All",  self.onShowAll, None)]
        if self._hidemesh:
           menus.append(("Show Mesh",  self.onShowMesh, None))
        else:
           menus.append(("Hide Mesh",  self.onHideMesh, None))
           
        if self.engine is not None and self.engine.get_mesh() is not None:
            mesh_dim = self.engine.get_mesh().Dimension()
            menus.append(("+Select", None, None))
            if mesh_dim == 3:
                menus.extend([("Volume", self.onSelVolume, None),
                              ("Face", self.onSelFace, None),])
            elif mesh_dim == 2:            
                menus.extend([("Face", self.onSelFace, None),
                              ("Edge", self.onSelEdge, None),])
            elif mesh_dim == 1:
                menus.extend([("Edge", self.onSelEdge, None),
                              ("Point", self.onSelPoint, None),])
            menus.extend([("Any", self.onSelAny, None), 
                         ("!", None, None),])

        if self.editdlg is not None:
            check, kind, cidxs, labels = self.editdlg.isSelectionPanelOpen()
            if check:
                if kind == 'domain': idx = self._dom_bdr_sel[0]
                elif kind == 'bdry': idx = self._dom_bdr_sel[1]
                elif kind == 'pair': idx = self._dom_bdr_sel[1]
                else:
                    idx = None
                k = 0
                for cidx, label in zip(cidxs, labels):
                    if label == '': continue
                    show_rm = any([x in cidx for x in idx])
                    show_add = any([not x in cidx for x in idx])
                    print cidx, label, show_rm, show_add                    
                    if show_add:
                       m = getattr(self, 'onAddSelection'+str(k))
                       txt = "Add to "+ label
                       menus.append((txt, m, None))
                    if show_rm:
                       m = getattr(self, 'onRmSelection'+str(k))
                       txt = "Remove from "+ label
                       menus.append((txt, m, None))
                    k = k + 1
        menus.extend([("!", None, None),
                      ("---", None, None),])
        return menus

    def onAddSelection(self, evt, flag=0):
        check, kind, cidx, labels= self.editdlg.isSelectionPanelOpen()
        if check:
            if kind == 'domain': idx = self._dom_bdr_sel[0]
            elif kind == 'bdry': idx = self._dom_bdr_sel[1]
            elif kind == 'pair': idx = self._dom_bdr_sel[1]
            else:
                idx = None
            if idx is not None:
                self.editdlg.add_remove_AreaSelection(idx, flag = flag)

    def onAddSelection0(self, evt):
        self.onAddSelection(evt, flag = 0)
    def onAddSelection1(self, evt):
        self.onAddSelection(evt, flag = 1)
    def onAddSelection2(self, evt):
        self.onAddSelection(evt, flag = 2)
        
    def onRmSelection(self, evt, flag = 0):
        check, kind, cidx, labels = self.editdlg.isSelectionPanelOpen()
        if check:
            if kind == 'domain': idx = self._dom_bdr_sel[0]
            elif kind == 'bdry': idx = self._dom_bdr_sel[1]
            elif kind == 'pair': idx = self._dom_bdr_sel[1]
            else:
                idx = None
            if idx is not None:
                self.editdlg.add_remove_AreaSelection(idx, rm=True, flag = flag)
                
    def onRmSelection0(self, evt):
        self.onRmSelection(evt, flag = 0)
    def onRmSelection1(self, evt):
        self.onRmSelection(evt, flag = 1)
    def onRmSelection2(self, evt):
        self.onRmSelection(evt, flag = 2)
        
    def onSelVolume(self, evt):
        self.set_picker_mask('face') # volume is not shown as figobj
        self._sel_mode = 'volume'
    def onSelFace(self, evt):
        self.set_picker_mask('face')
        self._sel_mode = 'face'        
    def onSelEdge(self, evt):
        self.set_picker_mask('edge')
        self._sel_mode = 'edge'                
    def onSelPoint(self, evt):
        self.set_picker_mask('point')
        self._sel_mode = 'point'                        
    def onSelAny(self, evt):
        self.set_picker_mask('')
        self._sel_mode = ''
        
    def set_picker_mask(self, key):
        for name, child in self.get_axes().get_children():
            child.set_pickmask(not name.startswith(key))
            
    def onShowMesh(self, evt):
        from petram.mesh.plot_mesh  import plot_domainmesh        
        mesh = self.engine.get_mesh()
        self._hidemesh = False
        self.update(False)
        
        if mesh.Dimension() == 3:      # 3D mesh
           children = [child
                    for name, child in self.get_axes().get_children()
                    if name.startswith('face')]
        else:                          # 2D mesh
           children = [child
                    for name, child in self.get_axes().get_children()
                    if name.startswith('face')]
        for child in children:
            if not child.isempty(): child.set_linewidth(1.0, child._artists[0])
        self.update(True)            
        self.draw_all()
        
    def onHideMesh(self, evt):
        self._hidemesh = True
        self.update(False)
        mesh = self.engine.get_mesh()
        if mesh.Dimension() == 3:        # 3D mesh
           children = [child
                    for name, child in self.get_axes().get_children()
                    if name.startswith('face')]
        else:
           children = [child
                    for name, child in self.get_axes().get_children()
                    if name.startswith('face')]
        for child in children:
                child.set_linewidth(0.0, child._artists[0])
        self.update(True)
        self.draw_all()
        
    def onHideBdry(self, evt):
        objs = [x().figobj for x in self.canvas.selection]        
        for o in objs:
            o.set_suppress(True)
        self.draw()
        
    def onHideDom(self, evt):
        mesh = self.engine.get_mesh()
        if not mesh.Dimension() == 3: return    # 3D mesh
        
        domidx = [x-1 for x in self._dom_bdr_sel[0]]

        sel0 = []  # hide this
        sel1 = []  # but show this...
        for k, bdrs in enumerate(self.dombdr):
            if k in domidx:
                sel0.extend(bdrs)
            else:
                sel1.extend(bdrs)
        sel = [x  for x in sel0 if not x in sel1]
        
        children = [child
                    for name, child in self.get_axes().get_children()
                    if name.startswith('face') and int(name.split('_')[1]) in sel]
        for o in children:
            o.set_suppress(True)
        self.draw()
        

    def onShowAll(self, evt):
        for obj in self.book.walk_tree():
            if obj.is_suppress():
                obj.set_suppress(False)
        self.draw()
        
    def onCallGlvis(self, evt):
        self.model.scripts.helpers.call_glvis.RunT()

    def _onDlgPlotExprClose(self, evt):
        wx.GetApp().rm_palette(self.plotexprdlg)        
        self.plotexprdlg.Destroy()                        
        self.plotexprdlg = None
        evt.Skip()
        
    def _doPlotExpr(self, value):
        expr = str(value[0])
        iattr = [int(x) for x in value[1].split(',')]
        phys = str(value[2])

        try:
           d = self.model.scripts.helpers.eval_expr(expr, iattr, phys = phys)
        except:
           dialog.showtraceback(parent = self,
                                txt='Failed to evauate expression',
                                title='Error',
                                traceback=traceback.format_exc())
           return
        from ifigure.interactive import figure
        v = figure()
        setup_figure(v)        
        v.update(False)
        v.suptitle(expr)        
        for k in d.keys():
            v.solid(d[k][0], cz=True, cdata= d[k][1])
        v.update(True)
        v.view('noclip')
        v.lighting(light = 0.5)
        

    def onPlotExpr(self, evt):
        try:
            self.plotexprdlg.Raise()
            return
        except AttributeError:
            pass
        except:
            import traceback
            traceback.print_exc()
            pass

        iattr = [x().figobj.name.split('_')[1]
                 for x in self.canvas.selection
                 if x().figobj.name.startswith('bdry')]


        choices = self.model.param.getvar('mfem_model')['Phys'].keys()
        if len(choices)==0: return
        
        ll = [['Expression', '', 0, {}],
              ['Boundary Index', ','.join(iattr), 0, {}],
              ['Physics', choices[0], 4, {'style':wx.CB_READONLY,
                                       'choices': choices}],]

        from ifigure.utils.edit_list import DialogEditListWithWindowList      
#        ret = DialogEditList(ll, modal=False, parent= self,
        ret = DialogEditListWithWindowList(ll, modal=False, parent= self,
                                            ok_cb = self._doPlotExpr, ok_noclose = True,
                                            close_cb = self._onDlgPlotExprClose,
                                            style=wx.DEFAULT_DIALOG_STYLE|wx.RESIZE_BORDER,
                                            add_palette = True)
        self.plotexprdlg = ret
        evt.Skip()        
    def onDlgPlotSol(self, evt):
        from petram.pi.dlg_plot_sol import DlgPlotSol
        try:
            if self.plotsoldlg is not None:            
                self.plotsoldlg.Raise()
            else:
                self.plotsoldlg = DlgPlotSol(self, wx.ID_ANY, "plot...")        
        except:
            self.plotsoldlg = DlgPlotSol(self, wx.ID_ANY, "plot...")
            
    def onDlgPlotSolClose(self, evt):
        self.plotsoldlg = None
        evt.Skip()
        
#    def doPlotSolBdr(self, value):
#        self._doPlotSolBdr(value)
        
    def onNewNS(self, evt):
        ret, txt = dialog.textentry(self, 
                                     "Enter namespace name", "New NS...", '')
        if not ret: return
        self.model.scripts.helpers.create_ns(txt)
        from ifigure.events import SendChangedEvent
        SendChangedEvent(self.model, w=self)
        evt.Skip()

    def onLoadNS(self, evt):
        from petram.pi.ns_utils import import_ns
        import_ns(self.model)
        from ifigure.events import SendChangedEvent
        SendChangedEvent(self.model, w=self)
        evt.Skip()        

    def onExportNS(self, evt):
        choices = ['_'.join(name.split('_')[:-1])
                   for name, child in self.model.namespaces.get_children()]
        if len(choices) == 0: return

        ll = [
              
              [None, choices[0], 4, {'style':wx.CB_READONLY,
                                       'choices': choices}],]        
        from ifigure.utils.edit_list import DialogEditList
        ret = DialogEditList(ll, modal=True, parent= self,
                             title = 'Select Namespace')
        if not ret[0]: return
        name = str(ret[1][0])

        from petram.pi.ns_utils import export_ns
        export_ns(self.model, name)
        from ifigure.events import SendChangedEvent
        SendChangedEvent(self.model, w=self)
        evt.Skip()
        
    def onRebuildNS(self, evt):
        self.model.scripts.helpers.rebuild_ns()
        evt.Skip()
        
    def onClearSol(self, evt):
        self.model.scripts.helpers.clear_sol(w = self)
        self.model.param.setvar('sol', None)
        evt.Skip()

    #def onActivate(self, evt):
    #    windows = [self.editdlg, self.plotsoldlg, self.plotexprdlg]
    #    for w in windows:           
    #        if w is not None:
    #            if not w.IsShown():
    #                w.Show()
    #    evt.Skip()
        
    def onWindowClose(self, evt=None):
        if self.editdlg is not None:
            try:
                self.editdlg.Destroy()
            except:
                pass
            self.editdlg = None
        if self.plotsoldlg is not None:
            try:
               self.plotsoldlg.Destroy()
            except:
               pass
            self.plotsoldlg = None
            
        super(MFEMViewer, self).onWindowClose(evt)

    def onServerSetting(self, evt):
        remote = self.model.param.eval('remote')
        if remote is not None:
            hostname = remote.param.eval('host').name
        else:
            hostname = ''
        ret, new_name=dialog.textentry(self,
                                       "Enter the name of new connection",
                                       "Add Connection",
                                       hostname)
        if ret:
            self.model.scripts.remote.setup_server(new_name)
            
    def onServerNewDir(self, evt):

        import datetime, socket
        from ifigure.widgets.dialog import textentry
        txt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        hostname = socket.gethostname()
        txt = txt + '_' + hostname
        f,txt = textentry(self, message = 'Enter remote directory name',
                          title = 'Creating remote directory', 
                          def_string = txt, center = True)
        if not f: return
        
        remote = self.model.param.eval('remote')
        remote.scripts.prepare_remote_dir(txt)
        
    def onServerSolve(self, evt):
        try:
            self.run_preprocess()
        except:
           dialog.showtraceback(parent = self,
                                txt='Failed to during pre-processing model data',
                                title='Error',
                                traceback=traceback.format_exc())
           return


        remote = self.model.param.eval('remote')
        if remote is None: return

        value = [1, 1, '00:60:00', False]
        keys = ['num_nodes', 'num_cores', 'walltime']
        for i in range(3):
            if remote.param.getvar(keys[i]) is not None:
                value[i] = remote.param.getvar(keys[i])
        from petram.pi.dlg_submit_job import get_job_submisson_setting
        setting = get_job_submisson_setting(self, 'using '+remote.name,
                                            value = value)
        for k in setting.keys():
            remote.param.setvar(k, setting[k])

        if len(setting.keys()) == 0: return
        if self.model.param.eval('sol') is None:
            folder = self.model.scripts.helpers.make_new_sol()
        else:
            folder = self.model.param.eval('sol')
        if remote.param.eval('rwdir') is None:
            remote.scripts.prepare_remote_dir()
            
        #remote.scripts.clean_remote_dir()

        self.model.param.eval('sol').clean_owndir()        
        res = self.model.scripts.remote.run_model.RunT(folder,
                             retrieve_files = setting["retrieve_files"])

    def onServerRetrieve(self, evt):
        self.model.scripts.remote.retrieve_files()

    def run_preprocess(self):
        model = self.model
        engine = self.engine
        engine.preprocess_ns(model.namespaces, model.datasets)
        engine.build_ns()
        engine.run_preprocess(model.namespaces, model.datasets)


