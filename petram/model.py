'''

 Model tree using OrderedDict.

 This is meant to be a generic layer

'''
from collections import OrderedDict
import traceback
from collections import MutableMapping
import os

from functools import reduce

from petram.namespace_mixin import NS_mixin

class Restorable(object):
    def __init__(self):
        self._requires_restoration = False
        
    def __setstate__(self, state):
        self.__dict__ = state
        self._requires_restoration = True
    def __getattribute__(self, item):
        if item == '_contents':
            if self._requires_restoration:
                self._requires_restoration = False
                self._restore(self._restoration_data)
                self._restoration_data = None
        return object.__getattribute__(self, item)

    def _restore(self, restoration_data):
        raise NotImplementedError(
             "you must specify the _restore method with the Restorable type")


class RestorableOrderedDict(MutableMapping, Restorable, object):
     def __init__(self, *args, **kwargs):
         self._contents = OrderedDict(*args, **kwargs)
         Restorable.__init__(self)
 
     def __setstate__(self, state):
         try:
             Restorable.__setstate__(self, {
                 '_contents' : OrderedDict(),
                 '_restoration_data' : state[0],
              })
             self._parent = None
             for attr, value in state[1]:
                 setattr(self, attr, value)
         except:
             traceback.print_exc()
         
     def __getstate__(self):
         st = [(x, self.__dict__[x]) for x in self.__dict__ if not x.startswith('_')]
#         st.append(('_parent', self._parent))
         return [ (key, value) for key, value in self._contents.iteritems() ], st
     
     def _restore(self, restoration_data):
         for (key, value) in restoration_data:
#             print key
             self._contents[key] = value
             value._parent = self

     def __getitem__(self, item):
         if (isinstance(item, list) or
             isinstance(item, tuple)):
             keys = [self]+list(item)
             return reduce(lambda x, y: x[y], keys)
         elif item.find('.') != -1:
             items = item.split('.')
             keys = [self]+list(items)
             return reduce(lambda x, y: x[y], keys)
         else:
             return self._contents[item]
 
     def __setitem__(self, key, value):
         self._contents[key] = value
         value._parent = self
 
     def __delitem__(self, key):
         del self._contents[key]
 
     def __iter__(self):
         return iter(self._contents)
     
     def __len__(self):
         return len(self._contents)
     
     def __repr__(self):
         return """RestorableOrderedDict{}""".format(repr(self._contents))
     
class Hook(object):
    def __init__(self, names):
        self.names = names

class Model(RestorableOrderedDict):
    can_delete = True
    has_2nd_panel = True
    has_3rd_panel = False
    mustbe_firstchild =False  

    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self._parent = None
        self._hook = None
        if not hasattr(self, 'init_attr'):
            self.update_attribute_set(kw = kwargs)
            self.init_attr = True
            
    def get_hook(self):
        if not hasattr(self, '_hook'): self._hook = None
        if self._hook is not None: return self._hook
        olist = [self]
        o = self
        while o._parent is not None:
            o = o._parent
            olist.append(o)
        names = [o.name() for o in olist]
        self._hook = Hook(names)
        return self._hook
        
    def __repr__(self):
         return self.__class__.__name__+'('+self.name()+':'+','.join(self.keys()) + ')'

    def attribute_set(self, v):
        v['enabled'] = True
        v['sel_readonly'] = False
        v['sel_index'] = ['remaining']
        return v

    def process_sel_index(self):
        if len(self.sel_index) == 1 and self.sel_index[0] == 'remaining':
            self._sel_index = []
            return None
        elif len(self.sel_index) == 0:
            self._sel_index = []            
        elif self.sel_index[0] == '':            
            self._sel_index = []
        else:
            self._sel_index = [long(i) for i in self.sel_index]
        return self._sel_index
            
    def update_attribute_set(self, kw = None):
        if kw is None: kw = {}
        d = self.attribute_set(kw)

        for k in d.keys():
           if not hasattr(self, k):
               setattr(self, k, d[k])
        
    def attribute(self, *args, **kwargs):
        if 'showall' in kwargs:
            return {x: self.attribute(x) for x in self.attribute_set({}).keys()}
        
        if len(args) == 0:
            return self.attribute_set({}).keys()
        elif len(args) == 1:
            if hasattr(self, args[0]): return getattr(self, args[0])
        elif len(args) == 2:
            if hasattr(self, args[0]): setattr(self, args[0], args[1])
        else:
            pass

    def get_editor_menus(self):
        '''
        show custom menu in editor panel
        '''
        return []
               
    def __setstate__(self, state):
        super(Model, self).__setstate__(state)
        self.update_attribute_set()
        
    def is_enabled(self):
        ''' 
        check if all parents are all enabled
        '''
        p = self
        while p is not None:
            if not p.enabled: return False
            p = p._parent
        return True
    
    @property
    def parent(self):
        return self._parent
    @property
    def parents(self):
        parents = []
        p = self
        while True:
           p = p._parent
           if p is None: break
           parents.insert(0, p)
        return parents
    
    def verify_setting(self):
        '''
        a check routine to seting verificaiton comes here.
        return flag,  text, long explanation
        '''
        return True, '', ''
    
    def GetItemText(self, indices):
        key = ''
        d0 = self
        for k in indices:
            key = d0.keys()[k]
            d0 = d0[key]
        return key          
    def GetChildrenCount(self, indices):
       d0 = self
       for k in indices:
           key = d0.keys()[k]
           d0 = d0[key]
       return len(d0.keys())
   
    def GetItem(self, indices):
       d0 = self
       for k in indices:
           key = d0.keys()[k]
           d0 = d0[key]
       return d0

    def get_child(self, id):
        return self[self.keys()[id]]
    
    def get_possible_child(self):
        return []
    
    def get_special_menu(self):
        return []

    def add_item(self, txt, cls,  **kwargs):
        
        m = []
        for k in self.keys():
            label = ''.join([x for x in k if not x.isdigit()])
            if txt == label:
                m.append(long(k[len(txt):]))
        if len(m) == 0:
           name = txt+str(1)
        else:
           name = txt + str(max(m)+1)
        obj = cls(**kwargs)
        if obj.mustbe_firstchild:
            old_contents = self._contents 
            self._contents = OrderedDict()
            self[name] = obj
            names = list(old_contents)
            for n in names:
                self[n] = old_contents[n]
        else:
            self[name] = obj
        return name
    
    def add_itemobj(self, txt, obj):
        
        m = []
        for k in self.keys():
            if k.startswith(txt):
                m.append(long(k[len(txt):]))
        if len(m) == 0:
           name = txt+str(1)
        else:
           name = txt + str(max(m)+1)
        self[name] = obj
        return txt+str(1)

    def panel1_param(self):
        return []
                         
    def panel2_param(self):
        return []

    def panel2_sel_labels(self):
        return ['selection']
    
    def panel2_all_sel_index(self):
        try:
            idx = [int(x) for x in self.sel_index]
        except:
            idx = []
        return [idx]
    def is_wildcard_in_sel(self):
        ans = [False,]
        try:
            idx = [int(x) for x in self.sel_index]
        except:
            ans[0] = True
        return ans
    
    def panel3_param(self):
        return []
                         
    def panel4_param(self):
        return []

    def panel1_tip(self):
        return None
                         
    def panel2_tip(self):
        return None
    
    def panel3_tip(self):
        return None
    
    def panel4_tip(self):
        return None

    def get_panel1_value(self):
        pass

    def get_panel2_value(self):
        return (','.join([str(x) for x in self.sel_index]),)
    
    def get_panel3_value(self):
        pass
    
    def get_panel4_value(self):
        pass

    def import_panel1_value(self, v):
        '''
        return value : gui_update_request
        '''
        return False
    
    def import_panel2_value(self, v):
        '''
        return value : gui_update_request
        '''
        if not self.sel_readonly:
           arr =  str(v[0]).split(',')
           arr = [x for x in arr if x.strip() != '']
           self.sel_index = arr
        return False
    
    def import_panel3_value(self, v):
        '''
        return value : gui_update_request
        '''
        return False        
    
    def import_panel4_value(self, v):
        '''
        return value : gui_update_request
        '''
        return False                

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('',  self)                                                
    
    def export_modeldata(self):
        pass

    def write_setting(self, fid):
        pass

    def preprocess_params(self, engine):
        pass

    def walk(self):
        yield self
        for k in self.keys():
            for x in self[k].walk():
                yield x

    def iter_enabled(self):
        for k in self.keys():
            if not self[k].enabled: continue
            yield self[k]
    enum_enabled = iter_enabled #backward compabibility.
    
    def name(self):
        if self._parent is None: return 'root'
        for k in self._parent.keys():
            if self._parent[k] is self: return k
   
    def split_digits(self):
        '''
        split tailing digits
        '''
        name = self.name()
        l = -1
        if not name[l].isdigit(): name, '0'

        while name[l].isdigit(): l =l-1
        l = l+1
        return name[:l], name[l:]

    def insert_item(self, index, name, item):
        items = self._contents.items()
        items.insert(index, (name, item))
        self._contents = OrderedDict(items)
        item.set_parent(self)

    def set_parent(self, parent):
        self._parent = parent

    def fullname(self):
        '''
        returns 'root.Phys.Boundary...'
        '''
        olist = [self]
        o = self
        while o._parent is not None:
            o = o._parent
            olist.append(o)
        names = [o.name() for o in olist]
        return '.'.join(reversed(names))

    def fullpath(self):
        '''
        returns 'Phys.Boundary...'
        similar to fullname but without "root"
        can be used to root[obj.fullpath()] to get model object
        '''
        olist = [self]
        o = self
        while o._parent is not None:
            o = o._parent
            olist.append(o)
        names = [o.name() for o in olist]
        return '.'.join(reversed(names[:-1]))
           
    def root(self):
        o = self
        while o._parent is not None:
            o = o._parent
        return o
    
    def has_ns(self):
        return isinstance(self, NS_mixin)

    def add_node(self, name = '', cls = ''):
        ''' 
        this is similar to add_item, but does not
        do anything to modify name
        '''
        if not  name in self.keys():
            self[name] = cls()
        return self[name]
    
    def set_script_idx(self, idx=1):
        self._script_name = 'obj'+str(idx)

        for name in self.keys():
            node = self[name]
            idx = idx + 1
            idx = node.set_script_idx(idx=idx)
        return idx

    def _generate_model_script(self, script = None,
                               skip_def_check = False, dir = None):
        # assigne script index if root node
        if script is None:
            self.set_script_idx()
            script = []
            script.append('obj1 = MFEM_ModelRoot()')
        for attr in self.attribute():
            defvalue = self.attribute_set(dict())
            value = self.attribute(attr)
            mycheck = True
            try:
                mycheck = any(value != defvalue[attr])  # for numpy array
            except TypeError:
                try:
                    mycheck = value != defvalue[attr]
                except:
                    pass
            if mycheck or skip_def_check:
                script.append(self._script_name + '.'+attr + ' = ' +
                              value.__repr__())
        if self.has_ns() and self.ns_name is not None:
            script.append(self._script_name + '.ns_name = "' +
                          self.ns_name + '"')

        ns_names = []        
        for name in self.keys():
            node = self[name]

            script.append(node._script_name + 
                  ' = ' + self._script_name +
                  '.add_node(name = "'+name + '"' + 
                  ', cls = '+node.__class__.__name__ +  ')')
            script = node._generate_model_script(script=script,
                                           skip_def_check = skip_def_check,
                                           dir=dir)
            if node.has_ns():
                if node.ns_name is None: continue
                if not node.ns_name in ns_names:
                    ns_names.append(node.ns_name)
                    node.write_ns_script_data(dir = dir)
                
        return script

    def generate_main_script(self):

        script  = []
        script.append('if mfem_config.use_parallel:')
        script.append('    from petram.engine import ParallelEngine as Eng')
        script.append('else:')                      
        script.append('    from petram.engine import SerialEngine as Eng')
        script.append('')        
        script.append('model = make_model()')
        script.append('')        
        script.append('eng = Eng(model = model)')
        script.append('')        
        script.append('solver = eng.preprocess_modeldata()')
        script.append('')        
        script.append('solver.run(eng)')
        
        return script
    
    def generate_script(self, skip_def_check = False, dir = None, nofile = False):
        if dir is None: dir = os.getcwd()        
        script = []
        script.append('import  petram.mfem_config as mfem_config')
        script.append('mfem_config.use_parallel = False')
        
        script.append('from petram import *')
        script.append('from petram.mesh import *')
        script.append('from petram.phys import *')
        script.append('from petram.solver import *')
        script.append('')

        script2 = self._generate_model_script(
                            skip_def_check = skip_def_check,
                            dir = dir)

        script.append('def make_model():')
        for x in script2: script.append(' '*4 + x)
        script.append(' '*4 + 'return obj1')

        script.append('')                      
        script.append('if __name__ == "__main__":')
        main_script = self.generate_main_script()
        for x in main_script:
            script.append(' '*4 + x)

        path1 = os.path.join(dir, 'model.py')
        fid = open(path1, 'w')
        fid.write('\n'.join(script))
        fid.close()
        
        return script
    
    def load_gui_figure_data(self, viewer):
        '''
        called when mfem_viewer opened to set inital figure (geometry)
        plottting data.

        return value : (view_mode, name, data)
        '''
        return None, None, None
    def is_viewmode_grouphead(self):
        return False
    
    def figure_data_name(self):
        return self.name()
    
class Bdry(Model):
    can_delete = True
    is_essential = False            
    def get_possible_child(self):
        return self.parent.get_possible_bdry()
    
    def panel2_param(self):
        return [["Boundary",  'remaining',  0, {'changing_event': True,
                                                'setfocus_event': True}] ]

class Pair(Model):
    can_delete = True
    is_essential = False            
    def attribute_set(self, v):
        v = super(Pair, self).attribute_set(v)
        v['sel_readonly'] = False
        v['src_index'] = []
        v['sel_index'] = []
        return v
        
    def get_possible_child(self):
        return self.parent.get_possible_pair()

    def panel2_param(self):
        return [["Destination",  '',  0, {'changing_event':True,
                                          'setfocus_event':True}],
                ["Source",  '',  0, {'changing_event':True, 
                                     'setfocus_event':True}] ]

    def panel2_sel_labels(self):
        return ['destination', 'source']

    def panel2_all_sel_index(self):
        try:
            idx = [int(x) for x in self.sel_index]
        except:
            idx = []
        try:
            idx2 = [int(x) for x in self.src_index]
        except:
            idx2 = []
            
        return [idx, idx2]
    
    def is_wildcard_in_sel(self):
        ans = [False, False]
        try:
            idx = [int(x) for x in self.sel_index]
        except:
            ans[0] = True
        try:
            idx2 = [int(x) for x in self.src_index]
        except:
            ans[1] = True
        return ans
        
    def get_panel2_value(self):
        return (','.join([str(x) for x in self.sel_index]),
                ','.join([str(x) for x in self.src_index]),)
    
    def import_panel2_value(self, v):
        arr =  str(v[0]).split(',')
        arr = [x for x in arr if x.strip() != '']
        self.sel_index = arr
        arr =  str(v[1]).split(',')
        arr = [x for x in arr if x.strip() != '']
        self.src_index = arr
        
    def process_sel_index(self):
        if len(self.sel_index) == 0:
            self._sel_index = []            
        elif self.sel_index[0] == '':            
            self._sel_index = []
        else:
            self._sel_index = [long(i) for i in self.sel_index]

        if len(self.src_index) == 0:
            self._src_index = []            
        elif self.src_index[0] == '':            
            self._src_index = []
        else:
            self._src_index = [long(i) for i in self.src_index]
        self._sel_index = self._sel_index + self._src_index
            
        return self._sel_index

   
class Domain(Model):
    can_delete = True
    is_essential = False            
    def attribute_set(self, v):
        v = super(Domain, self).attribute_set(v)
        v['sel_readonly'] = True
        return v
        
    def get_possible_child(self):
        return self.parent.get_possible_domain()
    
    def panel2_param(self):
        return [["Domain",  'remaining',  0, {'changing_event':True,
                                              'setfocus_event':True}, ]]          
      
class Edge(Model):
    can_delete = True
    is_essential = False            
    def attribute_set(self, v):
        v = super(Edge, self).attribute_set(v)
        v['sel_readonly'] = True
        return v
    
    def get_possible_child(self):
        return self.parent.get_possible_edge()
    
    def panel2_param(self):
        return [["Edge",  'remaining',  0, {'changing_event':True, 
                                            'setfocus_event':True}, ]]      
class Point(Model):
    can_delete = True
    is_essential = False        
    def attribute_set(self, v):
        v = super(Point, self).attribute_set(v)
        v['sel_readonly'] = True
        return v
    
    def get_possible_child(self):
        return self.parent.get_possible_point()
    
    def panel2_param(self):
        return [["Point",  'remaining',  0, {'changing_event':True,
                                             'setfocus_event':True}, ]]

        
