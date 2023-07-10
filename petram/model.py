
'''

 Model tree using OrderedDict.

 This is meant to be a generic layer

'''
from abc import ABC
from collections import OrderedDict
import traceback
from collections.abc import MutableMapping
import six
import os
import numpy as np
import weakref
from weakref import WeakKeyDictionary

from functools import reduce

from petram.namespace_mixin import NS_mixin, NSRef_mixin


def validate_sel(value, obj, w):
    g = obj._global_ns
    l = {"remaining": [], "all": []}
    try:
        value = eval(value, g, l)
        return True
    except:
        return False


def validate_sel2(value, obj, w):
    g = obj._global_ns
    l = {"remaining": [], "all": [], "internal_bdr": []}
    try:
        value = eval(value, g, l)
        return True
    except:
        return False


def convert_sel_txt(txt, g):
    if txt.strip() == 'remaining':
        return ['remaining']
    elif txt.strip() == 'all':
        return ['all']
    elif txt.strip() == 'internal_bdr':
        return ['internal_bdr']
    elif txt.strip() == '':
        arr = []
    else:
        arr = list(np.atleast_1d(eval(txt, g)))
    return arr


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


class RestorableOrderedDict(ABC, MutableMapping, Restorable, object):
    def __init__(self, *args, **kwargs):
        self._contents = OrderedDict(*args, **kwargs)
        Restorable.__init__(self)

    def __setstate__(self, state):
        try:
            Restorable.__setstate__(self, {
                '_contents': OrderedDict(),
                '_restoration_data': state[0],
            })
            self._parent = None
            for attr, value in state[1]:
                setattr(self, attr, value)
        except:
            traceback.print_exc()

    def __getstate__(self):
        st = [(x, self.__dict__[x])
              for x in self.__dict__ if not x.startswith('_')]

        # Note:
        #   we save _sel_index so that model_proc.pmfm has this
        #   informaiton, this way, evaluator does not need to re-run
        #   assign_sel_index, which requires loading base mesh (which is slow)

        if hasattr(self, '_sel_index'):
            st.append(('_sel_index', self._sel_index))

        return [(key, value) for key, value in self._contents.items()], st

    def _restore(self, restoration_data):
        for (key, value) in restoration_data:
            self._contents[key] = value
            value._parent = self

    def __getitem__(self, item):
        if (isinstance(item, list) or
                isinstance(item, tuple)):
            keys = [self]+list(item)
            return reduce(lambda x, y: x[y], keys)
        elif item is None:
            raise KeyError
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


class ModelDict(WeakKeyDictionary):
    '''
    Weak reference dictionary using Model Object (unhashable) as key
    '''

    def __init__(self, root):
        WeakKeyDictionary.__init__(self)
        self.root = weakref.ref(root)

    def __setitem__(self, key, value):
        hook = key.get_hook()
        return WeakKeyDictionary.__setitem__(self, hook, value)

    def __getitem__(self, key):
        hook = key.get_hook()
        return WeakKeyDictionary.__getitem__(self, hook)

    def __iter__(self):
        return [reduce(lambda x, y: x[y], [self.root()] + hook().names)
                for hook in self]


class Model(RestorableOrderedDict):
    can_delete = True
    has_2nd_panel = True
    has_3rd_panel = False
    _has_4th_panel = False
    mustbe_firstchild = False
    always_new_panel = True
    can_rename = False
    unique_child = False
    extra_diagnostic_print = False

    @classmethod
    def fancy_menu_name(cls):
        return cls.__name__.split('_')[-1]

    @classmethod
    def fancy_tree_name(cls):
        return cls.fancy_menu_name()

    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self._parent = None
        self._hook = None

        if not hasattr(self, 'init_attr'):
            self.init_attr = True
            self.update_attribute_set(kw=kwargs)

    @property
    def has_4th_panel(self):
        return self.has_3rd_panel and self._has_4th_panel

    def get_info_str(self):
        return ""

    def get_hook(self):
        if not hasattr(self, '_hook'):
            self._hook = None
        if self._hook is not None:
            return self._hook
        olist = [self]
        o = self
        while o._parent is not None:
            o = o._parent
            olist.append(o)
        names = [o.name() for o in olist]
        self._hook = Hook(names)
        return self._hook

    def __repr__(self):
        return self.__class__.__name__+'('+self.name()+':'+','.join(list(self)) + ')'

    def __eq__(self, x):
        try:
            return x.fullpath() == self.fullpath()
        except:
            return False

    def attribute_set(self, v):
        v['enabled'] = True
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['_sel_index'] = []
        if (hasattr(self, 'sel_index') and
                not hasattr(self, 'sel_index_txt')):
            v['sel_index_txt'] = ', '.join([str(x) for x in self.sel_index])
        elif not hasattr(self, 'sel_index_txt'):
            v['sel_index_txt'] = ''
        else:
            v['sel_index_txt'] = ''
        return v

    def process_sel_index(self, choice=None, internal_bdr=None):
        try:
            arr = convert_sel_txt(self.sel_index_txt, self._global_ns)
            self.sel_index = arr
        except:
            assert False, "failed to convert "+self.sel_index_txt

        if len(self.sel_index) == 1 and self.sel_index[0] == 'remaining':
            self._sel_index = []
            return None
        elif len(self.sel_index) == 1 and self.sel_index[0] == 'all':
            self._sel_index = list(choice)
            return -1
        elif len(self.sel_index) == 1 and self.sel_index[0] == 'internal_bdr':
            if internal_bdr is not None:
                self._sel_index = internal_bdr
            else:
                assert False, "Internal bdr is not defined"
        elif len(self.sel_index) == 0:
            self._sel_index = []
        elif self.sel_index[0] == '':
            self._sel_index = []
        else:
            self._sel_index = [int(i) for i in self.sel_index]
        if choice is not None:
            ret = np.array(self._sel_index)
            ret = list(ret[np.in1d(ret, choice)])
            self._sel_index = ret

        return self._sel_index

    def update_attribute_set(self, kw=None):
        if kw is None:
            kw = {}
        d = self.attribute_set(kw)
        self.do_update_attribute_set(d)

    def do_update_attribute_set(self, d):
        for k in d:
            if not hasattr(self, k):
                try:
                    setattr(self, k, d[k])
                except AttributeError:
                    print("Attribute Error", self, k, d[k])

    def attribute(self, *args, **kwargs):
        if 'showall' in kwargs:
            return {x: self.attribute(x) for x in self.attribute_set({}).keys()}

        if len(args) == 0:
            return list(self.attribute_set({}).keys())
        elif len(args) == 1:
            if hasattr(self, args[0]):
                return getattr(self, args[0])
        elif len(args) == 2:
            if hasattr(self, args[0]):
                setattr(self, args[0], args[1])
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
            if not p.enabled:
                return False
            p = p._parent
        return True

    def gather_enebled_flags(self, parent):
        enabled = {}
        for o in parent.walk():
            fname = o.fullname()
            enabled[fname] = o.enabled
        return enabled

    def apply_enebled_flags(self, parent, enabled):
        for o in parent.walk():
            fname = o.fullname()
            if fname in enabled:
                o.enabled = enabled[fname]

    @property
    def parent(self):
        return self._parent

    @property
    def parents(self):
        parents = []
        p = self
        while True:
            p = p._parent
            if p is None:
                break
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
            key = list(d0.keys())[k]
            d0 = d0[key]
        return key

    def GetChildrenCount(self, indices):
        d0 = self
        for k in indices:
            key = list(d0.keys())[k]
            d0 = d0[key]
        return len(d0)

    def GetItem(self, indices):
        d0 = self
        for k in indices:
            key = list(d0)[k]
            d0 = d0[key]
        return d0

    def GetIndices(self):
        parents = self.parents+[self, ]
        indices = [list(parents[i]).index(parents[i+1].name())
                   for i in range(len(parents)-1)]
        return indices

    def get_child(self, id):
        return self[list(self.keys())[id]]

    def get_children(self):
        return list(self.values())

    def get_possible_child(self):
        return []

    def get_possible_child_menu(self):
        return [('', cls) for cls in self.get_possible_child()]

    def get_special_menu(self, evt):
        return []

    def add_item(self, txt, cls,  **kwargs):
        after = kwargs.pop("after", None)
        before = kwargs.pop("before", None)

        if cls.unique_child:
            if txt in self:
                assert False, "this class (unique_child) already exists"
            obj = cls(**kwargs)
            self[txt] = obj
            return txt

        m = []
        for k in self.keys():
            ll = len(k)
            while ll >= 0:
                if k[ll-1].isdigit():
                    ll = ll-1
                else:
                    break

            name = k[:ll]
            if name == txt:
                if len(k) > len(name):
                    m.append(int(k[ll:]))

        if len(m) == 0:
            name = txt+str(1)
        else:
            name = txt + str(max(m)+1)

        obj = cls(**kwargs)
        done = False
        if obj.mustbe_firstchild:
            old_contents = self._contents
            self._contents = OrderedDict()
            self[name] = obj
            names = list(old_contents)
            for n in names:
                self[n] = old_contents[n]
            done = True
        elif after is not None:
            old_contents = self._contents
            self._contents = OrderedDict()
            names = list(old_contents)
            for n in names:
                self[n] = old_contents[n]
                if n == after.name():
                    self[name] = obj
                    done = True
                    # break
        elif before is not None:
            old_contents = self._contents
            self._contents = OrderedDict()
            names = list(old_contents)
            for n in names:
                if n == before.name():
                    self[name] = obj
                    done = True
                self[n] = old_contents[n]
                # if done: break

        if not done:
            self[name] = obj
        return name

    def postprocess_after_add(self, engine):
        pass

    def add_itemobj(self, txt, obj, nosuffix=False):

        m = []
        for k in self.keys():
            if k.startswith(txt):
                for x in k[len(txt):]:
                    if not x.isdigit():
                        break
                else:
                    m.append(int(k[len(txt):]))
        if len(m) == 0:
            if nosuffix:
                name = txt
            else:
                name = txt+str(1)
        else:
            name = txt + str(max(m)+1)
        self[name] = obj
        return name

    def panel2_tabname(self):
        return "Selection"

    def panel3_tabname(self):
        return "Init."

    def panel4_tabname(self):
        return "Time Dep./NL."

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
        ans = [False, ]
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
        return []

    def get_panel2_value(self):
        return (self.sel_index_txt,)

    def get_panel3_value(self):
        return []

    def get_panel4_value(self):
        return []

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
            self.sel_index_txt = str(v[0])
            try:
                arr = convert_sel_txt(str(v[0]), self._global_ns)
                self.sel_index = arr
            except:
                pass
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

    def walk_enabled(self, skip_self=False):
        '''
        skip_self: not return the top level model
        '''
        if not self.enabled:
            return
        if not skip_self:
            yield self
        for k in self.keys():
            if not self[k].enabled:
                continue
            for x in self[k].walk_enabled():
                yield x

    def iter_enabled(self):
        for child in self.values():
            if not child.enabled:
                continue
            yield child
    enum_enabled = iter_enabled  # backward compabibility.

    def name(self):
        if self._parent is None:
            return 'root'
        for k in self._parent.keys():
            if self._parent[k] is self:
                return k
        return "No Parent"

    def rename(self, new_name):
        if self.name() == 'root':
            assert False, "can't rename root"

        new_cnt = []
        for key in self._parent.keys():
            if self._parent[key] is self:
                new_cnt.append((new_name, self._parent[key]))
            else:
                new_cnt.append((key, self._parent[key]))

        parent = self._parent
        for key in list(self._parent):
            parent[key]._parent = None
            del parent[key]

        for key, value in new_cnt:
            parent[key] = value

    def split_digits(self):
        '''
        split tailing digits
        '''
        name = self.name()
        l = -1
        if not name[l].isdigit():
            return name, '0'

        while name[l].isdigit():
            l = l-1
        l = l+1
        return name[:l], name[l:]

    def insert_item(self, index, name, item):
        items = list(self._contents.items())
        items.insert(index, (name, item))
        self._contents = OrderedDict(items)
        item.set_parent(self)

    def set_parent(self, parent):
        self._parent = parent

    def on_created_in_tree(self):
        # called when item is newly created in Tree from GUI
        None

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

    def has_nsref(self):
        return isinstance(self, NSRef_mixin)

    def add_node(self, name='', cls=''):
        ''' 
        this is similar to add_item, but does not
        do anything to modify name
        '''
        if not name in self.keys():
            self[name] = cls()
        return self[name]

    def set_script_idx(self, idx=1):
        self._script_name = 'obj'+str(idx)

        for name in self.keys():
            node = self[name]
            idx = idx + 1
            idx = node.set_script_idx(idx=idx)
        return idx

    def save_attribute_set(self, skip_def_check):
        ans = []
        for attr in self.attribute():
            defvalue = self.attribute_set(dict())
            value = self.attribute(attr)
            mycheck = True

            try:
                # print attr, type(value), value, type(defvalue[attr]), defvalue[attr]
                if type(value) != type(defvalue[attr]):
                    mycheck = True
                else:  # for numpy array
                    mycheck = value != defvalue[attr]  # for numpy array
                    if isinstance(mycheck, np.ndarray):
                        mycheck = mycheck.any()
                    else:
                        mycheck = any(mycheck)
            except TypeError:
                try:
                    mycheck = value != defvalue[attr]
                except:
                    pass
            if mycheck or skip_def_check:
                ans.append(attr)
        return ans

    def _generate_model_script(self, script=None,
                               skip_def_check=False,
                               dir=None):
        # assigne script index if root node
        if script is None:
            self.set_script_idx()
            script = []
            script.append('obj1 = MFEM_ModelRoot()')

        attrs = self.save_attribute_set(skip_def_check)
        for attr in attrs:
            value = self.attribute(attr)
            script.append(self._script_name + '.'+attr + ' = ' +
                          value.__repr__())

        if (self.has_ns() or self.has_nsref()) and self.ns_name is not None:
            script.append(self._script_name + '.ns_name = "' +
                          self.ns_name + '"')

        ns_names = []
        for name in self.keys():
            node = self[name]

            script.append(node._script_name +
                          ' = ' + self._script_name +
                          '.add_node(name = "'+name + '"' +
                          ', cls = '+node.__class__.__name__ + ')')
            script = node._generate_model_script(script=script,
                                                 skip_def_check=skip_def_check,
                                                 dir=dir)
            if node.has_ns():
                if node.ns_name is None:
                    continue
                if not node.ns_name in ns_names:
                    ns_names.append(node.ns_name)
                    node.write_ns_script_data(dir=dir)

        return script

    def generate_main_script(self):

        script = []
        script.append('import time, datetime')
        script.append('stime = time.time()')
        script.append('if mfem_config.use_parallel:')
        script.append('    from petram.engine import ParallelEngine as Eng')
        script.append('else:')
        script.append('    from petram.engine import SerialEngine as Eng')
        script.append('')
        script.append('import petram.debug as debug')
        script.append('debug.set_debug_level(debug_level)')
        script.append('')
        script.append('model = make_model()')
        script.append('')
        script.append('eng = Eng(model = model)')
        script.append('')

        script.append('solvers = eng.run_build_ns()')
        # script.append('solvers = eng.preprocess_modeldata()')
        # script.append('if myid == 0: model.save_to_file("model_proc.pmfm", meshfile_relativepath = False)')
        script.append('')
        script.append('is_first = True')
        script.append('for s in solvers:')
        script.append('    s.run(eng, is_first=is_first)')
        script.append('    is_first=False')
        script.append('')
        script.append('if myid == 0:')
        script.append('    print("End Time " + ')
        script.append(
            '          datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))')
        script.append(
            '    print("Total Elapsed Time: " + str(time.time()-stime) + "s")')
        script.append('    print("Petra-M Normal End")')

        return script

    def generate_import_section(self, script):
        from petram.helper.variables import var_g
        from collections import OrderedDict

        d1 = OrderedDict()
        d2 = OrderedDict()
        for x in var_g.keys():
            err = False
            try:
                d1[x] = var_g[x].__module__
            except:
                try:
                    d1[x] = var_g[x].__class__.__module__
                except:
                    assert False, "can not generate script for " + x

        for x in self.walk():
            d2[x.__class__.__name__] = x.__class__.__module__

        for key in d1.keys():
            if d1[key] == '__builtin__':
                continue  # skip something like pi....(PY2)
            if d1[key] == 'builtins':
                continue  # skip something like pi....(PY3)
            script.append('from '+d1[key] + ' import ' + key)
        script.append('')
        for key in d2.keys():
            script.append('from '+d2[key] + ' import ' + key)

        script.append('from collections import OrderedDict')

    def generate_script(self, skip_def_check=False, dir=None, nofile=False,
                        parallel=False, filename='model.py'):
        if dir is None:
            dir = os.getcwd()
        script = []
        script.extend(['from __future__ import print_function',
                       'import os',
                       '',
                       'if __name__=="__main__":',
                       '    from mfem.common.arg_parser import ArgParser',
                       '    parser = ArgParser(description="PetraM sciprt")',
                       '    parser.add_argument("-s", "--force-serial", ',
                       '                     action = "store_true", ',
                       '                     default = True,',
                       '                     help="Use serial model even if nproc > 1.")',
                       '    parser.add_argument("-p", "--force-parallel", ',
                       '                     action = "store_true", ',
                       '                     default = False,',
                       '                     help="Use parallel model even if nproc = 1.")',
                       '    parser.add_argument("-d", "--debug-param", ',
                       '                     action = "store", ',
                       '                     default = 1, type=int) ',
                       '',
                       '    args = parser.parse_args()',
                       '    if args.force_parallel:',
                       '        use_parallel = True',
                       '        args.force_serial = False',
                       '    else:',
                       '        use_parallel = False',
                       '',
                       '    import  petram.mfem_config as mfem_config',
                       '    mfem_config.use_parallel = use_parallel',
                       '    debug_level=args.debug_param',
                       '',
                       '# this is needed if this file is being imported',
                       'if not "use_parallel" in locals():',
                       '    use_parallel = False',
                       '#set default parallel/serial flag',
                       'if use_parallel:',
                       '    from mpi4py import MPI',
                       '    num_proc = MPI.COMM_WORLD.size',
                       '    myid = MPI.COMM_WORLD.rank',
                       'else:',
                       '    myid = 0',
                       '    num_proc = 1', ])

        script.append('')
        self.generate_import_section(script)
        script.append('')

        script2 = self._generate_model_script(
            skip_def_check=skip_def_check,
            dir=dir)

        script.append('def make_model():')
        for x in script2:
            script.append(' '*4 + x)
        script.append(' '*4 + 'return obj1')

        script.append('')
        script.extend(['if __name__ == "__main__":',
                       '    if (myid == 0): parser.print_options(args)'
                       '', ])

        script.append('')
        main_script = self.generate_main_script()
        for x in main_script:
            script.append(' '*4 + x)

        path1 = os.path.join(dir, filename)
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

    def update_figure_data(self, *args, **kwargs):
        del args
        del kwargs

    def figure_data_name(self):
        return self.name()

    def update_after_ELChanged2(self, evt):
        '''
        return True if edit panel needs update
        '''
        return False

    def update_after_ELChanged(self, dlg):
        '''
        return True if edit panel needs update
        '''
        return False

    def use_essential_elimination(self):
        return True


class Bdry(Model):
    can_delete = True
    is_essential = False

    def attribute_set(self, v):
        v = super(Bdry, self).attribute_set(v)
        v['sel_readonly'] = True
        v['esse_elim_txt'] = True

        return v

    def get_possible_child(self):
        return self.parent.get_possible_bdry()

    def panel2_param(self):
        return [["Boundary",  'remaining',  0, {'changing_event': True,
                                                'setfocus_event': True,
                                                'validator': validate_sel2,
                                                'validator_param': self}]]

    def use_essential_elimination(self):
        return self.esse_elim_txt


class Pair(Model):
    can_delete = True
    is_essential = False

    def attribute_set(self, v):
        v = super(Pair, self).attribute_set(v)
        v['sel_readonly'] = False
        v['src_index'] = []
        v['sel_index'] = []
        v['src_index_txt'] = ''
        v['sel_index_txt'] = ''

        return v

    def get_possible_child(self):
        return self.parent.get_possible_pair()

    def panel2_param(self):
        return [["Source",  '',  0, {'changing_event': True,
                                     'setfocus_event': True,
                                     'validator': validate_sel,
                                     'validator_param': self}],
                ["Destination",  '',  0, {'changing_event': True,
                                          'setfocus_event': True,
                                          'validator': validate_sel,
                                          'validator_param': self}], ]

    def panel2_sel_labels(self):
        return ['source', 'destination']

    def panel2_all_sel_index(self):
        try:
            idx = [int(x) for x in self.sel_index]
        except:
            idx = []
        try:
            idx2 = [int(x) for x in self.src_index]
        except:
            idx2 = []

        return [idx2, idx]

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
        return (self.src_index_txt, self.sel_index_txt)

    def import_panel2_value(self, v):

        self.src_index_txt = str(v[0])
        self.sel_index_txt = str(v[1])

        try:
            arr = convert_sel_txt(str(v[0]), self._global_ns)
            self.src_index = arr
        except:
            pass
        try:
            arr = convert_sel_txt(str(v[1]), self._global_ns)
            self.sel_index = arr
        except:
            pass
        return False

    def process_sel_index(self, choice=None, internal_bdr=None):
        # interanl_bdr is ignored..
        try:
            arr = convert_sel_txt(self.src_index_txt, self._global_ns)
            self.src_index = arr
        except:
            assert False, "failed to convert "+self.src_index_txt

        try:
            arr = convert_sel_txt(self.sel_index_txt, self._global_ns)
            self.sel_index = arr
        except:
            assert False, "failed to convert "+self.sel_index_txt

        if len(self.sel_index) == 0:
            self._sel_index = []
        elif self.sel_index[0] == '':
            self._sel_index = []
        else:
            self._sel_index = [int(i) for i in self.sel_index]
        if len(self.src_index) == 0:
            self._src_index = []
        elif self.src_index[0] == '':
            self._src_index = []
        else:
            self._src_index = [int(i) for i in self.src_index]
        self._sel_index = self._sel_index + self._src_index

        if choice is not None:
            ret = np.array(self._sel_index)
            ret = list(ret[np.in1d(ret, choice)])
            self._sel_index = ret
            ret = np.array(self._src_index)
            ret = list(ret[np.in1d(ret, choice)])
            self._src_index = ret

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
        return [["Domain",  'remaining',  0, {'changing_event': True,
                                              'setfocus_event': True,
                                              'validator': validate_sel,
                                              'validator_param': self}], ]


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
        return [["Edge",  'remaining',  0, {'changing_event': True,
                                            'setfocus_event': True,
                                            'validator': validate_sel,
                                            'validator_param': self}], ]


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
        return [["Point",  'remaining',  0, {'changing_event': True,
                                             'setfocus_event': True,
                                             'validator': validate_sel,
                                             'validator_param': self}], ]
