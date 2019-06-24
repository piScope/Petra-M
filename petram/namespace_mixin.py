from __future__ import print_function

import os
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Namespace')

class NS_mixin(object):
    hide_ns_menu = False
    def __init__(self, *args, **kwargs):
        object.__init__(self)
        self.reset_ns()

    @property
    def namespace(self):
        return (self._global_ns, self._local_ns)
        
    def attribute_expr(self):
        '''
        define attributes evaluated as exprssion
        returns name and validator (float, int, complex or None)
        '''
        return [], []
    
    def attribute_mirror_ns(self):
        '''
        a list of attribute copied to ns
        '''
        return []
    
    def get_info_str(self):
        if self.ns_name is not None:
            return 'NS:'+self.ns_name
        return ""
        
    def reset_ns(self):
        self._global_ns = None
        self._local_ns = None
        self.ns_name = None
        self.ns_string = None
        self.dataset = None
        
    def get_ns_name(self):
        if not hasattr(self, 'ns_name'):self.reset_ns()
        return self.ns_name
    
    def get_ns_chain(self):
        chain = []
        p = self
        while p is not None:
            if isinstance(p, NS_mixin):
                if p.ns_name is not None: 
                    chain.append(p)
                elif len(p.get_default_ns().keys()) != 0:
                    chain.append(p)
            p = p.parent
            
        tmp = [x for x in reversed(chain)]
        gn = self.root()['General']
        if gn.get_ns_name() is not None:
            if len(tmp) == 0: tmp.append(gn)
            elif tmp[0] is not gn:
                tmp = [gn] + tmp
            else:
                pass
        return tmp
            
    def write_ns_script_data(self, dir = None):
        path1 = os.path.join(dir, self.ns_name+'_ns.py')
        path2 = os.path.join(dir, self.ns_name+'_ns.dat')
        fid = open(path1, 'w')
        fid.write(self.ns_string)
        fid.close()
        import cPickle as pickle
        pickle.dump(self.dataset, open(path2, 'wb'))
        
    def read_ns_script_data(self, dir = None):
        path1 = os.path.join(dir, self.ns_name+'_ns.py')
        path2 = os.path.join(dir, self.ns_name+'_ns.dat')
        fid = open(path1, 'r')
        self.ns_string = '\n'.join(fid.readlines())
        fid.close()
        import cPickle as pickle
        self.dataset = pickle.load(open(path2, 'rb'))
        
    def delete_ns(self):
        self._global_ns = None
        self._local_ns = None
        self.ns_name = None
        self.ns_string = None
        self.dataset = None

    def new_ns(self, name):
        self._global_ns = None
        self._local_ns = None
        self.ns_name = name
        self.ns_string = None
        self.dataset = None

    def preprocess_ns(self, ns_folder, data_folder):
        if self.get_ns_name() is None: return
        
        ns_script = ns_folder.get_child(name = self.ns_name+'_ns')
        if ns_script is None:
            raise ValueError("namespace script is not found")                   
        ns_script.reload_script()
        self.ns_string = ns_script._script._script
        
        data = data_folder.get_child(name = self.ns_name+'_data')
        if data is None:
            raise ValueError("dataset is not found")      
        d = data.getvar()
        self.dataset = {k:d[k] for k in d.keys()} # copy dict

    def get_default_ns(self):
        '''
        this method is overwriten when model wants to
        set its own default namespace. For example, when
        RF module set freq and omega
        '''
        return {}
                
    def eval_attribute_expr(self, targets=None):

        names, types = self.attribute_expr()
        exprs = [(x, x+'_txt', v) for x, v in zip(names, types)]

        invalid_expr = []
        result = {}
        for name, tname,  validator in exprs:
            if targets is not None and not tname in targets: continue
            void = {}
            try:
                x = eval(str(getattr(self, tname)), self._global_ns, void)
            except:
                import traceback
                traceback.print_exc()
                invalid_expr.append(tname)
                continue
            try:
                if validator is not None:
                     x = validator(x)
            except:
                import traceback
                traceback.print_exc()
                invalid_expr.append(tname)
                continue
            result[name] = x
            
        return result, invalid_expr
    
    def eval_ns(self):
        chain = self.get_ns_chain()
        l = self.get_default_ns()

        from petram.helper.variables import var_g
        g = var_g.copy()
        
        if self.root() is self:
            self._variables = {}
        else:
            self._local_ns = self.root()._variables
        if len(chain) == 0:
            raise ValueError("namespace chain is not found")
        # step1 (fill ns using upstream + constant (no expression)        
        if chain[-1] is not self:
            if len(l.keys()) == 0:
                self._global_ns = chain[-1]._global_ns
                #self._local_ns = chain[-1]._local_ns
            else:
                self._global_ns = g
                for k in l.keys():
                    g[k] = l[k]
                for k in chain[-1].keys():
                    g[k] = chain[-1]._global_ns[k]
                #self._local_ns = {}
        elif len(chain) > 1:
           # step 1-1 evaluate NS chain except for self and store dataset to
           # g including mine
           self._global_ns = g
           for p in chain[:-1]:#self.parents:
               if not isinstance(p, NS_mixin): continue
               ll = p.get_default_ns()               
               if (p.ns_string == '' or p.ns_string is None and
                   len(ll.keys()) == 0): continue
               for k in ll.keys():
                   g[k] = ll[k]
               if p.ns_name is not None:
                   try:
                       if p.dataset is not None:
                           for k in p.dataset.keys(): g[k] = p.dataset[k]
                       for k in p.attribute_mirror_ns():
                           g[k] = chain[-2]._global_ns[k]                   
                       ll = {}
                       if (p.ns_string != '' and p.ns_string is not None):
                           exec(p.ns_string, g, ll)
                           for k in ll.keys(): g[k] = ll[k]
                           
                   except Exception as e:
                       import traceback
                       assert False, traceback.format_exc()
           if self.dataset is not None:
               for k in self.dataset.keys(): g[k] = self.dataset[k]
        else:
           self._global_ns = g
           for k in l.keys():  g[k] = l[k]
           if self.dataset is not None:
               for k in self.dataset.keys(): g[k] = self.dataset[k]
        # step2 eval attribute using upstream + non-expression
        result, invalid =  self.eval_attribute_expr()
        for k in result.keys():
            setattr(self, k, result[k])

        # step 3 copy attributes to ns 
        attrs = self.attribute_mirror_ns()
        for a in attrs:
            if not a in invalid: g[a] = getattr(self, a)

        # step 4 run namespace scripts otherise exit
        for k in l.keys():
            g[k] = l[k]  # copying default ns

        import mfem
        if mfem.mfem_mode == 'serial':
            g['mfem'] = mfem.ser
        elif mfem.mfem_mode == 'parallel':
            g['mfem'] = mfem.par
        else:
            assert False, "PyMFEM is not loaded"

        try:
            l = {}
            if (self.ns_string != '' and self.ns_string is not None):
                 exec(self.ns_string, g, l)
            else:
                 pass ###return
        except Exception as e:
            import traceback
            assert False, traceback.format_exc()

        for k in l.keys():
            g[k] = l[k]
        
        # step 5  re-eval attribute with self-namespace
        #         passing previous invalid as a list of variables
        #         to evaluate
        result, invalid =  self.eval_attribute_expr(invalid)
        for k in result.keys():
            setattr(self, k, result[k])

        # if something is still not known,,, raise
        if len(invalid) != 0:
            raise ValueError("failed to evaluate variable "+ ', '.join(invalid))


    # parameters with validator
    def check_param_expr(self, value, param, ctrl):
        try:
            self.eval_param_expr(str(value), param)
            return True
        except:
            import petram.debug
            import traceback
            if petram.debug.debug_default_level > 2:
                traceback.print_exc()
            return False
        
    def eval_param_expr(self, value, param):
        x = eval(value, self._global_ns, self._local_ns)
        dprint2('Value Evaluation ', param, '=', x)            
        return x, None

    # note that physics modules overwrite this with more capablie version
    def make_param_panel(self, base_name, value):
        return  [base_name + "(=)",  value, 0,  
                 {'validator': self.check_param_expr,
                 'validator_param':base_name}]
     
