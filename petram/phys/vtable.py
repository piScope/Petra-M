'''

  variable-table

  a way to define model parameters and to generate associated
  methods

  definition

  name : 'tol'
  guilabel: 'tolelance (%)'
  type : 'any', 'float, 'int', 'complex'
  suffix : [('x', 'y', 'z'), ('x', 'y', 'z')]
  cb :   callback method name
  no_func: True: it can not become variable (use '=')
  tip : tip string
  example:

  Scalar : VtableElement('Tbdr',  type='float')
  Vector : VtableElement('Jsurf', type='float', 
                         suffix = ('x', 'y', 'z')
                         default = [0,0,0])
  Matrix : VtableElement('epsilonr', type='complex', 
                         suffix = [('x', 'y', 'z'), ('x', 'y', 'z')]
                         default = np.eye(3,3))

  StaticText :  VtableElement(None,
                              guilabel = 'Default Domain (Vac)',  
                              default =  'eps_r=1, mu_r=1, sigma=0')

                         
'''

import six
import numpy as np
import itertools
from collections import OrderedDict

class VtableElement(object):
    def __init__(self, name, type = '', 
                size  = (1,), suffix = None,
                cb = None, no_func = False,
                default = 0., guilabel = None, tip = '',
                chkbox = False):
        self.name = name
        if not isinstance(type, str):
            assert False, "data type should be given as str"
        self.type = type
        self.chkbox = chkbox
        if suffix is None:
            self.shape = ()
            self.suffix = []
            self.ndim = 0
        else:
            ndim = 1 if isinstance(suffix[0], str) else 2
            if ndim == 1:
                self.shape = (len(suffix),)
                suffix = (suffix,)
            else:
               self.shape = tuple([len(x) for x in suffix])
            self.ndim = ndim
            self.suffix = [''.join(tmp) for tmp in itertools.product(*suffix)]
        self.cb = cb
        self.no_func = no_func
        if name is not None:
            self.default = np.array(default, copy = False)
        else:
            self.default = default
        self.guilabel = guilabel if guilabel is not None else self.name
        self.tip = tip

    def txt2value(self, txt):
        if self.type == 'float': return float(txt)
        if self.type == 'complex': return complex(txt)
        if self.type == 'int': return int(txt)
        if self.type == 'long': return long(txt)
        return float(txt)
    
    def add_attribute(self, v):
        if self.name is None: return
        if len(self.shape) == 0:
            v[self.name] = self.txt2value(self.default)
            v[self.name + '_txt'] = self.default
            if self.chkbox:
                v['use_' + self.name + '_txt'] = False
        else:
            values = [str(x) for x in self.default.flatten()]
            if len(values) != len(self.suffix):
                 raise ValueError("Length of defualt value does not match")
            suffix = ['_'+x for x in self.suffix]             
            for x, v_txt  in zip(suffix, values):
                v[self.name + x] = self.txt2value(v_txt)
                v[self.name + x + '_txt'] = v_txt
            v[self.name + '_m'] =  self.default
            xxx = self.default.__repr__().split('(')[1].split(')')[0]            
            v[self.name + '_m_txt'] =  ''.join(xxx.split("\n"))
            v['use_m_'+self.name] = False
        return v
    
    def panel_param(self, obj, validator = None):
        if self.name is None:
            return [self.guilabel, self.default,  2, {}]   
        chk_float   = (self.type == 'float')
        chk_int     = (self.type == 'int')
        chk_complex = (self.type == 'complex')

        if len(self.shape) == 0:
            value = getattr(obj, self.name + '_txt')
            ret = obj.make_phys_param_panel(self.guilabel,
                                            value, 
                                             no_func = self.no_func,
                                             chk_float = chk_float,
                                             chk_int   = chk_int,
                                             chk_complex = chk_complex)
            if self.chkbox:
                ret =  [None, [True, [value]], 27, [{'text':'Use'},
                                                    {'elp': [ret]}],]
            return ret
        else:
            row = self.shape[0]
            col = self.shape[1] if self.ndim == 2 else 1
            suffix = ['_'+x for x in self.suffix]
            return obj.make_matrix_panel(self.name, suffix, row = row,
                                         col = col,
                                         validator = validator,
                                         chk_float = chk_float,
                                         chk_int   = chk_int, 
                                         chk_complex = chk_complex)

    def get_panel_value(self, obj):
        if self.name is None: return
        if len(self.shape) == 0:
            if self.chkbox:
                f = getattr(obj, 'use_' + self.name + '_txt')
                v = getattr(obj, self.name + '_txt')
                return [f, [v]]
            else:
                return getattr(obj, self.name + '_txt')
        else:
            suffix = ['_'+x for x in self.suffix]        
            flag = getattr(obj, 'use_m_'+self.name)
            cb_value =  'Array Form' if flag else 'Elemental Form'
            a = [cb_value,
                [[str(getattr(obj, self.name+n+'_txt')) for n in suffix]],
                [str(getattr(obj, self.name + '_m_txt'))]]
            return a
            
    def import_panel_value(self, obj, v):
        if self.name is None: return        
        if len(self.shape) == 0:
            if self.chkbox:
                setattr(obj, 'use_' + self.name + '_txt', v[0])
                setattr(obj, self.name + '_txt', str(v[1][0]))                
            else:
                setattr(obj, self.name + '_txt', str(v))
        else:
            suffix = ['_'+x for x in self.suffix]        
            setattr(obj, 'use_m_'+self.name,
                    (str(v[0]) == 'Array Form'))
            for k, n in enumerate(suffix):
                setattr(obj, self.name + n + '_txt', str(v[1][0][k]))
            setattr(obj, self.name + '_m_txt', str(v[2][0]))

    def preprocess_params(self, obj):
        '''
        if no_func, values are evaluated at this stage.
        otherwise, it only makes sure that values are string
        '''
        if self.name is None: return                
        if len(self.shape) == 0:
            if self.no_func:
                value = obj.eval_phys_expr(str(getattr(obj, self.name+'_txt')),
                                           self.name)[0]
                setattr(obj, self.name, value)
            else:
                setattr(obj, self.name,
                        str(getattr(obj, self.name + '_txt')))
        else:
            suffix = ['_'+x for x in self.suffix] + ['_m']
            for n in suffix:
                 setattr(obj, self.name + n,
                         str(getattr(obj, self.name + n + '_txt')))
                             
    def make_value_or_expression(self, obj):
        if self.name is None: return None                     
        if len(self.shape) == 0:
            if self.no_func:
                pass
            else:
                var, f_name0 = obj.eval_phys_expr(getattr(obj, self.name),
                                           self.name)
                if f_name0 is None: return var
                return f_name0
        else:
            if getattr(obj, 'use_m_'+self.name):
                suffix = ['_m']                          
                eval_expr = obj.eval_phys_array_expr
            else:
                suffix = ['_'+x for x in self.suffix] 
                eval_expr = obj.eval_phys_expr         
            f_name = []
            for n in suffix:
               var, f_name0 = eval_expr(getattr(obj, self.name+n), self.name + n)
               if f_name0 is None:
                   f_name.append(var)
               else:
                   f_name.append(f_name0)

            return f_name
        
    def panel_tip(self):
        if self.name is None: return None                       
        return self.tip
    
class Vtable(OrderedDict):
    def attribute_set(self, v, keys = None):
        keys = keys if keys is not None else self.keys()
        for key in keys:
            v = self[key].add_attribute(v)
        return v

    def panel_param(self, obj, keys = None, validator = None):
        keys = keys if keys is not None else self.keys()
        return [self[key].panel_param(obj, validator = validator)
                for key in keys]
    
    def panel_tip(self, keys = None):
        keys = keys if keys is not None else self.keys()
        return [self[key].panel_tip() for key in keys]
                    
    def get_panel_value(self, obj, keys = None):
        keys = keys if keys is not None else self.keys()
        return [self[key].get_panel_value(obj) for key in keys]
                    
    def import_panel_value(self, obj, values, keys = None):
        keys = keys if keys is not None else self.keys()
        for k, v in zip(keys, values):
            self[k].import_panel_value(obj, v)
                    
    def preprocess_params(self, obj, keys = None):
        keys = keys if keys is not None else self.keys()
        for k in keys:
            self[k].preprocess_params(obj)
                    
    def make_value_or_expression(self, obj, keys = None):    
        keys = keys if keys is not None else self.keys()
        return [self[key].make_value_or_expression(obj) for key in keys]
    
