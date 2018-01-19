def attribute_set_names(name, suffix):
    '''
    ex. epsilonr -> expilonr_xx, _xy,,,,
    '''
    return [name + '_'+ x for x in suffix]

def init_multivalidator_attribute_set(d, name, suffix, values, use):
    names = attribute_set_names(name, suffix)
    for n, v in names, values:
        d[n] = v
    d['use_m_'+name] = use

class ValidatorInputMixIn(object):        
    def validator_elp_panel(self, row, col, name):
        elp = [[None, None, 0, {'validator': self.check_phys_expr,
                            'validator_param': name},]]
    
        return elp
    
    def get_alidator_panel1_value(self, name, suffix):
        '''
        get attribute xxx_m, xxx_suffix, use_m_xxx for
        elp panel
        '''         
        return [[str(getattr(self, name))],]
                
    def import_validator_panel1_value(self, v, name, suffix):
        '''
        set attribute xxx_m, xxx_suffix, use_m_xxx using
        elp panel value
        '''         
        setattr(self, name,  str(v[0]))

    def init_multivalidator_attribute_set(self, d, name, suffix, values, use):
        names = attribute_set_names(name, suffix)
        for n, v in names, values:
            d[n] = v
        d['use_m_'+name] = use
                
    
    def multivalicator_elp_panel(self, row, col, name, suffix):
        '''
        generate elp (row x col)
        '''
        a = [{'validator': self.check_phys_expr,
              'validator_param':name + '_' + s} for s in suffix]
                   
        elp1 = [[None, None, 43, {'row': row,
                              'col': col,
                             'text_setting': a}],]
        elp2 = [[None, None, 0, {'validator': self.check_phys_array_expr,
                             'validator_param': name+'_m'},]]
                   
        elp = [None, None, 34, ({'text': name + '   ',
                             'choices': ['Elemental Form', 'Array Form'],
                             'call_fit': False},
                             {'elp': elp1},  
                             {'elp': elp2},),],
        return elp
   
    def form_name(self, v):
        if v: return 'Array Form'
        else: return 'Elemental Form'
        
    def get_multivalidator_panel1_value(self, name, suffix):
        '''
        get attribute xxx_m, xxx_suffix, use_m_xxx for
        elp panel
        '''         
        return [self.form_name(getattr(self, 'use_m_' + name)), 
                [[str(getattr(self, name + s)) for s in suffix]],
                [str(getattr(self, name + '_m'))]]

    def import_multivalidator_panel1_value(self, v, name, suffix):
        '''
        set attribute xxx_m, xxx_suffix, use_m_xxx using
        elp panel value
        '''         
        setattr(self, 'use_m_'+name,  (str(v[0]) == 'Array Form'))
        for k, s in enumerate(suffix):
            setattr(self, name + '_' + s, str(v[1][0][k]))
        setattr(self, name + '_m',  str(v[2][0]))
            

    def make_multivaliator_f_name(self, basename, suffix):
        if getattr(self, 'use_m_'+basename):
            names = ['_m']
            eval_expr = self.eval_phys_array_expr
        else:
            names = ['_' + x for x in suffix]
            eval_expr = self.eval_phys_expr            
              
        f_name = []
        for n in names:
           var, f_name0 = eval_expr(getattr(self, basename+n), basename + n)
           if f_name0 is None:
               f_name.append(var)
           else:
               f_name.append(f_name0)
        
        return f_name
