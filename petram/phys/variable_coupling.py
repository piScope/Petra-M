from petram.helper.matrix_file import write_coo_matrix, write_vector
import sys
import os
import numpy as np
import scipy.sparse
from collections import OrderedDict
from warnings import warn

from petram.mfem_config import use_parallel
if use_parallel:
    from petram.helper.mpi_recipes import *
    import mfem.par as mfem
else:
    import mfem.ser as mfem
import mfem.common.chypre as chypre

# these are only for debuging
from mfem.common.parcsr_extra import ToScipyCoo
from mfem.common.mpi_debug import nicePrint

from petram.phys.phys_model import Phys
from petram.model import Domain, Bdry, ModelDict
from petram.phys.vtable import VtableElement, Vtable

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('VariableCoupling')

# groups = ['Domain', 'Boundary', 'Edge', 'Point', 'Pair']
groups = ['Domain', 'Boundary', 'Pair']

data0 = [("oprt_t1_t2", VtableElement("oprt_t1_t2", type='array',
                                      guilabel="M_{v1, v2}", default="0.0",
                                      tip="oprator to couple (1) to (2)",)),
         ("oprt_t2_t1", VtableElement("rhs_vec", type='array',
                                      guilabel="M_{v2, v1}", default="0.0",
                                      tip="oprator to couple (2) to (1)",)),]


class VariableCoupling(Phys):
    vt = Vtable(data0)
    has_3rd_panel = True
    _has_4th_panel = True
    extra_diagnostic_print = False

    def attribute_set(self, v):
        v = super(VariableCoupling, self).attribute_set(v)
        v["variable_name1_txt"] = ""
        v["variable_name2_txt"] = ""
        v["jmatrix_config"] = None

        v = self.vt.attribute_set(v)

        return v

    def panel1_param(self):
        ll = [["variable(1)",   self.variable_name1_txt, 0, {}],
              ["variable(2)",   self.variable_name2_txt, 0, {}], ]
        ll2 = self.vt.panel_param(self)
        return ll + ll2

    def import_panel1_value(self, v):
        self.variable_name1_txt = str(v[0])
        self.variable_name2_txt = str(v[1])
        self.vt.import_panel_value(self, v[2:])

    def get_panel1_value(self):
        return ([str(self.variable_name1_txt),
                 str(self.variable_name2_txt)] +
                self.vt.get_panel_value(self))

    def panel2_param(self):
        return [[None, "Auxiriary varialbe coupling is global",  2,   {}], ]

    def import_panel2_value(self, v):
        pass

    def get_panel2_value(self):
        return [None]

    def panel3_param(self):
        return [[None, "Auxiriary varialbe coupling is linear/no init.",  2,   {}], ]

    def import_panel3_value(self, v):
        pass

    def get_panel3_value(self):
        return [None]

    def panel4_param(self):
        ll = super(VariableCoupling, self).panel4_param()
        ll.append(['Varying (in time/for loop) RHS', False, 3, {"text": ""}])
        return ll

    def panel4_tip(self):
        return None

    def import_panel4_value(self, value):
        super(VariableCoupling, self).import_panel4_value(value[:-1])
        self.isTimeDependent_RHS = value[-1]

    def get_panel4_value(self):
        value = super(VariableCoupling, self).get_panel4_value()
        value.append(self.isTimeDependent_RHS)
        return value

    def check_extra_update(self, mode):
        '''
        mode = 'B' or 'M'
        'M' return True, if M needs to be updated
        'B' return True, if B needs to be updated
        '''
        if self._update_flag:
            if mode == 'B':
                return self.isTimeDependent_RHS
            if mode == 'M':
                return self.isTimeDependent
        return False

    def has_extra_coupling(self):
        '''
        True if it define coupling between Lagrange multipliers
        '''
        name1 = self.variable_name1_txt.strip()
        names2 = [x.strip() for x in self.variable_name2_txt.split(',')]

        if len(name1) == 0 or len(names2) == 0:
            return False
        return True

    def extra_coupling_names(self):
        '''
        return its own extra_name, and paired (coupled) extra_names
        '''
        name1 = self.variable_name1_txt.strip()
        names2 = [x.strip() for x in self.variable_name2_txt.split(',')]
        return name1, names2

    def get_extra_coupling(self, target_name):
        '''
        [    t2][paired extra]
        [t1    ][extra]
        t1 = (size of extra, size of targert_extra, )
        t2 = (size of target_extra, size of extra, )        
        '''
        from petram.helper.densemat2pymat import Densemat2PyMat

        c1, c2 = self.vt.make_value_or_expression(self)
        c1 = np.atleast_2d(c1)
        c2 = np.atleast_2d(c2)

        ret1 = None
        ret2 = None
        if np.sum(np.abs(c1)) != 0:
            ret1 = Densemat2PyMat(c1)
        if np.sum(np.abs(c2)) != 0:
            ret2 = Densemat2PyMat(c2)
        return ret1, ret2
