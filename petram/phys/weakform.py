'''

   Weakform : interface to use MFEM integrator

'''
from petram.phys.coefficient import SCoeff, VCoeff, DCoeff, MCoeff
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import os
import numpy as np
from abc import abstractmethod

from petram.phys.phys_model import Phys
from petram.model import Domain, Bdry, Edge, Point, Pair

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('WeakForm')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


def get_integrators(filename, return_all=False):
    import petram.engine
    fid = open(os.path.join(os.path.dirname(
        petram.engine.__file__), 'data', filename), 'r')
    lines = fid.readlines()
    fid.close()

    lines = [l for l in lines if not l.startswith('#')]
    lines = [l.split("|")[1:] for l in lines]
    names = [l[0].strip() for l in lines]
    domains = [[x.strip() for x in l[1].split(',') if x.strip() != '']
               for l in lines]
    ranges = [[x.strip() for x in l[2].split(',') if x.strip() != '']
              for l in lines]
    for trial, test in zip(domains, ranges):
        if len(test) == 0:
            for x in trial:
                test.append(x)

    def x(txt):
        xx = [x.strip() for x in txt.split(",")]
        return xx

    coeffs = [x(l[3]) for l in lines]
    dims = [[int(x.strip()[0]) for x in l[6].split(',')] for l in lines]
    wf_forms = [l[4] for l in lines]
    strong_forms = [l[5] for l in lines]

    if return_all:
        return list(zip(names, domains, ranges, coeffs, dims, wf_forms, strong_forms))
    else:
        return list(zip(names, domains, ranges, coeffs, dims))


bilinintegs = get_integrators('BilinearOps')
linintegs = get_integrators('LinearOps')

data = [("coeff_lambda", VtableElement("coeff_lambda", type='array',
                                       guilabel="lambda", default=0.0, tip="coefficient",))]


class WeakIntegration(Phys):
    vt_coeff = Vtable(data)

    def attribute_set(self, v):
        v['use_src_proj'] = False
        v['use_dst_proj'] = False
        v['coeff_type'] = 'Scalar'
        v['integrator'] = 'MassIntegrator'
        v['test_idx'] = 0  # (index)
        self.vt_coeff.attribute_set(v)
        v = Phys.attribute_set(self, v)
        return v

    def get_panel1_value(self):
        dep_vars = self.get_root_phys().dep_vars
        return [dep_vars[self.test_idx],
                self.coeff_type,
                self.vt_coeff.get_panel_value(self)[0],
                self.integrator, ]
        # self.use_src_proj,
        # self.use_dst_proj,]

    def panel1_tip(self):
        pass

    def itg_choice_cb(self):
        names = [x[0] for x in self.itg_choice()]
        return names

    def panel1_param(self):
        import wx
        p = ["coeff. type", "Scalar", 4,
             {"style": wx.CB_READONLY, "choices": ["Scalar", "Vector", "Diagonal", "Matrix"]}]

        names = [x[0] for x in self.itg_choice()]

        import wx
        from petram.pi.widget_forms import WidgetForms

        p2 = ["integrator", None, 99,
              {"UI": WidgetForms,
               # "span": (1, 2),
               # "choices": names,
               "choices_cb": self.itg_choice_cb, }]

        dep_vars = self.get_root_phys().dep_vars
        panels = self.vt_coeff.panel_param(self)
        ll = [["test space (Rows)", dep_vars[0], 4,
               {"style": wx.CB_READONLY, "choices": dep_vars}],
              p,
              panels[0],
              p2, ]
        #["use src proj.",  self.use_src_proj,   3, {"text":""}],
        # ["use dst proj.",  self.use_dst_proj,   3, {"text":""}],  ]
        return ll

    def is_complex(self):
        return self.get_root_phys().is_complex_valued

    def import_panel1_value(self, v):
        dep_vars = self.get_root_phys().dep_vars
        self.test_idx = dep_vars.index(str(v[0]))
        self.coeff_type = str(v[1])
        self.vt_coeff.import_panel_value(self, (v[2],))
        self.integrator = str(v[3])
        #self.use_src_proj = v[4]
        #self.use_dst_proj = v[5]

    def preprocess_params(self, engine):
        self.vt_coeff.preprocess_params(self)
        super(WeakIntegration, self).preprocess_params(engine)

    def add_contribution(self, engine, a, real=True, is_trans=False, is_conj=False):
        c = self.vt_coeff.make_value_or_expression(self)[0]

        if real:
            dprint1("Add "+self.integrator + " contribution(real)" +
                    str(self._sel_index), "c", c)
        else:
            dprint1("Add "+self.integrator + " contribution(imag)" +
                    str(self._sel_index), "c", c)

        cotype = self.coeff_type[0]
        use_dual = False
        for b in self.itg_choice():
            if b[0] == self.integrator:
                use_dual = "S*2" in b[3]
                break

        c_coeff = self.get_coefficient_from_expression(c, cotype,
                                                       use_dual=use_dual,
                                                       real=real,
                                                       is_conj=is_conj)

        if self.integrator == 'DerivativeIntegrator1':
            integrator = getattr(mfem, 'DerivativeIntegrator')
            c_coeff = (c_coeff, 0)
        elif self.integrator == 'DerivativeIntegrator2':
            integrator = getattr(mfem, 'DerivativeIntegrator')
            c_coeff = (c_coeff, 1)
        elif self.integrator == 'DerivativeIntegrator3':
            integrator = getattr(mfem, 'DerivativeIntegrator')
            c_coeff = (c_coeff, 2)
        else:
            integrator = getattr(mfem, self.integrator)

        if isinstance(self, Bdry):
            # print "Bdry Integrator"
            adder = a.AddBoundaryIntegrator
        elif isinstance(self, Domain):
            # print "Domain Integrator"
            adder = a.AddDomainIntegrator
        else:
            assert False, "this class is not supported in weakform"
        self.add_integrator(engine, 'c', c_coeff,
                            adder, integrator, transpose=is_trans)

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        self.add_contribution(engine, a, real=real)

    def add_lf_contribution(self, engine, b, real=True, kfes=0):
        self.add_contribution(engine, b, real=real)

    def add_mix_contribution2(self, engine, mbf, r, c,  is_trans, is_conj,
                              real=True):
        self.add_contribution(engine, mbf, real=real,
                              is_trans=is_trans, is_conj=is_conj)

    @abstractmethod
    def itg_choice(self):
        return []


class WeakLinIntegration(WeakIntegration):
    def has_lf_contribution(self, kfes):
        return self.test_idx == kfes

    def itg_choice(self):
        t = self.get_root_phys().get_fec_type(self.test_idx)
        if len(t) > 2 and t[2] == "v":
            t = t[:3]
        bb = [b for b in linintegs if t in b[1]]

        if (t == 'RT' and self.get_root_phys().dim == 2 and
                self.get_root_phys().geom_dim == 3):
            t = 'L2'
            bb = bb + [b for b in linintegs if t in b[1]]

        dim = self.dim
        bb = [b for b in bb if dim in b[4]]

        return bb


class WeakBilinIntegration(WeakIntegration):
    def itg_choice(self):
        t = self.get_root_phys().get_fec_type(self.test_idx)
        if len(t) > 2 and t[2] == "v":
            t = t[:3]
        bb = [b for b in bilinintegs if t in b[2]]

        # adjust choices for RT-Trace (I need this to have MassIntegrator in menu)
        if (t == 'RT' and self.get_root_phys().dim == 2 and
                self.get_root_phys().geom_dim == 3):
            t = 'L2'
            bb = bb + [b for b in bilinintegs if t in b[2]]

        if self.paired_var is not None:
            paired_name, paired_idx = self.paired_var
        else:
            paired_name = self.get_root_phys().name()
            paired_idx = 0

        if paired_name in self.get_root_phys().parent:
            phys1 = (self.get_root_phys().parent)[paired_name]
            fes_idx1 = paired_idx

            from petram.helper.projection import find_fes_mapping
            t2, _order = find_fes_mapping(phys1, fes_idx1,
                                          self.get_root_phys(), self.test_idx)
            #t2 = (self.get_root_phys().parent)[paired_name].get_fec_type(paired_idx)
            if len(t2) > 2 and t2[2] == "v":
                t2 = t2[:3]
            else:
                t2 = t2[:2]
            bb2 = [b for b in bb if t2 in b[1]]

            # adjust choices for RT-Trace
            if (t2 == 'RT' and self.get_root_phys().dim == 2 and
                    self.get_root_phys().geom_dim == 3):
                t2 = 'L2'
                bb2 = bb2 + [b for b in bb if t2 in b[1]]
        else:
            bb2 = bb
        dim = self.dim

        bb = [b for b in bb2 if dim in b[4]]

        return bb

    def attribute_set(self, v):
        v = super(WeakBilinIntegration, self).attribute_set(v)
        v['paired_var'] = None  # (phys_name, index)
        v['use_symmetric'] = False
        v['use_conj'] = False
        return v

    def get_panel1_value(self):
        '''
        (old routine will be deleted)
        if self.paired_var is not None:
            try:
                mfem_physroot = self.get_root_phys().parent
                var_s = mfem_physroot[self.paired_var[0]].dep_vars
                n = var_s[self.paired_var[1]]
                p = self.paired_var[0]
            except:
                self.paired_var = None

        if self.paired_var is None:
            n = self.get_root_phys().dep_vars[0]
            p = self.get_root_phys().name()

        var = n + " ("+p + ")"
        '''
        from petram.utils import pv_get_gui_value
        gui_value, self.paired_var = pv_get_gui_value(self, self.paired_var)

        v1 = [gui_value]
        v2 = super(WeakBilinIntegration, self).get_panel1_value()
        v3 = [self.use_symmetric, self.use_conj]
        return v1 + v2 + v3

    def panel1_tip(self):
        pass

    def is_complex(self):
        return self.get_root_phys().is_complex_valued

    def import_panel1_value(self, v):
        '''
        (old routine will be deleted)
        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()

        if len(v[0]) == 0:
            # v[0] could be '' if object is based to a tree.
            idx = 0
        else:
            idx = names.index(str(v[0]).split("(")[0].strip())

        self.paired_var = (pnames[idx], pindex[idx])
        '''
        from petram.utils import pv_from_gui_value
        self.paired_var = pv_from_gui_value(self, v[0])

        super(WeakBilinIntegration, self).import_panel1_value(v[1:-2])
        self.use_symmetric = v[-2]
        self.use_conj = v[-1]

    def panel1_param(self):
        '''
        (old routine will be deleted)
        import wx
        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()
        names = [n+" ("+p + ")" for n, p in zip(names, pnames)]

        ll1 = [["paired variable", "S", 4,
                {"style": wx.CB_READONLY, "choices": names}]]
        '''
        from petram.utils import pv_panel_param
        ll1 = [pv_panel_param(self, "paird var."), ]

        ll2 = super(WeakBilinIntegration, self).panel1_param()
        ll3 = [["make symmetric",  self.use_symmetric,   3, {"text": ""}],
               ["use  conjugate",  self.use_conj,   3, {"text": ""}], ]
        ll = ll1 + ll2 + ll3

        return ll

    def has_bf_contribution(self, kfes):
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[self.paired_var[0]].dep_vars
        trialname = var_s[self.paired_var[1]]
        testname = self.get_root_phys().dep_vars[self.test_idx]

        return (trialname == testname) and (self.test_idx == kfes)

    def has_mixed_contribution(self):
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[self.paired_var[0]].dep_vars
        trialname = var_s[self.paired_var[1]]
        testname = self.get_root_phys().dep_vars[self.test_idx]

        return (trialname != testname)

    def get_mixedbf_loc(self):
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[self.paired_var[0]].dep_vars
        trialname = var_s[self.paired_var[1]]
        testname = self.get_root_phys().dep_vars[self.test_idx]

        loc = []
        is_trans = 1
        loc.append((testname, trialname, is_trans, 1))

        if self.use_symmetric and not self.use_conj:
            loc.append((testname, trialname, -1, -1))
        if self.use_symmetric and self.use_conj:
            loc.append((testname, trialname, -1, 1))

        return loc

    def get_projection(self):
        return 1


def add_delta_contribution(obj, engine, a, real=True, is_trans=False, is_conj=False):
    self = obj
    c = self.vt_coeff.make_value_or_expression(self)[0]
    if isinstance(c, str):
        c = [c]

    if real:
        dprint1("Add "+self.integrator + " delta (real)" +
                str(self._sel_index), "c", c)
    else:
        dprint1("Add "+self.integrator + " delta (imag)" +
                str(self._sel_index), "c", c)

    cotype = self.coeff_type[0]

    if self.get_root_phys().vdim > 1:
        dim = self.get_root_phys().vdim
    else:
        el_name = self.get_root_phys().element
        dim = self.get_root_phys().geom_dim
    sdim = self.get_root_phys().geom_dim

    integrator = getattr(mfem, self.integrator)
    adder = a.AddDomainIntegrator

    for pos in self.pos_value:
        args = list(pos[:sdim])
        if cotype == 'S':
            for b in self.itg_choice():
                if b[0] == self.integrator:
                    break

            if not "S*2" in b[3]:
                if isinstance(c[0], str):
                    c_coeff = None
                    # c_coeff = SCoeff(c[0],  self.get_root_phys().ind_vars,
                    #                 self._local_ns, self._global_ns,
                    #                 real = real, conj=is_conj)
                    value = eval(c[0], self._global_ns, self._local_ns)
                    dprint1("time dependent delta", value)
                    args.append(float(value))
                else:
                    c_coeff = None
                    if real:
                        args.append(float(np.array(c)[0].real))
                    else:
                        args.append(float(np.array(c)[0].imag))
                if args[-1] != 0:
                    d = mfem.DeltaCoefficient(*args)
                    if c_coeff is not None:
                        assert False, "This option needs update of PyMFEM"
                        d2 = mfem.ProductCoefficient(c_coeff, d)
                        d2._linked_c = (c_coeff, d)
                        adder(integrator(d2))
                    else:
                        adder(integrator(d))

            else:  # so far this is only for an elastic integrator
                if real:
                    args.append(float(np.array(c)[0].real))
                else:
                    args.append(float(np.array(c)[0].imag))
                d1 = mfem.DeltaCoefficient(*args)
                if real:
                    args.append(float(np.array(c)[1].real))
                else:
                    args.append(float(np.array(c)[1].imag))
                d2 = mfem.DeltaCoefficient(*args)
                adder(integrator(d1, d2))

        elif cotype == 'V':
            if real:
                direction = np.array(c).real
            else:
                direction = np.array(c).imag
            args.append(1.0)
            dir = mfem.Vector(direction)
            d = mfem.VectorDeltaCoefficient(dir, *args)
            adder(integrator(d))
        else:
            assert False, "M and D are not supported for delta coefficient"


def validate_sel(value, obj, w):
    g = obj._global_ns
    try:
        value = eval(value, g)
        pos = np.atleast_2d(value)
        return True
    except:
        return False


class WeakLinDeltaIntegration(WeakLinIntegration):
    def panel2_param(self):
        return [["Position",  "[[0,0,0],]",  0, {'changing_event': True,
                                                 'setfocus_event': True,
                                                 'validator': validate_pos,
                                                 'validator_param': self}]]

    def import_panel2_value(self, v):
        self.sel_index = ['all']
        self.sel_index_txt = '["all"]'
        self.pos_txt = v[0]
        g = self._global_ns
        try:
            value = eval(v[0], g)
            self.pos_value = np.atleast_2d(value)
        except:
            import traceback
            traceback.print_exc()
            pass

    def attribute_set(self, v):
        v = super(WeakLinDeltaIntegration, self).attribute_set(v)
        v['pos_txt'] = "[[ 0,0,0,],]"
        v['pos_value'] = np.array([[0, 0, 0, ], ])
        return v

    def get_panel2_value(self):
        return (self.pos_txt,)

    def add_contribution(self, engine, a, real=True, is_trans=False, is_conj=False):
        g = self._global_ns
        try:
            value = eval(self.pos_txt, g)
            self.pos_value = np.atleast_2d(value)
        except:
            import traceback
            traceback.print_exc()
            raise
        add_delta_contribution(self, engine, a, real=real, is_trans=is_trans,
                               is_conj=is_conj)


class WeakBilinDeltaIntegration(WeakBilinIntegration):
    def panel2_param(self):
        return [["Position",  "[[0,0,0],]",  0, {'changing_event': True,
                                                 'setfocus_event': True,
                                                 'validator': validate_pos,
                                                 'validator_param': self}]]

    def import_panel2_value(self, v):
        self.sel_index = ['all']
        self.sel_index_txt = '["all"]'
        self.pos_txt = v[0]
        g = self._global_ns
        try:
            value = eval(v[0], g)
            self.pos_value = np.atleast_2d(value)
        except:
            import traceback
            traceback.print_exc()

    def attribute_set(self, v):
        v = super(WeakBilinDeltaIntegration, self).attribute_set(v)
        v['pos_txt'] = "[[ 0,0,0,],]"
        v['pos_value'] = np.array([[0, 0, 0, ], ])
        return v

    def get_panel2_value(self):
        return (self.pos_txt,)

    def add_contribution(self, engine, a, real=True, is_trans=False, is_conj=False):
        g = self._global_ns
        try:
            value = eval(self.pos_txt, g)
            self.pos_value = np.atleast_2d(value)
        except:
            import traceback
            traceback.print_exc()
            raise
        add_delta_contribution(self, engine, a, real=real, is_trans=is_trans,
                               is_conj=is_conj)
