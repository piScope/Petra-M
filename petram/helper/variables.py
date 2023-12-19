'''
    Copyright (c) 2018-, S. Shiraiwa
    All Rights reserved. See file COPYRIGHT for details.

    Variables


    This modules interface string exression to MFEM

    for example, when a user write

       epsiolnr = 'x + 0.5y'

     and if epsilonr is one of Variable object, it will become

        call of epsilon(x,y,z) at integration points (matrix assembly)

        or

        many such calles for all nodal points (plot)

    about variable decorator:
       this class instance is used to convered a user written function
       to a Vriable object.

    from petram.helper.variables import variable

    @variable.float()
    def test(x, y, z):
       return 1-0.1j

    @variable.float(dependency = ("u",))
    def test(x, y, z):
       # u is FES variable solved in the previous space.
       value = u()
       return value

    @variable.complex()
    def ctest(x, y, z):
       return 1-0.1j

    @variable.array(complex=True,shape=(2,))
    def atest(x, y, z):
       return np.array([1-0.1j,1-0.1j])

'''
import numpy as np
import weakref
import types
import traceback
from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD

from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Variables')


class _decorator(object):
    @staticmethod
    def float(func=None, *, dependency=None, grad=None, curl=None, div=None, td=False):
        '''
        this form allows for using both
        @float
        @float()
        '''
        def wrapper(func):
            def dec(*args, **kwargs):
                obj = PyFunctionVariable(func,
                                         complex=False,
                                         dependency=dependency,
                                         grad=grad,
                                         curl=curl,
                                         div=div)
                return obj
            return dec(func)
        if func:
            return wrapper(func)
        else:
            return wrapper

    @staticmethod
    def complex(func=None, *, dependency=None, grad=None, curl=None, div=None, td=False):
        '''
        this form allows for using both
        @complex
        @complex()
        '''
        def wrapper(func):
            def dec(*args, **kwargs):
                obj = PyFunctionVariable(func,
                                         complex=True,
                                         dependency=dependency,
                                         grad=grad,
                                         curl=curl,
                                         div=div)

                return obj
            return dec(func)
        if func:
            return wrapper(func)
        else:
            return wrapper

    @staticmethod
    def array(complex=False, shape=(1,), dependency=None, grad=None,
              curl=None, div=None, td=False):
        def dec(func):
            obj = PyFunctionVariable(func,
                                     complex=complex,
                                     shape=shape,
                                     dependency=dependency,
                                     grad=grad,
                                     curl=curl,
                                     div=div,)
            return obj
        return dec


class _decorator_jit(object):

    @staticmethod
    def float(func=None, *, dependency=None, grad=None, curl=None, div=None, td=False, params=None):
        '''
        This form allows to use both with and without ()
        @float
        @float()
        @float(dependency....
        '''
        def wrapper(func):
            def dec(*args, **kwargs):
                obj = NumbaCoefficientVariable(func,
                                               complex=False,
                                               dependency=dependency,
                                               params=params,
                                               grad=grad,
                                               curl=curl,
                                               div=div,
                                               td=td,)
                return obj
            return dec(func)
        if func:
            return wrapper(func)
        else:
            return wrapper

    @staticmethod
    def complex(func=None, *, dependency=None, grad=None, curl=None, div=None, td=False, params=None):
        '''
        This form allows to use both with and without ()
        @complex
        @complex()
        @complex(dependency....
        '''
        def wrapper(func):
            def dec(*args, **kwargs):
                obj = NumbaCoefficientVariable(func,
                                               complex=True,
                                               dependency=dependency,
                                               params=params,
                                               grad=grad,
                                               curl=curl,
                                               div=div,
                                               td=td,)
                return obj
            return dec(func)
        if func:
            return wrapper(func)
        else:
            return wrapper

    @staticmethod
    def array(complex=False, shape=(1,), dependency=None, grad=None, curl=None, div=None, td=False, params=None):
        def dec(func):
            obj = NumbaCoefficientVariable(func,
                                           complex=complex,
                                           shape=shape,
                                           dependency=dependency,
                                           params=params,
                                           grad=grad,
                                           curl=curl,
                                           div=div,
                                           td=td,)

            return obj
        return dec


variable = _decorator()
variable.jit = _decorator_jit()


def eval_code(co, g, l, flag=None):
    if flag is not None:
        if not flag:
            return co
    else:
        if not isinstance(co, types.CodeType):
            return co
    try:
        a = eval(co, g, l)
    except NameError:
        dprint1("global names", g.keys())
        dprint1("local names", l.keys())
        raise
    if callable(a):
        return a()
    return a


def cosd(x): return np.cos(x * np.pi / 180.)
def sind(x): return np.sin(x * np.pi / 180.)
def tand(x): return np.tan(x * np.pi / 180.)


var_g = {'sin': np.sin,
         'cos': np.cos,
         'tan': np.tan,
         'cosd': cosd,
         'sind': sind,
         'tand': tand,
         'arctan': np.arctan,
         'arctan2': np.arctan2,
         'exp': np.exp,
         'log10': np.log10,
         'log': np.log,
         'log2': np.log2,
         'sqrt': np.sqrt,
         'abs': np.abs,
         'conj': np.conj,
         'real': np.real,
         'imag': np.imag,
         'sum': np.sum,
         'dot': np.dot,
         'vdot': np.vdot,
         'array': np.array,
         'cross': np.cross,
         'pi': np.pi,
         'min': np.max,
         'min': np.min,
         'sign': np.sign,
         'ones': np.ones,
         'diag': np.diag,
         'zeros': np.zeros,
         'nan': np.nan,
         'inf': np.inf,
         'inv': np.linalg.inv,
         'linspace': np.linspace,
         'logspace': np.logspace}


def check_vectorfe_in_lowdim(gf):
    # check if this is VectorFE in lower dimensionality

    fes = gf.FESpace()
    sdim = fes.GetMesh().SpaceDimension()

    assert fes.GetNE() > 0, "Finite Element space has zero elements"

    isVector = (fes.GetFE(0).GetRangeType() == fes.GetFE(0).VECTOR)
    dim = fes.GetFE(0).GetDim()

    if isVector and dim < sdim:
        assert False, "Nodal evaluator does not work for low dimenstional vector field (try without averaging)"


class Variables(dict):
    def __repr__(self):
        txt = []
        for k in self.keys():
            txt.append(k + ':' + str(self[k]))
        return "\n".join(txt)


class Variable():
    '''
    define everything which we define algebra
    '''

    def __init__(self, complex=False, dependency=None, grad=None, curl=None, div=None):
        self.complex = complex

        # dependency stores a list of Finite Element space discrite variable
        # names whose set_point has to be called
        self.dependency = [] if dependency is None else dependency
        self.div = [] if div is None else div
        self.curl = [] if curl is None else curl
        self.grad = [] if grad is None else grad

    def __call__(self):
        raise NotImplementedError("Subclass need to implement")

    def __add__(self, other):
        if isinstance(other, Variable):
            return self() + other()
        return self() + other

    def __sub__(self, other):
        if isinstance(other, Variable):
            return self() - other()
        return self() - other

    def __mul__(self, other):
        if isinstance(other, Variable):
            return self() * other()
        return self() * other

    def __div__(self, other):
        if isinstance(other, Variable):
            return self() / other()
        return self() / other

    def __truediv__(self, other):
        if isinstance(other, Variable):
            return self() / other()
        return self() / other

    def __floordiv__(self, other):
        if isinstance(other, Variable):
            return self() // other()
        return self() // other

    def __radd__(self, other):
        if isinstance(other, Variable):
            return self() + other()
        return self() + other

    def __rsub__(self, other):
        if isinstance(other, Variable):
            return other() - self()
        return other - self()

    def __rmul__(self, other):
        if isinstance(other, Variable):
            return self() * other()
        return self() * other

    def __rdiv__(self, other):
        if isinstance(other, Variable):
            return other() / self()
        return other / self()

    def __rtruediv__(self, other):
        if isinstance(other, Variable):
            return other() / self()
        return other / self()

    def __rfloordiv__(self, other):
        if isinstance(other, Variable):
            return other() // self()
        return other // self()

    def __divmod__(self, other):
        if isinstance(other, Variable):
            return self().__divmod__(other())
        return self().__divmod__(other)

    def __floordiv__(self, other):
        if isinstance(other, Variable):
            return self().__floordiv__(other())
        return self().__floordiv__(other)

    def __mod__(self, other):
        if isinstance(other, Variable):
            return self().__mod__(other())
        return self().__mod__(other)

    def __pow__(self, other):
        if isinstance(other, Variable):
            return self().__pow__(other())
        return self().__pow__(other)

    def __neg__(self):
        return self().__neg__()

    def __pos__(self):
        return self().__pos__()

    def __abs__(self):
        return self().__abs__()

    def __getitem__(self, idx):
        return self()[idx]

    def get_names(self):
        return []

    def get_emesh_idx(self, idx=None, g=None):
        if idx is None:
            idx = []
        return idx

    def ncface_values(self, ifaces=None, irs=None,
                      gtypes=None, **kwargs):
        raise NotImplementedError("Subclass need to implement")

    def ncedge_values(self, *args, **kwargs):
        return self.ncface_values(*args, **kwargs)

    def point_values(self, *args, **kwargs):
        print(self)
        raise NotImplementedError("Subclass need to implement")

    def add_topological_info(self, mesh):
        if not isinstance(self, DomainVariable):
            return
        if mesh.Dimension() == 3:
            self.topo_info = (3, mesh.extended_connectivity['surf2vol'])
        if mesh.Dimension() == 2:
            self.topo_info = (2, mesh.extended_connectivity['line2surf'])
        if mesh.Dimension() == 1:
            self.topo_info = (1, mesh.extended_connectivity['vert2line'])

    def get_jitted_coefficient(self, *args):
        return None
    '''
    def make_callable(self):
        raise NotImplementedError("Subclass need to implement")

    def make_nodal(self):
        raise NotImplementedError("Subclass need to implement")
    '''


class TestVariable(Variable):
    def __init__(self, comp=-1, complex=False):
        super(TestVariable, self).__init__(complex=complex)

    def set_point(self, T, ip, g, l, t=None):
        self.x = T.Transform(ip)

    def __call__(self, **kwargs):
        return 2.

    def nodal_values(self, locs=None, **kwargs):
        # iele = None, elattr = None, el2v = None,
        #  wverts = None, locs = None, g = None
        return locs[:, 0] * 0 + 2.0

    def ncface_values(self, locs=None, **kwargs):
        return locs[:, 0] * 0 + 2.0


class Constant(Variable):
    def __init__(self, value, comp=-1):
        super(Constant, self).__init__(complex=np.iscomplexobj(value))
        self.value = value

    def __repr__(self):
        return "Constant(" + str(self.value) + ")"

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def set_point(self, T, ip, g, l, t=None):
        self.x = T.Transform(ip)

    def __call__(self, **kwargs):
        return self.value

    def nodal_values(self, iele=None, el2v=None, locs=None,
                     wverts=None, elvertloc=None, **kwargs):

        size = len(wverts)
        shape = [size] + list(np.array(self.value).shape)

        dtype = np.complex128 if self.complex else np.float64
        ret = np.zeros(shape, dtype=dtype)
        wverts = np.zeros(size)

        for kk, m, loc in zip(iele, el2v, elvertloc):
            if kk < 0:
                continue
            for pair, xyz in zip(m, loc):
                idx = pair[1]
                ret[idx] = self.value

        return ret

    def ncface_values(self, locs=None, **kwargs):
        size = len(locs)
        shape = [size] + list(np.array(self.value).shape)
        return np.tile(self.value, shape)

    def point_values(self, locs=None, **kwargs):
        size = len(locs)
        shape = [size] + list(np.array(self.value).shape)
        return np.tile(self.value, shape)


class SumVariable(Variable):
    def __init__(self, variables, gvariables):
        iscomplex = any([x.complex for x in variables])
        super(SumVariable, self).__init__(complex=iscomplex)
        self.variables = list(variables)
        self.gvariables = list(gvariables)

    def get_names(self):
        ret = []
        for x in self.variables:
            ret.extend(x.get_names())

        return ret

    def add_expression(self, v, g):
        self.complex = self.complex or v.complex
        self.variables.append(v)
        self.gvariables.append(g)

    def set_point(self, T, ip, _g, l, t=None):
        for v, g in zip(self.variables, self.gvariables):
            v.set_point(T, ip, g, l, t=t)

    def __call__(self, **kwargs):
        g = kwargs.pop("g", {}).copy()

        g.update(self.gvariables[0])
        v1 = self.variables[0].__call__(g=g, **kwargs)
        for v, g2 in zip(self.variables[1:], self.gvariables[1:]):
            g.update(g2)
            v1 = v1 + v.__call__(g=g, **kwargs)
        return v1

    def nodal_values(self, **kwargs):
        g = kwargs.pop("g", {}).copy()

        g.update(self.gvariables[0])
        v1 = self.variables[0].nodal_values(g=g, **kwargs)
        for v, g2 in zip(self.variables[1:], self.gvariables[1:]):
            g.update(g2)
            v1 = v1 + v.nodal_values(g=g, **kwargs)
        return v1

    def ncface_values(self, **kwargs):
        g = kwargs.pop("g", {}).copy()

        g.update(self.gvariables[0])
        v1 = self.variables[0].ncface_values(g=g, **kwargs)
        for v, g2 in zip(self.variables[1:], self.gvariables[1:]):
            g.update(g2)
            v1 = v1 + v.ncface_values(g=g, **kwargs)
        return v1

    def point_values(self, **kwargs):
        g = kwargs.pop("g", {}).copy()

        g.update(self.gvariables[0])
        v1 = self.variables[0].point_values(g=g, **kwargs)
        for v, g2 in zip(self.variables[1:], self.gvariables[1:]):
            g.update(g2)
            v1 = v1 + v.point_values(g=g, **kwargs)
        return v1


class CoordVariable(Variable):
    def __init__(self, comp=-1, complex=False):
        super(CoordVariable, self).__init__(complex=complex)
        self.comp = comp

    def __repr__(self):
        return "Coordinates"

    def set_point(self, T, ip, g, l, t=None):
        self.x = T.Transform(ip)

    def __call__(self, **kwargs):
        if self.comp == -1:
            return self.x
        else:
            return self.x[self.comp - 1]

    def nodal_values(self, locs=None, **kwargs):
        # iele = None, elattr = None, el2v = None,
        #  wverts = None, locs = None, g = None
        if self.comp == -1:
            return locs
        else:
            return locs[:, self.comp - 1]

    def ncface_values(self, locs=None, **kwargs):
        if self.comp == -1:
            return locs
        else:
            return locs[:, self.comp - 1]

    def point_values(self, counts=None, locs=None, **kwargs):
        if self.comp == -1:
            return locs
        else:
            return locs[:, self.comp - 1]


class ExpressionVariable(Variable):
    def __init__(self, expr, ind_vars, complex=False):
        super(ExpressionVariable, self).__init__(complex=complex)

        variables = []

        expr = expr.strip()

        code = compile(expr, '<string>', 'eval')
        names = code.co_names
        self.co = code
        self.names = names
        self.expr = expr
        self.ind_vars = ind_vars
        self.variables = WVD()
        # print 'Check Expression', expr.__repr__(), names

    def get_names(self):
        return self.names

    def __repr__(self):
        return "Expression(" + self.expr + ")"

    def set_point(self, T, ip, g, l, t=None):
        self.x = T.Transform(ip)
        #print("setting x", self, self.x)
        for n in self.names:
            if (n in g and isinstance(g[n], Variable)):
                g[n].set_point(T, ip, g, l, t=t)
                self.variables[n] = g[n]

    def __call__(self, **kwargs):
        l = {}
        for k, name in enumerate(self.ind_vars):
            l[name] = self.x[k]
        keys = self.variables.keys()
        for k in keys:
            l[k] = self.variables[k]()
        return (eval_code(self.co, var_g, l))

    def get_emesh_idx(self, idx=None, g=None):
        if idx is None:
            idx = []
        for n in self.names:
            if n in g and isinstance(g[n], Variable):
                idx = g[n].get_emesh_idx(idx=idx, g=g)
        return idx

    def nodal_values(self, iele=None, el2v=None, locs=None,
                     wverts=None, elvertloc=None, g=None,
                     **kwargs):
        #print("Entering nodal(expr)", self.expr)
        size = len(wverts)
        dtype = np.complex128 if self.complex else np.float64
        ret = np.zeros(size, dtype=dtype)
        for kk, m, loc in zip(iele, el2v, elvertloc):
            if kk < 0:
                continue
            for pair, xyz in zip(m, loc):
                idx = pair[1]
                ret[idx] = 1

        l = {}
        ll_name = []
        ll_value = []
        var_g2 = var_g.copy()

        for n in self.names:
            if (n in g and isinstance(g[n], Variable)):
                l[n] = g[n].nodal_values(iele=iele, el2v=el2v, locs=locs,
                                         wverts=wverts, elvertloc=elvertloc,
                                         g=g, **kwargs)
                ll_name.append(n)
                ll_value.append(l[n])
            elif (n in g):
                var_g2[n] = g[n]
        if len(ll_name) > 0:
            value = np.array([eval(self.co, var_g2, dict(zip(ll_name, v)))
                              for v in zip(*ll_value)])
        else:
            for k, name in enumerate(self.ind_vars):
                l[name] = locs[..., k]
            value = np.array(eval_code(self.co, var_g2, l), copy=False)
            if value.ndim > 1:
                value = np.stack([value] * size)
        #value = np.array(eval_code(self.co, var_g, l), copy=False)
        from petram.helper.right_broadcast import multi

        ret = multi(ret, value)
        #print("return (expr)", ret.shape)

        return ret

    def _ncx_values(self, method, ifaces=None, irs=None, gtypes=None,
                    g=None, attr1=None, attr2=None, locs=None,
                    **kwargs):

        size = len(locs)
        dtype = np.complex127 if self.complex else np.float64
        ret = np.zeros(size, dtype=dtype)

        l = {}
        ll_name = []
        ll_value = []
        var_g2 = var_g.copy()
        for n in self.names:
            if (n in g and isinstance(g[n], Variable)):
                m = getattr(g[n], method)
                # l[n] = g[n].ncface_values(ifaces = ifaces, irs = irs,
                l[n] = m(ifaces=ifaces, irs=irs,
                         gtypes=gtypes, locs=locs,
                         attr1=attr1, attr2=attr2,
                         g=g, **kwargs)
                ll_name.append(n)
                ll_value.append(l[n])
            elif (n in g):
                var_g2[n] = g[n]

        if len(ll_name) > 0:
            value = np.array([eval(self.co, var_g2, dict(zip(ll_name, v)))
                              for v in zip(*ll_value)])
        else:
            for k, name in enumerate(self.ind_vars):
                l[name] = locs[..., k]
            value = np.array(eval_code(self.co, var_g2, l), copy=False)
            if value.ndim > 1:
                value = np.stack([value] * size)

        return value

    def ncface_values(self, *args, **kwargs):
        return self._ncx_values('ncface_values', *args, **kwargs)

    def ncedge_values(self, *args, **kwargs):
        return self._ncx_values('ncedge_values', *args, **kwargs)

    def point_values(self, counts=None, locs=None, points=None,
                     attrs=None, elem_ids=None,
                     mesh=None, int_points=None, g=None,
                     knowns=None, **kwargs):

        l = {}
        ll_name = []
        ll_value = []
        var_g2 = var_g.copy()
        for n in self.names:
            if (n in g and isinstance(g[n], Variable)):
                l[n] = g[n].point_values(counts=counts, locs=locs, points=points,
                                         attrs=attrs, elem_ids=elem_ids,
                                         mesh=mesh, int_points=int_points, g=g,
                                         knowns=knowns)
                ll_name.append(n)
                ll_value.append(l[n])
            elif (n in g):
                var_g2[n] = g[n]
        if len(ll_name) > 0:
            value = np.array([eval(self.co, var_g2, dict(zip(ll_name, v)))
                              for v in zip(*ll_value)])
        else:
            for k, name in enumerate(self.ind_vars):
                l[name] = locs[..., k]
            value = np.array(eval_code(self.co, var_g2, l), copy=False)
            if value.ndim > 1:
                value = np.stack([value] * size)

        return value


class DomainVariable(Variable):
    def __init__(self, expr='', ind_vars=None, domains=None,
                 complex=False, gdomain=None):
        super(DomainVariable, self).__init__(complex=complex)
        self.domains = {}
        self.gdomains = {}
        if expr == '':
            return
        domains = sorted(domains)
        self.gdomains[tuple(domains)] = gdomain
        self.domains[tuple(domains)] = ExpressionVariable(expr, ind_vars,
                                                          complex=complex)

    def get_names(self):
        ret = []
        for x in self.domains:
            ret.extend(self.domains[x].get_names())
        return ret

    def __repr__(self):
        return "DomainVariable"

    def _add_something(self, something, gsomething, domains):
        doms = tuple(sorted(domains))

        existing_domains = list(self.domains)
        new_domains = tuple(np.setdiff1d(domains, sum(existing_domains, ())))

        new_exprs = {}
        new_gexprs = {}

        if len(new_domains) > 0:
            new_exprs[new_domains] = something
            new_gexprs[new_domains] = gsomething

        for ed in existing_domains:
            diff = tuple(np.setdiff1d(ed, doms))
            if len(diff) != 0:
                new_exprs[diff] = self.domains[ed]
                new_gexprs[diff] = self.gdomains[ed]
            insct = tuple(np.intersect1d(ed, doms))
            if len(insct) != 0:
                if isinstance(ed, SumVariable):
                    self.domains[ed].add_expression(something, gsomething)
                    new_exprs[insct] = self.domains[ed]
                    new_gexprs[insct] = {}
                else:
                    s = SumVariable(
                        (self.domains[ed], something), (self.gdomains[ed], gsomething))
                    new_exprs[insct] = s
                    new_gexprs[insct] = {}

        self.domains = new_exprs
        self.gdomains = new_gexprs

    def add_expression(self, expr, ind_vars, domains, gdomain, complex=False):
        new_expr = ExpressionVariable(expr, ind_vars,
                                      complex=complex)
        self._add_something(new_expr, gdomain, domains)
        if complex:
            self.complex = True

    def add_const(self, value, domains, gdomain):
        new_expr = Constant(value)
        self._add_something(new_expr, gdomain, domains)

        if np.iscomplexobj(value):
            self.complex = self.complex and True

    def set_point(self, T, ip, g, l, t=None):
        attr = T.Attribute
        if T.GetDimension() == self.topo_info[0]:
            # domain mode
            attrs = [attr]

        elif T.GetDimension() == self.topo_info[0] - 1:
            # boundary mode
            attrs = self.topo_info[1][attr]

        self.domain_target = []
        for domains in self.domains.keys():
            for a in attrs:
                if a in domains:
                    self.domains[domains].set_point(T, ip, g, l, t=t)
                self.domain_target.append(domains)

    def __call__(self, **kwargs):
        if len(self.domain_target) == 0:
            return 0.0
        # we return average for now. when domain variable is
        # evaluated on the boundary, it computes the aveage on both side.
        values = [self.domains[x]() for x in self.domain_target]
        ans = values[0]
        for v in values[1:]:
            ans = ans + v
        return ans / len(values)

    def get_emesh_idx(self, idx=None, g=None):
        if idx is None:
            idx = []
        for domains in self.domains.keys():
            expr = self.domains[domains]
            if isinstance(expr, Variable):
                gdomain = g if self.gdomains[domains] is None else self.gdomains[domains]
                idx.extend(expr.get_emesh_idx(idx=idx, g=gdomain))
        return idx

    def nodal_values(self, iele=None, elattr=None, g=None,
                     current_domain=None, **kwargs):
        # iele = None, elattr = None, el2v = None,
        # wverts = None, locs = None, g = None):
        from petram.helper.right_broadcast import add

        ret = None
        w = None

        for domains in self.domains.keys():
            if (current_domain is not None and
                    domains != current_domain):
                continue

            iele0 = np.zeros(iele.shape, dtype=int) - 1
            for domain in domains:
                idx = np.where(np.array(elattr) == domain)[0]
                iele0[idx] = iele[idx]

            expr = self.domains[domains]

            if self.gdomains[domains] is None:
                gdomain = g
            else:
                gdomain = g.copy()
                for key in self.gdomains[domains]:
                    gdomain[key] = self.gdomains[domains][key]

            v = expr.nodal_values(iele=iele0, elattr=elattr,
                                  current_domain=domains,
                                  g=gdomain, **kwargs)

            if w is None:
                a = np.sum(np.abs(v.reshape(len(v), -1)), -1)
                w = (a != 0).astype(float)
            else:
                a = np.sum(np.abs(v.reshape(len(v), -1)), -1)
                w = w + (a != 0).astype(float)

            #ret = v if ret is None else add(ret, v)
            ret = v if ret is None else ret + v

        idx = np.where(w != 0)[0]
        #ret2 = ret.copy()
        from petram.helper.right_broadcast import div

        ret[idx, ...] = div(ret[idx, ...], w[idx])

        return ret

    def _ncx_values(self, method, ifaces=None, irs=None, gtypes=None,
                    g=None, attr1=None, attr2=None, locs=None,
                    current_domain=None, **kwargs):
        from petram.helper.right_broadcast import add, multi

        ret = None

        w = ifaces * 0  # w : 0 , 0.5, 1
        for domains in self.domains:
            idx = np.in1d(attr1, domains)
            w[idx] = w[idx] + 1.0
            idx = np.in1d(attr2, domains)
            w[idx] = w[idx] + 1.0
        w[w > 0] = 1. / w[w > 0]

        npts = [irs[gtype].GetNPoints() for gtype in gtypes]
        base_weight = np.repeat(w, npts)
        # 1 for exterior face, 0.5 for internal faces

        for domains in self.domains.keys():
            if (current_domain is not None and
                    domains != current_domain):
                continue

            w = np.zeros(ifaces.shape)
            w[np.in1d(attr1, domains)] += 1.0
            w[np.in1d(attr2, domains)] += 1.0
            w2 = base_weight * np.repeat(w, npts)

            expr = self.domains[domains]

            if self.gdomains[domains] is None:
                gdomain = g
            else:
                gdomain = g.copy()
                gdomain.update(self.gdomains[domains])

            m = getattr(expr, method)
            #kwargs['weight'] = w2

            v = m(ifaces=ifaces, irs=irs,
                  gtypes=gtypes, locs=locs, attr1=attr1,
                  attr2=attr2, g=gdomain,
                  current_domain=domains,
                  **kwargs)

            v = multi(v, w2)
            ret = v if ret is None else ret + v
            #ret = v if ret is None else add(ret, v)
        return ret

    def ncface_values(self, *args, **kwargs):
        return self._ncx_values('ncface_values', *args, **kwargs)

    def ncedge_values(self, *args, **kwargs):
        return self._ncx_values('ncedge_values', *args, **kwargs)

    def point_values(self, counts=None, locs=None, points=None,
                     attrs=None, elem_ids=None,
                     mesh=None, int_points=None, g=None,
                     knowns=None, current_domain=None):

        valid_idx = np.where(attrs != -1)[0]
        valid_attrs = attrs[attrs != -1]
        ret = None

        for domains in self.domains.keys():
            if (current_domain is not None and
                    domains != current_domain):
                continue

            '''
            iele0 = np.zeros(iele.shape) - 1
            for domain in domains:
                idx = np.where(np.array(elattr) == domain)[0]
                iele0[idx] = iele[idx]
            '''
            expr = self.domains[domains]

            if self.gdomains[domains] is None:
                gdomain = g
            else:
                gdomain = g.copy()
                for key in self.gdomains[domains]:
                    gdomain[key] = self.gdomains[domains][key]

            idx = np.in1d(valid_attrs, domains)
            attrs2 = attrs.copy()
            attrs2[np.in1d(attrs, domains)] = -1
            v = expr.point_values(counts=counts, locs=locs, points=points,
                                  attrs=attrs, elem_ids=elem_ids,
                                  mesh=mesh, int_points=int_points, g=gdomain,
                                  current_domain=domains,
                                  knowns=knowns)
            if ret is None:
                ret = v
            else:
                idx = np.in1d(valid_attrs, domains)
                ret[idx] = v[idx]

        return ret


class PyFunctionVariable(Variable):
    def __init__(self, func, complex=False, shape=tuple(), dependency=None,
                 grad=None, curl=None, div=None):
        super(
            PyFunctionVariable,
            self).__init__(complex=complex,
                           dependency=dependency,
                           grad=grad,
                           curl=curl,
                           div=div)

        self.func = func
        self.t = None
        self.x = (0, 0, 0)
        self.shape = shape

    def __repr__(self):
        return "PyFunction"

    def set_point(self, T, ip, g, l, t=None):
        self.x = T.Transform(ip)
        self.t = t

    def __call__(self, **kwargs):
        if self.t is not None:
            args = tuple(np.hstack((self.x, t)))
        else:
            args = tuple(self.x)

        #kwargs = {n: locals()[n]() for n in self.dependency}
        # return np.array(self.func(*args, **kwargs), copy=False)
        return np.array(self.func(*args, **kwargs), copy=False)

    def nodal_values(self, iele=None, el2v=None, locs=None,
                     wverts=None, elvertloc=None, g=None, knowns=None,
                     **kwargs):
        # elattr = None, el2v = None,
        # wverts = None, locs = None, g = None
        if locs is None:
            return
        if g is None:
            g = {}
        if knowns is None:
            knowns = WKD()

        size = len(wverts)
        shape = [size] + list(self.shape)

        dtype = np.complex128 if self.complex else np.float64
        ret = np.zeros(shape, dtype=dtype)
        wverts = np.zeros(size)

        for kk, m, loc in zip(iele, el2v, elvertloc):
            if kk < 0:
                continue
            for pair, xyz in zip(m, loc):
                idx = pair[1]
                for n in self.dependency:
                    if not g[n] in knowns:
                        knowns[g[n]] = g[n].nodal_values(iele=iele, el2v=el2v,
                                                         locs=locs, wverts=wverts,
                                                         elvertloc=elvertloc,
                                                         g=g, knows=knowns,
                                                         **kwargs)
                kwargs = {n: knowns[g[n]][idx] for n in self.dependency}
                ret[idx] = ret[idx] + self.func(*xyz, **kwargs)
                wverts[idx] = wverts[idx] + 1
        ret = np.stack([x for x in ret if x is not None])

        idx = np.where(wverts == 0)[0]
        wverts[idx] = 1.0

        from petram.helper.right_broadcast import div
        ret = div(ret, wverts)

        return ret

    def _ncx_values(self, method, ifaces=None, irs=None, gtypes=None,
                    g=None, attr1=None, attr2=None, locs=None,
                    knowns=None, **kwargs):
        if locs is None:
            return
        if g is None:
            g = {}
        if knowns is None:
            knowns = WKD()

        dtype = np.complex128 if self.complex else np.float64

        ret = [None] * len(locs)
        for idx, xyz in enumerate(locs):
            '''
            for n in self.dependency:
                g[n].local_value = knowns[g[n]][idx]
                # putting the dependency variable to functions global.
                # this may not ideal, since there is potential danger
                # of name conflict?
                self.func.func_globals[n] = g[n]
            '''
            for n in self.dependency:
                if not g[n] in knowns:
                    m = getattr(g[n], method)
                    knowns[g[n]] = m(ifaces=ifaces, irs=irs,
                                     gtypes=gtypes, g=g,
                                     attr1=attr1, attr2=attr2,
                                     locs=locs, knows=knowns,
                                     **kwargs)

            kwargs = {n: knowns[g[n]][idx] for n in self.dependency}
            ret[idx] = self.func(*xyz, **kwargs)

        ret = np.stack(ret).astype(dtype, copy=False)

        return ret

    def ncface_values(self, *args, **kwargs):
        return self._ncx_values('ncface_values', *args, **kwargs)

    def ncedge_values(self, *args, **kwargs):
        return self._ncx_values('ncedge_values', *args, **kwargs)

    def point_values(self, counts=None, locs=None, points=None,
                     attrs=None, elem_ids=None,
                     mesh=None, int_points=None, g=None,
                     knowns=None, current_domains=None):

        if locs is None:
            return
        if g is None:
            g = {}
        if knowns is None:
            knowns = WKD()

        shape = [counts] + list(self.shape)
        dtype = np.complex128 if self.complex else np.float64
        ret = np.zeros(shape, dtype=dtype)

        valid_attrs = attrs[attrs != -1]

        jj = 0
        for i in range(counts):
            if valid_attrs[i] == -1:
                continue
            if (current_domains is not None and
                    not valid_attrs[i] in current_domains):
                continue

            for n in self.dependency:
                if not g[n] in knowns:
                    knowns[g[n]] = g[n].point_values(counts=counts, locs=locs, points=points,
                                                     attrs=attrs, elem_ids=elem_ids,
                                                     mesh=mesh, int_points=int_points, g=g,
                                                     knowns=knowns, current_domains=current_domains)

            xyz = tuple(locs[i])
            kwargs = {n: knowns[g[n]][i] for n in self.dependency}
            value = self.func(*xyz, **kwargs)

            ret[i, ...] = value

        return ret


class CoefficientVariable(Variable):
    def __init__(self, coeff_gen, l, g=None, coeff=None):
        self.coeff = coeff_gen(l, g)
        self.kind = coeff_gen.kind

        complex = not (self.coeff[1] is None)
        super(CoefficientVariable, self).__init__(complex=complex)

    @property
    def local_value(self):
        return self._local_value

    @local_value.setter
    def local_value(self, value):
        self._local_value = value

    def __call__(self, **kwargs):
        return self._local_value

    def set_point(self, T, ip, g, l, t=None):
        self.T = T
        self.ip = ip
        self.t = t
        self.set_local_from_T_ip()

    def set_local_from_T_ip(self):
        self.local_value = self.eval_local_from_T_ip()

    def eval_local_from_T_ip(self):
        call_eval = self.get_call_eval()

        T, ip = self.T, self.ip
        if (self.coeff[0] is not None and
                self.coeff[1] is not None):
            value = (np.array(call_eval(self.coeff[0], T, ip)) +
                     1j * np.array(self.coeff[1], T, ip))
        elif self.coeff[0] is not None:
            value = np.array(call_eval(self.coeff[0], T, ip))
        elif self.coeff[1] is not None:
            value = 1j * np.array(call_eval(self.coeff[1], T, ip))
        else:
            assert False, "coeff is (None, None)"
        return value

    def nodal_values(self, iele=None, ibele=None, elattr=None, el2v=None,
                     locs=None, elvertloc=None, wverts=None, mesh=None,
                     iverts_f=None, g=None, knowns=None,
                     edge_evaluator=False, **kwargs):

        g = mfem.Geometry()
        size = len(iverts_f)
        #wverts = np.zeros(size)
        ret = None
        if ibele is None:
            return

        if mesh.Dimension() == 3:
            if edge_evaluator:
                assert False, "EdgeNodal Evaluator does not supported dim=3"
            getelement = mesh.GetBdrElement
            gettransformation = mesh.GetBdrElementTransformation
        elif mesh.Dimension() == 2:
            if edge_evaluator:
                getelement = mesh.GetBdrElement
                gettransformation = mesh.GetBdrElementTransformation
            else:
                getelement = mesh.GetElement
                gettransformation = mesh.GetElementTransformation
        else:
            if edge_evaluator:
                getelement = mesh.GetElement
                gettransformation = mesh.GetElementTransformation
            else:
                assert False, "BdrNodal Evaluator does not support dim=1"

        call_eval = self.get_call_eval()

        for ibe in ibele:
            el = getelement(ibe)
            rule = g.GetVertices(el.GetGeometryType())
            nv = rule.GetNPoints()

            T = gettransformation(ibe)
            bverts = el.GetVerticesArray()

            for i in range(nv):
                ip = rule.IntPoint(i)
                T.SetIntPoint(ip)

                if (self.coeff[0] is not None and
                        self.coeff[1] is not None):
                    value = (np.array(call_eval(self.coeff[0], T, ip)) +
                             1j * np.array(call_eval(self.coeff[1], T, ip)))
                elif self.coeff[0] is not None:
                    value = np.array(call_eval(self.coeff[0], T, ip))
                elif self.coeff[1] is not None:
                    value = 1j * np.array(call_eval(self.coeff[1], T, ip))
                else:
                    assert False, "coeff is (None, None)"

                if ret is None:
                    if len(value.shape) == 0:
                        shape = np.hstack((size, ))
                    else:
                        shape = np.hstack((size, np.atleast_1d(value).shape))
                    ret = np.zeros(shape, dtype=value.dtype)

                idx = np.searchsorted(iverts_f, bverts[i])
                ret[idx, ...] = value

        return ret

    def ncface_values(self, ifaces=None, irs=None, gtypes=None, mesh=None,
                      **kwargs):

        call_eval = self.get_call_eval()

        if mesh.Dimension() == 3:
            m = mesh.GetFaceTransformation
        elif mesh.Dimension() == 2:
            m = mesh.GetElementTransformation

        data = []
        dtype = np.complex128 if self.complex else np.float64

        for i, gtype, in zip(ifaces, gtypes):
            T = m(i)
            ir = irs[gtype]
            nv = ir.GetNPoints()

            for j in range(nv):
                ip = ir.IntPoint(j)

                if (self.coeff[0] is not None and
                        self.coeff[1] is not None):
                    value = (np.array(call_eval(self.coeff[0], T, ip)) +
                             1j * np.array(call_eval(self.coeff[1], T, ip)))
                elif self.coeff[0] is not None:
                    value = np.array(call_eval(self.coeff[0], T, ip))
                elif self.coeff[1] is not None:
                    value = 1j * np.array(call_eval(self.coeff[1], T, ip))
                else:
                    assert False, "coeff is (None, None)"
                data.append(value)

        ret = np.stack(data).astype(dtype, copy=False)

        return ret

    def ncedge_values(self, ifaces=None, irs=None, gtypes=None, mesh=None,
                      **kwargs):

        call_eval = self.get_call_eval()

        if mesh.Dimension() == 2:
            m = mesh.GetFaceTransformation
        elif mesh.Dimension() == 1:
            m = mesh.GetElementTransformation
        else:
            assert False, "NCEdge Evaluator is not supported for this dimension"

        data = []
        dtype = np.complex128 if self.complex else np.float64

        for i, gtype, in zip(ifaces, gtypes):
            T = m(i)
            ir = irs[gtype]
            nv = ir.GetNPoints()

            for j in range(nv):
                ip = ir.IntPoint(j)

                if (self.coeff[0] is not None and
                        self.coeff[1] is not None):
                    value = (np.array(call_eval(self.coeff[0], T, ip)) +
                             1j * np.array(call_eval(self.coeff[1], T, ip)))
                elif self.coeff[0] is not None:
                    value = np.array(call_eval(self.coeff[0], T, ip))
                elif self.coeff[1] is not None:
                    value = 1j * np.array(call_eval(self.coeff[1], T, ip))
                else:
                    assert False, "coeff is (None, None)"
                data.append(value)

        ret = np.stack(data).astype(dtype, copy=False)

        return ret

    def point_values(self, counts=None, locs=None, points=None,
                     attrs=None, elem_ids=None,
                     mesh=None, int_points=None, g=None,
                     knowns=None, **kwargs):

        call_eval = self.get_call_eval()

        data = []
        dtype = np.complex128 if self.complex else np.float64

        for i in range(len(attrs)):
            if attrs[i] == -1:
                continue

            iele = elem_ids[i]
            T = mesh.GetElementTransformation(iele)
            ip = int_points[i]

            if (self.coeff[0] is not None and
                    self.coeff[1] is not None):
                value = (np.array(call_eval(self.coeff[0], T, ip)) +
                         1j * np.array(call_eval(self.coeff[1], T, ip)))
            elif self.coeff[0] is not None:
                value = np.array(call_eval(self.coeff[0], T, ip))
            elif self.coeff[1] is not None:
                value = 1j * np.array(call_eval(self.coeff[1], T, ip))
            else:
                assert False, "coeff is (None, None)"
            data.append(value)

        ret = np.stack(data).astype(dtype, copy=False)

        return ret

    def get_call_eval(self):
        if self.kind == "scalar":
            def call_eval(c, T, ip):
                return c.Eval(T, ip)
        elif self.kind == "vector":
            def call_eval(c, T, ip):
                v = mfem.Vector()
                c.Eval(v, T, ip)
                return v.GetDataArray().copy()
        elif self.kind == "matrix":
            def call_eval(c, T, ip):
                m = mfem.DenseMatrix()
                c.Eval(m, T, ip)
                return m.GetDataArray().copy()
        else:
            assert False, "unknown kind of Coefficient. Must be scalar/vector/matrix"
        return call_eval


class NumbaCoefficientVariable(CoefficientVariable):
    def __init__(self, func, complex=False, shape=tuple(), dependency=None,
                 grad=None, curl=None, div=None, td=False, params=None):
        super(
            CoefficientVariable,
            self).__init__(complex=complex,
                           dependency=dependency,
                           grad=grad,
                           curl=curl,
                           div=div)

        self.func = func
        self.params = params
        self.t = None
        self.x = (0, 0, 0)
        self.shape = shape
        self.td = td
        if len(self.shape) == 0:
            self.kind = 'scalar'
        elif len(self.shape) == 1:
            self.kind = 'vector'
        elif len(self.shape) == 2:
            self.kind = 'matrix'
        else:
            assert False, "unsupported shape"

        self._jitted = None

    def get_jitted_coefficient(self, ind_vars, locals):

        if self._jitted is not None:
            dprint1("(Note) this numba coefficient is already compiled", self.func)
            return self._jitted

        from petram.phys.numba_coefficient import NumbaCoefficient
        if isinstance(self.func, NumbaCoefficient):
            return self.func.mfem_numba_coeff

        from petram.helper.numba_utils import (generate_caller_scalar,
                                               generate_caller_array,
                                               generate_signature_scalar,
                                               generate_signature_array,)
        sdim = len(ind_vars)

        if len(self.shape) == 0:
            jitter = mfem.jit.scalar

            def gen_caller(setting):
                return generate_caller_scalar(setting, sdim)

            def gen_sig(setting):
                return generate_signature_scalar(setting, sdim)

            kwargs = {}

        elif len(self.shape) == 1:
            jitter = mfem.jit.vector

            def gen_caller(setting):
                return generate_caller_array(setting, sdim)

            def gen_sig(setting):
                return generate_signature_array(setting, sdim)

            kwargs = {"shape": self.shape}

        elif len(self.shape) == 2:
            jitter = mfem.jit.matrix

            def gen_caller(setting):
                return generate_caller_array(setting, sdim)

            def gen_sig(setting):
                return generate_signature_array(setting, sdim)

            kwargs = {"shape": self.shape}

        else:
            assert False, "unsupported shape"

        dep = []

        for d in self.dependency:
            dd = locals[d].get_jitted_coefficient(ind_vars, locals)
            if dd is None:
                return
            dep.append(dd)

        from petram.mfem_config import numba_debug, use_parallel
        if use_parallel:
            from mpi4py import MPI
            myid = MPI.COMM_WORLD.rank
        else:
            myid = 0
        numba_debug = False if myid != 0 else numba_debug

        wrapper = jitter(sdim=sdim,
                         complex=self.complex,
                         td=self.td,
                         params=self.params,
                         dependency=dep,
                         interface=(gen_caller, gen_sig),
                         debug=numba_debug,
                         **kwargs)
        self._jitted = wrapper(self.func)
        return self._jitted

    def set_coeff(self, ind_vars, locals):
        coeff = self.get_jitted_coefficient(ind_vars, locals)
        if self.complex:
            self.coeff = (coeff.real, coeff.imag)
        else:
            self.coeff = (coeff, None)


class GridFunctionVariable(Variable):
    def __init__(self, gf_real, gf_imag=None, comp=1,
                 deriv=None, complex=False):

        complex = not (gf_imag is None)
        super(GridFunctionVariable, self).__init__(complex=complex)
        self.dim = gf_real.VectorDim()
        self.comp = comp
        self.isGFSet = False
        self.isDerived = False
        self.deriv = deriv if deriv is not None else self._def_deriv
        self.deriv_args = (gf_real, gf_imag)

        self._curl = None
        self._div = None
        self._grad = None

    def _def_deriv(self, *args):
        return args[0], args[1], None

    def eval_grad(self):
        if self._grad is None:
            grad_r = mfem.GradientGridFunctionCoefficient(self.gfr)
            if self.gfi is not None:
                grad_i = mfem.GradientVectorGridFunctionCoefficient(self.gfi)
            else:
                grad_i = None
            self._grad_gf = (grad_r, grad_i)
        v = mfem.Vector()
        self._grad_gf[0].Eval(v, self.T, self.ip)
        ret = v.GetDataArray().copy()
        if self._grad_gf[1] is not None:
            self._grad_gf[1].Eval(v, self.T, self.ip)
            ret = ret + 1j*v.GetDataArray()
        return ret

    def eval_curl(self):
        if self._curl is None:
            curl_r = mfem.CurlGridFunctionCoefficient(self.gfr)
            if self.gfi is not None:
                curl_i = mfem.CurlVectorGridFunctionCoefficient(self.gfi)
            else:
                curl_i = None
            self._curl_gf = (curl_r, curl_i)
        v = mfem.Vector()
        self._curl_gf[0].Eval(v, self.T, self.ip)
        ret = v.GetDataArray().copy()
        if self._curl_gf[1] is not None:
            self._curl_gf[1].Eval(v, self.T, self.ip)
            ret = ret + 1j*v.GetDataArray()
        return ret

    def eval_div(self):
        if self._div is None:
            div_r = mfem.DivergenceGridFunctionCoefficient(self.gfr)
            if self.gfi is not None:
                div_i = mfem.DivergenceVectorGridFunctionCoefficient(self.gfi)
            else:
                div_i = None
            self._div_gf = (div_r, div_i)
        v = mfem.Vector()
        self._div_gf[0].Eval(v, self.T, self.ip)
        ret = v.GetDataArray().copy()
        if self._div_gf[1] is not None:
            self._div_gf[1].Eval(v, self.T, self.ip)
            ret = ret + 1j*v.GetDataArray()
        return ret

    def get_gf_real(self):
        if not self.isGFSet:
            self.set_gfr_gfi()
        return self.gfr

    def get_gf_imag(self):
        if not self.isGFSet:
            self.set_gfr_gfi()
        return self.gfi

    def set_gfr_gfi(self):
        gf_real, gf_imag, extra = self.deriv(*self.deriv_args)
        self.gfr = gf_real
        self.gfi = gf_imag
        self.extra = extra
        self.isGFSet = True
        return gf_real, gf_imag

    def set_point(self, T, ip, g, l, t=None):
        self.T = T
        self.ip = ip
        self.t = t
        self.set_local_from_T_ip()

    @property
    def local_value(self):
        return self._local_value

    @local_value.setter
    def local_value(self, value):
        self._local_value = value

    def __call__(self, **kwargs):
        return self._local_value

    def set_local_from_T_ip(self):
        self.local_value = self.eval_local_from_T_ip()

    def get_emesh_idx(self, idx=None, g=None):
        if idx is None:
            idx = []
        gf_real, gf_imag = self.deriv_args

        if gf_real is not None:
            if not gf_real._emesh_idx in idx:
                idx.append(gf_real._emesh_idx)
        elif gf_imag is not None:
            if not gf_imag._emesh_idx in idx:
                idx.append(gf_imag._emesh_idx)
        else:
            pass

        return idx

    def FESpace(self, check_parallel=True):
        gf_real, gf_imag = self.deriv_args
        if gf_real is not None:
            if hasattr(gf_real, "ParFESpace"):
                return gf_real.ParFESpace()
            else:
                return gf_real.FESpace()
        if gf_imag is not None:
            if hasattr(gf_imag, "ParFESpace"):
                return gf_imag.ParFESpace()
            else:
                return gf_imag.FESpace()


class GFScalarVariable(GridFunctionVariable):
    def __repr__(self):
        return "GridFunctionVariable (Scalar)"

    def set_funcs(self):
        # I should come back here to check if this works
        # with vector gf and/or boundary element. probably not...
        if not self.isGFSet:
            gf_real, gf_imag = self.set_gfr_gfi()
        else:
            gf_real, gf_imag = self.gfr, self.gfi

        name = gf_real.FESpace().FEColl().Name()
        if name.startswith("ND") or name.startswith("RT"):
            self.isVectorFE = True
            self.func_r = mfem.VectorGridFunctionCoefficient(gf_real)
            if gf_imag is not None:
                self.func_i = mfem.VectorGridFunctionCoefficient(gf_imag)
            else:
                self.func_i = None
        else:
            self.isVectorFE = False
            self.func_r = mfem.GridFunctionCoefficient(gf_real)
            if gf_imag is not None:
                self.func_i = mfem.GridFunctionCoefficient(gf_imag)
            else:
                self.func_i = None
        self.isDerived = True

    def eval_local_from_T_ip(self):
        if not self.isDerived:
            self.set_funcs()
        if self.isVectorFE:
            if self.func_i is None:
                v = mfem.Vector()
                self.func_r.Eval(v, self.T, self.ip)
                return v.GetDataArray()[self.comp - 1]
            else:
                v1 = mfem.Vector()
                v2 = mfem.Vector()
                self.func_r.Eval(v1, self.T, self.ip)
                self.func_i.Eval(v2, self.T, self.ip)
                return (v1.GetDataArray() + 1j *
                        v2.GetDataArray())[self.comp - 1]
        else:
            if self.func_i is None:
                return self.func_r.Eval(self.T, self.ip)
            else:
                return (self.func_r.Eval(self.T, self.ip) +
                        1j * self.func_i.Eval(self.T, self.ip))

    def nodal_values(self, iele=None, el2v=None, wverts=None,
                     **kwargs):
        if iele is None:
            return

        if not self.isDerived:
            self.set_funcs()

        # check if this is VectorFE in lower dimensionality
        gf = self.gfr if self.gfr is not None else self.gfi
        check_vectorfe_in_lowdim(gf)

        size = len(wverts)
        if self.gfi is None:
            ret = np.zeros(size, dtype=np.float64)
        else:
            ret = np.zeros(size, dtype=np.complex128)
        wverts = np.zeros(size)

        for kk, m in zip(iele, el2v):
            if kk < 0:
                continue
            values = mfem.doubleArray()

            self.gfr.GetNodalValues(kk, values, self.comp)

            for k, idx in m:
                ret[idx] = ret[idx] + values[k]
                wverts[idx] += 1
            if self.gfi is not None:
                arr = mfem.doubleArray()
                self.gfi.GetNodalValues(kk, arr, self.comp)
                for k, idx in m:
                    ret[idx] = ret[idx] + arr[k] * 1j

        ret = ret / wverts

        return ret

    def ncface_values(self, ifaces=None, irs=None,
                      gtypes=None, **kwargs):
        if not self.isDerived:
            self.set_funcs()

        name = self.gfr.FESpace().FEColl().Name()
        ndim = self.gfr.FESpace().GetMesh().Dimension()

        isVector = False
        if (name.startswith('RT') or
                name.startswith('ND')):
            d = mfem.DenseMatrix()
            p = mfem.DenseMatrix()
            isVector = True
        else:
            d = mfem.Vector()
            p = mfem.DenseMatrix()
        data = []

        def get_method(gf, ndim, isVector):
            if gf is None:
                return None
            if ndim == 3:
                if isVector:
                    return gf.GetFaceVectorValues
                elif gf.VectorDim() > 1:
                    def func(i, side, ir, vals, tr, in_gf=gf):
                        in_gf.GetFaceValues(
                            i, side, ir, vals, tr, vdim=self.comp)
                    return func
                else:
                    return gf.GetFaceValues
            elif ndim == 2:
                if isVector:
                    def func(i, side, ir, vals, tr, in_gf=gf):
                        in_gf.GetVectorValues(i, ir, vals, tr)
                    return func
                elif gf.VectorDim() > 1:
                    def func(i, side, ir, vals, tr, in_gf=gf):
                        #in_gf.GetValues(i, ir, vals, tr, vdim=self.comp - 1)
                        in_gf.GetValues(i, ir, vals, tr, self.comp-1)
                    return func
                else:
                    def func(i, side, ir, vals, tr, in_gf=gf):
                        in_gf.GetValues(i, ir, vals, tr)
                        return
                    return func
            else:
                assert False, "ndim = 1 has no face"
            return None

        getvalr = get_method(self.gfr, ndim, isVector)
        getvali = get_method(self.gfi, ndim, isVector)

        for i, gtype, in zip(ifaces, gtypes):
            ir = irs[gtype]
            getvalr(i, 2, ir, d, p)  # side = 2 (automatic?)
            v = d.GetDataArray().copy()
            if isVector:
                v = v[self.comp - 1, :]

            if getvali is not None:
                getvali(i, 2, ir, d, p)  # side = 2 (automatic?)
                vi = d.GetDataArray().copy()
                if isVector:
                    vi = vi[self.comp - 1, :]
                v = v + 1j * vi
            data.append(v)
        data = np.hstack(data)

        return data

    def ncedge_values(self, ifaces=None, irs=None,
                      gtypes=None, **kwargs):
        if not self.isDerived:
            self.set_funcs()

        name = self.gfr.FESpace().FEColl().Name()
        ndim = self.gfr.FESpace().GetMesh().Dimension()

        isVector = False
        if name.startswith('RT'):
            d = mfem.DenseMatrix()
            p = mfem.DenseMatrix()
            isVector = True
        elif name.startswith('ND'):
            d = mfem.DenseMatrix()
            p = mfem.DenseMatrix()
            isVector = True
        else:
            d = mfem.Vector()
            p = mfem.DenseMatrix()
        data = []

        def get_method(gf, ndim, isVector):
            if gf is None:
                return None
            if ndim == 1:
                # if isVector:
                #    def func(i, ir, vals, tr, in_gf=gf):
                if gf.VectorDim() > 1:
                    def func(i, ir, vals, tr, in_gf=gf):
                        in_gf.GetValues(i, ir, vals, tr, self.comp - 1)
                    return func
                else:
                    def func(i, ir, vals, tr, in_gf=gf):
                        in_gf.GetValues(i, ir, vals, tr)
                        return
                    return func
            elif ndim == 2:
                side = 2
                if isVector:
                    def func(i, ir, vals, tr, in_gf=gf):
                        in_gf.GetFaceVectorValues(i, side, ir, vals, tr)
                    return func
                elif gf.VectorDim() > 1:
                    def func(i, ir, vals, tr, in_gf=gf):
                        in_gf.GetFaceValues(
                            i, side, ir, vals, tr, self.comp - 1)
                    return func
                else:
                    def func(i, ir, vals, tr, in_gf=gf):
                        in_gf.GetFaceValues(i, side, ir, vals, tr)
                        return
                    return func

            else:
                assert False, "ndim = 3 is not supported"
            return None

        getvalr = get_method(self.gfr, ndim, isVector)
        getvali = get_method(self.gfi, ndim, isVector)

        for i, gtype, in zip(ifaces, gtypes):
            ir = irs[gtype]
            getvalr(i, ir, d, p)  # side = 2 (automatic?)
            v = d.GetDataArray().copy()
            if isVector:
                v = v[self.comp - 1, :]

            if getvali is not None:
                getvali(i, ir, d, p)  # side = 2 (automatic?)
                vi = d.GetDataArray().copy()
                if isVector:
                    vi = vi[self.comp - 1, :]
                v = v + 1j * vi
            data.append(v)

        data = np.hstack(data)
        return data

    def point_values(self, counts=None, locs=None, points=None,
                     attrs=None, elem_ids=None,
                     mesh=None, int_points=None, g=None,
                     knowns=None, **kwargs):

        if not self.isDerived:
            self.set_funcs()
        gf = self.gfr if self.gfr is not None else self.gfi

        name = gf.FESpace().FEColl().Name()
        ndim = gf.FESpace().GetMesh().Dimension()

        if (name.startswith('RT') or
                name.startswith('ND')):
            isVector = True
        else:
            isVector = False

        d = mfem.Vector()

        if self.complex:
            dtype = complex
            val = complex(0.0)
        else:
            dtype = float
            val = float(0.0)

        data = np.zeros(counts, dtype=dtype)

        jj = 0
        for i in range(len(attrs)):
            if attrs[i] == -1:
                continue

            ip = int_points[i]
            iele = elem_ids[i]

            val = val * 0.

            if self.gfr is not None:
                if isVector:
                    self.gfr.GetVectorValue(iele, ip, d)
                    val = d[self.comp - 1]
                else:
                    val = self.gfr.GetValue(iele, ip, self.comp - 1)

            if self.gfi is not None:
                if isVector:
                    self.gfi.GetVectorValue(iele, ip, d)
                    val += 1j * d[self.comp - 1]
                else:
                    val += 1j * self.gfi.GetValue(iele, ip, self.comp - 1)

            data[jj] = val
            jj = jj + 1

        return data

    def get_jitted_coefficient(self, ind_vars, locals):
        if not self.isDerived:
            self.set_funcs()

        if isinstance(self.func_r, mfem.VectorCoefficient):
            v = [0] * self.func_r.GetVDim()
            v[self.comp-1] = 1
            c2 = mfem.VectorConstantCoefficient(v)
            # the value of c2 will be copied.
            ret1 = mfem.InnerProductCoefficient(self.func_r, c2)
            if self.func_i is not None:
                ret2 = mfem.InnerProductCoefficient(self.func_i, c2)
                return (ret1, ret2)
            else:
                return ret1

        if self.func_i is None:
            return self.func_r
        else:
            return (self.func_r, self.func_i)


class GFVectorVariable(GridFunctionVariable):
    def __repr__(self):
        return "GridFunctionVariable (Vector)"

    def set_funcs(self):
        if not self.isGFSet:
            gf_real, gf_imag = self.set_gfr_gfi()
        else:
            gf_real, gf_imag = self.gfr, self.gfi

        self.dim = gf_real.VectorDim()
        name = gf_real.FESpace().FEColl().Name()
        if name.startswith("ND") or name.startswith("RT"):
            self.isVectorFE = True
            self.func_r = mfem.VectorGridFunctionCoefficient(gf_real)
            if gf_imag is not None:
                self.func_i = mfem.VectorGridFunctionCoefficient(gf_imag)
            else:
                self.func_i = None

        else:
            self.isVectorFE = False
            self.func_r = [mfem.GridFunctionCoefficient(gf_real, comp=k + 1)
                           for k in range(self.dim)]

            if gf_imag is not None:
                self.func_i = [mfem.GridFunctionCoefficient(gf_imag, comp=k + 1)
                               for k in range(self.dim)]
            else:
                self.func_i = None
        self.isDerived = True

    def eval_local_from_T_ip(self):
        if not self.isDerived:
            self.set_funcs()
        if self.isVectorFE:
            if self.func_i is None:
                v = mfem.Vector()
                self.func_r.Eval(v, self.T, self.ip)
                return v.GetDataArray().copy()
            else:
                v1 = mfem.Vector()
                v2 = mfem.Vector()
                self.func_r.Eval(v1, self.T, self.ip)
                self.func_i.Eval(v2, self.T, self.ip)
                return v1.GetDataArray().copy() + 1j * v2.GetDataArray().copy()
        else:
            if self.func_i is None:
                return np.array([func_r.Eval(self.T, self.ip) for
                                 func_r in self.func_r])
            else:
                return np.array([(func_r.Eval(self.T, self.ip) +
                                  1j * func_i.Eval(self.T, self.ip))
                                 for func_r, func_i
                                 in zip(self.func_r, self.func_i)])

    def nodal_values(self, iele=None, el2v=None, wverts=None,
                     **kwargs):
        # iele = None, elattr = None, el2v = None,
        # wverts = None, locs = None, g = None
        #print("grid vector nodal")
        if iele is None:
            return
        if not self.isDerived:
            self.set_funcs()

        size = len(wverts)

        # check if this is VectorFE in lower dimensionality
        gf = self.gfr if self.gfr is not None else self.gfi
        check_vectorfe_in_lowdim(gf)

        ans = []
        for comp in range(self.dim):
            if self.gfi is None:
                ret = np.zeros(size, dtype=np.float64)
            else:
                ret = np.zeros(size, dtype=np.complex128)

            wverts = np.zeros(size)
            for kk, m in zip(iele, el2v):
                if kk < 0:
                    continue
                values = mfem.doubleArray()
                self.gfr.GetNodalValues(kk, values, comp + 1)
                for k, idx in m:
                    ret[idx] = ret[idx] + values[k]
                    wverts[idx] += 1
                if self.gfi is not None:
                    arr = mfem.doubleArray()
                    self.gfi.GetNodalValues(kk, arr, comp + 1)
                    for k, idx in m:
                        ret[idx] = ret[idx] + arr[k] * 1j
            # print(list(wverts))
            ans.append(ret / wverts)
        ret = np.transpose(np.vstack(ans))
        return ret

    def ncface_values(self, ifaces=None, irs=None,
                      gtypes=None, **kwargs):

        if not self.isDerived:
            self.set_funcs()
        ndim = self.gfr.FESpace().GetMesh().Dimension()

        d = mfem.DenseMatrix()
        p = mfem.DenseMatrix()
        data = []

        def get_method(gf, ndim):
            if gf is None:
                return None
            if ndim == 3:
                return gf.GetFaceVectorValues
            elif ndim == 2:
                return gf.GetVectorValues
            else:
                assert False, "ndim = 1 has no face"
        getvalr = get_method(self.gfr, ndim)
        getvali = get_method(self.gfi, ndim)

        for i, gtype, in zip(ifaces, gtypes):
            ir = irs[gtype]
            getvalr(i, 2, ir, d, p)  # side = 2 (automatic?)
            v = d.GetDataArray().copy()

            if getvali is not None:
                getvali(i, 2, ir, d, p)
                vi = d.GetDataArray().copy()
                v = v + 1j * vi
            data.append(v)
        ret = np.hstack(data).transpose()

        return ret

    def point_values(self, counts=None, locs=None, points=None,
                     attrs=None, elem_ids=None,
                     mesh=None, int_points=None, g=None,
                     knowns=None, **kwargs):

        if not self.isDerived:
            self.set_funcs()
        gf = self.gfr if self.gfr is not None else self.gfi

        name = gf.FESpace().FEColl().Name()
        ndim = gf.FESpace().GetMesh().Dimension()

        if (name.startswith('RT') or
                name.startswith('ND')):
            isVector = True
            vdim = gf.FESpace().GetMesh().SpaceDimension()
        else:
            isVector = False
            vdim = gf.VectorDim()

        d = mfem.Vector()

        if self.complex:
            dtype = complex

        else:
            dtype = float

        data = np.zeros((counts, vdim), dtype=dtype)

        jj = 0
        for i in range(len(attrs)):
            if attrs[i] == -1:
                continue

            ip = int_points[i]
            iele = elem_ids[i]

            val = 0.0

            if self.gfr is not None:
                self.gfr.GetVectorValue(iele, ip, d)
                val = d.GetDataArray().copy()

            if self.gfi is not None:
                self.gfi.GetVectorValue(iele, ip, d)
                dd = 1j * d.GetDataArray().copy()
                val = val + dd

            data[jj, :] = val
            jj = jj + 1

        return data


'''

Surf Variable:
 Regular Variable + Surface Geometry (n, nx, ny, nz)

'''


class SurfVariable(Variable):
    def __init__(self, sdim, complex=False):
        self.sdim = sdim
        super(SurfVariable, self).__init__(complex=complex)


class SurfNormal(SurfVariable):
    def __init__(self, sdim, comp=-1, complex=False):
        self.comp = comp
        SurfVariable.__init__(self, sdim, complex=complex)

    def __repr__(self):
        return "SurfaceNormal (nx, ny, nz)"

    def set_point(self, T, ip, g, l, t=None):
        nor = mfem.Vector(self.sdim)
        mfem.CalcOrtho(T.Jacobian(), nor)
        self.nor = nor.GetDataArray().copy()

    def get_jitted_coefficient(self, ind_vars, locals):
        norm = mfem.VectorBdrNormalCoefficient(len(ind_vars))
        return norm

    def __call__(self, **kwargs):
        if self.comp == -1:
            return self.nor
        else:
            return self.nor[self.comp - 1]

    def nodal_values(self, ibele=None, mesh=None, iverts_f=None,
                     **kwargs):
        # iele = None, elattr = None, el2v = None,
        # wverts = None, locs = None, g = None

        g = mfem.Geometry()
        size = len(iverts_f)
        #wverts = np.zeros(size)
        ret = np.zeros((size, self.sdim))
        if ibele is None:
            return

        for ibe in ibele:
            el = mesh.GetBdrElement(ibe)
            rule = g.GetVertices(el.GetGeometryType())
            nv = rule.GetNPoints()

            T = mesh.GetBdrElementTransformation(ibe)
            bverts = mesh.GetBdrElement(ibe).GetVerticesArray()

            for i in range(nv):
                nor = mfem.Vector(self.sdim)
                T.SetIntPoint(rule.IntPoint(i))
                mfem.CalcOrtho(T.Jacobian(), nor)
                idx = np.searchsorted(iverts_f, bverts[i])

                ret[idx, :] += nor.GetDataArray().copy()
                #wverts[idx] = wverts[idx] + 1

        #for i in range(self.sdim): ret[:,i] /= wvert
        # normalize to length one.
        ret = ret / np.sqrt(np.sum(ret**2, 1)).reshape(-1, 1)

        if self.comp == -1:
            return ret
        return ret[:, self.comp - 1]

    def ncface_values(self, ifaces=None, irs=None, gtypes=None,
                      locs=None, mesh=None, **kwargs):

        size = len(locs)
        ret = np.zeros((size, self.sdim))
        if ifaces is None:
            return

        nor = mfem.Vector(self.sdim)

        if mesh.Dimension() == 3:
            m = mesh.GetFaceTransformation
        elif mesh.Dimension() == 2:
            m = mesh.GetElementTransformation
        idx = 0
        for i, gtype, in zip(ifaces, gtypes):
            ir = irs[gtype]
            nv = ir.GetNPoints()
            T = m(i)
            for j in range(nv):
                T.SetIntPoint(ir.IntPoint(i))
                mfem.CalcOrtho(T.Jacobian(), nor)
                ret[idx, :] = nor.GetDataArray().copy()
                idx = idx + 1

        from petram.helper.right_broadcast import div

        ret = div(ret, np.sqrt(np.sum(ret**2, -1)))
        if self.comp == -1:
            return ret
        return ret[:, self.comp - 1]

    def ncedge_values(self, *args, **kwargs):
        raise NotImplementedError("Normal is not defined on Edge")


class SurfExpressionVariable(ExpressionVariable, SurfVariable):
    '''
    expression valid on surface
    '''

    def __init__(self, expr, ind_vars, sdim, complex=False):
        ExpressionVariable.__init__(self, expr, ind_vars, complex=complex)
        SurfVariable.__init__(self, sdim, complex=complex)

    def __repr__(self):
        return "SurfaceExpression(" + self.expr + ")"

    def set_point(self, T, ip, g, l, t=None):
        self.x = T.Transform(ip)
        self.t = t
        T.SetIntPoint(ip)
        nor = mfem.Vector(self.sdim)
        mfem.CalcOrtho(T.Jacobian(), nor)
        self.nor = nor.GetDataArray().copy()

    def __call__(self, **kwargs):
        l = {}
        for k, name in enumerate(self.ind_vars):
            l[name] = self.x[k]
        l['n'] = self.nor
        for k, name in enumerate(self.ind_vars):
            l['n' + name] = self.nor[k]
        keys = self.variables.keys()
        for k in keys:
            l[k] = self.variables[k]()
        return (eval_code(self.co, var_g, l))

    def nodal_values(self, **kwargs):
        # this may not be used al all??

        l = {}
        for n in self.names:
            if (n in g and isinstance(g[n], Variable)):
                l[n] = g[n].nodal_values(**kwargs)
        for k, name in enumerate(self.ind_vars):
            l[name] = locs[..., k]
        for k, name in enumerate(self.ind_vars):
            l['n' + name] = nor[..., k]
        return (eval_code(self.co, var_g, l))

    def ncface_values(self, **kwargs):
        assert False, "ncface in SurfaceExpressionVariable must be added"


'''
 Bdr Variable = Surface Variable defined on particular boundary
'''


class BdrVariable(ExpressionVariable, SurfVariable):
    pass


def append_suffix_to_expression(expr, vars, suffix):
    if vars is None:
        return expr + suffix
    for v in vars:
        expr = expr.replace(v, v + suffix)
    return expr


def add_scalar(solvar, name, suffix, ind_vars, solr,
               soli=None, deriv=None, vars=None):
    name = append_suffix_to_expression(name, vars, suffix)
    solvar[name] = GFScalarVariable(solr, soli, comp=1, deriv=deriv)


def add_components(solvar, name, suffix, ind_vars, solr,
                   soli=None, deriv=None, vars=None):
    name = append_suffix_to_expression(name, vars, suffix)
    solvar[name] = GFVectorVariable(solr, soli, deriv=deriv)
    for k, p in enumerate(ind_vars):
        solvar[name + p] = GFScalarVariable(solr, soli, comp=k + 1,
                                            deriv=deriv)


def add_elements(solvar, name, suffix, ind_vars, solr,
                 soli=None, deriv=None, elements=None):
    elements = elements if elements is not None else []
    for k, p in enumerate(ind_vars):
        solvar[name + suffix + p] = GFScalarVariable(solr, soli, comp=k + 1,
                                                     deriv=deriv)


def add_component_expression(solvar, name, suffix, ind_vars, expr, vars,
                             componentname,
                             domains=None, bdrs=None, complex=None,
                             gdomain=None, gbdr=None):
    expr = append_suffix_to_expression(expr, vars, suffix)

    if isinstance(componentname, int):
        componentname = ind_vars[componentname]

    cname = name + suffix + componentname
    if domains is not None:
        if (cname) in solvar:
            solvar[cname].add_expression(expr, ind_vars, domains,
                                         gdomain,
                                         complex=complex)
        else:
            solvar[cname] = DomainVariable(expr, ind_vars,
                                           domains=domains,
                                           complex=complex,
                                           gdomain=gdomain)
    elif bdrs is not None:
        assert False, "BoundaryVariable not implemented."

    else:
        solvar[cname] = ExpressionVariable(expr, ind_vars,
                                           complex=complex)


def add_expression(solvar, name, suffix, ind_vars, expr, vars,
                   domains=None, bdrs=None, complex=None,
                   gdomain=None, gbdr=None):

    expr = append_suffix_to_expression(expr, vars, suffix)

    if domains is not None:
        if (name + suffix) in solvar:
            solvar[name + suffix].add_expression(expr, ind_vars, domains,
                                                 gdomain,
                                                 complex=complex)

        else:
            solvar[name + suffix] = DomainVariable(expr, ind_vars,
                                                   domains=domains,
                                                   complex=complex,
                                                   gdomain=gdomain)
    elif bdrs is not None:
        assert False, "BoundaryVariable not implemented."

    else:
        solvar[name + suffix] = ExpressionVariable(expr, ind_vars,
                                                   complex=complex)


def add_constant(solvar, name, suffix, value, domains=None,
                 gdomain=None, bdrs=None, gbdr=None):

    if domains is not None:
        if (name + suffix) in solvar:
            solvar[name + suffix].add_const(value, domains, gdomain)
        else:
            solvar[name + suffix] = DomainVariable('')
            solvar[name + suffix].add_const(value, domains, gdomain)
    elif bdrs is not None:
        pass
    else:
        solvar[name + suffix] = Constant(value)
        #solvar[name + suffix] = value


def add_surf_normals(solvar, ind_vars):
    sdim = len(ind_vars)
    solvar['n'] = SurfNormal(sdim, comp=-1)
    for k, p in enumerate(ind_vars):
        solvar['n' + p] = SurfNormal(sdim, comp=k + 1)


def add_coordinates(solvar, ind_vars):
    for k, p in enumerate(ind_vars):
        solvar[p] = CoordVariable(comp=k + 1)


def project_variable_to_gf(c, ind_vars, gfr, gfi,
                           global_ns=None, local_ns=None):

    if global_ns is None:
        global_ns = {}
    if local_ns is None:
        local_ns = {}

    from petram.phys.weakform import VCoeff, SCoeff

    fes = gfr.FESpace()
    ndim = fes.GetMesh().Dimension()
    sdim = fes.GetMesh().SpaceDimension()
    vdim = fes.GetVDim()
    fec = fes.FEColl().Name()

    if (fec.startswith('ND') or fec.startswith('RT')):
        coeff_dim = sdim
    else:
        coeff_dim = vdim

    def project_coeff(gf, coeff_dim, c, ind_vars, real):
        if coeff_dim > 1:
            #print("vector coeff", c)
            coeff = VCoeff(coeff_dim, c, ind_vars,
                           local_ns, global_ns, real=real)
        else:
            #print("coeff", c)
            coeff = SCoeff(c, ind_vars,
                           local_ns, global_ns, real=real)

        gf.ProjectCoefficient(coeff)

    project_coeff(gfr, coeff_dim, c, ind_vars, True)
    if gfi is not None:
        project_coeff(gfi, coeff_dim, c, ind_vars, False)


'''

   NativeCoefficient class

   This class opens the possibility ot use mfem native coefficient (C++)
   class object in BF/LF.

   We can define  math operatios between native coefficent class objects
   in the way to map the operatio to SumCoefficient/ProductCofficient/...
   recently added in MFEM.

   Full implementation needs to wait update of PyMFEM. Eventually, this
   class may move to PyMFEM (under mfem.common)

   from petram.helper.variables import variable, coefficient
   @coefficient.complex()
   def ksqrt():
       coeff1 = mfem.ConstantCoefficient(900)
       coeff2 = mfem.ConstantCoefficient(200)
       return coeff1, coeff2
'''


class _coeff_decorator(object):
    def float(self, dependency=None, td=False, jit=False):
        def dec(func):
            obj = NativeCoefficientGen(
                func, dependency=dependency, td=td, jit=jit)
            return obj
        return dec

    def complex(self, dependency=None, td=False, jit=False):
        def dec(func):
            obj = ComplexNativeCoefficientGen(
                func, dependency=dependency, td=td, jit=jit)
            return obj
        return dec

    def array(self, complex=False, shape=(1,),
              dependency=None, td=False, jit=False):
        def dec(func):
            if len(shape) == 1:
                if complex:
                    obj = VectorComplexNativeCoefficientGen(
                        func, dependency=dependency, shape=shape, jit=jit, td=td)
                else:
                    obj = VectorNativeCoefficientGen(
                        func, dependency=dependency, shape=shape, jit=jit, td=td)
            elif len(shape) == 2:
                if complex:
                    obj = MatrixComplexNativeCoefficientGen(
                        func, dependency=dependency, shape=shape, jit=jit, td=td)
                else:
                    obj = MatrixNativeCoefficientGen(
                        func, dependency=dependency, shape=shape, jit=jit, td=td)
            return obj
        return dec


coefficient = _coeff_decorator()


class NativeCoefficientGenBase(object):
    '''
    define everything which we define algebra
    '''

    def __init__(self, fgen, igen=None, complex=False,
                 dependency=None, shape=None, td=False, jit=False):
        self.complex = complex
        # dependency stores a list of Finite Element space discrite variable
        # names whose set_point has to be called
        self.dependency = [] if dependency is None else dependency
        self.fgen = fgen
        self.shape = shape
        self.complex = complex
        self.jit = jit
        self.td = False
        self._generated = None

    def __call__(self, l, g=None):
        '''
        call fgen to generate coefficient

        '''
        if self._generated is not None:
            return self._generated
        m = getattr(self, 'fgen')
        if not self.jit:
            args = []
            for n in self.dependency:
                if self.complex:
                    args.append((l[n].get_gf_real(), l[n].get_gf_imag()))
                else:
                    args.append(l[n].get_gf_real())

            rc = m(*args)
            if self.complex:
                if len(rc) != 2:
                    assert False, "generator must return real/imag parts"
                self._generated = rc
                return self._generated
            else:
                self._generated = rc, None
                return self._generated
        else:
            assert False, "coefficient(jit=True) is not valid anymore. jit is supported by @variable"

    def scale_coeff(self, coeff, scale):
        if self.shape is None:
            c2 = mfem.ConstantCoefficient(scale)
            ret = mfem.ProductCoefficient(coeff, c2)
            ret._c2 = c2
            ret._coeff = coeff
            return ret
        elif len(self.shape) == 1:  # Vector
            c2 = mfem.ConstantCoefficient(scale)
            ret = mfem.ScalarVectorProductCoefficient(c2, coeff)
            ret._c2 = c2
            ret._coeff = coeff
            return ret
        elif len(self.shape) == 2:  # Matrix
            c2 = mfem.ConstantCoefficient(scale)
            ret = mfem.ScalarMatrixProductCoefficient(c2, coeff)
            ret._c2 = c2
            ret._coeff = coeff
            return ret
        else:
            assert False, "dim >= 3 is not supported"

    def get_emesh_idx(self, idx=None, g=None):
        if idx is None:
            idx = []
        return idx


class NativeCoefficientGen(NativeCoefficientGenBase):
    kind = "scalar"

    def __init__(self, func, dependency=None, td=False, jit=False):
        NativeCoefficientGenBase.__init__(
            self, func, complex=False, dependency=dependency, td=td, jit=jit)


class ComplexNativeCoefficientGen(NativeCoefficientGenBase):
    kind = "scalar"

    def __init__(self, func, dependency=None, td=False, jit=False):
        NativeCoefficientGenBase.__init__(
            self, func, complex=True, dependency=dependency, td=td, jit=jit)


class VectorNativeCoefficientGen(NativeCoefficientGenBase):
    kind = "vector"

    def __init__(self, func, dependency=None, shape=None, td=False, jit=False):
        NativeCoefficientGenBase.__init__(
            self, func, complex=False, dependency=dependency, shape=shape, td=td, jit=jit)


class VectorComplexNativeCoefficientGen(NativeCoefficientGenBase):
    kind = "vector"

    def __init__(self, func, dependency=None, shape=None, td=False, jit=False):
        NativeCoefficientGenBase.__init__(
            self, func, complex=True, dependency=dependency, shape=shape, td=td, jit=jit)


class MatrixNativeCoefficientGen(NativeCoefficientGenBase):
    kind = "matrix"

    def __init__(self, func, dependency=None, shape=None, td=False, jit=False):
        NativeCoefficientGenBase.__init__(
            self, func, complex=False, dependency=dependency, shape=shape, td=td, jit=jit)


class MatrixComplexNativeCoefficientGen(NativeCoefficientGenBase):
    kind = "matrix"

    def __init__(self, func, dependency=None, shape=None, td=False, jit=False):
        NativeCoefficientGenBase.__init__(
            self, func, complex=True, dependency=dependency, shape=shape, td=td, jit=jit)
