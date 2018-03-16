'''

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
    @variable.float
    def test(x, y, z):
       return 1-0.1j
    @variable.complex
    def ctest(x, y, z):
       return 1-0.1j
    @variable.array(complex=True,shape=(2,))
    def atest(x, y, z):
       return np.array([1-0.1j,1-0.1j])
     
'''
import numpy as np
import parser
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

class _decorator(object):
    def float(self, func):
        obj = PyFunctionVariable(func, complex = False)
        return obj
    def complex(self, func):
        obj = PyFunctionVariable(func, complex = True)
        return obj
    def array(self, complex=False, shape = (1,)):
        def dec(func):
            #print "inside dec", complex, shape
            obj = PyFunctionVariable(func, complex = complex, shape = shape)
            return obj
        return dec
        
variable = _decorator()        


def eval_code(co, g, l):
    if not isinstance(co, types.CodeType): return co
    a = eval(co, g, l)
    if callable(a): return a()
    return a

var_g = {'sin':  np.sin,
         'cos':  np.cos,
         'tan':  np.tan,
         'arctan':  np.arctan,                                      
         'arctan2':  np.arctan2,
         'exp': np.exp,
         'log10':  np.log10,
         'log':  np.log,
         'log2':  np.log2,
         'sqrt':  np.sqrt,
         'abs':  np.abs,                   
         'conj': np.conj,
         'real': np.real,
         'imag': np.imag,
         'dot': np.dot,
         'vdot': np.vdot,
         'array': np.array,
         'cross': np.cross, 
         'pi': np.pi,}

class Variables(dict):
    def __repr__(self):
        txt = []
        for k in self.keys():
            txt.append(k + ':' + str(self[k]))
        return "\n".join(txt)

class Variable(object):
    '''
    define everything which we define algebra
    '''
    def __init__(self, complex=False):
        self.complex = complex

    def __add__(self, other):
        if isinstance(other, Variable):
            return self() + other()
        else:
            return self() + other
        
    def __sub__(self, other):
        if isinstance(other, Variable):
            return self() - other()
        else:
            return self() - other
    def __mul__(self, other):
        if isinstance(other, Variable):
            return self() * other()
        else:
            return self() * other
    def __div__(self, other):
        if isinstance(other, Variable):
            return self() / other()
        else:
            return self() / other

    def __radd__(self, other):
        if isinstance(other, Variable):
            return self() + other()
        else:
            return self() + other

    def __rsub__(self, other):
        if isinstance(other, Variable):
            return other() - self()
        else:
            return other - self()
        
    def __rmul__(self, other):
        if isinstance(other, Variable):
            return self() * other()
        else:
            return self() * other
        
    def __rdiv__(self, other):
        if isinstance(other, Variable):
            return other()/self()
        else:
            return other/self()

    def __divmod__(self, other):
        if isinstance(other, Variable):
            return self().__divmod__(other())
        else:
            return self().__divmod__(other)

    def __floordiv__(self, other):
        if isinstance(other, Variable):
            return self().__floordiv__(other())
        else:
            return self().__floordiv__(other)
        
    def __mod__(self, other):
        if isinstance(other, Variable):
            return self().__mod__(other())
        else:
            return self().__mod__(other)
        
    def __pow__(self, other):
        if isinstance(other, Variable):
            return self().__pow__(other())
        else:
            return self().__pow__(other)
        
    def __neg__(self):
        return self().__neg__()
        
    def __pos__(self):
        return self().__pos__()
    
    def __abs__(self):
        return self().__abs__()

    def __getitem__(self, idx):
        #print idx
        #print self().shape
        return self()[idx]

    def get_emesh_idx(self, idx = None, g=None):
        if idx is None: idx = []
        return idx

    def make_callable(self):
        raise NotImplementedError("Subclass need to implement")
    
    def make_nodal(self):
        raise NotImplementedError("Subclass need to implement")
    def ncface_values(self, ifaces = None, irs = None,
                      gtypes = None, **kwargs):
        raise NotImplementedError("Subclass need to implement")
    
class TestVariable(Variable):
    def __init__(self, comp = -1, complex=False):
        super(TestVariable, self).__init__(complex = complex)
        
    def set_point(self,T, ip, g, l, t = None):
        self.x = T.Transform(ip)        
               
    def __call__(self):
        return 2.
    
    def nodal_values(self, locs = None,  **kwargs):
                    # iele = None, elattr = None, el2v = None,
                    #  wverts = None, locs = None, g = None
        return locs[:, 0]*0 + 2.0
    
    def ncface_values(self, locs = None,  **kwargs):
        return locs[:, 0]*0 + 2.0        
    
class Constant(Variable):
    def __init__(self, value, comp = -1):
        super(Constant, self).__init__(complex = np.iscomplexobj(value))
        self.value = value
        
    def __repr__(self):
        return "Constant("+str(self.value)+")"
        
    def set_point(self,T, ip, g, l, t = None):
        self.x = T.Transform(ip)        
               
    def __call__(self):
        return self.value
    
    def nodal_values(self, iele = None, el2v = None, locs = None,
                     wverts = None, elvertloc = None, **kwargs):

        size = len(wverts)        
        shape = [size] + list(np.array(self.value).shape)

        dtype = np.complex if self.complex else np.float
        ret = np.zeros(shape, dtype = dtype)
        wverts = np.zeros(size)
        
        for kk, m, loc in zip(iele, el2v, elvertloc):
            if kk < 0: continue
            for pair, xyz in zip(m, loc):
                idx = pair[1]
                ret[idx] = self.value

        return ret
    
    def ncface_values(self, locs = None,  **kwargs):
        size = len(locs)        
        shape = [size]+list(np.array(self.value).shape)
        return np.tile(self.value, shape)
        
class CoordVariable(Variable):
    def __init__(self, comp = -1, complex=False):
        super(CoordVariable, self).__init__(complex = complex)
        self.comp = comp
        
    def __repr__(self):
        return "Coordinates"
        
    def set_point(self,T, ip, g, l, t = None):
        self.x = T.Transform(ip)        
               
    def __call__(self):
        if self.comp == -1:
            return self.x
        else:
            return self.x[self.comp-1]
    
    def nodal_values(self, locs = None,  **kwargs):
                    # iele = None, elattr = None, el2v = None,
                    #  wverts = None, locs = None, g = None
        if self.comp == -1:
            return locs
        else:
            return locs[:, self.comp-1]

    def ncface_values(self, locs = None,  **kwargs):
        if self.comp == -1:
            return locs
        else:
            return locs[:, self.comp-1]
    
class ExpressionVariable(Variable):
    def __init__(self, expr, ind_vars, complex=False):
        super(ExpressionVariable, self).__init__(complex = complex)


        variables = []
        st = parser.expr(expr)
        code= st.compile('<string>')
        names = code.co_names
        self.co = code
        self.names = names
        self.expr = expr
        self.ind_vars = ind_vars
        self.variables = WVD()
        #print 'Check Expression', expr.__repr__(), names
    def __repr__(self):
        return "Expression("+self.expr + ")"
    
    def set_point(self,T, ip, g, l, t = None):
        self.x = T.Transform(ip)        
        for n in self.names:
            if (n in g and isinstance(g[n], Variable)):
               g[n].set_point(T, ip, g, l, t=t)
               self.variables[n] = g[n]
               
    def __call__(self):
        l = {}
        for k, name in enumerate(self.ind_vars):
           l[name] = self.x[k]
        keys = self.variables.keys()
        for k in keys:
           l[k] = self.variables[k]()
        return (eval_code(self.co, var_g, l))

    def get_emesh_idx(self, idx = None, g = None):
        if idx is None: idx = []
        for n in self.names:
            if (n in g and isinstance(g[n], Variable)):
                idx = get_emesh_idx(self, idx=idx, g = g)
        return idx
    
    def nodal_values(self, iele = None, el2v = None, locs = None,
                     wverts = None, elvertloc = None, g = None,
                     **kwargs):

        size = len(wverts)        
        dtype = np.complex if self.complex else np.float
        ret = np.zeros(size, dtype = dtype)
        for kk, m, loc in zip(iele, el2v, elvertloc):
            if kk < 0: continue
            for pair, xyz in zip(m, loc):
                idx = pair[1]
                ret[idx] = 1

        l = {}
        ll_name = []
        ll_value = []
        var_g2 = var_g.copy()
        for n in self.names:
            if (n in g and isinstance(g[n], Variable)):
                l[n] = g[n].nodal_values(iele = iele, el2v = el2v, locs = locs,
                                         wverts = wverts, elvertloc = elvertloc,
                                         g = g, **kwargs)
                ll_name.append(n)
                ll_value.append(l[n])
            elif (n in g):
                var_g2[n] = g[n]
        if len(ll_name) > 0:
            value = np.array([eval(self.co, var_g2, dict(zip(ll_name, v)))
                        for v in zip(*ll_value)])
        else:
            for k, name in enumerate(self.ind_vars):
                l[name] = locs[...,k]
            value = np.array(eval_code(self.co, var_g2, l), copy=False)
            if value.ndim > 1:
                value = np.stack([value]*size)
        #value = np.array(eval_code(self.co, var_g, l), copy=False)

        from petram.helper.right_broadcast import multi

        ret = multi(ret, value)
        return ret
    
    def ncface_values(self, ifaces = None, irs = None, gtypes = None,
                      g=None, attr1 = None, attr2 = None, locs = None,
                      **kwargs):
        
        size = len(locs)
        dtype = np.complex if self.complex else np.float
        ret = np.zeros(size, dtype = dtype)

        l = {}
        ll_name = []
        ll_value = []
        var_g2 = var_g.copy()
        for n in self.names:
            if (n in g and isinstance(g[n], Variable)):
                l[n] = g[n].ncface_values(ifaces = ifaces, irs = irs,
                                          gtypes = gtypes, locs = ptx,
                                          attr1 = attr1, attr2 = attr2, 
                                          g = g, **kwargs)
                ll_name.append(n)
                ll_value.append(l[n])
            elif (n in g):
                var_g2[n] = g[n]
                
        if len(ll_name) > 0:
            value = np.array([eval(self.co, var_g2, dict(zip(ll_name, v)))
                        for v in zip(*ll_value)])
        else:
            for k, name in enumerate(self.ind_vars):
                l[name] = locs[...,k]
            value = np.array(eval_code(self.co, var_g2, l), copy=False)
            if value.ndim > 1:
                value = np.stack([value]*size)

        return value

class DomainVariable(Variable):
    def __init__(self, expr = '', ind_vars = None, domains = None,
                 complex = False, gdomain = None):
        super(DomainVariable, self).__init__(complex = complex)
        self.domains = {}
        self.gdomains = {}        
        if expr == '': return
        domains = sorted(domains)
        self.gdomains[tuple(domains)] = gdomain
        self.domains[tuple(domains)] = ExpressionVariable(expr, ind_vars,
                                                  complex = complex)
    def __repr__(self):
        return "DomainVariable"
        
    def add_expression(self, expr, ind_vars, domains, gdomain, complex = False):
        domains = sorted(domains)
        #print 'adding expression expr',expr, domains
        self.domains[tuple(domains)] = ExpressionVariable(expr, ind_vars,
                                                  complex = complex)
        self.gdomains[tuple(domains)] = gdomain        
        if complex: self.complex = True

    def add_const(self, value, domains, gdomain):
        domains = sorted(domains)        
        self.domains[tuple(domains)] = Constant(value)
        self.gdomains[tuple(domains)] = gdomain
        if np.iscomplexobj(value):self.complex = True
        
    def set_point(self,T, ip, g, l, t = None):
        attr = T.Attribute
        self.domain_target = None
        for domains in self.domains.keys():
           if attr in domains:
               self.domains[domains].set_point(T, ip, g, l, t=t)
           self.domain_target = domains
               
    def __call__(self):
        if self.domain_target is None: return 0.0
        return self.domains[self.domain_target]()
    
    def get_emesh_idx(self, idx = None, g = None):
        if idx is None: idx = []        
        for domains in self.domains.keys():
            expr = self.domains[domains]
            gdomain = g if self.gdomains[domains] is None else self.gdomains[domains]
            idx = expr.get_emesh_idx(self, idx=idx, g=gdomain)
        return idx
    
    def nodal_values(self, iele = None, elattr = None, g = None,
                     **kwargs):
                     #iele = None, elattr = None, el2v = None,
                     #wverts = None, locs = None, g = None):

        from petram.helper.right_broadcast import add
        
        ret = None
        w = None
        for domains in self.domains.keys():
            iele0 = np.zeros(iele.shape)-1
            for domain in domains:
                idx = np.where(np.array(elattr) == domain)[0]
                iele0[idx] = iele[idx]

            expr = self.domains[domains]
            gdomain = g if self.gdomains[domains] is None else self.gdomains[domains]
            v = expr.nodal_values(iele = iele0, elattr = elattr,
                                  g = gdomain, **kwargs)
                                  #iele = iele, elattr = elattr,
                                  #el2v = el2v, wvert = wvert,
                                  #locs = locs, g = g
            if w is None:
                a  = np.sum(np.abs(v.reshape(len(v), -1)), -1)
                w = (a != 0).astype(float)
            else:
                a  = np.sum(np.abs(v.reshape(len(v), -1)), -1)                
                w = w + (a != 0).astype(float)                
            ret = v if ret is None else add(ret, v)

        idx = np.where(w != 0)[0]
        #ret2 = ret.copy()
        from petram.helper.right_broadcast import div                
        ret[idx, ...] = div(ret[idx, ...], w[idx])
        return ret
    
    def ncface_values(self, ifaces = None, irs = None, gtypes = None,
                      g=None, attr1 = None, attr2 = None, locs = None,
                      **kwargs):
        
        from petram.helper.right_broadcast import add, multi
        
        ret = None
        
        w = ifaces*0 # w : 0 , 0.5, 1
        for domains in self.domains.keys():
            for domain in domains:
                idx = np.where(np.array(attr1) == domain)[0]
                w[idx] = w[idx] + 1.0
                idx = np.where(np.array(attr2) == domain)[0]
                w[idx] = w[idx] + 2.0
        w[w>0] = 1./w[w>0]

        npts = [irs[gtype].GetNPoints() for gtype in gtypes]
        weight = np.repeat(w, npts)
        
        for domains in self.domains.keys():
            w = np.zeros(ifaces.shape)
            for domain in domains:
                idx = np.where(np.array(attr1) == domain)[0]
                w[idx] = 1.0
            w2 = weight * np.repeat(w, npts)

            
            expr = self.domains[domains]
            gdomain = g if self.gdomains[domains] is None else self.gdomains[domains]
            v = expr.ncface_values(ifaces = ifaces, irs = irs,
                                   gtypes = gtypes, locs=locs, attr1 = attr1,
                                   attr2 = attr2, g = gdomain, 
                                   weight = w2, **kwargs)
            v = multi(v, w2)
            ret = v if ret is None else add(ret, v)

        return ret
        
        
class PyFunctionVariable(Variable):
    def __init__(self, func, complex=False, shape = tuple()):
        super(PyFunctionVariable, self).__init__(complex = complex)
        self.func = func
        self.t = None
        self.x = (0,0,0)
        self.shape = shape
        
    def __repr__(self):
        return "PyFunction"
        
    def set_point(self,T, ip, g, l, t = None):
        self.x = T.Transform(ip)
        self.t = t
        
    def __call__(self):
        if self.t is not None:
           args = tuple(np.hstack((self.x, t)))
        else:
           args = tuple(self.x)
        return np.array(self.func(*args), copy=False)
       
    def nodal_values(self, iele = None, el2v = None, locs = None,
                     wverts = None, elvertloc = None, **kwargs):
                     # elattr = None, el2v = None,
                     # wverts = None, locs = None, g = None
        if locs is None: return

        size = len(wverts)        
        shape = [size] + list(self.shape)

        dtype = np.complex if self.complex else np.float
        ret = np.zeros(shape, dtype = dtype)
        wverts = np.zeros(size)
        for kk, m, loc in zip(iele, el2v, elvertloc):
            if kk < 0: continue
            for pair, xyz in zip(m, loc):
                idx = pair[1]
                ret[idx] = ret[idx] + self.func(*xyz)
                wverts[idx] = wverts[idx] + 1
        ret = np.stack([x for x in ret if x is not None])


        idx = np.where(wverts == 0)[0]        
        wverts[idx] = 1.0
        
        from petram.helper.right_broadcast import div        
        ret = div(ret, wverts)
        #print("PyFunctionVariable", ret)
        return ret
    
    def ncface_values(self, ifaces = None, irs = None, gtypes = None,
                      g=None, attr1 = None, attr2 = None, locs = None,
                      **kwargs):
        if locs is None: return
        dtype = np.complex if self.complex else np.float
        ret = [self.func(*xyz) for xyz in locs]
        ret = np.stack(ret).astype(dtype, copy=False)
        return ret


class GridFunctionVariable(Variable):
    def __init__(self, gf_real, gf_imag = None, comp = 1,
                 deriv = None, complex = False):
        
        super(GridFunctionVariable, self).__init__(complex = complex)
        self.dim = gf_real.VectorDim()
        self.comp = comp
        self.isDerived = False
        self.deriv = deriv if deriv is not None else self._def_deriv
        self.deriv_args = (gf_real, gf_imag)
        
    def _def_deriv(self, *args):
        return args[0], args[1], None
    
    def set_point(self,T, ip, g, l, t = None):
        self.T = T
        self.ip = ip
        self.t = t
        
    def get_emesh_idx(self, idx = None, g=None):
        if idx is None: idx = []
        gf_real, gf_imag = self.deriv_args
        if gf_real is not None:
            idx.append(gf_real._emesh_idx)
        elif gf_imag is not None:
            idx.append(gf_imag._emesh_idx)
        else:
            pass
            
        return idx
        
class GFScalarVariable(GridFunctionVariable):
    def __repr__(self):
        return "GridFunctionVariable (Scalar)"

    def set_funcs(self):
        # I should come back here to check if this works
        # with vector gf and/or boundary element. probably not...
        gf_real, gf_imag, extra = self.deriv(*self.deriv_args)
        self.gfr = gf_real
        self.gfi = gf_imag
        self.func_r = mfem.GridFunctionCoefficient(gf_real,
                                                   comp = self.comp)
        if gf_imag is not None:
            self.func_i = mfem.GridFunctionCoefficient(gf_imag,
                                                       comp = self.comp)
        else:
            self.func_i = None
        self.isDerived = True
        self.extra = extra            
    def __call__(self):
        if not self.isDerived: self.set_funcs()
        if self.func_i is None:
            return self.func_r.Eval(self.T, self.ip)
        else:
            return (self.func_r.Eval(self.T, self.ip) +
                    1j*self.func_i.Eval(self.T, self.ip))

    def nodal_values(self, iele = None, el2v = None, wverts = None,
                     **kwargs):
        if iele is None: return        
        if not self.isDerived: self.set_funcs()
        
        size = len(wverts)
        if self.gfi is None:
            ret = np.zeros(size, dtype = np.float)
        else:
            ret = np.zeros(size, dtype = np.complex)
        for kk, m in zip(iele, el2v):
            if kk < 0: continue            
            values = mfem.doubleArray()
            self.gfr.GetNodalValues(kk, values, self.comp)
            for k, idx in m:
                ret[idx] = ret[idx] + values[k]
            if self.gfi is not None:
                arr = mfem.doubleArray()                
                self.gfi.GetNodalValues(kk, arr, self.comp)
                for k, idx in m:
                    ret[idx] = ret[idx] + arr[k]*1j
        ret = ret / wverts
        return ret

    def ncface_values(self, ifaces = None, irs = None,
                      gtypes = None, **kwargs):
        if not self.isDerived: self.set_funcs()
        
        name = self.gfr.FESpace().FEColl().Name()
        ndim = self.gfr.FESpace().GetMesh().Dimension()
        
        isVector = False
        if (name.startswith('RT') or 
            name.startswith('ND')):
            d = mfem.DenseMatrix()
            p  = mfem.DenseMatrix()
            isVector = True
        else:
            d = mfem.Vector()
            p  = mfem.DenseMatrix()
        data = []

        def get_method(gf, ndim, isVector):
            if gf is None: return None
            if ndim == 3:
                if isVector:
                    return gf.GetFaceVectorValues
                else:
                    return gf.GetFaceValues
            elif ndim == 2:
                if isVector:
                    return gf.GetVectorValues
                else:
                    def func(i, side, ir, vals, tr, in_gf = gf):
                        in_gf.GetValues(i, ir, vals, tr)
                        return
                    return func
            else:
                assert False, "ndim = 1 has no face"
            return None
        
        getvalr = get_method(self.gfr, ndim, isVector)
        getvali = get_method(self.gfi, ndim, isVector)        
            
        for i, gtype,  in zip(ifaces, gtypes):
            ir = irs[gtype]
            getvalr(i, 2, ir,  d, p) # side = 2 (automatic?)
            v = d.GetDataArray().copy()
            if isVector:  v = v[self.comp-1,:]

            if getvali is not None:
                getvali(i, 2, ir,  d, p) # side = 2 (automatic?)
                vi = d.GetDataArray().copy()
                if isVector:  vi = vi[self.comp-1,:]
                v = v + 1j*vi
            data.append(v)
        data = np.hstack(data)                
        return data
    
            
class GFVectorVariable(GridFunctionVariable):
    def __repr__(self):
        return "GridFunctionVariable (Vector)"
    
    def set_funcs(self):
        gf_real, gf_imag, extra = self.deriv(*self.deriv_args)
        self.gfr = gf_real
        self.gfi = gf_imag
        self.dim = gf_real.VectorDim()        
        self.func_r = [mfem.GridFunctionCoefficient(gf_real, comp = k+1)
                          for k in range(self.dim)]
           
        if gf_imag is not None:
            self.func_i = [mfem.GridFunctionCoefficient(gf_imag, comp = k+1)
                           for k in range(self.dim)]

        self.isDerived = True
        self.extra = extra                    
    def __call__(self):
        if self.func_i is None:
            return np.array([func_r.Eval(self.T, self.ip) for
                                 func_r in self.func_r])
        else:
            return np.array([(func_r.Eval(self.T, self.ip) +
                                  1j*func_i.Eval(self.T, self.ip))
                                 for func_r, func_i
                                 in zip(self.func_r, self.func_i)])
            
    def nodal_values(self, iele = None, el2v = None, wverts = None,
                     **kwargs):
                    # iele = None, elattr = None, el2v = None,
                    # wverts = None, locs = None, g = None
       
        if iele is None: return        
        if not self.isDerived: self.set_funcs()

        size = len(wverts)

        ans = []
        for comp in range(self.dim):
            if self.gfi is None:
                ret = np.zeros(size, dtype = np.float)
            else:
                ret = np.zeros(size, dtype = np.complex)
            for kk, m in zip(iele, el2v):
                if kk < 0: continue  
                values = mfem.doubleArray()
                self.gfr.GetNodalValues(kk, values, comp+1)
                for k, idx in m:
                    ret[idx] = ret[idx] + values[k]
                if self.gfi is not None:
                    arr = mfem.doubleArray()                
                    self.gfi.GetNodalValues(kk, arr, comp+1)
                    for k, idx in m:
                        ret[idx] = ret[idx] + arr[k]*1j
            ans.append(ret / wverts)
        ret =np.transpose(np.vstack(ans))
        return ret
    
    def ncface_values(self, ifaces = None, irs = None,
                      gtypes = None, **kwargs):

        if not self.isDerived: self.set_funcs()
        ndim = self.gfr.FESpace().GetMesh().Dimension()
        
        d = mfem.DenseMatrix()
        p  = mfem.DenseMatrix()
        data = []
        
        def get_method(gf, ndim):
            if gf is None: return None
            if ndim == 3:
                return gf.GetFaceVectorValues
            elif ndim == 2:
                return gf.GetVectorValues
            else:
                assert False, "ndim = 1 has no face"
        getvalr = get_method(self.gfr, ndim)
        getvali = get_method(self.gfi, ndim)
        
        for i, gtype,  in zip(ifaces, gtypes):
            ir = irs[gtype]
            getvalr(i, 2, ir,  d, p) # side = 2 (automatic?)
            v = d.GetDataArray().copy()

            if getvali is not None:
                getvali(i, 2, ir,  d, p)
                vi = d.GetDataArray().copy()
                v = v + 1j*vi
            data.append(v)
        return np.hstack(data)                
    
'''

Surf Variable:
 Regular Variable + Surface Geometry (n, nx, ny, nz)

'''    
class SurfVariable(Variable):
    def __init__(self, sdim, complex = False):
        self.sdim = sdim
        super(SurfVariable, self).__init__(complex = complex)
        
class SurfNormal(SurfVariable):
    def __init__(self, sdim, comp = -1, complex = False):
        self.comp = comp
        SurfVariable.__init__(self, sdim, complex = complex)
        
    def __repr__(self):
        return "SurfaceNormal (nx, ny, nz)"
        
    def set_point(self, T, ip, g, l, t = None):
        nor = mfem.Vector(self.sdim)
        mfem.CalcOrtho(T.Jacobian(), nor)
        self.nor =nor.GetDataArray().copy()
        
    def __call__(self):
        if self.comp == -1:
            return self.nor
        else:
            return self.nor[self.comp-1]
        
    def nodal_values(self, ibele = None, mesh = None, iverts_f = None,
                     **kwargs):
                    # iele = None, elattr = None, el2v = None,
                    # wverts = None, locs = None, g = None
      
        g = mfem.Geometry()
        size = len(iverts_f)
        #wverts = np.zeros(size)
        ret = np.zeros((size, self.sdim))
        if ibele is None: return               
                       
        ibe  = ibele[0]
        el = mesh.GetBdrElement(ibe)
        rule = g.GetVertices(el.GetGeometryType())
        nv = rule.GetNPoints()
        

        for ibe in ibele:
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
        ret = ret / np.sqrt(np.sum(ret**2, 1)).reshape(-1,1)
        
        if self.comp == -1: return ret
        return ret[:, self.comp-1]

    def ncface_values(self, ifaces = None, irs = None, gtypes = None,
                      locs = None, mesh = None, **kwargs):
    
        size = len(locs)
        ret = np.zeros((size, self.sdim))
        if ifaces is None: return
        
        nor = mfem.Vector(self.sdim)
         
        if mesh.Dimension() == 3:
            m = mesh.GetFaceTransformation
        elif mesh.Dimension() == 2:
            m = mesh.GetElementTransformation            
        idx = 0
        for i, gtype,  in zip(ifaces, gtypes):
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
        if self.comp == -1: return ret
        return ret[:, self.comp-1]

class SurfExpressionVariable(ExpressionVariable, SurfVariable):
    '''
    expression valid on surface
    '''    
    def __init__(self, expr, ind_vars, sdim, complex=False):
        ExpressionVariable.__init__(self, expr, ind_vars, complex=complex)
        SurfVariable.__init__(self, sdim, complex = complex)
        
    def __repr__(self):
        return "SurfaceExpression("+self.expr+")"
        
    def set_point(self, T, ip, g, l, t = None):
        self.x = T.Transform(ip)
        self.t = t
        T.SetIntPoint(ip)
        nor = mfem.Vector(self.sdim)
        mfem.CalcOrtho(T.Jacobian(), nor)
        self.nor =nor.GetDataArray().copy()
        
    def __call__(self):
        l = {}
        for k, name in enumerate(self.ind_vars):
           l[name] = self.x[k]
        l['n'] = self.nor
        for k, name in enumerate(self.ind_vars):
           l['n'+name] = self.nor[k]
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
           l[name] = locs[...,k]
        for k, name in enumerate(self.ind_vars):
           l['n'+name] = nor[...,k]
        return (eval_code(self.co, var_g, l))
    
    def ncface_values(self, **kwargs):
        assert False, "ncface in SurfaceExpressionVariable must be added"
'''
 Bdr Variable = Surface Variable defined on particular boundary
'''    
class BdrVariable(ExpressionVariable, SurfVariable):
    pass

def append_suffix_to_expression(expr, vars, suffix):
    for v in vars:
        expr = expr.replace(v, v+suffix)
    return expr

def add_scalar(solvar, name, suffix, ind_vars, solr, soli=None, deriv = None):
    solvar[name + suffix] = GFScalarVariable(solr, soli, comp=1,
                                                deriv = deriv)

def add_components(solvar, name, suffix, ind_vars, solr,
                   soli=None, deriv = None):
    solvar[name + suffix] = GFVectorVariable(solr, soli, deriv = deriv)
    for k, p in enumerate(ind_vars):
       solvar[name + suffix + p] = GFScalarVariable(solr, soli, comp=k+1,
                                                    deriv = deriv)

def add_expression(solvar, name, suffix, ind_vars, expr, vars,
                   domains = None, bdrs = None, complex = None,
                   gdomain = None, gbdr = None):
    expr = append_suffix_to_expression(expr, vars, suffix)
    if domains is not None:
        if (name + suffix) in solvar:
            solvar[name + suffix].add_expression(expr, ind_vars, domains,
                                                 gdomain,
                                                 complex = complex)
        else:
            solvar[name + suffix] = DomainVariable(expr, ind_vars,
                                                   domains = domains,
                                                   complex = complex,
                                                   gdomain = gdomain)
    elif bdrs is not None:
        pass
    else:
        solvar[name + suffix] = ExpressionVariable(expr, ind_vars,
                                                   complex = complex)
        
def add_constant(solvar, name, suffix, value, domains = None,
                 gdomain = None, bdrs = None, gbdr = None):
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
        

def add_surf_normals(solvar, ind_vars):
    sdim = len(ind_vars)                         
    solvar['n'] = SurfNormal(sdim, comp = -1)
    for k, p in enumerate(ind_vars):
       solvar['n'+p] = SurfNormal(sdim, comp = k+1)

def add_coordinates(solvar, ind_vars):
    for k, p in enumerate(ind_vars):    
       solvar[p] = CoordVariable(comp = k+1)

       


       
  

    


            
   
