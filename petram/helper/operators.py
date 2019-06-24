from __future__ import print_function

import weakref
import numpy as np

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
   from mpi4py import MPI
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   comm = MPI.COMM_WORLD
   from petram.helper.mpi_recipes import  gather_vector, allgather_vector
   from mfem.common.mpi_debug import nicePrint   
else:
   import mfem.ser as mfem

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints("Operators")
rprint = debug.regular_print('Operators')

class Operator(object):
    def __repr__(self):
        return self.__class__.__name__ + "("+",".join(self.sel)+")"

class Operator(object):
    def __init__(self, **kwargs):
        self._sel  = kwargs.pop("sel", "all")
        self._ssel = kwargs.pop("src", "all")        
        self._sel_mode = kwargs.pop("sel_mode", "domain")
        self._fes1 = None
        self._fes2 = None
        self._engine = None
        self._transpose = False
        self._trial_ess_tdof = None
        self._test_ess_tdof = None
        
    def __call__(self, *args, **kwargs):
        return self.assemble(*args, **kwargs )
    
    def assemble(self, fes):
        raise NotImplementedError("Subclass needs to implement this")

    ''' 
    def get_restriction_array(self, fes):
        mesh = fes.GetMesh()
        intArray = mfem.intArray

        if self._sel_mode == 'domain':
            size = np.max(mesh.GetAttributeArray())
        else:
            size = np.max(mesh.GetBdrAttributeArray())

        if size == 0: return None

        if self._sel[0] == "all":
            arr = [1]*size
        else:
            if len(self._sel) > 0:
                size = np.max((size, np.max(self._sel)))
            arr = [0]*size           
            for k in self._sel: arr[k-1] = 1
        return intArray(arr)
    
    def restrict_coeff(self, coeff, fes, vec = False, matrix=False):
        if self._sel == 'all': 
           return coeff

        arr = self.get_restriction_array(fes)
        
        if arr is None:
           # this could happen when local mesh does not have Bdr/Domain attribute
           return coeff
        
        if vec:
            return mfem.VectorRestrictedCoefficient(coeff, arr)
        elif matrix:
            return mfem.MatrixRestrictedCoefficient(coeff, arr)           
        else:
            return mfem.RestrictedCoefficient(coeff, arr)
    '''

    def process_kwargs(self, engine, kwargs):
        '''
        kwargs in expression can overwrite this.
        '''
        fes1 = kwargs.pop('test', None)
        fes2 = kwargs.pop('trial', None)
        self._transpose = kwargs.pop('transpose', self._transpose)
        self._sel_mode = kwargs.pop('sel_mode', self._sel_mode)
        self._sel   = kwargs.pop('sel', self._sel)
        self._ssel  = kwargs.pop('src', self._ssel)        
        if fes1 is not None:
           if isinstance(fes1, 'str'):
               fes1 = engine.fespaces[fes]
           self._fes1 = weakref.ref(fes1)        
        if fes2 is not None:
           if isinstance(fes2, 'str'):
               fes2 = engine.fespaces[fes]
           self._fes2 = weakref.ref(fes2)        

    @property
    def fes1(self):
        return self._fes1()
    @property
    def fes2(self):
        return self._fes2()
    @property
    def sel_mode(self):
        return self._sel_mode

class LoopIntegral(Operator):
    def assemble(self, *args, **kwargs):
        '''
        looplintegral(18)   # integrate around the bdr 18

        # this mode is only avaiable on serial version.
        looplintegral([1,2,3], [4, 5]) # loop defined by two groups of boundaries 
        '''
        coeff = kwargs.pop("coeff", "1")
        coeff_type = kwargs.pop("coeff_type", "S")

        engine = self._engine()        
        self.process_kwargs(engine, kwargs)

        face = args
        
        if not self.fes1.FEColl().Name().startswith('ND'):
            assert False, "line integration is only implemented for ND"

        fes1 = self.fes1
        info1 = engine.get_fes_info(self.fes1)
        emesh_idx = info1['emesh_idx']
        mesh = engine.emeshes[emesh_idx]

        if use_parallel:
            from petram.mesh.find_loop import find_loop_par
            idx, signs = find_loop_par(mesh, *face)
            fesize1 = fes1.GetTrueVSize()
            P = fes1.Dof_TrueDof_Matrix()
            from mfem.common.parcsr_extra import ToScipyCoo
            P = ToScipyCoo(P).tocsr()
            VDoFtoGTDoF = P.indices
            rstart = fes1.GetMyTDofOffset()            
        else:
            from petram.mesh.find_loop import find_loop_ser
            idx, signs = find_loop_ser(mesh, *face)            
            fesize1 = fes1.GetNDofs()
            
        map = np.zeros((fesize1,1), dtype=float)       

        w = []
        for sign, ie in zip(signs, idx):
           dofs = fes1.GetEdgeDofs(ie)
           # don't put this Tr outside the loop....
           Tr = mesh.GetEdgeTransformation(ie)
           weight = Tr.Weight()
           w.append(Tr.Weight())
           for dof in dofs:
              if use_parallel:
                 dof = VDoFtoGTDoF[dof] - rstart
              map[dof] = sign
              
        #if len(w) > 0: print("weight", w)
        
        from mfem.common.chypre import PyVec2PyMat, CHypreVec
        if use_parallel:
            v1 = CHypreVec(map.flatten(), None)
        else:
            v1 = map.reshape((-1, 1))

        v1 = PyVec2PyMat(v1)
        if not self._transpose:        
            v1 = v1.transpose()
        return v1

class Integral(Operator):
    def assemble(self, *args, **kwargs):
        '''
        integral()
        integral('boundary', [1,3])  # selection type and selection
        integral('boundary', [1,3], integrator = 'auto' or other MFEM linearform integrator
        integral('boundary', [1,3], weight = '1')

        '''

        coeff = kwargs.pop("coeff", "1")
        coeff_type = kwargs.pop("coeff_type", "S")

        engine = self._engine()        
        self.process_kwargs(engine, kwargs)
        
        if len(args)>0: self._sel_mode = args[0]
        if len(args)>1: self._sel = args[1]

        if (self.fes1.FEColl().Name().startswith('ND') or
            self.fes1.FEColl().Name().startswith('RT')):
            cdim = self.fes1.GetMesh().SpaceDimension()
        else:
            cdim = self.fes1.GetVDim()

        info1 = engine.get_fes_info(self.fes1)
        emesh_idx = info1['emesh_idx']

        isDomain = (self._sel_mode == 'domain')

        from petram.helper.phys_module_util import default_lf_integrator        
        integrator = kwargs.pop("integrator", "Auto")
        if integrator == 'Auto':
            integrator = default_lf_integrator(info1, isDomain)
        
        global_ns = globals()
        local_ns = {}
        real = True

        try:
            is_complex = np.iscomplex(complex(eval(coeff), global_ns, local_ns))
        except:
            is_complex = kwargs.pop('is_complex', False)

        if isDomain:
            adder = 'AddDomainIntegrator'
        else:
            adder = 'AddBoundaryIntegrator'
            
        from petram.helper.phys_module_util import restricted_integrator
        from mfem.common.chypre import LF2PyVec, PyVec2PyMat, MfemVec2PyVec

        itg = restricted_integrator(engine, integrator, self._sel,
                                    coeff, coeff_type, cdim, 
                                    emesh_idx,
                                    isDomain,
                                    self._ind_vars, local_ns, global_ns, True)
        
        lf1 = engine.new_lf(self._fes1())                    
        getattr(lf1, adder)(itg)
        lf1.Assemble()
        
        if is_complex:
            itg = restricted_integrator(engine, integrator, self._sel,
                                        coeff, coeff_type, cdim, 
                                        emesh_idx, isDomain,
                                        self._ind_vars, local_ns, global_ns, False)
            
            lf2 = engine.new_lf(self._fes1())
            getattr(lf2, adder)(itg)            
            lf2.Assemble()
            v1 = MfemVec2PyVec(engine.b2B(lf1), engine.b2B(lf2))
        else:
            v1 = MfemVec2PyVec(engine.b2B(lf1), None)                       
            lf2 = None            

        v1 = PyVec2PyMat(v1)
        if not self._transpose:        
            v1 = v1.transpose()
        return v1
     
def make_diagonal_mat(engine, fes1, fes2, value):
    if fes1 == fes2:
        bf = engine.new_bf(fes1)
        bf.Assemble()
        bf.Finalize()
        mat = engine.a2A(bf)
    else:
        bf = engine.new_mixed_bf(fes1, fes2)
        #one = mfem.ConstantCoefficient(0.0001)
        #itg = mfem.MixedScalarMassIntegrator()
        #bf.AddDomainIntegrator(itg)
        bf.Assemble()
        mat = engine.a2Am(bf)

    if use_parallel:
        mat.CopyRowStarts()
        mat.CopyColStarts()
            
    from mfem.common.chypre import MfemMat2PyMat
    m1 = MfemMat2PyMat(mat, None)
        
    if not use_parallel:
        from petram.helper.block_matrix import convert_to_ScipyCoo
        m1 = convert_to_ScipyCoo(m1)
    shape = m1.shape
    assert shape[0]==shape[1], "Identity Operator must be square"

    idx = range(shape[0])
    m1.setDiag(idx, value=value)
    
    return m1

class Identity(Operator):    
    def assemble(self, *args, **kwargs):
        engine = self._engine()
        self.process_kwargs(engine, kwargs)
        
        fes1 = self._fes1()
        fes2 = fes1 if self._fes2 is None else self._fes2()
        return make_diagonal_mat(engine, fes1, fes2, 1.0)
        '''
        if fes1 == fes2:
            bf = engine.new_bf(fes1)
            #one = mfem.ConstantCoefficient(0.0001)
            #itg = mfem.MassIntegrator()
            #bf.AddDomainIntegrator(itg)
            bf.Assemble()
            bf.Finalize()
            mat = engine.a2A(bf)
        else:
            bf = engine.new_mixed_bf(fes1, fes2)
            #one = mfem.ConstantCoefficient(0.0001)
            #itg = mfem.MixedScalarMassIntegrator()
            #bf.AddDomainIntegrator(itg)
            bf.Assemble()
            mat = engine.a2Am(bf)

        if use_parallel:
            mat.CopyRowStarts()
            mat.CopyColStarts()
            
        from mfem.common.chypre import MfemMat2PyMat
        m1 = MfemMat2PyMat(mat, None)
        
        if not use_parallel:
            from petram.helper.block_matrix import convert_to_ScipyCoo
            m1 = convert_to_ScipyCoo(m1)
        shape = m1.shape
        assert shape[0]==shape[1], "Identity Operator must be square"

        idx = range(shape[0])
        m1.setDiag(idx)

        return m1
        '''
     
class Zero(Operator):
    def assemble(self, *args, **kwargs):
        engine = self._engine()
        self.process_kwargs(engine, kwargs)
        
        fes1 = self._fes1()
        fes2 = fes1 if self._fes2 is None else self._fes2()
        return make_diagonal_mat(engine, fes1, fes2, 0.0)

        '''
        if fes1 == fes2:
            bf = engine.new_bf(fes1)
            bf.Assemble()
            bf.Finalize()
            mat = engine.a2A(bf)
        else:
            bf = engine.new_mixed_bf(fes1, fes2)
            bf.Assemble()
            mat = engine.a2Am(bf)

        if use_parallel:
            mat.CopyRowStarts()
            mat.CopyColStarts()
            
        from mfem.common.chypre import MfemMat2PyMat
        m1 = MfemMat2PyMat(mat, None)
        
        if not use_parallel:
            from petram.helper.block_matrix import convert_to_ScipyCoo
            m1 = convert_to_ScipyCoo(m1)
        shape = m1.shape
        assert shape[0]==shape[1], "Zero Operator must be square"
        idx = range(shape[0])
        m1.setDiag(idx, value=0.0)
        
        return m1
        '''
class Delta(Operator):
    '''
    Delta function

      delta(x)
      delta(x, y)
      delta(x, y, z)

      delta(x, y, z, direction=[1,0,0]))
      delta(array) to set multiple points at once
    '''
    def assemble(self, *args, **kwargs):
        engine = self._engine()
        direction = kwargs.pop("direction", 0)
        
        self.process_kwargs(engine, kwargs)

        x = args[0]
        y = args[1] if len(args)>1 else 0
        z = args[2] if len(args)>2 else 0        
        
        sdim = self.fes1.GetMesh().SpaceDimension()
        if direction == 0:
            if sdim == 3:
               d = mfem.DeltaCoefficient(x, y, z, 1)
            elif sdim == 2:
               d = mfem.DeltaCoefficient(x, y, 1)
            elif sdim == 1:
               d = mfem.DeltaCoefficient(x, 1)
            else:
                assert False, "unsupported dimension"
            intg = mfem.DomainLFIntegrator(d)                
        else:
            dir = mfem.Vector(direction)
            if sdim == 3:
               d = mfem.VectorDeltaCoefficient(dir, x, y, z, 1)
            elif sdim == 2:
               d = mfem.VectorDeltaCoefficient(dir, x, y, 1)
            elif sdim == 1:
               d = mfem.VectorDeltaCoefficient(dir,x, 1)
            else:
                assert False, "unsupported dimension"
                
            if self.fes1.FEColl().Name().startswith('ND'):
                intg = mfem.VectorFEDomainLFIntegrator(d)                
            elif self.fes1.FEColl().Name().startswith('RT'):
                intg = mfem.VectorFEDomainLFIntegrator(d)            
            else:    
                intg = mfem.VectorDomainLFIntegrator(d)                                

        lf1 = engine.new_lf(self.fes1)
        lf1.AddDomainIntegrator(intg)
        lf1.Assemble()
        
        from mfem.common.chypre import LF2PyVec, PyVec2PyMat, MfemVec2PyVec
        v1 = MfemVec2PyVec(engine.b2B(lf1), None)

        v1 = PyVec2PyMat(v1)
        if not self._transpose:        
            v1 = v1.transpose()
        return v1

class DeltaM(Operator):
    '''
    Delta function

      delta(array, direction = None, weight = None, sum = False) to set multiple points at once

      array = [x1, y1, z1, x2, y2, z2,....]
      direction = [dx1, dy1, dz1, dx2, dy2, dz2,....]
      weight = [w1, w2, w3,...]
      sum = if True, all points are corrapsed to one array.

    '''
    def assemble(self, *args, **kwargs):
        engine = self._engine()
        direction = kwargs.pop("direction", None)
        weight = kwargs.pop("weight", 1)
        weight = np.atleast_1d(weight)
        do_sum = kwargs.pop("sum", False)
        
        info1 = engine.get_fes_info(self.fes1)
        sdim = info1['sdim']
        vdim = info1['vdim']

        if (info1['element'].startswith('ND') or
            info1['element'].startswith('RT')):
            vdim = sdim

        if vdim > 1:
            if direction is None:
                direction = [0]*vdim
                direction[0] = 1
            direction = np.atleast_1d(direction).astype(float, copy=False)
            direction = direction.reshape(-1, sdim)
        
        self.process_kwargs(engine, kwargs)

        pts = np.array(args[0], copy = False).reshape(-1, sdim)

        from mfem.common.chypre import LF2PyVec, PyVec2PyMat, MfemVec2PyVec, HStackPyVec
        vecs = []

        for k, pt in enumerate(pts):
            w = weight[0] if len(weight) == 1 else weight[k]
            if vdim == 1:
                if sdim == 3:
                    x, y, z = pt
                    d = mfem.DeltaCoefficient(x, y, z, w)
                elif sdim == 2:
                    x, y = pt               
                    d = mfem.DeltaCoefficient(x, y, w)
                elif sdim == 1:
                    x = pt                              
                    d = mfem.DeltaCoefficient(x, w)
                else:
                     assert False, "unsupported dimension"
                intg = mfem.DomainLFIntegrator(d)                
            else:
                dir = direction[0] if len(direction) == 1 else direction[k]               
                dd = mfem.Vector(dir)
                if sdim == 3:
                    x, y, z = pt               
                    d = mfem.VectorDeltaCoefficient(dd, x, y, z, w)
                elif sdim == 2:
                    x, y = pt                              
                    d = mfem.VectorDeltaCoefficient(dd, x, y, w)
                elif sdim == 1:
                    x = pt                                             
                    d = mfem.VectorDeltaCoefficient(dd,x, w)
                else:
                    assert False, "unsupported dimension"

                if self.fes1.FEColl().Name().startswith('ND'):
                    intg = mfem.VectorFEDomainLFIntegrator(d)                
                elif self.fes1.FEColl().Name().startswith('RT'):
                    intg = mfem.VectorFEDomainLFIntegrator(d)            
                else:    
                    intg = mfem.VectorDomainLFIntegrator(d)                                



            if do_sum:
                if k == 0:
                   lf1 = engine.new_lf(self.fes1)
                lf1.AddDomainIntegrator(intg)            
            else:
                lf1 = engine.new_lf(self.fes1)
                lf1.AddDomainIntegrator(intg)                        
                lf1.Assemble()            
                vecs.append(LF2PyVec(lf1))


        if do_sum: 
            lf1.Assemble()                        
            v1 = MfemVec2PyVec(engine.b2B(lf1), None)
            v1 = PyVec2PyMat(v1)
        else:
            v1 = HStackPyVec(vecs)

        if not self._transpose:        
            v1 = v1.transpose()
        return v1
     
class Projection(Operator):
    '''
    DoF mapping (Dof of fes1 is mapped to fes2)
   
    fes1 is trial space
    fes2 is test  space
 
    example
    # selection mode is interpreted in the test fes.
    # map all booundary 
    projection("boundary", "all")     
    # map domain 1
    projection("domain", [1])         
    # to give FEspace explicitly
    projection("boundary", [3], trial="u", test="v")
    # to project on a different location
    projection("boundary", [1], src=[2], trans="(x, y, 0)", srctrans="(x, y, 0)")
    '''
    def assemble(self, *args, **kwargs):
        engine = self._engine()        
        self.process_kwargs(engine, kwargs)
        if len(args)>0: self._sel_mode = args[0]
        if len(args)>1: self._sel = args[1]

        trans1 = kwargs.pop("trans",   None)     # transformation of fes1 (or both)
        trans2 = kwargs.pop("srctrans", trans1)  # transformation of fes2
        srcsel = kwargs.pop("src", self._ssel)    # source idx
        tol    = kwargs.pop("tol", 1e-5)         # projectio tolerance

        dim1 = self.fes1.GetMesh().Dimension()
        dim2 = self.fes2.GetMesh().Dimension()

        projmode = ""
        if dim2 == 3:
           if self.sel_mode == "domain":
               projmode = "volume"
           elif self.sel_mode == "boundary":
               projmode = "surface"
        elif dim2 == 2:
           if self.sel_mode == "domain":
               projmode = "surface"
           elif self.sel_mode == "boundary":
               projmode = "edge"
        elif dim2 == 1:
           if self.sel_mode == "domain":
               projmode = "edge"
           elif self.sel_mode == "boundary":
               projmode = "vertex"
        assert projmode != "", "unknow projection mode"
        if self._sel == 'all' and self._ssel == 'all':
            if self.sel_mode == "domain":
                if dim1 == dim2:
                    idx1 = np.unique(self.fes1.GetMesh().GetAttributeArray())
                elif dim1 == dim2+1:
                    idx1 = np.unique(self.fes1.GetMesh().GetBdrAttributeArray())
                else:
                    assert False, "unsupported mode"
                idx2 = np.unique(self.fes2.GetMesh().GetAttributeArray()) 
            else:
                idx1 = np.unique(self.fes1.GetMesh().GetBdrAttributeArray())
                idx2 = np.unique(self.fes2.GetMesh().GetBdrAttributeArray())

            if use_parallel:
                idx1 = list(idx1)
                idx2 = list(idx2)
                idx1 = list(set(sum(comm.allgather(idx1),[])))
                idx2 = list(set(sum(comm.allgather(idx2),[])))
            idx = np.intersect1d(idx1, idx2)
            idx1 = list(idx)
            idx2 = list(idx)
            
        elif self._ssel == 'all':
            if self.sel_mode == "domain":
                idx2 = np.unique(self.fes2.GetMesh().GetAttributeArray())
            else:
                idx2 = np.unique(self.fes2.GetMesh().GetBdrAttributeArray())
            idx1 = self._sel
            idx2 = list(idx2)            
            if use_parallel:
                idx2 =  list(set(sum(comm.allgather(idx2),[])))

            
        elif self._sel == 'all':
            if self.sel_mode == "domain":
                idx1 = np.unique(self.fes1.GetMesh().GetAttributeArray())
            else:
                idx1 = np.unique(self.fes1.GetMesh().GetBdrAttributeArray())

            idx1 = list(idx1)
            if use_parallel:
                idx1 =  list(set(sum(comm.allgather(idx1),[])))
            idx2 = srcsel
            
        else:
            idx1 = self._sel
            idx2 = srcsel
            
        if use_parallel:
            # we may not need this?
            idx1 =  list(set(sum(comm.allgather(idx1),[])))
            idx2 =  list(set(sum(comm.allgather(idx2),[])))

        dprint1("projection index ", idx1, idx2)
        
        from petram.helper.dof_map import notrans

        sdim1 = self.fes1.GetMesh().SpaceDimension()
        sdim2 = self.fes2.GetMesh().SpaceDimension()        

        lns = {}
        if trans1 is not None:
            if sdim1 == 3:
                trans1= ['def trans1(xyz):',
                         '    import numpy as np',
                         '    x = xyz[0]; y=xyz[1]; z=xyz[2]',
                         '    return np.array(['+trans1+'])']
            elif sdim1 == 2:
                trans1= ['def trans1(xyz):',
                         '    import numpy as np',
                         '    x = xyz[0]; y=xyz[1]',
                         '    return np.array(['+trans1+'])']
            else: # sdim1 == 3  
                trans1= ['def trans1(xyz):',
                         '    import numpy as np',
                         '    x = xyz[0]',
                         '    return np.array(['+trans1+'])']
            exec('\n'.join(trans1) , self._global_ns, lns)
            trans1 = lns['trans1']            
        else:
            trans1 = notrans
            
        if trans2 is not None:
            if sdim2 == 3:
                trans2 = ['def trans2(xyz):',
                         '    import numpy as np',
                         '    x = xyz[0]; y=xyz[1]; z=xyz[2]',
                         '    return np.array(['+trans2+'])']
            elif sdim2 == 2:
                trans2 = ['def trans2(xyz):',
                         '    import numpy as np',
                         '    x = xyz[0]; y=xyz[1]',
                         '    return np.array(['+trans2+'])']
            else: # sdim1 == 3  
                trans2 = ['def trans2(xyz):',
                         '    import numpy as np',
                         '    x = xyz[0]',
                         '    return np.array(['+trans2+'])']

            trans2 = lns['trans2']                    
        else:
            trans2 = notrans
                  


        from petram.helper.dof_map import projection_matrix as pm
        # matrix to transfer unknown from trail to test
        M, row, col = pm(idx2, idx1, self.fes2, [], fes2=self.fes1,
                         trans1=trans2, trans2=trans1,
                         mode=projmode, tol=tol, filldiag=False)
        return M
        
# for now we assemble matrix whcih mapps essentials too...        
#                        tdof1=self._test_ess_tdof,
#                        tdof2=self._trial_ess_tdof)


               
class Gradient(Operator):               
    '''


    '''
    pass


        
    
    
