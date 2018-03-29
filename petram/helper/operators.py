import weakref
import numpy as np

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
   from mpi4py import MPI
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   comm = MPI.COMM_WORLD   
else:
   import mfem.ser as mfem

class Operator(object):
    def __repr__(self):
        return self.__class__.__name__ + "("+",".join(self.sel)+")"

class Operator(object):
    def __init__(self, **kwargs):
        self._sel = kwargs.pop("sel", "all")
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
    
    def get_restriction_array(self, fes):
        mesh = fes.GetMesh()
        intArray = mfem.intArray

        if isinstance(self, Domain):
            size = np.max(mesh.GetAttributeArray())
        else:
            size = np.max(mesh.GetBdrAttributeArray())

        if self._sel[0] == "all":
            arr = [0]*size
        else:
            for k in self._sel: arr[k-1] = 1
        return intArray(arr)
    
    def restrict_coeff(self, coeff, fes, vec = False, matrix=False):
        if self._sel == 'all': 
           return coeff
        print self._sel
        arr = self.get_restriction_array(fes, self._sel)
        if vec:
            return mfem.VectorRestrictedCoefficient(coeff, arr)
        elif matrix:
            return mfem.MatrixRestrictedCoefficient(coeff, arr)           
        else:
            return mfem.RestrictedCoefficient(coeff, arr)


    def process_kwargs(self, engine, kwargs):
        '''
        kwargs in expression can overwrite this.
        '''
        fes1 = kwargs.pop('test', None)
        fes2 = kwargs.pop('trial', None)
        self._sel_mode = kwargs.pop('sel_mode', self._sel_mode)
        self._sel  = kwargs.pop('sel', self._sel)
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
    
class Integral(Operator):
    def assemble(self, *args, **kwargs):
        '''
        integral()
        integral('boundary', [1,3])  # selection type and selection
        '''
        engine = self._engine()        
        self.process_kwargs(engine, kwargs)
        
        if len(args)>0: self._sel_mode = args[0]
        if len(args)>1: self._sel = args[1]

        lf1 = engine.new_lf(self._fes1())
        one = mfem.ConstantCoefficient(1.0)        
        coff = self.restrict_coeff(one, self._fes1())

        if self._sel_mode == 'domain':
            intg = mfem.DomainLFIntegrator(coff)
        elif self._sel_mode == 'boundary':
            intg = mfem.BoundaryLFIntegrator(coff)
        else:
            assert False, "Selection Type must be either domain or boundary"
        lf1.AddDomainIntegrator(intg)
        lf1.Assemble()
        
        from mfem.common.chypre import LF2PyVec, PyVec2PyMat, MfemVec2PyVec
        v1 = MfemVec2PyVec(engine.b2B(lf1), None)
        

        v1 = PyVec2PyMat(v1)
        if not self._transpose:        
            v1 = v1.transpose()
        return v1

class Identity(Operator):    
    def assemble(self, *args, **kwargs):
        engine = self._engine()
        self.process_kwargs(engine, kwargs)
        
        fes1 = self._fes1()
        fes2 = fes1 if self._fes2 is None else self._fes2()
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


class Projection(Operator):
    '''
    DoF mapping (Dof of fes1 is mapped to fes2)
 
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
        srcsel = kwargs.pop("src", self._sel)    # transformation of fes2
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

        if self._sel == 'all':
            if self.sel_mode == "domain":
                idx = np.unique(self.fes2.GetMesh().GetAttributeArray())
            else:
                idx = np.unique(self.fes2.GetMesh().GetBdrAttributeArray())
            idx1 = list(idx)
            idx2 = list(idx)
        else:
            idx1 = self._sel
            idx2 = srcsel
        if use_parallel:
            idx1 =  list(set(sum(comm.allgather(idx1),[])))
            idx2 =  list(set(sum(comm.allgather(idx2),[])))
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
            exec '\n'.join(trans1) in self._global_ns, lns
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


               
               
           

        
    
    
