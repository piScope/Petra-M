import weakref

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
   '''
   from mpi4py import MPI
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   '''
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


    def process_kwargs(self, kwargs):
        '''
        kwargs in expression can overwrite this.
        '''
        fes1 = kwargs.pop('test', None)
        fes2 = kwargs.pop('range', None)
        self._sel_mode = kwargs.pop('sel_mode', self._sel_mode)
        self._sel  = kwargs.pop('sel', self._sel)
        if fes1 is not None: 
           self._fes1 = weakref.ref(fes1)        
        if fes2 is not None: 
           self._fes2 = weakref.ref(fes2)        
        
class Integral(Operator):
    def assemble(self, *args, **kwargs):
        '''
        integral()
        integral('boundary', [1,3])  # selection type and selection
        '''
        self.process_kwargs(kwargs)
        if len(args)>0: self._sel_mode = args[0]
        if len(args)>1: self._sel = args[1]
        engine = self._engine()
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
        self.process_kwargs(kwargs)
        
        engine = self._engine()
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
          
