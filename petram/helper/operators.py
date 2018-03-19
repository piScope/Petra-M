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

operators = {"integd":IntegD,
             "ingegb":IntegB}
class Operator(object):
    def __repr__(self):
        return self.__class__.__name__ + "("+",".join(self.sel)+")"

class Operator(object):
    def __init__(self, **kwargs):
        self.sel = kwargs.pop("sel", "all")

    def __call__(self):
        fespace = globals()["fespace"]
        return self.assemble(fespace)
    
    def assemble(self, fes):
        raise NotImplementedError("Subclass needs to implement this")
    
    def get_restriction_array(self, fes):
        mesh = fes.GetMesh()
        intArray = mfem.intArray

        if isinstance(self, Domain):
            size = mesh.attributes.Size()
        else:
            size = mesh.bdr_attributes.Size()


        if self.sel[0] == "all":
            arr = [0]*size
        else:
            for k in self.sel: arr[k-1] = 1
        return intArray(arr)
    
    def restrict_coeff(self, coeff, fes, vec = False, matrix=False):
        if self.sel == 'all': return coeff
           return coeff
        arr = self.get_restriction_array(fes, idx)
        if vec:
            return mfem.VectorRestrictedCoefficient(coeff, arr)
        elif matrix:
            return mfem.MatrixRestrictedCoefficient(coeff, arr)           
        else:
            return mfem.RestrictedCoefficient(coeff, arr)


class IntegD(Operator):
    def assemble(self, engine, fes):
        lf1 = engine.new_lf(fes)
        one = mfem.ConstantCoefficient(1.0)        
        coff = self.restrict_coeff(one, fes)
        
        intg = mfem.DomainLFIntegrator(coff)        
        lf1.AddDomainIntegrator(intg)
        lf1.Assemble()
        
        from mfem.common.chypre import LF2PyVec, PyVec2PyMat,
        v1 = LF2PyVec(lf1, None)
        v1 = PyVec2PyMat(v1)        
        return v1
    
class IntegB(Operator):
    def assemble(self, engine, fes):    
        lf1i = engine.new_lf(fes)
        one = mfem.ConstantCoefficient(1.0)        
        coff = self.restrict_coeff(one, fes)
        
        intg = mfem.BoundaryLFIntegrator(coeff)
        lf1.AddBoundaryIntegrator(intg)
        lf1.Assemble()
        lf1i = engine.new_lf(fes)

        from mfem.common.chypre import LF2PyVec, PyVec2PyMat,
        v1 = LF2PyVec(lf1, None)
        v1 = PyVec2PyMat(v1)        
        return v1
        
                 
