import weakref
from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD

class EvaluatorAgent(object):
    def __init__(self):
        object.__init__(self)
        self.mesh = None
        self.knowns = WKD()
        
    def forget_knowns(self):
        self.knowns = WKD()
        
    def set_mesh(self, mesh):
        self.mesh = weakref.ref(mesh)

    def preprocess_geometry(self):
        raise NotImplementedError("subclass needs to implelment this")

    def eval(self, expr, solvars, phys):    
        raise NotImplementedError("subclass needs to implelment this")        
    
