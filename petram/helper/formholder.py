'''
   FormHolder is a block structure to maintain a biuch of lf, bf, and gf.

   Each block can have a multiple bf/lf. So, each element is a dictionary, 
   whose key is a projection operator. 
   Value of dictionary is a two element list. The first one is lf/bf/gf,
   and the second place is for Vector/Matrix 
'''
from itertools import product

class FormBlock(object):
    def __init__(self, shape, new=None, mixed_new=None):
        '''
        kind : scipy
                  stores scipy sparse or numpy array
               hypre
        '''
        try:
            x = len(shape)
        except:
            shape = [shape]
            
        if len(shape) == 2:
            r, c = shape
            self.block = [[None]*c for x in range(r)]
            self.ndim = 2
        elif len(shape) == 1:
            r = shape[0]
            c = 1
            self.ndim = 1
            self.block = [[None]*r]
        self._shape = (r,c)
        self.allocator1 = new
        if mixed_new is None:
            self.allocator2 = new
        else:
            self.allocator2 = mixed_new
    @property
    def shape(self):
        return self._shape
    
    def set_allocator(self, alloc):
        self.allocator1 = alloc
    def set_mixed_allocator(self, alloc):
        self.allocator2 = alloc
        
    def set_no_allocator(self):        
        self.allocator1 = None
        self.allocator2 = None

    def __iter__(self):
        assert (self.allocator1 is None and self.allocator2 is None), "FormBlock must be fixed"

        all_forms = []
        for r, c in product(range(self.shape[0]),range(self.shape[1])):
            if self.block[r][c] is None: continue
            for key in self.block[r][c].keys():
                all_forms.append(self.block[r][c][key][0])

        return all_forms.__iter__()

    def __getitem__(self, idx):
        if self.ndim == 2:
            try:
                r, c, projector = idx
            except:
                r, c = idx
                projector = 1
        else:
            c = 0            
            try:
                r, projector = idx
            except:
                r = idx
                projector = 1
                        
        if self.block[r][c] is None: self.block[r][c] = {}
        
        if not projector in self.block[r][c]:
            if self.allocator1 is None:
                return None
            else:
                if r == c:
                    form = self.allocator1(r)
                else:
                    form = self.allocator2(r, c)                    
                self.block[r][c][projector] = [form, None]
        return self.block[r][c][projector][0]
                                    

    def __setitem__(self, idx, v):
        if self.ndim == 2:
            try:
                r, c, projector = idx
            except:
                r, c = idx
                projector = 1
        else:
            c = 0
            try:
                r, projector = idx
            except:
                r = idx
                projector = 1
        
        if self.block[r][c] is None: self.block[r][c] = {}
        
        self.block[r][c][projector] = [v, None]

    def get_projections(self, r, c):
        if self.block[r][c] is None: return []
        return self.block[r][c].keys()
        
    def get_matvec(self, r, c, p):
        return self.block[r][c][p][1]
    
    def set_matvec(self, r, c, p, v):
        self.block[r][c][p][1] = v               

    def generateMatVec(self, converter1, converter2=None):
        if converter2 is None: converter2 = converter1

        for i, j in product(range(self.shape[0]),range(self.shape[1])):
            projs = self.get_projections(i, j)
            print("debug", i, j, self.block[i][j])
            for p in projs:
                form = self.block[i][j][p][0]
                print form
                if form is not None:
                    if i == j:
                        self.set_matvec(i, j, p, converter1(form))
                    else:
                        self.set_matvec(i, j, p, converter2(form))
                else:
                    self.set_matvec(i, j, p, None)
                
def convertElement(Mreal, Mimag, i, j, converter):
    '''
    Generate PyVec/PyMat format data.
    It takes two FormBlocks. One for real and the other for imag.
    '''
    keys = set(Mreal.get_projections(i,j)+
               Mimag.get_projections(i,j))

    term = None
    print "real", Mreal.block[i][j]
    print "imag", Mimag.block[i][j]    
    for k in keys:
        if Mreal.block[i][j] is not None:
            rmatvec = Mreal.block[i][j][k][1] if k in Mreal.block[i][j] else None
        else:
            rmatvec = None
        if Mimag.block[i][j] is not None:
            imatvec = Mimag.block[i][j][k][1] if k in Mimag.block[i][j] else None
        else:
            imatvec = None
            
        m = converter(rmatvec, imatvec)
        if k!=1:
            pos, projector = k
            if pos > 0:
                m = m.dot(projector)
            else:
                m = projector.dot(m)
        if term is None:
            term = m
        else:
            term = term + m
    return term
