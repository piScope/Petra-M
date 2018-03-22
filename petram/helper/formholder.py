'''
   block of linear/bilinear forms

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

    def get_projections(self, idx):
        if self.shape[1] != 1:
            r, c = idx
        else:
            c = 0
            r = idx
        return self.block[r][c].keys()
        
    def get_matvec(self, idx):
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
        return self.block[r][c][projector][1]
    
    def set_matvec(self, idx, v):
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
        self.block[r][c][projector][1] = v               
    
                   

def generateMatVect(M, converter1, converter2=None):
    if converter2 is None: converter2 = converter1

    for i, j in product(range(M.shape[0]),range(M.shape[1])):
        projs = M.get_projections(i, j)
        for p in projs:
            form = M[i, j, p]
            if i == j:
                M.set_matvec(i, j, converter1(form))
            else:
                M.set_matvec(i, j, converter2(form))
                
def convertElement(Mreal, Mimag, i, j, converter):
    keys = set(Mreal.block[i][j].keys() + Mimag.block[i][j].keys())

    term = None
    for k in keys:

        rmatvec = Mreal.block[i][j][k][1] if k in Mreal.block[i][j] else None
        imatvec = Mimag.block[i][j][k][1] if k in Mimag.block[i][j] else None

        m = converter(rform, iform)
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
    return m
