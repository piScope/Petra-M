'''
   FormBBlock

      Block-Block strcturue of Forms

      Data strucutre to handle non-conforming block-matrix

'''
from itertools import product
import numpy as np
import itertools

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('FromHolder')


class FormBBlock2D(object):
    def __init__(self, phys_target, phys_range, new=None, mixed_new=None, diag=None):


        def __init__(self, engine, phys_target, phys_range, def_bf_alloc, def_mixed_alloc, ):

        all_connection = {}

        
        target_phys_list = [[phys1]*len(phys1.dep_vars) for phys1 in phys_target]
        range_phys_list = [[phys1]*len(phys1.dep_vars) for phys1 in phys_range]
        
            for phys2 in phys_range:
                dv2 = phys2.dep_vars
                forms = {}
                if phys1 == phys2:
                    diag_forms = phys1.allocate_diag_forms(def_bf_alloc)
                    for a, b in itertools.product(enumerate(dv1), enumerate(dv2),):
                        forms[(a, b)] = phys1.pick_diag_form(diag_forms, a[0], b[0])                        
                else:
                    mixed_forms = phys1.allocate_mixed_forms(phys2, def_mix_alloc))
                    for a, b in itertools.product(enumerate(dv1), enumerate(dv2),):
                        forms[(a, b)] = phys1.pick_mixed_form(mixed_forms, phys2, a[0], b[0])                        
                   
                all_connections[(phys1, phys2)] = forms

        for phys1, phys2 in all_connections
                   
                
            for print(en._rdep_vars)
        print(en._rdep_vars)        
        n_fes = len(en.fes_vars)
            mask = [False]*len(en._rdep_vars)
        else:
            mask = [False]*len(en._dep_vars)
        
        n_rfes = len(en.r_fes_vars)
        diag = [self.r_fes_vars.index(n) for n in self.fes_vars]
        n_mat = self.n_matrix


        en.
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
            self.block = [[None]*c for x in range(r)]
        self._shape = (r, c)
        self._diag = diag
        self.allocator1 = new
        if mixed_new is None:
            self.allocator2 = new
        else:
            self.allocator2 = mixed_new

        self.no_allocation = False

    def __repr__(self):

        text2 = []
        for x in self.block:
            text = []
            for y in x:
                if y is None:
                    text.append("None")
                else:
                    text.append(str(list(y)))
            text2.append(",".join(text))
        full_repr = "\n".join(text2)

        return "Formblock" + str(self._shape) + ":\n" + full_repr
        # return "Formblock" + str(self._shape) + ":\n" + str(self.block)

    @property
    def shape(self):
        return self._shape

    @property
    def diag(self):
        return self._diag

    def set_allocator(self, alloc):
        self.allocator1 = alloc

    def set_mixed_allocator(self, alloc):
        self.allocator2 = alloc

    def set_no_allocator(self):
        self.no_allocation = True
        #self.allocator1 = None
        #self.allocator2 = None

    def __iter__(self):
        assert self.no_allocation, "FormBlock must be fixed"

        all_forms = []
        for r, c in product(range(self.shape[0]), range(self.shape[1])):
            if self.block[r][c] is None:
                continue
            for key in self.block[r][c].keys():
                all_forms.append((r, c, self.block[r][c][key][0]))

        return all_forms.__iter__()

    def __getitem__(self, idx):
        r, c, projector = self.allocate_block(idx)
        if not projector in self.block[r][c]:
            return None

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

        if self.block[r][c] is None:
            self.block[r][c] = {}

        self.block[r][c][projector] = [v, None]

    def allocate_block(self, idx, reset=False):
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
        if self.block[r][c] is None:
            self.block[r][c] = {}

        if reset:
            keys = list(self.block[r][c])
            for p in keys:
                del self.block[r][c][p]

        if len(self.block[r][c]) == 0:
            if self.no_allocation and not reset:
                pass
            else:
                if self._diag is None or self._diag[r] == c:
                    form = self.allocator1(r)
                    self.block[r][c][1] = [form, None]
                    projector = 1
                else:
                    form, projector = self.allocator2(r, c)
                    self.block[r][c][projector] = [form, None]
        elif len(self.block[r][c]) == 1:
            projector = list(self.block[r][c])[0]
        else:
            assert False, "should not come here? having two differnt projection in the same block"
        return r, c, projector

    def renew(self, idx):
        self.allocate_block(idx, reset=True)

    def get_projections(self, r, c):
        if self.block[r][c] is None:
            return []
        return list(self.block[r][c])

    def get_matvec(self, r, c=0, p=1):
        return self.block[r][c][p][1]

    def set_matvec(self, r, *args):
        # set_matvec(self, r, c=0, p=1, v):
        if len(args) < 1:
            assert False, "need  a value to set"
        v = args[-1]
        if len(args) == 2:
            c = args[0]
            p = 1
        if len(args) == 3:
            c = args[0]
            p = args[1]
        self.block[r][c][p][1] = v

    def generateMatVec(self, converter1, converter2=None, verbose=False):
        if converter2 is None:
            converter2 = converter1

        for i, j in product(range(self.shape[0]), range(self.shape[1])):
            projs = self.get_projections(i, j)
            for p in projs:
                form = self.block[i][j][p][0]
                if verbose:
                    dprint1("generateMatVec", i, j, form)

                if form is not None:
                    if self._diag is None or self._diag[i] == j:
                        self.set_matvec(i, j, p, converter1(form))
                    else:
                        self.set_matvec(i, j, p, converter2(form))
                else:
                    self.set_matvec(i, j, p, None)

