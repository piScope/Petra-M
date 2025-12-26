'''
   FormBBlock

      Block-Block strcturue of Forms

      Data strucutre to handle non-conforming block-matrix

'''
import itertools
import weakref

import numpy as np
from itertools import product
import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('FromBHolder')


class FormBBlock(object):
    def __init__(self, phys_target, phys_range, new=None, mixed_new=None,
                 converter1=None, converter2=None):

        self.target_phys_list = sum([
            [phys1.name()]*len(phys1.dep_vars) for phys1 in phys_target], [])
        self.range_phys_list = sum([
            [phys1.name()]*len(phys1.dep_vars) for phys1 in phys_range], [])
        self.target_kfes_list = sum([list(range(len(phys1.dep_vars)))
                                     for phys1 in phys_target], [])
        self.range_kfes_list = sum([list(range(len(phys1.dep_vars)))
                                    for phys1 in phys_range], [])

        shape = (len(self.target_phys_list), len(self.range_phys_list))

        r, c = shape
        self._shape = shape

        self.block = [[None]*c for x in range(r)]
        self.allocator1 = new
        if mixed_new is None:
            self.allocator2 = new
        else:
            self.allocator2 = mixed_new

        self.no_allocation = False

        # storage for physics-specific forms for diagnals
        #
        #   a physics-specific form typically generate block-matrix
        #   by itself.

        self._diagform = weakref.WeakValueDictionary()
        self._diagform2mat = [[None]*c for x in range(r)]
        self._diag_callable = [[None]*c for x in range(r)]
        self._diag_callable_mask = [[False]*c for x in range(r)]        
        self._converters = (converter1, converter2)
        
    def set_diag_callable_mask(self, ifess, rifess):
        for i in ifess:
            for j in rifess:
                self._diag_callable_mask[i][j] = True

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
    def diag_callable(self):
        return self._diag_callable
    @property
    def diag_callable_mask(self):
        return self._diag_callable_mask
    @property
    def diagform2mat(self):
        return self._diagform2mat

    def set_allocator(self, alloc):
        self.allocator1 = alloc

    def set_mixed_allocator(self, alloc):
        self.allocator2 = alloc

    def set_no_allocator(self):
        self.no_allocation = True

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
        try:
            r, c, projector = idx
        except:
            r, c = idx
            projector = 1

        if self.block[r][c] is None:
            self.block[r][c] = {}

        self.block[r][c][projector] = [v, None]

    def allocate_block(self, idx, reset=False):
        try:
            r, c, projector = idx
        except:
            r, c = idx
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
                callable = self.diag_callable[r][c]
                callable_mask = self.diag_callable_mask[r][c]                
                physname = self.target_phys_list[r]
                if callable is not None:
                    if physname not in self._diagform:
                        form = callable()
                        if form is not None:
                             self._diagform[physname] = form

                    if physname in self._diagform:                             
                        self.block[r][c][1] = [self._diagform[physname], None]
                    else:
                        self.block[r][c][1] = [None, None]
                elif callable_mask:
                    self.block[r][c][1] = [None, None]                    
                else:
                    if (self.target_phys_list[r] == self.range_phys_list[c] and
                            self.target_kfes_list[r] == self.range_kfes_list[c]):
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

    def generateMatVec(self, verbose=False):
        converter1, converter2 = self._converters

        for i, j in product(range(self.shape[0]), range(self.shape[1])):
            projs = self.get_projections(i, j)            
            diag_callable = self.diag_callable[i][j]            
            diag_callable_mask = self.diag_callable_mask[i][j]
            
            for p in projs:
                form = self.block[i][j][p][0]
                if verbose:
                   dprint1("generateMatVec", i, j, form)
                if form is None:
                    pass
                elif diag_callable is not None:
                    # this returns block matrix generated by
                    diagform2mat = self.diagform2mat[i][j]
                    mats = diagform2mat(form)
                    size = int(np.sqrt(len(mats)))
                    k = 0
                    for ii in range(size):
                        for jj in range(size):
                            # sibling blocks are None. need to fill
                            # default setting
                            if self.block[i+ii][j+jj] is None:
                                self.block[i+ii][j+jj] = {1:[None, None]}
                                
                            self.set_matvec(i+ii, j+jj, p, mats[k])
                            k = k + 1

                elif diag_callable_mask:
                    pass
                else:
                    if (self.target_phys_list[i] == self.range_phys_list[j] and
                            self.target_kfes_list[i] == self.range_kfes_list[j]):
                        self.set_matvec(i, j, p, converter1(form))
                    else:
                        self.set_matvec(i, j, p, converter2(form))


