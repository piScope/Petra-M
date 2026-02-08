'''
   FormHolder is a block structure to maintain a biuch of lf, bf, and gf.

   Each block can have a multiple bf/lf. So, each element is a dictionary,
   whose key is a projection operator.
   Value of dictionary is a two element list. The first one is lf/bf/gf,
   and the second place is for Vector/Matrix
'''
from itertools import product
import numpy as np

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('FromHolder')


class FormBlock(object):
    def __init__(self, shape, new=None):
        '''
        kind : scipy
                  stores scipy sparse or numpy array
               hypre
        '''
        r = shape
        c = 1
        self.ndim = 1
        self.block = [[None]*c for x in range(r)]
        self._diag_vec = [False]*r
        self._diag_callable_mask = [False]*r
        self.diag_b_mode = False

        self._shape = (r, c)
        self.allocator1 = new
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
    def diag_callable_mask(self):
        return self._diag_callable_mask

    def set_diag_callable_mask(self, ifess):
        for i in ifess:
            self._diag_callable_mask[i] = True

    def set_allocator(self, alloc):
        self.allocator1 = alloc

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
                form = self.allocator1(r)
                self.block[r][c][1] = [form, None]
                projector = 1

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
        from petram.debug import debug_dpg_essential

        if converter2 is None:
            converter2 = converter1

        for i, j in product(range(self.shape[0]), range(self.shape[1])):
            projs = self.get_projections(i, j)
            diag_callable_mask = self.diag_callable_mask[i]

            for p in projs:
                form = self.block[i][j][p][0]
                if verbose:
                    dprint1("generateMatVec", i, j, form)

                if diag_callable_mask:
                    if self.diag_b_mode:
                        continue

                    dest = self.block[i][j][p][1]
                    if dest is None:
                        continue
                    if debug_dpg_essential:
                        vec = converter1(form)
                        dest.GetDataArray()[:] = vec.GetDataArray()
                    else:
                        # this information is already in X
                        pass

                elif form is not None:
                    self.set_matvec(i, j, p, converter1(form))
                else:
                    self.set_matvec(i, j, p, None)

    def set_vec_from_blk(self, i, vec, vblk):
        # this returns block matrix generated by
        self._diag_vec[i] = vec
        size = len(vblk)
        k = 0
        for ii in range(size):
            # sibling blocks are None. need to fill
            # default setting
            if self.block[i+ii] is None:
                 self.block[i+ii] = {1:[None, None]}
            self.set_matvec(i+ii, 0, 1, vblk[k])
            k = k + 1

def convertElement(Mreal, Mimag, i, j, converter, projections=None, verbose=False):
    '''
    Generate PyVec/PyMat format data.
    It takes two FormBlocks. One for real and the other for imag.

    M = sum(Opr*proj or proj*Opr), Opr is made from BilinearForms.

    We use the same data structure for lf/gf, for which projector
    will be always 1.
    '''
    keys = set(Mreal.get_projections(i, j) +
               Mimag.get_projections(i, j))

    term = None
    for k in keys:
        args = []
        if Mreal.block[i][j] is not None:
            rmatvec = Mreal.block[i][j][k][1] if k in Mreal.block[i][j] else None
        else:
            rmatvec = None

        if Mimag.block[i][j] is not None:
            imatvec = Mimag.block[i][j][k][1] if k in Mimag.block[i][j] else None
        else:
            imatvec = None

        args = (rmatvec, imatvec)

        if rmatvec is None and imatvec is None:
            return None

        m = converter(*args)

        if k != 1:
            pos, projector = k
            projections, projections_hash = projections

            h = projections_hash[projector]
            p = projections[h]
            if pos > 0:
                m = m.dot(p)
            else:
                pp = p.transpose()
                m = pp.dot(m)

        if term is None:
            term = m
        else:
            term = term + m
    return term
