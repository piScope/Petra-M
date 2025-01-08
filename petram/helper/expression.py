import weakref

import petram.helper.operators as ops
operators = {"integral": ops.Integral,
             "loopintegral": ops.LoopIntegral,
             "identity": ops.Identity,
             "projection": ops.Projection,
             "delta": ops.Delta,
             "deltam": ops.DeltaM,
             "zero": ops.Zero,
             "curl": ops.Curl,
             "grad": ops.Gradient,
             "div": ops.Divergence,
             "convolve": ops.Convolve,
             "hcurln": ops.Hcurln}


def get_operators():
    return {key: operators[key]() for key in operators}

# this is used in vtable to evaluate the expresion containing the operators


def dummy(*args, **kwargs):
    return 1


def get_dummy_operators():
    return {key: dummy for key in operators}


'''
   exp = Expression("integral("domain", "all")")
   exp.set_range_space(...)
   exp.set_test_space(...)
   exp.set_engine(...)
   # exp.set_sel_mode(...)
   # exp.set_sel(...)
   operator = exp.assemble(g = None)

'''


class Expression(object):
    def __init__(self, expr, **kwargs):
        self._expr = expr

        engine = kwargs.pop('engine', None)
        fes1 = kwargs.pop('trial', None)
        fes2 = kwargs.pop('test', None)

        self._sel_mode = kwargs.pop("_sel_mode", "domain")
        self._sel = kwargs.pop("_sel", "all")
        self._fes1 = None
        self._fes2 = None
        self._engine = None
        self._transpose = kwargs.pop('transpose', False)
        # False (horizontal vector if it is vector,,,)
        self._trial_ess_tdof = kwargs.pop('trial_ess_tdof', None)
        self._test_ess_tdof = kwargs.pop('test_ess_tdof', None)
        self._ind_vars = kwargs.pop('ind_vars', 'x, y, z')
        self._is_complex = kwargs.pop('is_complex', False)
        self._c_coeff = kwargs.pop('c_coeff', False)

        super(Expression, self).__init__()
        variables = []
        code = compile(expr, '<string>', 'eval')
        names = code.co_names
        self.co = code

        if fes1 is not None:
            self.set_trial_space(fes1)
        if fes2 is not None:
            self.set_test_space(fes2)
        if engine is not None:
            self.set_engine(engine)

    @property
    def names(self):
        return self._names

    def isOperatorType(self, g):

        from petram.helper.operators import Operator, operators

        # make sure that basic operators are there (if not overwritten)x
        for n in names:
            if n in operators:
                return True

        return False

    def isVariableType(self, g):
        return not isVariableType(g)

    def set_trial_space(self, fes):
        self._fes1 = weakref.ref(fes)

    def set_test_space(self, fes):
        self._fes2 = weakref.ref(fes)

    def set_sel_mode(self, mode):
        self._sel_mode = mode

    def set_sel(self, mode):
        self._sel = mode

    def set_engine(self, en):
        self._engine = weakref.ref(en)

    def assemble(self, g=None):
        g = {} if g is None else g
        operators = get_operators()
        attrs = [
            '_fes1',
            '_fes2',
            '_sel',
            '_sel_mode',
            '_engine',
            '_transpose',
            '_trial_ess_tdof',
            '_test_ess_tdof',
            '_ind_vars',
            '_is_complex',
            '_c_coeff', ]

        for op in operators:
            for attr in attrs:
                setattr(operators[op], attr, getattr(self, attr))
        g2 = g.copy()
        for op in operators:
            g2[op] = operators[op]

        op = eval(self.co, g2)

        return op
