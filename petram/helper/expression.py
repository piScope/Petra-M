import parser

class Expression(str):
    def __init__(self, *args, **kwargs):
        super(Expression, self).__init__(*args, **kwargs)
        variables = []
        st = parser.expr(expr)
        code= st.compile('<string>')
        names = code.co_names
        self.co = code
        self._names = names

    @property
    def names(self):
        return self._names

    def isOperatorType(self, g):
        
        from petram.helper.operators import Operator, operators

        # make sure that basic operators are there (if not overwritten)x
        g2 = g[:]
        for key in operators:
            if not key in g2: g2[key] = operators[key]
        
        for n in self.names:
            if (n in g and isinstance(g[n], Operator)):
                return True
        return False
        
    def isVariableType(self, g):
        return not isVariableType(g)
