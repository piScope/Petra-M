"""
Policies for model verification in the executing environment.
"""
import ast
from importlib import import_module

from petram.mfem_config import get_use_parallel


class BasePolicy:
    """
    Dispatch to a class handler, falling back to the model's own check.

    Handlers return (success, text, detail) and replace verify_setting().
    A handler can call obj.verify_setting() explicitly to extend that check.
    """

    def check(self, obj):
        for cls in type(obj).__mro__:
            method = getattr(self, 'check_' + cls.__name__, None)
            if method is not None:
                return method(obj)
        return obj.verify_setting()


class ClusterSubmission(BasePolicy):
    """
    Validate model settings without requiring locally installed solvers.
    """


class RuntimeVerification(BasePolicy):
    """
    Extend model validation with native solver import checks.
    """

    def _check_solver(self, obj, solver):
        result = obj.verify_setting()
        if not result[0]:
            return result

        module = 'petram.ext.' + solver + '.' + solver + '_solve'
        try:
            import_module(module)
        except (ImportError, OSError) as err:
            text = 'Cannot load ' + solver.upper()
            detail = ('Required package: Petra-M--DS\n'
                      'Module: ' + module + '\n' +
                      type(err).__name__ + ': ' + str(err))
            return False, text, detail
        return result

    def check_MUMPSBase(self, obj):
        return self._check_solver(obj, 'mumps')

    def check_Strumpack(self, obj):
        return self._check_solver(obj, 'strumpack')

    def _uses_mumps(self, obj):
        if (getattr(obj, 'solver_type', '').startswith('Nested')
                and obj.solver_type_in == 'MUMPS'):
            return True

        if getattr(obj, 'adv_mode', False):
            expressions = [obj.adv_prc]
        else:
            index = 1 if get_use_parallel() else 0
            expressions = [choices[index] for name, choices
                           in obj.preconditioners]

        for expression in expressions:
            # Legacy GMRES refers to configurations such as MUMPS1.
            if expression.rstrip('0123456789') == 'MUMPS':
                return True
            try:
                tree = ast.parse(expression, mode='eval')
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == 'mumps'):
                    return True
        return False

    def check_Iterative(self, obj):
        if self._uses_mumps(obj):
            return self._check_solver(obj, 'mumps')
        return obj.verify_setting()

    check_GMRES = check_Iterative
    check_DiagonalPreconditioner = check_Iterative
