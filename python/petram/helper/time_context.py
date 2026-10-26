"""
Shared runtime time and numeric model namespaces.

Runtime time changes go through TimeContext. Direct MFEM SetTime calls remain
possible and are the caller's responsibility. Function-level Python global
assignment can bypass dict overrides; use the namespace mapping or context.
"""

import keyword
import weakref

class _IdentityRegistry:
    """
    Hold weak references without requiring hashable registered objects.
    """

    def __init__(self):
        self._refs = {}

    def add(self, obj):
        key = id(obj)
        if key not in self._refs:
            self._refs[key] = weakref.ref(
                obj, lambda ref: self._refs.pop(key, None))

    def values(self):
        return [obj for ref in list(self._refs.values())
                if (obj := ref()) is not None]

    def clear(self):
        self._refs.clear()


class TimeContext:
    """
    Process-wide time owner shared by all active model runtime objects.

    Each MPI rank owns its own instance; the solver supplies the same physical
    time on participating ranks. Model lifecycle code resets registrations.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_namespaces'):
            return
        self._namespaces = _IdentityRegistry()
        self._variables = _IdentityRegistry()
        self._coefficients = _IdentityRegistry()
        self._time = 0.0
        self._time_name = 't'
        self._updating = False

    @property
    def time(self):
        return self._time

    @property
    def time_name(self):
        return self._time_name

    def configure(self, time_name='t'):
        """
        Select a valid reserved time name and update registered namespaces.
        """
        if (not isinstance(time_name, str) or not time_name.isidentifier()
                or keyword.iskeyword(time_name)):
            raise ValueError('The time variable must be a Python identifier, '
                             'not a keyword')
        old_name = self._time_name
        if time_name == old_name:
            return
        namespaces = self._namespaces.values()
        for namespace in namespaces:
            if time_name in namespace:
                raise ValueError('The time variable conflicts with namespace '
                                 'name: ' + time_name)
        self._time_name = time_name
        for namespace in namespaces:
            dict.pop(namespace, old_name, None)
            dict.__setitem__(namespace, time_name, self.time)

    def set_time(self, value):
        """
        Synchronize numeric namespaces, Variables, and MFEM coefficients.
        """
        value = float(value)
        if self._updating:
            if value != self.time:
                raise RuntimeError('Cannot change time during time propagation')
            return
        if value == self.time:
            return
        self._time = value
        self._updating = True
        try:
            for namespace in self._namespaces.values():
                dict.__setitem__(namespace, self.time_name, value)
            for variable in self._variables.values():
                variable.set_time(value)
            for coefficient in self._coefficients.values():
                coefficient.SetTime(value)
        finally:
            self._updating = False

    def register_namespace(self, namespace):
        self._namespaces.add(namespace)
        dict.__setitem__(namespace, self.time_name, self.time)

    def register_variable(self, variable):
        self._variables.add(variable)
        variable.set_time(self.time)

    def register_coefficient(self, coefficient):
        self._coefficients.add(coefficient)
        coefficient.SetTime(self.time)

    def reset(self, time=0.0, time_name='t'):
        """
        Detach obsolete runtime objects when starting a model execution.
        """
        # Validate before discarding an active execution's registrations.
        if (not isinstance(time_name, str) or not time_name.isidentifier()
                or keyword.iskeyword(time_name)):
            raise ValueError('The time variable must be a Python identifier, '
                             'not a keyword')
        value = float(time)
        self._namespaces.clear()
        self._variables.clear()
        self._coefficients.clear()
        self._time_name = time_name
        self._time = value


def get_time_context():
    """
    Return the process-wide runtime time owner.
    """
    return TimeContext()


class TimeNamespace(dict):
    """
    Keep a numeric time entry synchronized through the shared context.

    Construction and copying inherit current runtime time; explicit assignment
    or update of the reserved entry requests a runtime time change.
    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self._time_context = get_time_context()
        self._time_context.register_namespace(self)

    def __setitem__(self, key, value):
        if key == self._time_context.time_name:
            try:
                self._time_context.set_time(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "The reserved time variable %r must hold a numeric time"
                    % key) from error
            # Also restore the entry after a Python operation bypassed us.
            dict.__setitem__(self, key, self._time_context.time)
        else:
            dict.__setitem__(self, key, value)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def copy(self):
        return type(self)(self)

    def __ior__(self, other):
        self.update(other)
        return self

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]
