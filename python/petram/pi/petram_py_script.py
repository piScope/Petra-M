from __future__ import print_function

import traceback

from ifigure.mto.py_script import PyScript


class PetraMPyScript(PyScript):
    """PyScript specialization for Petra-M namespace scripts."""

    def classimage(self):
        if PetraMPyScript._image_load_done is False:
            PetraMPyScript._image_id = self.load_classimage()
            PetraMPyScript._image_load_done = True

        if self.getvar('pathmode') == 'owndir':
            return PetraMPyScript._image_id[0]
        else:
            return PetraMPyScript._image_id[1]

    def load_classimage(self):
        import ifigure.utils.cbook as cbook
        import petram.pi
        from petram.utils import get_pkg_datafile

        path = get_pkg_datafile(petram.pi, 'icon')
        print(path)
        idx = cbook.LoadImageFile(path, 'petram_script.png')
        print(idx)
        return [idx, idx]

    def _get_ns_name_for_editor(self):
        name = getattr(self, 'name', None)
        if isinstance(name, str) and name.endswith('_ns'):
            return name[:-3]
        return None

    def _get_dataset_namespace(self, ns_name):
        if ns_name is None:
            return {}

        try:
            parent = self.get_parent()
            if parent is None:
                return {}
            container = parent.get_parent()
            if container is None or not container.has_child('datasets'):
                return {}

            data_folder = container.datasets
            data_obj = data_folder.get_child(name=ns_name + '_data')
            if data_obj is None:
                data_obj = data_folder.get_child(name=ns_name + 'data')
            if data_obj is None:
                return {}

            data = data_obj.getvar()
            if isinstance(data, dict):
                return data.copy()
        except Exception:
            traceback.print_exc()
            pass

        return {}

    def _find_ns_owner(self, ns_name):
        if ns_name is None:
            return None

        try:
            from petram.namespace_mixin import NS_mixin
            pymodel = self.get_pymodel()
            if pymodel is None or not hasattr(pymodel, 'param'):
                return None
            root = pymodel.param.getvar('mfem_model')
            if root is None:
                return None
            for obj in root.walk():
                if not isinstance(obj, NS_mixin):
                    continue
                if obj.get_ns_name() == ns_name:
                    return obj
        except Exception:
            traceback.print_exc()
            pass

        return None

    def _build_petram_base_ns(self):
        g = {}

        try:
            from petram.helper.variables import var_g
            g.update(var_g.copy())
        except Exception:
            traceback.print_exc()
            pass

        try:
            import mfem
            if mfem.mfem_mode == 'serial':
                g['mfem'] = mfem.ser
            elif mfem.mfem_mode == 'parallel':
                g['mfem'] = mfem.par
        except Exception:
            traceback.print_exc()
            pass

        try:
            import numpy
            g['np'] = numpy
        except Exception:
            traceback.print_exc()
            pass

        try:
            from petram.helper.variables import variable, coefficient
            g['variable'] = variable
            g['coefficient'] = coefficient
        except Exception:
            traceback.print_exc()
            pass

        return g

    def provide_ns_for_editor(self, editor=None):
        ns = super(PetraMPyScript, self).provide_ns_for_editor(editor=editor)

        petram_ns = self._build_petram_base_ns()
        ns.update(petram_ns)

        ns_name = self._get_ns_name_for_editor()
        owner = self._find_ns_owner(ns_name)

        if owner is not None:
            try:
                if not isinstance(owner._global_ns, dict):
                    owner.eval_ns()
            except Exception:
                traceback.print_exc()
                pass

            try:
                if isinstance(owner._global_ns, dict):
                    ns.update(owner._global_ns)
            except Exception:
                traceback.print_exc()
                pass

            try:
                if isinstance(owner.dataset, dict):
                    ns.update(owner.dataset)
            except Exception:
                traceback.print_exc()
                pass

            ns['obj'] = owner

        ns.update(self._get_dataset_namespace(ns_name))

        return ns


def migrate_namespace_scripts(model):
    """Upgrade legacy namespace script nodes from PyScript to PetraMPyScript."""
    if model is None or not model.has_child('namespaces'):
        return False

    migrated = False
    for _name, child in model.namespaces.get_children():
        if isinstance(child, PetraMPyScript):
            continue
        if isinstance(child, PyScript):
            # Safe in-place upgrade: subclass only adds behavior, no new state.
            child.__class__ = PetraMPyScript
            migrated = True
    return migrated
