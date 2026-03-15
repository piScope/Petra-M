def module_file_exists(module_name):
    import importlib, os
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError):
        return False
    return bool(spec and spec.origin and os.path.isfile(spec.origin))
