def module_file_exists(module_name):
    import importlib, os
    spec = importlib.util.find_spec(module_name)
    return bool(spec and spec.origin and os.path.isfile(spec.origin))
