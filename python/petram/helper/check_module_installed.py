def module_file_exists(module_name):
    import importlib, os
    spec = importlib.util.find_spec(module_name)
    if spec is not None and spec.origin is not None:
        # Check if the path points to a file and it is not a built-in module
        if os.path.isfile(spec.origin):
            return spec.origin
    return None
