"""
Configuration shared by the build backend and setup.py, following Petra-M--DS.
"""
import os
from shutil import which as find_command

__all__ = ["print_config", "initialize_cmd_opts", "configure_build",
           "superlu_mt_enabled"]

cmd_opts = [
    ('with-superlu-mt=', 'YES', 'build the SuperLU_MT native backend'),
    ('CC=', find_command('cc'), 'C compiler'),
    ('CXX=', find_command('c++'), 'C++ compiler'),
    ('SWIG=', find_command('swig'), 'SWIG wrapper generator'),
]


def initialize_cmd_opts(bglb):
    """
    Initialize build options from defaults and environment variables.
    """
    for param, value, _help in cmd_opts:
        attr = param.rstrip('=').replace('-', '_')
        value = os.getenv(attr, value)
        setattr(bglb, attr.lower(), value)


def _process_cmd_opts(bglb, cfs):
    """
    Apply frontend settings without modifying the caller's dictionary.
    """
    settings = dict(cfs)
    for param, _default, _help in cmd_opts:
        param = param.rstrip('=')
        if param in settings:
            value = settings.pop(param)
            if isinstance(value, list):
                value = value[-1]
            setattr(bglb, param.replace('-', '_').lower(), value)
    if settings:
        raise ValueError("Unknown build settings: " + str(settings))


def configure_build(bglb):
    """
    Resolve the supported native build options.
    """
    initialize_cmd_opts(bglb)
    _process_cmd_opts(bglb, bglb.cfs)
    value = str(bglb.with_superlu_mt).lower()
    if value not in ("yes", "true", "1", "no", "false", "0"):
        raise ValueError("with-superlu-mt must be Yes or No")
    bglb.with_superlu_mt = value in ("yes", "true", "1")


def superlu_mt_enabled(config_settings=None):
    """
    Read the native-build switch using the shared option-processing flow.
    """
    from dataclasses import replace
    from build_globals import bglb

    config = replace(bglb, cfs=bglb.cfs if config_settings is None else config_settings)
    configure_build(config)
    return config.with_superlu_mt


def print_config(bglb):
    """
    Display the selected native backend and build tools.
    """
    print("----configuration----")
    print(" build SuperLU_MT : " + ("Yes" if bglb.with_superlu_mt else "No"))
    print(" C compiler : " + str(bglb.cc))
    print(" C++ compiler : " + str(bglb.cxx))
    print(" SWIG : " + str(bglb.swig))
