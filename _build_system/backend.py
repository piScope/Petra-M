"""
Setuptools backend with Petra-M native-build settings, following PyMFEM/DS.
"""
from setuptools import build_meta as _orig
from setuptools.build_meta import *

from build_config import superlu_mt_enabled
from build_globals import bglb


def get_requires_for_build_wheel(config_settings=None):
    """
    Request native tools only when the native component is enabled.
    """
    bglb.cfs = dict(config_settings or {})
    requirements = _orig.get_requires_for_build_wheel(config_settings)
    if superlu_mt_enabled():
        requirements += ["numpy>=2.0", "cmake>=3.21", "swig>=4.3"]
    return requirements


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """
    Pass frontend settings to setup.py before building the wheel.
    """
    bglb.cfs = dict(config_settings or {})
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """
    Keep metadata configuration consistent with the requested wheel.
    """
    bglb.cfs = dict(config_settings or {})
    return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)
