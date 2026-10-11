"""Deprecated alias: MetalPlanet is now Elómë -- ``import elome``.

Elómë (Quenya: Él "star" + lómë "dimness") is the same package under its
new name. This shim keeps ``import metalplanet`` and every
``metalplanet.<module>`` import working, as the very same objects, so
code written against MetalPlanet runs unchanged; it warns
(DeprecationWarning) and will be removed in a future release.

An environment whose editable install predates the rename maps only the
name ``metalplanet`` (to this folder); ``elome`` is then loaded from the
folder beside it, without touching sys.path. Reinstalling
(``pip install -e .``) registers ``elome`` itself.
"""
import importlib
import importlib.util
import os
import pkgutil
import sys
import warnings

warnings.warn(
    "the 'metalplanet' package is now 'elome' (Elómë); `import metalplanet` "
    "still works but will be removed in a future release -- use "
    "`import elome`.", DeprecationWarning, stacklevel=2)

try:
    import elome as _elome
except ImportError:
    _here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "elome")
    _spec = importlib.util.spec_from_file_location(
        "elome", os.path.join(_here, "__init__.py"),
        submodule_search_locations=[_here])
    if _spec is None:
        raise
    _elome = importlib.util.module_from_spec(_spec)
    sys.modules["elome"] = _elome
    _spec.loader.exec_module(_elome)

from elome import *  # noqa: E402,F401,F403
from elome import __all__, __version__  # noqa: E402,F401

# metalplanet.<module> is elome.<module> (one module object, shared caches)
for _m in pkgutil.iter_modules(_elome.__path__):
    _mod = importlib.import_module(f"elome.{_m.name}")
    sys.modules[f"{__name__}.{_m.name}"] = _mod
    globals()[_m.name] = _mod
