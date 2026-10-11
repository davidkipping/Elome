"""MetalPlanet is now Elómë: `import metalplanet` is a deprecated alias.

The shim must hand back the very same objects (shared kernel caches, and
monkeypatches through either name reach the code), warn, and keep every
`metalplanet.<module>` import of downstream code working.
"""
import importlib
import subprocess
import sys

import pytest

import elome


def test_alias_is_the_same_package_and_warns():
    sys.modules.pop("metalplanet", None)
    with pytest.warns(DeprecationWarning, match="now 'elome'"):
        import metalplanet
    assert metalplanet.__version__ == elome.__version__
    assert metalplanet.TransitModel is elome.TransitModel
    assert set(metalplanet.__all__) == set(elome.__all__)


@pytest.mark.parametrize("mod", ["anvil", "hybrid", "metal", "metal_oblate",
                                 "oblate", "oblate_tau", "api", "exposure"])
def test_submodules_are_the_same_objects(mod):
    old = importlib.import_module(f"metalplanet.{mod}")
    assert old is importlib.import_module(f"elome.{mod}")


def test_old_import_forms_in_a_fresh_interpreter():
    code = ("import warnings; warnings.simplefilter('error', DeprecationWarning)\n"
            "try:\n    import metalplanet\nexcept DeprecationWarning:\n    pass\n"
            "warnings.simplefilter('ignore')\n"
            "import metalplanet\n"
            "from metalplanet.anvil import make_quad_transit_flux\n"
            "from metalplanet.metal import flux_dev_from_tau\n"
            "import metalplanet.hybrid as H, elome\n"
            "assert H is elome.hybrid\nprint('ok')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
