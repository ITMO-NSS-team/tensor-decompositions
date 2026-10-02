"""Compare native build metadata and smoke-test the wheel outside the checkout."""
from argparse import ArgumentParser
from email import message_from_bytes
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from packaging.specifiers import SpecifierSet


def metadata(path):
    with ZipFile(path) as archive:
        names = archive.namelist()
        metadata_path, = [name for name in names if name.endswith(".dist-info/METADATA")]
        result = message_from_bytes(archive.read(metadata_path))
        if any("__pycache__" in name or ".egg-info/" in name for name in names):
            raise AssertionError("Generated Python artifacts leaked into wheel")
        return result, names


def main():
    parser = ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist_wheel", type=Path)
    parser.add_argument("--require-license", action="store_true")
    args = parser.parse_args()
    original, names = metadata(args.wheel)
    rebuilt, rebuilt_names = metadata(args.sdist_wheel)
    for field in ("Name", "Version", "Requires-Python", "Requires-Dist", "Provides-Extra", "License", "License-Expression", "License-File", "Classifier"):
        assert original.get_all(field, []) == rebuilt.get_all(field, []), field
    assert original["Version"] == "0.2.4"
    assert SpecifierSet(original["Requires-Python"]) == SpecifierSet(">=3.10,<3.13")
    assert any(name.endswith("tdecomp/api.py") for name in names)
    if args.require_license:
        assert original.get_all("License-File"), "License text is required before release"
        assert all(any(name.endswith("/" + license_name) for name in names) for license_name in original.get_all("License-File"))
    smoke = '''
import importlib, pkgutil, sys
from pathlib import Path
import numpy as np
import torch
import tensorly as tl
tl.set_backend('numpy')
import tdecomp
assert Path(tdecomp.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
from tdecomp.api import SVDRequest, compute_svd, save_svd, load_svd
for X in (np.eye(3), torch.eye(3, dtype=torch.float64)):
    result = compute_svd(X, SVDRequest(rank=2))
    assert result.components == 2
    save_svd(result, 'smoke.npz')
    assert load_svd('smoke.npz').components == 2
from tdecomp.matrix.functional import rsvd
with tl.backend_context('numpy'):
    assert rsvd(np.eye(3), rank=2, random_state=7)[1].shape == (2,)
for item in pkgutil.walk_packages(tdecomp.__path__, prefix='tdecomp.'):
    importlib.import_module(item.name)
assert tl.get_backend() == 'numpy'
print('Native installed wheel smoke passed:', tdecomp.__file__)
'''
    with TemporaryDirectory(prefix="tdecomp-installed-") as temporary:
        root = Path(temporary)
        installed = root / "site"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(installed), str(args.wheel.resolve())], check=True)
        # -I bypasses the checkout and PYTHONPATH. Add only this installed target.
        isolated = "import sys; sys.path.insert(0, sys.argv[1]); " + smoke
        subprocess.run([sys.executable, "-I", "-c", isolated, str(installed)], cwd=root, check=True)
    print("Wheel/sdist metadata match; installed native public paths pass")


if __name__ == "__main__":
    main()
