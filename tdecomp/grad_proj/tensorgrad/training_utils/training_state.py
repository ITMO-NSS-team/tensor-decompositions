"""Versioned ordinary CPU/Torch checkpoints using the standard state_dict protocol."""
import copy
import os
import random
import tempfile
import uuid
from pathlib import Path
from typing import NamedTuple
import numpy as np
import torch
import torch.distributed as dist


class TrainingState(NamedTuple):
    model: object
    optimizer: object
    scheduler: object
    regularizer: object
    epoch: object


_COMPONENTS = ("model", "optimizer", "scheduler", "regularizer")


def _cpu_path(distributed=False):
    if distributed or dist.is_initialized():
        raise NotImplementedError("ordinary training_state supports a single process; use a separately validated distributed checkpoint implementation")


def _rng_state():
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(),
            np_state[2], np_state[3], np_state[4]], "torch": torch.random.get_rng_state(),
            "cuda": []}  # This ordinary format captures CPU RNG only; no CUDA initialization.


def _set_rng(state):
    if state.get("cuda"):
        raise ValueError("CUDA RNG restoration is outside the ordinary CPU checkpoint profile")
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    torch.random.set_rng_state(state["torch"].cpu())


def _atomic_save(value, path):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_training_state(save_dir, save_name, model=None, optimizer=None, scheduler=None,
                        regularizer=None, epoch=None, save_rng=True):
    """Save one checkpoint per directory. None is explicit absence, including epoch.

    New generation filenames keep the previous complete checkpoint valid if a
    write fails; the manifest is published last. Process RNG covers CPU only.
    """
    _cpu_path()
    if not isinstance(save_name, str) or not save_name or Path(save_name).name != save_name:
        raise ValueError("save_name must be a nonempty filename stem")
    if epoch is not None and (isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0):
        raise ValueError("epoch must be None or a nonnegative integer")
    components = dict(model=model, optimizer=optimizer, scheduler=scheduler, regularizer=regularizer)
    # Build every payload before touching the directory or writing its manifest.
    payloads = {}
    for name, component in components.items():
        if component is not None:
            if not callable(getattr(component, "state_dict", None)) or not callable(getattr(component, "load_state_dict", None)):
                raise TypeError(f"{name} must implement state_dict and load_state_dict")
            payloads[name] = component.state_dict()
    directory = Path(save_dir)
    directory.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    manifest = {"version": 1, "save_name": save_name, "epoch": epoch,
                "components": {name: f"{save_name}_{generation}_{name}.pt" if name in payloads else None for name in _COMPONENTS},
                "rng": _rng_state() if save_rng else None}
    for name, payload in payloads.items():
        _atomic_save(payload, directory / manifest["components"][name])
    _atomic_save(manifest, directory / "manifest.pt")


def load_training_state(save_dir, save_name, model=None, optimizer=None, scheduler=None,
                        regularizer=None, map_location=None, distributed=False, restore_rng=True):
    """Restore into supplied components and return the backward-compatible named tuple.

    A missing optional component is left unchanged. An existing saved component
    can be skipped by passing None. Unknown/incomplete manifests fail before any
    load; component load failures roll back every supplied state and process RNG.
    Legacy manifests/custom save_checkpoint protocols are rejected explicitly.
    """
    _cpu_path(distributed)
    directory = Path(save_dir)
    manifest = torch.load(directory / "manifest.pt", map_location="cpu", weights_only=True)
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("unsupported training manifest version; legacy manifests are not supported")
    if not {"version", "save_name", "epoch", "components", "rng"} <= manifest.keys():
        raise ValueError("incomplete training manifest")
    if manifest["save_name"] != save_name or set(manifest["components"]) != set(_COMPONENTS):
        raise ValueError("incompatible checkpoint name or component manifest")
    if restore_rng and manifest["rng"] is not None and manifest["rng"].get("cuda"):
        raise ValueError("CUDA RNG restoration is outside the ordinary CPU checkpoint profile")
    epoch = manifest["epoch"]
    if epoch is not None and (isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0):
        raise ValueError("invalid epoch in manifest")
    components = dict(model=model, optimizer=optimizer, scheduler=scheduler, regularizer=regularizer)
    loaded = {}
    for name, filename in manifest["components"].items():
        if filename is not None:
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("manifest component filenames must be local filenames")
            if not (directory / filename).is_file():
                raise ValueError(f"missing checkpoint component: {name}")
            # Read all files, including skipped components, to detect partial saves early.
            payload = torch.load(directory / filename, map_location=map_location or "cpu", weights_only=True)
            if components[name] is not None:
                loaded[name] = payload
    backups = {name: copy.deepcopy(components[name].state_dict()) for name in loaded}
    rng_backup = _rng_state()
    try:
        for name, payload in loaded.items():
            components[name].load_state_dict(payload)
        if restore_rng and manifest["rng"] is not None:
            _set_rng(manifest["rng"])
    except Exception:
        for name, state in backups.items():
            components[name].load_state_dict(state)
        _set_rng(rng_backup)
        raise
    return TrainingState(model, optimizer, scheduler, regularizer, epoch)
