"""AdamW with explicit single or composite projections and portable checkpoints."""
import copy
import math
import torch
from torch.optim import Optimizer
from .projectors.projector_utils import get_projector, normalize_group, projector_from_state
from .projectors._common import tensors_to


class TensorGRaD(Optimizer):
    """Ordinary groups match Torch AdamW (including complex real-view moments).

    Groups containing rank use one or two compressed Adam branches. Sequential
    branches receive g - back(first.project(g)); p.grad is never overwritten.
    Scheduled projection refits reset both moments and each branch's bias clock.
    'correct_bias=False' deliberately selects uncorrected compressed/ordinary Adam.
    """
    _method_fields = ("matrix_only", "support_complex", "use_sum", "enforce_full_complex_precision")

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, correct_bias=True, matrix_only=True,
                 support_complex=False, use_sum=False, enforce_full_complex_precision=False,
                 run_name=None, verbose=False):
        self._validate_adam(dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        for name, value in (("matrix_only", matrix_only), ("support_complex", support_complex),
                            ("use_sum", use_sum), ("enforce_full_complex_precision", enforce_full_complex_precision),
                            ("correct_bias", correct_bias)):
            if type(value) is not bool:
                raise ValueError(f"{name} must be bool")
        raw = list(params)
        if raw and isinstance(raw[0], dict):
            groups = [dict(group, params=list(group["params"])) for group in raw]
        else:
            groups = [{"params": raw}]
        validated = []
        for group in groups:
            self._validate_adam(dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay) | group)
            if "rank" in group:
                group = normalize_group(group)
                for p in group["params"]:
                    local = dict(group, dim=p.ndim)
                    projectors = get_projector(local, matrix_only, support_complex)
                    self._validate_shapes(p, projectors, enforce_full_complex_precision)
                if not math.isfinite(group["lambda_sparse"]) or group["lambda_sparse"] < 0:
                    raise ValueError("lambda_sparse must be finite and nonnegative")
            validated.append(group)
        super().__init__(validated, dict(lr=lr, betas=betas, eps=eps,
                                         weight_decay=weight_decay, correct_bias=correct_bias))
        self.matrix_only, self.support_complex, self.use_sum = matrix_only, support_complex, use_sum
        self.enforce_full_complex_precision = enforce_full_complex_precision
        self.run_name, self.verbose = run_name, verbose

    @staticmethod
    def _validate_adam(group):
        if "correct_bias" in group and type(group["correct_bias"]) is not bool:
            raise ValueError("correct_bias must be bool")
        for key in ("lr", "eps", "weight_decay"):
            if not isinstance(group[key], (int, float)) or not math.isfinite(group[key]) or group[key] < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if len(group["betas"]) != 2 or any(not 0 <= beta < 1 for beta in group["betas"]):
            raise ValueError("betas must contain two values in [0, 1)")

    @staticmethod
    def _validate_shapes(parameter, projectors, enforce_full_complex_precision=False):
        for projector in projectors if isinstance(projectors, tuple) else (projectors,):
            projector._check_input(parameter.to(torch.complex64) if enforce_full_complex_precision and parameter.dtype == torch.complex32 else parameter)
            rank = getattr(projector, "rank", None)
            ratio = getattr(projector, "sparse_ratio", None)
            if isinstance(rank, (tuple, list)) and len(rank) != parameter.ndim:
                raise ValueError("tensor rank must have one value per mode")
            if isinstance(ratio, (tuple, list)) and len(ratio) != parameter.ndim:
                raise ValueError("sparse_ratio must have one value per mode")
            if parameter.is_complex() and hasattr(projector, "support_complex") and not projector.support_complex:
                raise ValueError("complex projection requires support_complex=True")

    def _adam_update(self, grad, exp_avg, exp_avg_sq, beta1, beta2, eps, step, correct_bias=True):
        # Complex AdamW treats real and imaginary components independently.
        is_complex = grad.is_complex()
        if is_complex:
            grad, exp_avg, exp_avg_sq = map(torch.view_as_real, (grad, exp_avg, exp_avg_sq))
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias1 = 1 - beta1**step if correct_bias else 1.0
        bias2 = 1 - beta2**step if correct_bias else 1.0
        normalized = exp_avg / (exp_avg_sq.sqrt() / math.sqrt(bias2) + eps) / bias1
        if is_complex:
            normalized = torch.view_as_complex(normalized.contiguous())
        return normalized, 1.0

    def _branch_update(self, state, branch, projector, grad, group):
        avg, square, clock = branch + "_exp_avg", branch + "_exp_avg_sq", branch + "_step"
        if avg not in state or state[avg].shape != grad.shape or projector.should_update:
            state[avg], state[square], state[clock] = torch.zeros_like(grad), torch.zeros_like(grad), 0
        state[clock] += 1
        return self._adam_update(grad, state[avg], state[square], *group["betas"],
                                 group["eps"], state[clock], group["correct_bias"])[0]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Check all groups/gradients before changing any parameter, state or schedule.
        for group in self.param_groups:
            self._validate_adam(group)
            if "rank" in group:
                normalize_group(group)
            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.layout != torch.strided:
                        raise RuntimeError("TensorGRaD does not support sparse gradients")
                    if "rank" in group:
                        if not torch.isfinite(p.grad).all():
                            raise ValueError("projected gradients must be finite")
                        configured = get_projector(dict(group, dim=p.ndim), self.matrix_only, self.support_complex)
                        self._validate_shapes(p, configured, self.enforce_full_complex_precision)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if self.enforce_full_complex_precision and grad.dtype == torch.complex32:
                    grad = grad.to(torch.complex64)
                state = self.state[p]
                step = state.get("step", 0)
                if "rank" in group:
                    if "first_proj" not in state:
                        projectors = get_projector(dict(group, dim=p.ndim), self.matrix_only, self.support_complex)
                        if isinstance(projectors, tuple):
                            state["first_proj"], state["second_proj"] = projectors
                        else:
                            state["first_proj"] = projectors
                    first = state["first_proj"]
                    first_grad = first.project(grad, step)
                    first_update = self._branch_update(state, "first", first, first_grad, group)
                    first_weight = group["lambda_sparse"] if hasattr(first, "sparse_ratio") else 1.0
                    update = first.project_back(first_update, alpha=first_weight)
                    if "second_proj" in state:
                        second = state["second_proj"]
                        second_input = grad if self.use_sum else grad - first.project_back(first_grad)
                        second_grad = second.project(second_input, step)
                        second_update = self._branch_update(state, "second", second, second_grad, group)
                        second_weight = group["lambda_sparse"] if hasattr(second, "sparse_ratio") else 1.0
                        second.project_back(second_update, output_buffer=update, alpha=second_weight, accumulate=True)
                else:
                    if "exp_avg" not in state:
                        state["exp_avg"], state["exp_avg_sq"] = torch.zeros_like(grad), torch.zeros_like(grad)
                    update, _ = self._adam_update(grad, state["exp_avg"], state["exp_avg_sq"],
                        *group["betas"], group["eps"], step + 1, group["correct_bias"])
                # Decoupled decay acts on the old parameter, before the Adam update.
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(dtype=p.dtype), alpha=-group["lr"])
                state["step"] = step + 1
        return loss

    def state_dict(self):
        payload = super().state_dict()
        payload = dict(payload, state={key: dict(value) for key, value in payload["state"].items()})
        projectors = {}
        for parameter_id, state in payload["state"].items():
            for branch in ("first_proj", "second_proj"):
                if branch in state:
                    projectors.setdefault(parameter_id, {})[branch] = state.pop(branch).state_dict()
        for group in payload["param_groups"]:
            if callable(group.get("svd_type")):
                raise ValueError("custom SVD callables cannot be serialized; use a registered SVD name")
        payload["tensorgrad"] = dict(version=1, method={key: getattr(self, key) for key in self._method_fields},
                                     projectors=projectors)
        return copy.deepcopy(payload)

    def load_state_dict(self, state_dict):
        """Versioned portable format only; legacy pickled projector objects are rejected."""
        payload = copy.deepcopy(state_dict)
        metadata = payload.pop("tensorgrad", None)
        if not isinstance(metadata, dict) or metadata.get("version") != 1:
            raise ValueError("unsupported TensorGRaD state version; legacy object checkpoints are not supported")
        if set(metadata.get("method", {})) != set(self._method_fields) or any(type(value) is not bool for value in metadata["method"].values()):
            raise ValueError("incomplete or invalid TensorGRaD method state")
        if not isinstance(metadata.get("projectors"), dict) or "state" not in payload or "param_groups" not in payload:
            raise ValueError("incomplete TensorGRaD state")
        if len(payload["param_groups"]) != len(self.param_groups):
            raise ValueError("checkpoint parameter group count differs")
        parameter_map = {}
        group_map = {}
        restored = {}
        for saved, current in zip(payload["param_groups"], self.param_groups):
            if len(saved["params"]) != len(current["params"]):
                raise ValueError("checkpoint parameter group size differs")
            self._validate_adam(saved)
            if "rank" in saved:
                normalize_group(saved)
            parameter_map.update(zip(saved["params"], current["params"]))
            group_map.update({parameter_id: saved for parameter_id in saved["params"]})
        for parameter_id, branches in metadata["projectors"].items():
            if parameter_id not in parameter_map or not set(branches) <= {"first_proj", "second_proj"} or "first_proj" not in branches:
                raise ValueError("invalid checkpoint projector mapping")
            parameter = parameter_map[parameter_id]
            dtype = torch.complex64 if metadata["method"]["enforce_full_complex_precision"] and parameter.dtype == torch.complex32 else parameter.dtype
            restored[parameter_id] = {name: projector_from_state(data, parameter) for name, data in branches.items()}
            for projector in restored[parameter_id].values():
                projector.load_state_dict(projector.state_dict(), parameter.device, dtype)
                if projector._orig_shape != tuple(parameter.shape):
                    raise ValueError("checkpoint projector shape differs from parameter")
        for parameter_id, saved_state in payload["state"].items():
            if parameter_id not in parameter_map or any(key.endswith("_proj") for key in saved_state):
                raise ValueError("invalid optimizer state mapping or legacy projector object")
            if ("first_exp_avg" in saved_state) != (parameter_id in restored):
                raise ValueError("compressed moment state is missing projector data")
            self._validate_saved_moments(saved_state, parameter_map[parameter_id],
                restored.get(parameter_id), group_map[parameter_id])
        if set(restored) - payload["state"].keys():
            raise ValueError("projector state is missing moment state")
        super().load_state_dict(payload)
        for parameter_id, branches in restored.items():
            self.state[parameter_map[parameter_id]].update(branches)
        self.__dict__.update(metadata["method"])
        if self.enforce_full_complex_precision:
            for parameter, state in self.state.items():
                if parameter.dtype == torch.complex32:
                    for key in list(state):
                        if isinstance(state[key], torch.Tensor) and state[key].is_complex():
                            state[key] = state[key].to(torch.complex64)

    @staticmethod
    def _validate_saved_moments(state, parameter, projectors, group):
        if type(state.get("step")) is not int or state["step"] <= 0:
            raise ValueError("optimizer step must be a positive integer")
        if projectors is None:
            if "rank" in group:
                raise ValueError("projected group is missing projector state")
            required = {"step", "exp_avg", "exp_avg_sq"}
            shapes = {"exp_avg": tuple(parameter.shape), "exp_avg_sq": tuple(parameter.shape)}
        else:
            expected_count = 2 if group.get("projection_mode", "composite") == "composite" else 1
            if len(projectors) != expected_count:
                raise ValueError("checkpoint projection mode and branches differ")
            required, shapes = {"step"}, {}
            for name, projector in projectors.items():
                branch = name.removesuffix("_proj")
                clock = branch + "_step"
                if type(state.get(clock)) is not int or not 0 < state[clock] <= state["step"]:
                    raise ValueError("compressed branch step must be positive and no greater than optimizer step")
                shape = TensorGRaD._compressed_shape(projector, parameter)
                required.update({clock, branch + "_exp_avg", branch + "_exp_avg_sq"})
                shapes.update({branch + "_exp_avg": shape, branch + "_exp_avg_sq": shape})
        if set(state) != required:
            raise ValueError("incomplete or incompatible optimizer moment state")
        for name, shape in shapes.items():
            value = state[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not (value.is_floating_point() or value.is_complex()) or value.is_complex() != parameter.is_complex():
                raise ValueError(f"incompatible moment tensor: {name}")
            if not torch.isfinite(value).all():
                raise ValueError(f"nonfinite moment tensor: {name}")
            if name.endswith("_sq"):
                real_value = torch.view_as_real(value) if value.is_complex() else value
                if (real_value < 0).any():
                    raise ValueError("second moment must be nonnegative")

    @staticmethod
    def _compressed_shape(projector, parameter):
        from .projectors.galore_projector import GaLoreProjector
        from .projectors.tensor_lowrank_projector import TensorGradLowRankProjector
        from .projectors.tensor_unstructured_sparse_projector import TensorGradUnstructuredProjector
        if isinstance(projector, GaLoreProjector):
            basis = projector.ortho_matrix
            m, n = parameter.shape[0], parameter.numel() // parameter.shape[0]
            if projector.galore_2d_proj_type == "left":
                if not isinstance(basis, torch.Tensor) or basis.ndim != 2 or basis.shape[0] != m:
                    raise ValueError("incompatible left projection basis")
                return (basis.shape[1], n)
            if projector.galore_2d_proj_type == "right":
                if not isinstance(basis, torch.Tensor) or basis.ndim != 2 or basis.shape[1] != n:
                    raise ValueError("incompatible right projection basis")
                return (m, basis.shape[0])
            if not isinstance(basis, list) or len(basis) != 2 or any(not isinstance(b, torch.Tensor) or b.ndim != 2 for b in basis) or basis[0].shape[0] != m or basis[1].shape[1] != n:
                raise ValueError("incompatible full projection basis")
            return (basis[0].shape[1], basis[1].shape[0])
        if isinstance(projector, TensorGradLowRankProjector):
            factors = projector.proj_tensor
            if not isinstance(factors, list) or len(factors) != parameter.ndim or any(not isinstance(f, torch.Tensor) or f.ndim != 2 or f.shape[0] != size for f, size in zip(factors, parameter.shape)):
                raise ValueError("incompatible Tucker projection factors")
            return tuple(f.shape[1] for f in factors)
        if isinstance(projector, TensorGradUnstructuredProjector):
            TensorGRaD._validate_indices(projector._indices, parameter.numel())
            return (projector._indices.numel(),)
        if not isinstance(projector.indices, list) or len(projector.indices) != parameter.ndim or not isinstance(projector.masks, list) or len(projector.masks) != parameter.ndim:
            raise ValueError("incompatible structured projection masks")
        for index, mask, size in zip(projector.indices, projector.masks, parameter.shape):
            TensorGRaD._validate_indices(index, size)
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (size,) or not torch.equal(mask.nonzero().flatten(), index):
                raise ValueError("mask and selected indices differ")
        return tuple(index.numel() for index in projector.indices)

    @staticmethod
    def _validate_indices(indices, length):
        if not isinstance(indices, torch.Tensor) or indices.dtype != torch.long or indices.ndim != 1 or indices.numel() == 0 or indices.unique().numel() != indices.numel() or (indices < 0).any() or (indices >= length).any():
            raise ValueError("invalid projection indices")
