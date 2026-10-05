from dataclasses import asdict
import torch
from .tensorgrad import TensorGRaD
from .config import DataConfig, OptimizerConfig, TensorGRaDConfig
from .training_utils.get_scheduler import get_scheduler

SMALLNESS_THRESHOLD = 1000


def is_small(p):
    return p.ndim < 2


def setup_optimizer_and_scheduler(config, model, logging_name=None):
    """Build the selected algorithm. Only ndim>=2 weights receive projections.

    All trainable parameters are included once. Explicit exclude_first_parameter
    puts the first weight in the ordinary AdamW group rather than dropping it.
    """
    # Revalidate mutable dataclasses before any optimizer/scheduler allocation.
    opt = OptimizerConfig(**vars(config.opt))
    data = DataConfig(**vars(config.data))
    if opt.tensor_network_type is not None or opt.tensor_network_chi is not None:
        raise ValueError("tensor network optimization is not implemented")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    base = dict(lr=opt.learning_rate, weight_decay=opt.weight_decay)
    if opt.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(parameters, **base, momentum=opt.momentum)
    elif opt.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(parameters, **base, betas=opt.betas, eps=opt.eps)
    else:
        by_dimension, ordinary = {}, []
        for index, parameter in enumerate(parameters):
            if parameter.ndim < 2 or (opt.exclude_first_parameter and index == 0):
                ordinary.append(parameter)
            else:
                by_dimension.setdefault(parameter.ndim, []).append(parameter)
        settings = asdict(opt)
        settings.update(batch_size=data.batch_size, training_samples=data.n_train, epochs=opt.n_epochs,
                        lambda_sparse=opt.tensorgrad_sum_lambda_sparse)
        groups = [{"params": ordinary}] if ordinary else []
        groups.extend(settings | {"params": values, "dim": dimension} for dimension, values in by_dimension.items())
        optimizer = TensorGRaD(groups, **base, betas=opt.betas, eps=opt.eps,
            matrix_only=opt.naive_galore, support_complex=opt.adamw_support_complex,
            use_sum=opt.optimizer_type == "tensorgrad_sum", run_name=logging_name,
            enforce_full_complex_precision=opt.enforce_full_complex_precision)
    scheduler = get_scheduler(opt.scheduler, optimizer, opt.gamma, opt.scheduler_patience,
                              opt.scheduler_T_max, opt.step_size)
    return optimizer, scheduler
