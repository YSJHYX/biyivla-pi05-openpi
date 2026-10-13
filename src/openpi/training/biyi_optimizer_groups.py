"""Biyi Stage-1 AdamW groups: decay trainable matrices, including AdaRMS projections."""

from collections import OrderedDict

import torch


def _is_normalization_scale(name: str, modules: dict[str, torch.nn.Module]) -> bool:
    parent_name, _, leaf_name = name.rpartition(".")
    parent = modules[parent_name]
    return leaf_name in {"weight", "scale"} and type(parent).__name__.lower().endswith("norm")


def stage1_adamw_groups(model: torch.nn.Module, weight_decay: float) -> tuple[list[dict], dict[str, int]]:
    """Assign each distinct trainable tensor once; exclude bias and 0D/1D parameters."""
    modules = dict(model.named_modules(remove_duplicate=False))
    unique: OrderedDict[int, tuple[torch.nn.Parameter, list[str]]] = OrderedDict()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        entry = unique.get(id(parameter))
        if entry is None:
            unique[id(parameter)] = (parameter, [name])
        else:
            entry[1].append(name)

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    frozen_count = 0
    for parameter, names in unique.values():
        if not parameter.requires_grad:
            frozen_count += parameter.numel()
            continue
        norm_scales = [name for name in names if _is_normalization_scale(name, modules)]
        if parameter.ndim >= 2 and norm_scales:
            raise ValueError(f"Stage-1 normalization scale has ndim >= 2; review separately: {norm_scales}")
        no_decay_candidate = parameter.ndim < 2 or any(name.rsplit(".", 1)[-1] == "bias" for name in names)
        (no_decay if no_decay_candidate else decay).append(parameter)

    trainable_ids = {key for key, (parameter, _) in unique.items() if parameter.requires_grad}
    assigned_ids = [id(parameter) for parameter in (*decay, *no_decay)]
    if len(assigned_ids) != len(set(assigned_ids)) or set(assigned_ids) != trainable_ids:
        raise RuntimeError("Stage-1 AdamW groups duplicated or omitted a trainable parameter")
    if sum(parameter.numel() for parameter in (*decay, *no_decay)) != sum(
        parameter.numel() for parameter, _ in unique.values() if parameter.requires_grad
    ):
        raise RuntimeError("Stage-1 AdamW groups do not cover all trainable parameter elements")

    summary = {
        "total_trainable_tensors": len(trainable_ids),
        "decay_tensors": len(decay),
        "no_decay_tensors": len(no_decay),
        "total_trainable_parameters": sum(parameter.numel() for parameter in (*decay, *no_decay)),
        "decay_parameter_count": sum(parameter.numel() for parameter in decay),
        "no_decay_parameter_count": sum(parameter.numel() for parameter in no_decay),
        "frozen_parameter_count": frozen_count,
    }
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], summary
