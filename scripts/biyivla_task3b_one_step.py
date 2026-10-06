"""One real-data Task 3B optimizer step; no inference or hardware commands."""

import dataclasses
import math
import os
from pathlib import Path

import jax
import safetensors.torch
import torch
from train_pytorch import save_checkpoint

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training.config import get_config
from openpi.training.data_loader import create_data_loader


def main():
    config = get_config("pi05_biyivla_smoke")
    torch.manual_seed(config.seed)
    device = torch.device("cuda:0")
    loader = create_data_loader(config, framework="pytorch", shuffle=False, num_batches=1)
    data_config = loader.data_config()
    observation, actions = next(iter(loader))
    observation = jax.tree.map(lambda x: x.to(device), observation)
    actions = actions.to(device=device, dtype=torch.float32)
    print("BATCH", tuple(observation.state.shape), tuple(actions.shape), flush=True)

    model_config = dataclasses.replace(config.model, dtype=config.pytorch_training_precision)
    model = PI0Pytorch(model_config).to(device)
    model.gradient_checkpointing_enable()
    missing, unexpected = safetensors.torch.load_model(
        model, str(Path(config.pytorch_weight_path) / "model.safetensors"), strict=True
    )
    print("STRICT_LOAD", len(missing), len(unexpected), flush=True)
    model.train()
    print("GPU_AFTER_LOAD_MIB", torch.cuda.memory_allocated() // 2**20, flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    loss = model(observation, actions).mean()
    print(
        "FORWARD",
        float(loss.detach()),
        str(loss.dtype),
        bool(torch.isfinite(loss)),
        "GPU_PEAK_MIB",
        torch.cuda.max_memory_allocated() // 2**20,
        flush=True,
    )
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite loss")
    if os.environ.get("BIYIVLA_FORWARD_ONLY") == "1":
        return
    loss.backward()
    for label, pred in (
        ("action_in_proj", lambda n: n.startswith("action_in_proj.")),
        ("action_out_proj", lambda n: n.startswith("action_out_proj.")),
        ("gemma_expert", lambda n: "gemma_expert" in n),
        ("backbone", lambda n: "paligemma_with_expert.paligemma." in n),
    ):
        found = [(n, p) for n, p in model.named_parameters() if pred(n) and p.grad is not None]
        if not found:
            print("GRAD", label, "MISSING", flush=True)
            continue
        n, p = found[0]
        g = p.grad
        print(
            "GRAD",
            label,
            n,
            "FINITE",
            bool(torch.isfinite(g).all()),
            "NORM",
            float(torch.linalg.vector_norm(g.float())),
            flush=True,
        )
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.clip_gradient_norm)
    print(
        "BACKWARD",
        "GRAD_NORM",
        float(grad_norm),
        "GPU_PEAK_MIB",
        torch.cuda.max_memory_allocated() // 2**20,
        flush=True,
    )
    if not math.isfinite(float(grad_norm)):
        raise RuntimeError("nonfinite gradient norm")
    if os.environ.get("BIYIVLA_BACKWARD_ONLY") == "1":
        return
    old = model.action_out_proj.weight.detach().float().clone()
    optimizer.step()
    delta = torch.linalg.vector_norm(model.action_out_proj.weight.detach().float() - old)
    print(
        "OPTIMIZER_STEP",
        "LR",
        optimizer.param_groups[0]["lr"],
        "DELTA_NORM",
        float(delta),
        "GPU_PEAK_MIB",
        torch.cuda.max_memory_allocated() // 2**20,
        flush=True,
    )
    if not float(delta) > 0:
        raise RuntimeError("optimizer did not change action_out_proj")
    save_checkpoint(
        model,
        optimizer,
        1,
        config,
        is_main=True,
        data_config=data_config,
    )
    print("CHECKPOINT_SAVE", config.checkpoint_dir / "1", flush=True)


if __name__ == "__main__":
    main()
