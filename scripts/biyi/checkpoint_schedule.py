"""Standard-library parsing shared by the formal launcher and PyTorch trainer."""

import re


def parse_checkpoint_steps(values, num_train_steps=None):
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    steps = []
    for value in values:
        if type(value) is int:
            tokens = [str(value)]
        elif isinstance(value, str):
            tokens = re.split(r"[,\s]+", value.strip())
        else:
            raise ValueError("Checkpoint steps must be positive integers")
        for token in tokens:
            if not token:
                continue
            if not re.fullmatch(r"[0-9]+", token) or int(token) <= 0:
                raise ValueError(f"Invalid checkpoint step: {token}")
            step = int(token)
            if num_train_steps is not None and step > num_train_steps:
                raise ValueError(f"Checkpoint step {step} exceeds num_train_steps={num_train_steps}")
            steps.append(step)
    if not steps:
        raise ValueError("Checkpoint step list must not be empty")
    return tuple(sorted(set(steps)))


def normalize_checkpoint_argv(argv):
    """Accept --checkpoint_step 500, 1000 or the canonical Tyro spelling."""
    result = []
    index = 0
    aliases = {"--checkpoint_step", "--checkpoint-step", "--checkpoint_steps", "--checkpoint-steps"}
    while index < len(argv):
        option, separator, inline = argv[index].partition("=")
        if option not in aliases:
            result.append(argv[index])
            index += 1
            continue
        values = [inline] if separator else []
        index += 1
        while index < len(argv) and not argv[index].startswith("--"):
            values.append(argv[index])
            index += 1
        result.extend(["--checkpoint-steps", *map(str, parse_checkpoint_steps(values))])
    return result
