"""BiyiVLA physical 17D adapter. State and action use distinct coordinate spaces."""

import dataclasses

import numpy as np

from openpi import transforms


PHYSICAL_DIM = 17


def _image_to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected RGB image, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3 RGB channels, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class BiyiVLAInputs(transforms.DataTransformFn):
    """Map head and wrist; missing right wrist is masked, never duplicated."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape != (PHYSICAL_DIM,):
            raise ValueError(f"Expected physical state[17], got {state.shape}")
        head = _image_to_hwc_uint8(data["images"]["head"])
        wrist = _image_to_hwc_uint8(data["images"]["wrist"])
        result = {
            "state": state,
            "image": {
                "base_0_rgb": head,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(head),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != PHYSICAL_DIM:
                raise ValueError(f"Expected physical actions[...,17], got {actions.shape}")
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class BiyiVLAOutputs(transforms.DataTransformFn):
    """Crop after the stock policy has unnormalized the 32D model output."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] != 32:
            raise ValueError(f"Expected stock actions[...,32], got {actions.shape}")
        return {"actions": actions[..., :PHYSICAL_DIM]}
