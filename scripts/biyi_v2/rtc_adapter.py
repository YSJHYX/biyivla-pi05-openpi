import numpy as np
import torch

from openpi import transforms as _transforms
from openpi.models_pytorch.rtc import sample_actions_rtc


def prepare_prefix(
    action_prefix,
    normalize_actions,
    action_horizon=50,
    model_action_dim=32,
):
    # TODO:将 action_prefix 转为 float32 NumPy 数组,并复制
    action_prefix = np.asarray(action_prefix, dtype=np.float32).copy()
    # TODO:检查形状为 [prefix_length,17],且 1 <= prefix_length <= action_horizon
    if not (action_prefix.ndim == 2 and action_prefix.shape[1] == 17):
        raise ValueError("Action_prefix must be a 2D array.")
    # TODO:记录有效前缀长度 prefix_length
    prefix_length = action_prefix.shape[0]
    # TODO:检查所有值都是有限值;不满足时抛出 ValueError
    if (not (prefix_length >= 1 and action_horizon >= prefix_length)) or (not np.isfinite(action_prefix).all()):
        raise ValueError("Action_prefix is not an finite array.")
    # TODO:使用传入的 normalize_actions 对物理动作进行归一化
    # 提示:它接收 {"actions": 数组},返回字典
    #       从返回字典中取出 "actions"
    normalized_prefix = normalize_actions({"actions": action_prefix})["actions"]
    # TODO:创建全零 float32 数组,形状为
    #       (action_horizon, model_action_dim)
    new_array = np.zeros((action_horizon, model_action_dim), dtype=np.float32)
    # TODO:将归一化后的前缀填入前 prefix_length 行、前 17 列
    new_array[:prefix_length, :17] = normalized_prefix
    # TODO:返回 (填充后的数组, prefix_length)
    return (new_array, prefix_length)


def get_action_normalizer(policy):
    # TODO:遍历 policy._input_transform.transforms
    for transform in policy._input_transform.transforms:  # noqa: SLF001 -- reuse checkpoint normalization
        # TODO:用 isinstance 判断当前对象是否为 _transforms.Normalize
        if isinstance(transform, _transforms.Normalize):
            # TODO:找到后直接返回该对象,供 prepare_prefix 调用
            return transform
    # TODO:遍历结束仍未找到时,抛出 ValueError
    raise ValueError("No Normalize transform found in policy._input_transform.transforms.")


def sample_rtc_chunk(
    model,
    device,
    observation,
    physical_prefix,
    normalize_actions,
    *,
    inference_delay,
    prefix_horizon,
    num_steps=10,
    max_guidance_weight=10.0,
    noise=None,
):
    # TODO:调用 prepare_prefix,取得 padded_prefix 和有效长度 prefix_length
    # action_horizon 使用 model.config.action_horizon
    # model_action_dim 使用 model.config.action_dim
    padded_prefix, prefix_length = prepare_prefix(
        physical_prefix,
        normalize_actions,
        action_horizon=model.config.action_horizon,
        model_action_dim=model.config.action_dim,
    )

    # TODO:检查 0 <= inference_delay <= prefix_horizon <= prefix_length
    # 不满足时抛出 ValueError,避免引导补零的无效时间段
    if not (inference_delay >= 0 and inference_delay <= prefix_horizon and prefix_horizon <= prefix_length):
        raise ValueError("Invalid inference_delay or prefix_horizon.")
    # TODO:将 padded_prefix 转为 device 上的 float32 Torch tensor
    # 增加 batch 维度,使形状成为 [1, action_horizon, model_action_dim]
    action_prefix_tensor = torch.from_numpy(padded_prefix).to(device=device, dtype=torch.float32).unsqueeze(0)
    # TODO:调用 sample_actions_rtc
    # 传入 model、device、observation
    # action_prefix 使用上面的 tensor;physical_action_dim 使用 17
    # 其余参数使用本函数同名参数
    # prefix_attention_schedule 使用 "exp"
    # 注意:它返回 (actions, metrics)
    actions, metrics = sample_actions_rtc(
        model=model,
        device=device,
        observation=observation,
        action_prefix=action_prefix_tensor,
        physical_action_dim=17,
        inference_delay=inference_delay,
        prefix_horizon=prefix_horizon,
        num_steps=num_steps,
        max_guidance_weight=max_guidance_weight,
        prefix_attention_schedule="exp",
        noise=noise,
    )
    # TODO:只返回 actions,保留 Torch tensor 和 batch 维度
    return actions


def predict_rtc_chunk(
    policy,
    observation,
    physical_prefix,
    *,
    inference_delay,
    prefix_horizon,
    max_guidance_weight=10.0,
    noise=None,
):
    # TODO:通过 get_action_normalizer(policy) 取得归一化器
    normalizer = get_action_normalizer(policy)
    # TODO:保存 policy._sample_actions,供调用结束后恢复
    original_sample_actions = policy._sample_actions  # noqa: SLF001 -- per-instance native sampler adaptation

    def rtc_sampler(device, processed_observation, **sample_kwargs):
        # TODO:调用并返回 sample_rtc_chunk 的结果
        # model 使用 policy._model
        # observation 使用 processed_observation
        # 传入 physical_prefix、归一化器及本函数的 RTC 参数
        # 用 **sample_kwargs 转交原生 Policy 传来的 num_steps、noise
        return sample_rtc_chunk(
            model=policy._model,  # noqa: SLF001
            device=device,
            observation=processed_observation,
            physical_prefix=physical_prefix,
            normalize_actions=normalizer,
            inference_delay=inference_delay,
            prefix_horizon=prefix_horizon,
            max_guidance_weight=max_guidance_weight,
            **sample_kwargs,
        )

    # TODO:使用 try / finally
    # try 中:
    #   将当前 policy._sample_actions 临时设置为 rtc_sampler
    #   调用 policy.infer(observation, noise=noise)
    #   返回结果字典中的 "actions"
    # finally 中:
    #   恢复之前保存的 policy._sample_actions
    try:
        policy._sample_actions = rtc_sampler  # noqa: SLF001
        return policy.infer(observation, noise=noise)["actions"]
    finally:
        policy._sample_actions = original_sample_actions  # noqa: SLF001
