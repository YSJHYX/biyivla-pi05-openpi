from openpi.policies.policy_config import create_trained_policy
from openpi.training import config as _config


def load_native_policy(
    checkpoint_dir: str,  # checkpoint 目录
    device: str = "cuda:0",
    num_steps: int = 10,
):
    # TODO:获取 pi05_biyi_stage1 配置
    config = _config.get_config("pi05_biyi_stage1")
    # TODO:调用原生 create_trained_policy,并返回 policy
    return create_trained_policy(
        train_config=config,
        checkpoint_dir=checkpoint_dir,
        pytorch_device=device,
        sample_kwargs={"num_steps": num_steps},
    )


def predict_native_chunk(policy, observation: dict):
    # TODO:通过 policy 的公开接口进行一次推理
    result = policy.infer(observation)
    # TODO:取出并返回 actions
    return result["actions"]
