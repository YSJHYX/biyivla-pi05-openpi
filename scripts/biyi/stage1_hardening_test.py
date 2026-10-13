"""CPU regression tests for Stage-1 pre-smoke infrastructure."""

import dataclasses

import numpy as np
import pytest
import torch

from openpi import transforms
from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from scripts import train_pytorch
from scripts.train_pytorch import _mean_loss_for_logging
from scripts.train_pytorch import _should_save_checkpoint


class IndexDataset:
    def __len__(self):
        return 32

    def __getitem__(self, index):
        return {"sample_id": np.int64(index), "actions": np.zeros((1, 1), dtype=np.float32)}


@pytest.mark.parametrize("rank", [0, 1])
def test_distributed_loader_advances_reproducible_epochs(monkeypatch, rank):
    dataset = IndexDataset()
    monkeypatch.setattr(_data_loader, "create_torch_dataset", lambda *_: dataset)
    monkeypatch.setattr(_data_loader, "transform_dataset", lambda dataset, *_, **__: dataset)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(_model.Observation, "from_dict", staticmethod(lambda batch: batch))

    def make_loader():
        return _data_loader.create_torch_data_loader(
            _config.DataConfig(),
            _config.get_config("pi05_biyi_stage1").model,
            action_horizon=50,
            batch_size=8,
            shuffle=True,
            seed=42,
            framework="pytorch",
        )

    def read_pass(iterator, num_batches):
        return [sample for _ in range(num_batches) for sample in next(iterator)[0]["sample_id"].tolist()]

    loader = make_loader()
    assert len(loader) == 4
    iterator = iter(loader)
    orders = [read_pass(iterator, len(loader)) for _ in range(3)]
    assert len({tuple(order) for order in orders}) == 3
    expected_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=2, rank=rank, shuffle=True, drop_last=True, seed=42
    )
    for epoch, order in enumerate(orders):
        expected_sampler.set_epoch(epoch)
        assert order == list(expected_sampler)

    replay = make_loader()
    replay_iterator = iter(replay)
    assert [read_pass(replay_iterator, len(replay)) for _ in range(3)] == orders
    replay.set_epoch(1)
    assert read_pass(iter(replay), len(replay)) == orders[1]


@pytest.mark.parametrize("repo_id", ["org/repo-a", "org/repo-b", ""])
def test_stage1_asset_id_is_independent_of_repo(monkeypatch, tmp_path, repo_id):
    monkeypatch.setattr(_config.DataConfigFactory, "_load_norm_stats", lambda *_: None)
    monkeypatch.setattr(_config, "ModelTransformFactory", lambda: lambda _: transforms.Group())
    config = _config.get_config("pi05_biyi_stage1")
    factory = dataclasses.replace(config.data, repo_id=repo_id)
    data = factory.create(tmp_path, config.model)
    assert data.repo_id == repo_id
    assert data.asset_id == "biyi_stage1"


@pytest.mark.parametrize("num_train_steps", [10000, 10500])
def test_checkpoint_steps_include_exact_final_step(num_train_steps):
    saves = [step for step in range(num_train_steps + 1) if _should_save_checkpoint(step, num_train_steps, 1000)]
    expected = list(range(1000, 10001, 1000))
    if num_train_steps == 10500:
        expected.append(10500)
    assert saves == expected
    assert 999 not in saves
    assert 9999 not in saves


@pytest.mark.parametrize("distributed", [False, True])
def test_logging_loss_is_detached_global_mean(monkeypatch, distributed):
    calls = []

    def reduce_with_other_rank(log_loss, op):
        assert not log_loss.requires_grad
        assert log_loss.ndim == 0
        assert op == torch.distributed.ReduceOp.SUM
        calls.append(True)
        log_loss.add_(6.0)

    monkeypatch.setattr(train_pytorch.dist, "all_reduce", reduce_with_other_rank)
    loss = torch.tensor(2.0, requires_grad=True)
    result = _mean_loss_for_logging(loss, distributed=distributed, world_size=2 if distributed else 1)
    assert result == (4.0 if distributed else 2.0)
    assert len(calls) == int(distributed)
    assert loss.item() == 2.0
    loss.backward()
    assert loss.grad.item() == 1.0
