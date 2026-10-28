"""Offline launch/storage regressions. Fixtures are not model acceptance evidence."""
# ruff: noqa: PT009, PT027

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.biyi import checkpoint_store as store
from scripts.biyi import checkpoint_store_test as fixtures
from scripts.biyi import train_pi05_100ep as formal
from scripts.biyi import training_runtime as runtime


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {
            "BIYI_PYTHON": sys.executable,
            "BIYI_BASE_CHECKPOINT_DIR": "/persistent/base",
            "BIYI_DATASET_REPO_ID": store.DATASET_REPO,
            "BIYI_DATASET_LOCAL_ROOT": str(self.root / "dataset"),
            "BIYI_ASSETS_BASE_DIR": str(self.root / "assets"),
            "BIYI_CHECKPOINT_BASE_DIR": str(self.root / "checkpoints"),
            "BIYI_DATASET_REVISION": "b" * 40,
            "BIYI_EXP_NAME": "portable_test",
            "BIYI_NUM_TRAIN_STEPS": "10000",
            "BIYI_NUM_GPUS": "1",
            "BIYI_BATCH_SIZE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }

    def shell(self, *args, updates=None):
        env = {**os.environ, **self.env, **(updates or {})}
        return subprocess.run(
            ["bash", "scripts/biyi/train_pi05_stage1.sh", *args],
            cwd=store.REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_one_two_four_gpu_shell_plans_and_resume(self):
        for count, batch in ((1, 1), (2, 4), (4, 8)):
            for resume in ("0", "1"):
                with self.subTest(count=count, resume=resume):
                    result = self.shell(
                        "train",
                        "--dry-run",
                        updates={
                            "BIYI_NUM_GPUS": str(count),
                            "BIYI_BATCH_SIZE": str(batch),
                            "BIYI_RESUME": resume,
                        },
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    plan = json.loads(result.stdout)
                    command = plan["train"]
                    self.assertEqual(plan["batch_per_gpu"], batch // count)
                    self.assertEqual("--resume" in command, resume == "1")
                    self.assertEqual("torch.distributed.run" in command, count > 1)
                    self.assertEqual(f"--nproc_per_node={count}" in command, count > 1)
        self.assertFalse((self.root / "checkpoints").exists())

    def test_formal_shell_and_direct_plan_use_same_stock_trainer(self):
        result = self.shell(
            "train",
            "--dry-run",
            updates={
                "BIYI_EXP_NAME": "pi05_biyi_100ep_fft_v2",
                "BIYI_STORAGE_MODE": "backup",
                "BIYI_BACKUP_ROOT": "/mnt/persistent/backups",
                "BIYI_NUM_GPUS": "2",
                "BIYI_BATCH_SIZE": "4",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["environment"]["BIYI_DATASET_LOCAL_ROOT"], self.env["BIYI_DATASET_LOCAL_ROOT"])
        self.assertIn("--nproc_per_node=2", plan["train"])
        self.assertIn("scripts/train_pytorch.py", plan["train"])
        self.assertEqual(plan["environment"]["BIYI_CHECKPOINT_IO"], "1")
        self.assertFalse((self.root / "checkpoints").exists())

    def test_smoke_and_formal_dry_runs_never_import_torch(self):
        trap = self.root / "import-trap"
        trap.mkdir()
        (trap / "torch.py").write_text("raise RuntimeError('Dry-run imported torch')\n")
        for profile, extra in (
            ("gpu-smoke", {"BIYI_NUM_TRAIN_STEPS": "3"}),
            ("train", {}),
            ("train", {"BIYI_EXP_NAME": "pi05_biyi_100ep_fft_v1", "BIYI_STORAGE_MODE": "backup"}),
        ):
            result = self.shell(profile, "--dry-run", updates={"PYTHONPATH": str(trap), **extra})
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "checkpoints").exists())
        result = self.shell("train", "--dry-run", updates={"BIYI_NUM_GPUS": "auto", "PYTHONPATH": str(trap)})
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Dry-run imported torch", result.stderr)

    def test_invalid_batch_and_gpu_counts_rejected_before_launch(self):
        for count, batch in (("2", "3"), ("0", "1"), ("-1", "2"), ("bad", "2"), ("1", "0")):
            result = self.shell("train", "--dry-run", updates={"BIYI_NUM_GPUS": count, "BIYI_BATCH_SIZE": batch})
            self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "checkpoints").exists())

    def test_auto_visibility_and_explicit_count_bounds_without_cuda(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 3))
        with mock.patch.dict(sys.modules, {"torch": fake}):
            self.assertEqual(runtime.gpu_count("auto"), 3)
            self.assertEqual(runtime.gpu_count(2, verify=True), 2)
            with self.assertRaisesRegex(ValueError, "visible"):
                runtime.gpu_count(4, verify=True)
            fake.cuda.device_count = lambda: 0
            with self.assertRaisesRegex(ValueError, "visible"):
                runtime.gpu_count("auto")

    def test_unset_temporary_and_overlapping_storage_rejected(self):
        with self.assertRaisesRegex(ValueError, "BIYI_STORAGE_MODE"):
            runtime.storage_policy(self.root / "checkpoints", environment={})
        with self.assertRaisesRegex(ValueError, "persistent"):
            runtime.storage_policy("/tmp/runtime", environment={"BIYI_STORAGE_MODE": "persistent"})
        with self.assertRaisesRegex(ValueError, "BIYI_BACKUP_ROOT"):
            runtime.storage_policy("/tmp/runtime", environment={"BIYI_STORAGE_MODE": "backup"})
        with (
            mock.patch.object(store, "storage_root", side_effect=lambda p: Path(p).resolve()),
            self.assertRaisesRegex(ValueError, "overlapping"),
        ):
            runtime.storage_policy(
                self.root,
                environment={"BIYI_STORAGE_MODE": "backup", "BIYI_BACKUP_ROOT": str(self.root / "backup")},
            )
        with self.assertRaisesRegex(ValueError, "Mount Google Drive"):
            store.storage_root("/content/drive/MyDrive/biyi")

    def test_storage_probe_reads_back_and_cleans_temporary_files(self):
        with mock.patch.object(store, "storage_root", side_effect=lambda p: Path(p).resolve()):
            policy = runtime.storage_policy(
                self.root / "runtime",
                environment={
                    "BIYI_STORAGE_MODE": "backup",
                    "BIYI_BACKUP_ROOT": str(self.root / "durable"),
                },
                probe=True,
            )
        self.assertEqual(policy["mode"], "backup")
        self.assertEqual(list((self.root / "runtime").iterdir()), [])
        self.assertEqual(list((self.root / "durable").iterdir()), [])

    def test_disappeared_or_changed_persistent_mount_is_rejected(self):
        durable = self.root / "durable"
        durable.mkdir()
        with (
            mock.patch.object(store, "storage_root", side_effect=lambda p: Path(p).resolve()),
            self.assertRaisesRegex(ValueError, "disappeared or changed"),
        ):
            runtime.storage_policy(
                self.root / "runtime",
                environment={
                    "BIYI_STORAGE_MODE": "backup",
                    "BIYI_BACKUP_ROOT": str(durable),
                    "BIYI_STORAGE_DEVICE": "-1",
                },
            )

    def test_memory_smoke_gate_binds_actual_hardware_batch_and_provenance(self):
        signature = {"source_commit": "a" * 40, "batch_size": 1, "num_gpus": 1}
        record = {
            "signature": signature,
            "completed_steps": 3,
            "gpu_name": "fixture GPU",
            "total_memory": 80,
            "peak_allocated": 40,
            "peak_reserved": 50,
        }
        path = self.root / "memory_smoke_rank0.json"
        store.write_json(path, record)
        fake = SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_properties=lambda rank: SimpleNamespace(name="fixture GPU", total_memory=80)
            )
        )
        with (
            mock.patch.dict(sys.modules, {"torch": fake}),
            mock.patch.object(runtime, "smoke_signature", return_value=signature),
        ):
            with self.assertRaisesRegex(ValueError, "gpu-smoke"):
                runtime.check_memory_smoke(1, 1, environment={})
            env = {"BIYI_MEMORY_SMOKE_RUN_DIR": str(self.root)}
            runtime.check_memory_smoke(1, 1, environment=env)
            record["total_memory"] = 40
            store.write_json(path, record)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                runtime.check_memory_smoke(1, 1, environment=env)
            record["total_memory"] = 80
            record["signature"]["batch_size"] = 2
            store.write_json(path, record)
            with (
                mock.patch.object(runtime, "smoke_signature", return_value={**signature, "batch_size": 1}),
                self.assertRaisesRegex(ValueError, "differ"),
            ):
                runtime.check_memory_smoke(1, 1, environment=env)

    def test_formal_guard_failure_does_not_call_preflight_or_trainer(self):
        with (
            mock.patch.dict(
                os.environ, {**self.env, "BIYI_STORAGE_MODE": "backup", "BIYI_BACKUP_ROOT": "/mnt/persistent/backup"}
            ),
            mock.patch.object(
                sys, "argv", ["formal", "train", "--persistent-root", "/tmp/run", "--dataset-revision", "b" * 40]
            ),
            mock.patch.object(runtime, "gpu_count", return_value=1),
            mock.patch.object(runtime, "storage_policy", side_effect=ValueError("failed storage probe")),
            mock.patch.object(formal.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(ValueError, "failed storage"):
                formal.main()
            run.assert_not_called()


class DurableLifecycleTest(unittest.TestCase):
    def setUp(self):
        # Reuse existing stock-shaped, byte-only checkpoint fixture.
        fixture = fixtures.LifecycleTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        self.run = self.root / "runtime/checkpoints/pi05_biyi_stage1" / fixture.run.name
        self.run.parent.mkdir(parents=True)
        fixture.run.rename(self.run)
        self.env = {"BIYI_STORAGE_MODE": "backup", "BIYI_BACKUP_ROOT": str(self.root / "durable")}
        patch = mock.patch.object(store, "storage_root", side_effect=lambda p: Path(p).resolve())
        patch.start()
        self.addCleanup(patch.stop)

    def test_complete_checkpoint_backup_restore_and_resume_seal(self):
        target = runtime.backup_checkpoint(self.run, 1000, environment=self.env)
        self.assertEqual(runtime.backup_checkpoint(self.run, 1000, environment=self.env), target)
        restored = self.root / "restored" / self.run.name
        runtime.restore_backup(target, restored)
        store.seal_checkpoint(restored, 1000)
        for name in ("model.safetensors", "optimizer.pt", "metadata.pt", store.NORM):
            self.assertEqual(store.digest(self.run / "1000" / name), store.digest(restored / "1000" / name))
        with self.assertRaisesRegex(ValueError, "overwrites"):
            runtime.restore_backup(target, restored)

    def test_backup_failure_never_publishes_a_partial_save(self):
        with (
            mock.patch.object(runtime.shutil, "copyfile", side_effect=OSError("destination full")),
            self.assertRaisesRegex(OSError, "full"),
        ):
            runtime.backup_checkpoint(self.run, 1000, environment=self.env)
        self.assertFalse(list((self.root / "durable").glob("**/step-1000")))
        self.assertFalse(list((self.root / "durable").glob("**/.backup-*")))

    def test_corruption_and_missing_optimizer_block_restore(self):
        target = runtime.backup_checkpoint(self.run, 1000, environment=self.env)
        (target / "1000/optimizer.pt").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "checksum"):
            runtime.restore_backup(target, self.root / "restore" / self.run.name)
        with self.assertRaisesRegex(ValueError, "checksum"):
            runtime.backup_checkpoint(self.run, 1000, environment=self.env)

    def test_unexpected_backup_manifest_paths_block_restore(self):
        target = runtime.backup_checkpoint(self.run, 1000, environment=self.env)
        receipt = store.read_json(target / "backup_manifest.json")
        receipt["files"]["../escape"] = {"sha256": "a" * 64, "bytes": 1}
        store.write_json(target / "backup_manifest.json", receipt)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            runtime.restore_backup(target, self.root / "restore" / self.run.name)

    def test_restore_assets_rebinds_only_verified_dataset_path(self):
        target = runtime.backup_checkpoint(self.run, 1000, environment=self.env)
        restored = self.root / "new/checkpoints" / self.run.name
        dataset = self.root / "new/dataset"
        assets = self.root / "new/assets"
        with mock.patch.object(store, "dataset_identity", return_value=self.fixture.record["dataset"]):
            runtime.restore_backup(target, restored, assets_base=assets, dataset_root=dataset)
        manifest = store.read_json(assets / "pi05_biyi_stage1/biyi_stage1/stage1_train_stats_manifest.json")
        self.assertEqual(manifest["local_root"], str(dataset))
        self.assertEqual(
            (restored / "train_stats_manifest.json").read_bytes(), (self.run / "train_stats_manifest.json").read_bytes()
        )
        with (
            mock.patch.object(store, "dataset_identity", return_value={}),
            self.assertRaisesRegex(ValueError, "frozen dataset"),
        ):
            runtime.restore_backup(
                target, self.root / "other" / self.run.name, assets_base=assets, dataset_root=dataset
            )

    def test_symlinked_backup_payload_rejected(self):
        target = runtime.backup_checkpoint(self.run, 1000, environment=self.env)
        optimizer = target / "1000/optimizer.pt"
        optimizer.unlink()
        optimizer.symlink_to(self.run / "1000/optimizer.pt")
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            runtime.restore_backup(target, self.root / "restore" / self.run.name)


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main()
