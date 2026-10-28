"""117-episode lifecycle coverage; legacy 100-episode tests remain unchanged."""
# ruff: noqa: PT009, PT027

import copy
import os
from pathlib import Path
import unittest
from unittest import mock

from scripts.biyi import checkpoint_store as store
from scripts.biyi import checkpoint_store_test as fixtures
from scripts.biyi import train_pi05_100ep as launcher
from scripts.biyi import training_runtime as runtime


class Lifecycle117Test(fixtures.LifecycleTest):
    def setUp(self):
        super().setUp()
        self.settings = store.recipe(store.RECIPE_117)
        old = self.run
        self.run = self.root / "checkpoints/pi05_biyi_stage1" / self.settings["experiment"]
        self.run.parent.mkdir(parents=True)
        old.rename(self.run)
        self.ckpt = self.run / "1000"
        self.record["kind"] = "formal_117ep_fft"
        self.record["recipe"] = self.settings
        self.record["dataset"]["episodes"] = list(range(117))
        self.record["dataset"]["revision"] = self.settings["dataset_revision"]
        store.write_json(self.run / "run_manifest.json", self.record)
        path = self.run / "train_stats_manifest.json"
        stats = store.read_json(path)
        stats["train_episode_indices"] = list(range(117))
        store.write_json(path, stats)

    # Old tests below assert the legacy recipe's exact name/count or use a
    # deliberately different data revision. The 117 variants are explicit here.
    def test_plan_is_single_gpu_stock_and_resume_is_explicit(self):
        with mock.patch.dict(os.environ, {"BIYI_BASE_CHECKPOINT_DIR": "/persistent/base"}):
            plan = launcher.build_plan("/mnt/persistent/biyi", self.settings["dataset_revision"], self.settings)
        self.assertEqual(plan["environment"]["BIYI_TRAIN_EPISODES"], ",".join(map(str, range(117))))
        for key, value in (
            ("--batch-size", "8"),
            ("--num-train-steps", "20000"),
            ("--save-interval", "2000"),
            ("--lr-schedule.warmup-steps", "2000"),
        ):
            self.assertEqual(plan["train"][plan["train"].index(key) + 1], value)
        with self.assertRaisesRegex(ValueError, "revision"):
            launcher.build_plan("/mnt/persistent/biyi", "b" * 40, self.settings)

    def test_count_kind_name_and_provenance_cannot_cross(self):
        for count in (100, 116, 118):
            bad = copy.deepcopy(self.record)
            bad["dataset"]["episodes"] = list(range(count))
            with self.assertRaises(ValueError):
                store.validate_run(bad)
        bad = copy.deepcopy(self.record)
        bad["kind"] = "formal_100ep_fft"
        with self.assertRaisesRegex(ValueError, "kind"):
            store.validate_run(bad)

    def test_durable_backup_seals_117_checkpoint(self):
        environment = {"BIYI_STORAGE_MODE": "backup", "BIYI_BACKUP_ROOT": str(self.root / "durable")}
        with mock.patch.object(store, "storage_root", side_effect=Path):
            backup = runtime.backup_checkpoint(self.run, 1000, environment=environment)
        self.assertTrue((backup / "1000/checkpoint_manifest.json").is_file())
        target = self.root / "restored" / self.settings["experiment"]
        runtime.restore_backup(backup, target)
        store.inspect_checkpoint(target, 1000)


if __name__ == "__main__":
    unittest.main()
