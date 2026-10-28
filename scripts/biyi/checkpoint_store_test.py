"""CPU/standard-library lifecycle tests; synthetic tensors, no model or training."""

# Standard-library tests intentionally run without pytest or the training environment.
# ruff: noqa: PT009, PT027

import ast
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.biyi import checkpoint_store as store
from scripts.biyi import train_pi05_100ep as launcher


def tiny_stock_shaped_file(path):
    header = {}
    cursor = 0
    shapes = {"action_in_proj.weight": [1024, 32], "action_out_proj.weight": [32, 1024]}
    shapes.update({f"fixture.{i}": [1] for i in range(101)})
    for name, shape in shapes.items():
        count = 1
        for dimension in shape:
            count *= dimension
        header[name] = {"shape": shape, "dtype": "BF16", "data_offsets": [cursor, cursor + count * 2]}
        cursor += count * 2
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(cursor))


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = store.recipe()
        self.run = self.root / self.settings["experiment"]
        self.ckpt = self.run / "1000"
        self.ckpt.mkdir(parents=True)
        tiny_stock_shaped_file(self.ckpt / "model.safetensors")
        (self.ckpt / "optimizer.pt").write_bytes(b"synthetic optimizer; never unpickle")
        (self.ckpt / "metadata.pt").write_bytes(b"synthetic metadata; never unpickle")
        norm = self.ckpt / store.NORM
        norm.parent.mkdir(parents=True)
        store.write_json(
            norm,
            {
                "norm_stats": {
                    key: {field: [1.0] * 17 for field in ("mean", "std", "q01", "q99")} for key in ("state", "actions")
                }
            },
        )
        self.record = {
            "kind": "formal_100ep_fft",
            "source_commit": "a" * 40,
            "recipe": self.settings,
            "dataset": {
                "repo_id": store.DATASET_REPO,
                "revision": "b" * 40,
                "episodes": list(range(100)),
                "lock_sha256": "c" * 64,
            },
            "norm_stats_sha256": store.digest(norm),
            "base_model_sha256": "d" * 64,
            "uv_lock_sha256": "e" * 64,
        }
        store.write_json(self.run / "run_manifest.json", self.record)
        store.write_json(
            self.run / "train_stats_manifest.json",
            {
                "local_root": "/old/machine",
                "repo_id": store.DATASET_REPO,
                "train_episode_indices": list(range(100)),
                "norm_stats_sha256": store.digest(norm),
            },
        )
        (self.run / "wandb_id.txt").write_text("synthetic-wandb-id")
        self.report = self.root / "acceptance.json"
        (self.root / "inference.json").write_text('{"fixture_only": true}')
        self.bundle = self.root / "bundle"

    def accepted(self):
        store.seal_checkpoint(self.run, 1000)
        store.write_json(
            self.report,
            {
                "accepted": True,
                "reviewer": "UNIT TEST FIXTURE ONLY",
                "reviewed_at": "2026-10-28T00:00:00Z",
                "checkpoint_manifest_sha256": store.digest(self.ckpt / "checkpoint_manifest.json"),
                "checks": dict.fromkeys(("offline_policy_load", "finite_actions", "action_shape_50_17"), True),
                "evidence": ["inference.json"],
            },
        )

    def pack(self, *, resume=False):
        self.accepted()
        store.pack(self.run, 1000, self.report, self.bundle, include_optimizer=resume)

    def test_inference_excludes_optimizer_and_verifies_offline(self):
        self.pack()
        manifest = store.verify_bundle(self.bundle)
        self.assertEqual(manifest["kind"], "inference")
        self.assertFalse((self.bundle / "optimizer.pt").exists())
        self.assertFalse((self.bundle / "metadata.pt").exists())
        self.assertTrue((self.bundle / store.NORM).is_file())

    def test_complete_resume_package(self):
        self.pack(resume=True)
        self.assertEqual(store.verify_bundle(self.bundle)["kind"], "resume")
        for name in ("optimizer.pt", "metadata.pt", "wandb_id.txt"):
            self.assertTrue((self.bundle / name).is_file())

    def test_tampered_weights_fail(self):
        self.pack()
        with (self.bundle / "model.safetensors").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            store.verify_bundle(self.bundle)

    def test_unexpected_credentials_file_fails(self):
        self.pack()
        (self.bundle / "token").write_text("synthetic non-secret")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            store.verify_bundle(self.bundle)

    def test_checkpoint_bound_acceptance(self):
        self.accepted()
        report = store.read_json(self.report)
        report["checkpoint_manifest_sha256"] = "0" * 64
        store.write_json(self.report, report)
        with self.assertRaisesRegex(ValueError, "another checkpoint"):
            store.pack(self.run, 1000, self.report, self.bundle)

    def test_failed_inference_acceptance_rejected(self):
        self.accepted()
        report = store.read_json(self.report)
        report["checks"]["finite_actions"] = False
        store.write_json(self.report, report)
        with self.assertRaisesRegex(ValueError, "not accepted"):
            store.pack(self.run, 1000, self.report, self.bundle)

    def test_smoke_manifest_rejected(self):
        self.record["recipe"]["experiment"] = "local_smoke"
        store.write_json(self.run / "run_manifest.json", self.record)
        with self.assertRaisesRegex(ValueError, "formal name"):
            store.seal_checkpoint(self.run, 1000)

    def test_lora_or_ahfe_rejected(self):
        for field in ("lora", "ahfe"):
            settings = store.recipe()
            settings[field] = True
            path = self.root / "recipe.json"
            store.write_json(path, settings)
            with self.assertRaisesRegex(ValueError, "stock"):
                store.recipe(path)

    def test_seal_detects_post_acceptance_change(self):
        self.accepted()
        (self.ckpt / "optimizer.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            store.seal_checkpoint(self.run, 1000)

    def test_symlink_escape_rejected(self):
        (self.ckpt / "metadata.pt").unlink()
        (self.ckpt / "metadata.pt").symlink_to(self.root / "inference.json")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            store.inspect_checkpoint(self.run, 1000)
        for name in ("../token", "/token", "a/../token"):
            with self.assertRaises(ValueError):
                store.file_at(self.root, name)

    def test_missing_optimizer_rejected(self):
        (self.ckpt / "optimizer.pt").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            store.seal_checkpoint(self.run, 1000)

    def test_explicit_upload_confirmation_before_network_import(self):
        self.pack()
        with self.assertRaisesRegex(ValueError, "Explicit confirmation"):
            store.upload(self.bundle, "yes")

    def test_upload_private_model_type_parent_and_allowlist(self):
        self.pack()
        api = mock.Mock()
        api.repo_info.return_value = SimpleNamespace(private=True, sha="f" * 40)
        api.list_repo_files.return_value = ["storage_contract.json"]
        api.create_commit.return_value = SimpleNamespace(oid="1" * 40)
        sdk = SimpleNamespace(HfApi=lambda: api, CommitOperationAdd=lambda **kw: kw)
        phrase = f"{store.MODEL_REPO}:{self.settings['experiment']}:1000:inference"
        with mock.patch.dict(sys.modules, {"huggingface_hub": sdk}):
            store.upload(self.bundle, phrase)
        api.repo_info.assert_called_once_with(store.MODEL_REPO, repo_type="model")
        args, kwargs = api.create_commit.call_args
        self.assertEqual(args[0], store.MODEL_REPO)
        self.assertEqual(kwargs["repo_type"], "model")
        self.assertEqual(kwargs["parent_commit"], "f" * 40)
        names = [operation["path_in_repo"] for operation in kwargs["operations"]]
        self.assertTrue(all(name.startswith(f"inference/{self.settings['experiment']}/step-1000/") for name in names))
        self.assertFalse(any("optimizer.pt" in name or "metadata.pt" in name for name in names))
        self.assertTrue(self.bundle.with_name("bundle.hf-receipt.json").is_file())

    def test_upload_rejects_public_or_existing_release(self):
        self.pack()
        api = mock.Mock()
        sdk = SimpleNamespace(HfApi=lambda: api, CommitOperationAdd=lambda **kw: kw)
        phrase = f"{store.MODEL_REPO}:{self.settings['experiment']}:1000:inference"
        with mock.patch.dict(sys.modules, {"huggingface_hub": sdk}):
            api.repo_info.return_value = SimpleNamespace(private=False, sha="f" * 40)
            with self.assertRaisesRegex(ValueError, "private MODEL"):
                store.upload(self.bundle, phrase)
            api.repo_info.return_value.private = True
            api.list_repo_files.return_value = [f"inference/{self.settings['experiment']}/step-1000/model.safetensors"]
            with self.assertRaisesRegex(ValueError, "already exists"):
                store.upload(self.bundle, phrase)
        api.create_commit.assert_not_called()

    def test_inference_cannot_restore(self):
        self.pack()
        with self.assertRaisesRegex(ValueError, "no optimizer"):
            store.restore(self.bundle, self.root / "other")

    def test_restore_rebinds_only_local_stats_path(self):
        self.pack(resume=True)
        target = self.root / "other"
        with (
            mock.patch.object(store, "git", side_effect=lambda *args: "a" * 40 if args[0] == "rev-parse" else ""),
            mock.patch.object(store, "dataset_identity", return_value=self.record["dataset"]),
        ):
            store.restore(self.bundle, target)
            with self.assertRaisesRegex(ValueError, "overwrites"):
                store.restore(self.bundle, target)
        restored = target / "checkpoints/pi05_biyi_stage1" / self.settings["experiment"]
        store.inspect_checkpoint(restored, 1000)
        self.assertEqual((restored / "run_manifest.json").read_bytes(), (self.run / "run_manifest.json").read_bytes())
        stats_path = target / "assets/pi05_biyi_stage1/biyi_stage1/stage1_train_stats_manifest.json"
        rebound = store.read_json(stats_path)
        self.assertEqual(rebound["local_root"], str((target / "datasets" / self.record["dataset"]["revision"]).resolve()))
        self.assertEqual(rebound["norm_stats_sha256"], self.record["norm_stats_sha256"])

    def test_truncated_safetensors_rejected(self):
        path = self.ckpt / "model.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "payload length"):
            store.check_safetensors(path)

    def test_plan_is_single_gpu_stock_and_resume_is_explicit(self):
        with mock.patch.dict(os.environ, {"BIYI_BASE_CHECKPOINT_DIR": "/persistent/base"}):
            plan = launcher.build_plan("/mnt/persistent/biyi", "b" * 40, self.settings)
            resumed = launcher.build_plan("/mnt/persistent/biyi", "b" * 40, self.settings, resume=True)
        self.assertNotIn("--resume", plan["train"])
        self.assertIn("--resume", resumed["train"])
        self.assertIn("scripts/train_pytorch.py", plan["train"])
        self.assertNotIn("torch.distributed.run", plan["train"])
        self.assertEqual(plan["environment"]["BIYI_HF_CHECKPOINT_REPO_ID"], store.MODEL_REPO)
        self.assertTrue(plan["run_dir"].endswith("/checkpoints/pi05_biyi_stage1/pi05_biyi_100ep_fft_v1"))

    def test_dataset_lock_detects_change_and_enforces_100_episodes(self):
        data = self.root / "dataset"
        (data / "meta").mkdir(parents=True)
        store.write_json(data / "meta/info.json", {"total_episodes": 100})
        (data / "meta/episodes.jsonl").write_text("".join(json.dumps({"episode_index": i}) + "\n" for i in range(100)))
        lock = {
            "repo_id": store.DATASET_REPO,
            "revision": "b" * 40,
            "files": store.inventory(data, ["meta/info.json", "meta/episodes.jsonl"]),
        }
        store.write_json(data / "dataset_lock.json", lock)
        self.assertEqual(len(store.dataset_identity(data, "b" * 40)["episodes"]), 100)
        store.write_json(data / "meta/info.json", {"total_episodes": 99})
        with self.assertRaisesRegex(ValueError, "checksum"):
            store.dataset_identity(data, "b" * 40)
        lock["files"] = store.inventory(data, ["meta/info.json", "meta/episodes.jsonl"])
        store.write_json(data / "dataset_lock.json", lock)
        with self.assertRaisesRegex(ValueError, "exactly 100"):
            store.dataset_identity(data, "b" * 40)

    def test_existing_stock_save_schedule_without_importing_trainer(self):
        module = ast.parse((store.REPO_ROOT / "scripts/train_pytorch.py").read_text())
        function = next(
            node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_should_save_checkpoint"
        )
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<stock schedule>", "exec"), namespace)
        should_save = namespace["_should_save_checkpoint"]
        self.assertEqual([s for s in range(10501) if should_save(s, 10500, 1000)], [*range(1000, 10001, 1000), 10500])


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main()
