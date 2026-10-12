"""Read-only Hugging Face and W&B authentication gate for Biyi Stage-1."""

import argparse
import os

from huggingface_hub import HfApi
from huggingface_hub import get_token
import wandb

WANDB_PROJECT = "BiyiVLA-Stage1"


def check_cloud_auth(dataset_repo_id: str | None = None, checkpoint_repo_id: str | None = None) -> bool:
    """Verify live SDK access without creating a W&B run or remote resource."""
    hf_ok = False
    dataset_ok = not dataset_repo_id
    checkpoint_ok = not checkpoint_repo_id
    try:
        # get_token checks both HF_TOKEN and the official cached login token.
        # whoami below is still the actual online authentication test.
        if not get_token():
            raise ValueError("No Hugging Face SDK credential is configured")
        hf_api = HfApi()
        account = hf_api.whoami()
        username = account.get("name") or account.get("fullname")
        if not username:
            raise ValueError("Hugging Face returned no authenticated account")
        print(f"HF_ACCOUNT           = {username}")
        hf_ok = True
        if dataset_repo_id:
            info = hf_api.dataset_info(dataset_repo_id, timeout=15)
            dataset_ok = bool(info.id)
        if checkpoint_repo_id:
            info = hf_api.repo_info(checkpoint_repo_id, repo_type="model", timeout=15)
            checkpoint_ok = bool(info.id)
    except Exception:
        # SDK exceptions can contain request details. Never print credentials or headers.
        pass

    wandb_ok = False
    mode = os.environ.get("WANDB_MODE", "online").lower()
    disabled = os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes")
    if mode == "online" and not disabled:
        try:
            # Avoid the SDK's interactive login path when no env/netrc key exists.
            if not wandb.api.api_key:
                raise ValueError("No W&B credential is configured")
            viewer = wandb.Api(timeout=10).viewer
            username = getattr(viewer, "username", None)
            if not username:
                raise ValueError("W&B returned no authenticated viewer")
            print(f"WANDB_ACCOUNT        = {username}")
            wandb_ok = True
        except Exception:
            pass

    print(f"HF_AUTH              = {'PASS' if hf_ok else 'FAIL'}")
    print(
        f"HF_DATASET_ACCESS    = {'PASS' if dataset_ok and dataset_repo_id else 'FAIL' if dataset_repo_id else 'SKIP'}"
    )
    if checkpoint_repo_id:
        print(f"HF_CHECKPOINT_ACCESS = {'PASS' if checkpoint_ok else 'FAIL'}")
    print(f"WANDB_AUTH           = {'PASS' if wandb_ok else 'FAIL'}")
    print(f"WANDB_PROJECT        = {WANDB_PROJECT}")
    passed = hf_ok and dataset_ok and checkpoint_ok and wandb_ok
    print(f"FORMAL_AUTH_GATE     = {'PASS' if passed else 'FAIL'}")
    if not hf_ok or not dataset_ok or not checkpoint_ok:
        print("HF help: set HF_TOKEN or run hf auth login; verify Hub access to the configured repo.")
    if not wandb_ok:
        print("W&B help: set WANDB_API_KEY or run wandb login; use online mode for formal training.")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", default=os.environ.get("BIYI_DATASET_REPO_ID") or None)
    parser.add_argument("--checkpoint-repo-id", default=os.environ.get("BIYI_HF_CHECKPOINT_REPO_ID") or None)
    args = parser.parse_args()
    if not check_cloud_auth(args.dataset_repo_id, args.checkpoint_repo_id):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
