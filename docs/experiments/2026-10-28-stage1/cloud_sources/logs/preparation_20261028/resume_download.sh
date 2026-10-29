#!/usr/bin/env bash
source /root/autodl-tmp/biyivla/stage1_env.sh
source /root/autodl-tmp/biyivla/stage1_network_env.sh
export HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1
hf download PPPPPilot/BiyiVLA --type dataset --revision 5ffc4e58e4307fab8a02a9f34aa4d975008ffa5d --local-dir /root/autodl-tmp/biyivla/datasets/frozen/5ffc4e58e4307fab8a02a9f34aa4d975008ffa5d --max-workers 4 --format quiet >/dev/null 2>&1
status=$?
printf '%s\n' "$status" > /root/autodl-tmp/biyivla/logs/preparation_20261028/download_cli.exit
