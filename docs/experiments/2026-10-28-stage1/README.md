# BiyiVLA Stage-1 experiment source archive (2026-10-28)

This directory preserves the 32 distinct cloud-side helper/script/patch contents
used or prepared during the Stage-1 experiment and earlier related checks.
The exact completed recipe is tracked at
`scripts/biyi/configs/pi05_biyi_117ep_fft_v2.json`.
`source_manifest.json` maps every published file to its original cloud paths,
size and SHA256. Three duplicate final-report copies are aliases, not extra
distinct scripts. All 33 requested source/config files retain their original bytes.

## Reproduce the completed experiment

- Trained source: `dd8a85b16f46d081eb3f7813e87763fb0cb3c1aa`.
- Dataset: `PPPPPilot/BiyiVLA@5ffc4e58e4307fab8a02a9f34aa4d975008ffa5d`;
  117 episodes, 77,975 eligible starting frames, all training.
- Recipe: global batch 8, 9,747 optimizer steps, seed 42, one GPU,
  saves 3,249 / 6,498 / 9,747. This is one nominal starting-frame exposure,
  not evidence of sufficient convergence or held-out task success.
- Norm SHA256: `aea055f4b45f0c055d5d65be6db29ac54eab15057336af278a76b787f9dd00f3`.
- Base SHA256: `dbcbe3ea978ce1d6f791582fa034e7ece8a41feaad2d3c9a5b014c1938610e2a`.
- Stats used the stock effective batch 32; formal training used batch 8.
- Frozen runtime: Python 3.11, Torch 2.7.1/CUDA 12.6, Transformers 4.53.2 with
  the repository replacements; keep uv.lock unchanged.
- Completed full resume artifacts:
  `https://huggingface.co/PPPPPilot/BiyiVLA-pi05-checkpoints/tree/b6433d277226ac28b344a5d53d59a3ed922a5630/resume/pi05_biyi_117ep_fft_v2`.

Keep the archived recipe outside the source checkout when checking out the
exact trained commit: this archive was added after training. Its publication
does not change the source identity of that run. A fresh repeat must use a new
versioned experiment/output destination and pass the existing formal preflight.
Use the stock `bash scripts/biyi/train_pi05_stage1.sh` launcher with explicit
`BIYI_FORMAL_RECIPE`; the archived scripts do not replace its default entry.
Restores must preserve or explicitly rebind the data/assets/source provenance.
Stock checkpoints do not save all RNG/data-iterator state, so bitwise resume
is not guaranteed.

## Snapshot scope and historical variants

These files retain original absolute cloud paths and dependencies. They are
historical source snapshots and are not automatically executed or imported.
The source catalog labels superseded variants. In particular:

- `biyivla_archive_final.py` was an unexecuted proposal that included full
  readback. It was superseded when the user explicitly declined full payload
  download/readback. Actual uploads used metadata/hash and small manifest checks.
- The private-only intermediate variants were superseded after the repository
  became Public and the user explicitly authorized Public checkpoint/evidence
  upload. The final public helper requires `--allow-public` and an exact
  repository/experiment/step confirmation. The stock upload guard remains private-only.
- The original HTTP proxy environment is preserved as history. After the desktop
  VPN was closed, recovery used a bounded SSH remote dynamic SOCKS forward through
  the local Wi-Fi router VPN, with cloud requests using `socks5h://127.0.0.1:17890`.
  The retained runtime already had PySocks and the official hf_transfer backend;
  the final retry limited multipart concurrency to 8. Temporary tunnels were closed.
- `stage1_env.sh` references a separate credential file with mode 0600. That file, real keys,
  HF tokens, shell history, raw datasets and checkpoint binaries are not included.
  Authenticate separately and inject W&B through a secure exported environment.

Automatic formatting excludes only `cloud_sources/` so snapshot hashes stay
unchanged. Publication validates hashes, Python syntax, shell syntax and recipe
schema without executing these historical programs, starting training or
connecting to robot hardware.

## Source inventory

| Repository path | Role | SHA256 |
|---|---|---|
| `cloud_sources/stage1_env.sh` | historical_experiment_helper | `d3a5340382aea9ecd7e6b0f4633ba3777da2fa0463a871dfcbf665a324b1ea5f` |
| `cloud_sources/stage1_network_env.sh` | historical_http_proxy_binding | `bad6631d507784d2eccf514fc109e1f34bd1cb8fe7d5b7db7f692b510485b5cf` |
| `cloud_sources/logs/action_trajectory_export_step2/export.py` | historical_experiment_helper | `1ee3ece940103248a525b445bb6376b07976e93d9ceff1cd48de2ba705ed8b73` |
| `cloud_sources/logs/action_trajectory_export_step2/export.v2.py` | historical_experiment_helper | `34580b1e3a8202a3361fe6d563b2103a7e788a3d9aba4d0528b34d56754a8855` |
| `cloud_sources/logs/archive_preflight_20261020/preflight.py` | historical_experiment_helper | `7c92c8422e211260c2377a773d3da3cec4ce9a678a8a46bb0db3515f6f721325` |
| `cloud_sources/logs/checkpoint_reload_inference/validate.py` | historical_experiment_helper | `95b8184cc83249cc85004520b1c0ee48c781066839609981a9a37e238b46fc19` |
| `cloud_sources/logs/preparation_20261028/biyivla_full_audit.py` | historical_experiment_helper | `9b203a0ecd52d6941c3bada3257e53c6f0fced9af98fd3068eb0e204767699e3` |
| `cloud_sources/logs/preparation_20261028/biyivla_stats_entry.patch` | historical_patch_not_a_runtime_entry | `d1f787290ec3d3397754798c9ca79318d07f373f281a5234bd6afb07c92095b5` |
| `cloud_sources/logs/preparation_20261028/biyivla_gpu_sizing.py` | historical_experiment_helper | `9895e4872ab4cc2942eeac64dd81e5f933ba65e1bdcfc2ae3b3b0352f8b33543` |
| `cloud_sources/logs/preparation_20261028/preparation_env.sh` | experiment_paths_and_original_network_binding | `2afb345b2a16329a5985880f133a1f2087ab8fb54c6060cbc4971734a75633d4` |
| `cloud_sources/logs/preparation_20261028/resume_download.sh` | historical_experiment_helper | `1cbe54d0d2eef71dd91bff38061b62ad1db3e4615372fee984030df48374f589` |
| `cloud_sources/logs/preparation_20261028/biyivla_pack_only.py` | historical_experiment_helper | `241ab7ba8cee9677e6c20f50c2985c94870e19d4795f5041da6889e7ba6490c5` |
| `cloud_sources/logs/preparation_20261028/biyivla_verify_norm_and_loader.py` | historical_experiment_helper | `639b5ca0a4433d14b4aecad25a9a13057e672a81726cf5eaaa3503349487a01f` |
| `cloud_sources/logs/preparation_20261028/run_full_audit.sh` | historical_experiment_helper | `93f9583733d29d50920b9f3e12fe7f1f28b9653175d5362ca36d36af04e0aaf7` |
| `cloud_sources/logs/preparation_20261028/run_norm_and_sizing.sh` | historical_experiment_helper | `d08011ce1017ac0f178f52a585f067a9dc01431936f0b1e9b33619d57f588919` |
| `cloud_sources/logs/preparation_20261028/biyivla_gpu_sizing_optional8.py` | historical_experiment_helper | `78f4f3658adc71d6479d7223d671c2a1a47e8c2e13ea3081ed00effa39dfd692` |
| `cloud_sources/logs/preparation_20261028/run_optional_batch8.sh` | historical_experiment_helper | `d68efe1c4f20688593002b7d8aed224e69294a0fc9f3f7e051f2cd0b54c13fa3` |
| `cloud_sources/logs/preparation_20261028/biyivla_formal_experiment_monitor.py` | historical_experiment_helper | `0770d5b608760fabbf5c4dc7aae2468e7948534b5bffb0aab043e66b90a81e88` |
| `cloud_sources/logs/preparation_20261028/run_formal_experiment_monitor.sh` | historical_experiment_helper | `d6c1690ef4df64f5a8de962ba7ee1ccf4b725c2067ec8f9df5a01453965782a0` |
| `cloud_sources/logs/preparation_20261028/biyivla_final_technical_check.py` | historical_experiment_helper | `e4ca930bcd2ca2d4c5650f410bfd98e39245f159e45dfa05f518fa8c9e97ab0d` |
| `cloud_sources/logs/preparation_20261028/biyivla_summarize_experiment.py` | historical_experiment_helper | `cc2c5e14b38a557023ffadc6bfc7b162736e86b8522594add96372a181fc1a4b` |
| `cloud_sources/logs/preparation_20261028/biyivla_wandb_metric_audit.py` | historical_experiment_helper | `7ac8b20590f368889a43a9b5f5de4dc92c48924ac3e9cb70c00d4fa7d2a71804` |
| `cloud_sources/logs/preparation_20261028/run_final_checks.sh` | historical_experiment_helper | `8241e9b02c6dc74f8aadc8ad1ebe144557bfccf4a9a8e5eb57d886575cb81348` |
| `cloud_sources/logs/preparation_20261028/biyivla_build_archive_evidence.py` | historical_experiment_helper | `b544cfa838534737d39f2f3a31345ad61f0b89b5dcb85dab942f0a5deb5f75fa` |
| `cloud_sources/logs/preparation_20261028/biyivla_augment_timing_report.py` | historical_experiment_helper | `0164d93fd7e5f91d13f3194d4209087b432a075590b14ff7b173273aa07a8bb0` |
| `cloud_sources/logs/preparation_20261028/biyivla_archive_final.py` | unexecuted_superseded_full_readback_proposal | `7190483aecea8c6b43afc6ef0c91e0e024fd6a24cc1b74bb16a4bed00280b585` |
| `cloud_sources/logs/preparation_20261028/biyivla_upload_metadata_verify.py` | final_9747_metadata_verification_before_visibility_change | `5518dd2264ae182c4905915b8754567d6bbd81a146dfd578f8d443b4103ae775` |
| `cloud_sources/logs/preparation_20261028/biyivla_checkpoint_technical_check.py` | checkpoint_specific_offline_technical_checks | `199aa79cbfdd0da93c2fddfc544f2343e3b67c61361b458df2b5e89d1619de1e` |
| `cloud_sources/logs/preparation_20261028/biyivla_upload_metadata_step.py` | superseded_private_only_upload_helper | `406a9f357bfc41705ee5a5b685b64147c053de36464870c1854168cc5ff4f3cd` |
| `cloud_sources/logs/preparation_20261028/biyivla_archive_intermediate.py` | superseded_private_only_archive_controller | `1073a8b3e44a6f783a16a74d81d1cafece73591d4de0807d3c1fc35ef86b5776` |
| `cloud_sources/logs/preparation_20261028/biyivla_upload_metadata_public_step.py` | final_explicit_public_upload_with_bounded_existing_transport | `e7eb35afbddec4d26e1c47b324300c6d80fb35af898f5a7e05aa721bab8bc782` |
| `cloud_sources/logs/preparation_20261028/biyivla_archive_intermediate_public.py` | final_explicit_public_archive_controller | `c876ed490442da6e8b572f342445c910552136737a377661f0fb88c5070a31ab` |
| `scripts/biyi/configs/pi05_biyi_117ep_fft_v2.json` | exact_completed_fft_recipe | `5c70f3514bd7d645f49636f7eceac904a835ee881562ce61a695e6c41ff10868` |
