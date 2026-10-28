# BiyiVLA 正式 100-episode checkpoint 存储

本流程固定使用 stock **PyTorch π0.5 full fine-tuning**，AHFE 关闭、LoRA 关闭。
保留现有 `scripts/train_pytorch.py`、模型、优化器、数据变换、Stage-1 preflight、
smoke 启动器和 `uv.lock`。新增入口默认只输出计划，不启动训练。

## 三层边界

| 位置 | 内容 | 规则 |
|---|---|---|
| 明确挂载的 GPU 持久数据盘 / 云盘 / 可靠远端文件系统 | 完整 stock checkpoints：`model.safetensors`、`optimizer.pt`、`metadata.pt`（step 与完整 dataclass 配置）、`assets/biyi_stage1/norm_stats.json`；运行记录、冻结数据与 norm stats 来源 | 恢复训练的主副本；stock trainer 使用临时目录后 rename 保存；不自动清理旧 checkpoint |
| 私有 HF **Model** `PPPPPilot/BiyiVLA-pi05-checkpoints` | 已验收推理包：模型、norm stats、recipe、dataset revision、source commit、基础权重和 lockfile 哈希、校验和、验收证据；按需额外上传含 optimizer 的完整恢复包 | 默认离线；必须先验收再显式确认；拒绝覆盖已有 release |
| GitHub `YSJHYX/biyivla-pi05-openpi` | 配置、脚本、说明、HF 不可变 revision 引用 | 不提交权重、optimizer、token 或实际训练输出 |

生产 Dataset Repo 仍为 `PPPPPilot/BiyiVLA`，本流程不向它写入。
HF Model 仓库已于 2026-10-28 建为 private，通过存储契约文件写入和下载回读验证；
初始化 commit 为 `09cee256d594ee62cd04ed96c7aa742172475c4e`，没有模型权重。

## 持久路径和命名

必须先确认路径位于真正的持久挂载上。脚本要求绝对路径，拒绝仓库内路径、`/tmp` 和
`/content` 临时目录，允许明确的 `/content/drive/...` 持久挂载；脚本本身不能证明云盘的
生命周期或性能。不要使用临时 Colab runtime 作为主副本。没有挂载盘时，先准备可靠的
持久文件系统；也可通过受控传输将完整恢复包放入安全远端存储。

下面均为**未来操作命令**，本次任务不执行训练、数据下载或模型归档。先使用项目已验证的
Python 3.11 环境（下文 `python` 指该环境）；不更换依赖或 `uv.lock`。

```bash
export BIYI_PERSISTENT_ROOT=/mnt/persistent/biyivla
export BIYI_DATASET_REVISION=REPLACE_WITH_40_CHARACTER_DATASET_COMMIT
export BIYI_BASE_CHECKPOINT_DIR=/mnt/persistent/base/pi05_base_pytorch
```

不绑定 AutoDL。以后 Colab 可将首个路径改为已挂载的云盘；本任务不接入 Colab。
不要把 token 写进配置、命令参数、Git 或归档包。使用 `hf auth login` 或安全注入 `HF_TOKEN`；
已有官方 token 文件也可通过 `HF_TOKEN_PATH` 指定，不打印文件内容。W&B 使用原有认证。

默认布局：

```text
$BIYI_PERSISTENT_ROOT/
  datasets/<dataset_commit>/                  # 全量冻结文件 + dataset_lock.json
  assets/pi05_biyi_stage1/biyi_stage1/         # 该冻结训练集 norm stats 与来源 manifest
  checkpoints/pi05_biyi_stage1/pi05_biyi_100ep_fft_v1/
    run_manifest.json                        # source commit、dataset commit、recipe、SHA256
    train_stats_manifest.json
    wandb_id.txt
    1000/                                    # 已完成 step；忽略 tmp_* 中间目录
      model.safetensors
      optimizer.pt
      metadata.pt
      assets/biyi_stage1/norm_stats.json
      checkpoint_manifest.json               # seal 时生成
  bundles/                                   # 离线 staging，默认不上传
```

配置是 `configs/pi05_biyi_100ep_fft_v1.json`。默认单 GPU、global batch 1、10000 steps、
每 1000 steps 保存；这些是可审阅的起始值，尚未做正式 GPU 内存/训练时长验证。
现有 Stage-1 模型、AdamW、BF16、seed 42、10% warmup 和 LR 2.5e-5 → 2.5e-6 保持原样。
多 GPU 可在 recipe 中设置 `num_gpus` 及能整除它的 global batch，然后提交配置。
首次运行后，recipe/source/data/norm/base 的变化都会阻止恢复；新实验请使用新的 `..._v2`。

## 冻结数据、统计量与只读计划

当前任务不选择旧数据 revision 冒充正式 100 条数据。等 100 条数据验收后明确填入完整 commit。
`freeze-dataset` 只读 HF Dataset，并下载到**新的**持久目录；逐文件 SHA256 写入 lock。
已有目录拒绝覆盖；中断留下的 `.freeze-*` 可经人工检查后处理。

```bash
python -m scripts.biyi.checkpoint_store freeze-dataset \
  --persistent-root "$BIYI_PERSISTENT_ROOT" --revision "$BIYI_DATASET_REVISION"

export BIYI_DATASET_REPO_ID=PPPPPilot/BiyiVLA
export BIYI_DATASET_LOCAL_ROOT="$BIYI_PERSISTENT_ROOT/datasets/$BIYI_DATASET_REVISION"
export BIYI_TRAIN_EPISODES="$(seq -s, 0 99)"
export BIYI_ASSETS_BASE_DIR="$BIYI_PERSISTENT_ROOT/assets"
export BIYI_CHECKPOINT_BASE_DIR="$BIYI_PERSISTENT_ROOT/checkpoints"
python -m scripts.biyi.compute_stage1_norm_stats

python -m scripts.biyi.train_pi05_100ep plan
python -m scripts.biyi.train_pi05_100ep preflight
```

`plan` 不联网、不导入 torch、不创建运行目录。`preflight` 不训练，但运行已有正式认证、
数据、stats、base weights、完整 FFT 检查；要求 HF/W&B 可达，严格验证冻结目录全部哈希、
恰好 100 episodes（0..99，全用于此版训练）、干净的 source commit。改变 root 时需重新绑定
stats 来源的机器路径，恢复工具会在完整数据 lock 验证后处理这一个字段。
此 v1 不建立独立验证 split；策略验收需要另外的离线样本/证据，不能将训练 loss 当成验收。

**只有以后明确决定开始训练时才运行：**

```bash
python -m scripts.biyi.train_pi05_100ep train
# 同机恢复最新已完成的数字 step；仍经过预检
python -m scripts.biyi.train_pi05_100ep train --resume
```

新运行不会覆盖已有目录；resume 要求原配置、数据、源码、base、norm、W&B run ID 一致。
运行记录与 stock checkpoint 放在一起；恢复会先离线检查/封存最新 checkpoint。
直接绕过新增入口调用旧 trainer 仍能工作，但没有正式运行记录的 checkpoint 无法经归档门禁。

## 离线核验、验收和归档

```bash
export RUN_DIR="$BIYI_CHECKPOINT_BASE_DIR/pi05_biyi_stage1/pi05_biyi_100ep_fft_v1"
python -m scripts.biyi.checkpoint_store seal --run-dir "$RUN_DIR" --step 1000
```

`seal` 不执行模型、不初始化 CUDA、不反序列化 pickle。它检查完整四件套、stock-shaped
SafeTensors header/offset、17D finite norm stats、source/data/recipe 身份与 norm 哈希，
写入逐文件 SHA256 manifest。它证明静态结构与文件完整性，**不证明模型推理质量**，也不检查
optimizer 内部语义。损坏/错误 `.pt` 内容须由可信的离线加载和恢复验证发现；仅加载可信来源。
它不把 smoke 自动升级成正式结果。`tmp_*`、非数字 step、缺失正式运行记录均不归档。

复制 `configs/acceptance.example.json` 到持久存储的 review 目录，在真正的离线策略加载和
样本推理通过后，由审阅者填写 reviewer、UTC 时间、seal 输出的精确 manifest SHA256，
把三项 checks 和 accepted 改为 true，并附相对路径的 JSON/txt/md 证据文件。
示例默认 false，不能直接通过。检查应使用 pinned source 的
`openpi.policies.policy_config.create_trained_policy`、`pi05_biyi_stage1` 配置与 checkpoint norm
stats，验证 actions 为 `[50,17]` 且有限；不连接机器人。这是明确的人工验收声明，未实现自动
质量评估或签名认证；脚本不伪造验收结果。证据只放审阅后的指标/报告，不能带凭证或原始日志。

```bash
export ACCEPTANCE="$BIYI_PERSISTENT_ROOT/reviews/step-1000/acceptance.json"
export BUNDLE="$BIYI_PERSISTENT_ROOT/bundles/pi05_biyi_100ep_fft_v1-step-1000"
python -m scripts.biyi.checkpoint_store pack --run-dir "$RUN_DIR" --step 1000 \
  --acceptance "$ACCEPTANCE" --output "$BUNDLE"
python -m scripts.biyi.checkpoint_store verify --bundle "$BUNDLE"

# 上传是单独、显式的外部写操作；不存在默认自动上传。
python -m scripts.biyi.checkpoint_store upload --bundle "$BUNDLE" \
  --confirm 'PPPPPilot/BiyiVLA-pi05-checkpoints:pi05_biyi_100ep_fft_v1:1000:inference'
```

包严格按文件白名单构建；拒绝额外文件、symlink、路径穿越、哈希不符、未接受/错 checkpoint
的验收报告、smoke/LoRA/AHFE recipe、已有 bundle 和已有 HF release。上传再次核验所有哈希，
要求既有 repo 为 private **model**，以 HF 当前 HEAD 为 parent（有并发变更时失败）。
上传不修改 repo 可见性，不写 Dataset，不上传 optimizer/metadata，除非创建完整恢复包。
回执写在 bundle **旁边**的 `.hf-receipt.json`，包含 model commit、path 与 manifest SHA256。
将回执的 repo/revision/path/hash 存入 Git 跟踪的发布引用即可，不跟踪二进制包。
可以在隔离 CPU 环境用 `verify` 重复离线检查；它不导入 HF SDK。

## 跨机完整恢复

需要异机续训时，额外创建完整恢复包。它包含 optimizer、metadata、同一个验收报告和 W&B ID，
独立置于 HF 的 `resume/...` 路径；推理包单独置于 `inference/...`。二者均禁止覆盖已有版本。

```bash
export RESUME_BUNDLE="$BIYI_PERSISTENT_ROOT/bundles/pi05_biyi_100ep_fft_v1-step-1000-resume"
python -m scripts.biyi.checkpoint_store pack --run-dir "$RUN_DIR" --step 1000 \
  --acceptance "$ACCEPTANCE" --output "$RESUME_BUNDLE" --include-optimizer
python -m scripts.biyi.checkpoint_store upload --bundle "$RESUME_BUNDLE" \
  --confirm 'PPPPPilot/BiyiVLA-pi05-checkpoints:pi05_biyi_100ep_fft_v1:1000:resume'
```

也可用 `rsync`/安全对象存储传输整个已封存包，再在目标机器 `verify`，不必使用 HF。
普通完整备份只保留这里的 allowlist 工件，不打包凭证或整个开发工作区。
新机器先 checkout `run_manifest.json` 里的**精确源码 commit**、恢复锁定环境和已验证 base
weights、登录 HF/W&B，设置新的 `BIYI_PERSISTENT_ROOT`、同一个 dataset revision 和 base 路径，
先 `freeze-dataset` 恢复冻结数据（也可安全复制原冻结目录再核验），然后：

```bash
python -m scripts.biyi.checkpoint_store download \
  --persistent-root "$BIYI_PERSISTENT_ROOT" \
  --revision REPLACE_WITH_MODEL_COMMIT_FROM_RECEIPT \
  --prefix resume/pi05_biyi_100ep_fft_v1/step-1000 \
  --output "$BIYI_PERSISTENT_ROOT/bundles/restored-step-1000"
python -m scripts.biyi.checkpoint_store verify \
  --bundle "$BIYI_PERSISTENT_ROOT/bundles/restored-step-1000"
python -m scripts.biyi.checkpoint_store restore \
  --bundle "$BIYI_PERSISTENT_ROOT/bundles/restored-step-1000" \
  --persistent-root "$BIYI_PERSISTENT_ROOT"
python -m scripts.biyi.train_pi05_100ep preflight --resume
# 以后明确要继续训练时：
python -m scripts.biyi.train_pi05_100ep train --resume
```

`restore` 拒绝 inference-only 包、错误 source commit、错误数据、已有 run 或冲突 stats。
恢复保留原 provenance，只有 assets stats 来源中的 local_root 会重新绑定新数据目录。
stock trainer 从最高数字目录恢复 model/optimizer/global_step，学习率由 global_step 重新计算。
**stock 路径没有保存 RNG、DataLoader 位置/游标；不承诺 bitwise 或样本序列无缝续训。**
本改动保留该既有行为，没有修改模型或训练算法。W&B ID 用于延续同一记录，须有相同项目权限。

## 测试与已知边界

```bash
python -m unittest scripts.biyi.checkpoint_store_test -v
```

测试使用小型 synthetic stock-shaped tensors 和假的 `.pt` 字节，覆盖静态封存、推理/恢复白名单、
损坏拒绝、验收绑定、smoke/LoRA/AHFE 拒绝、路径安全、数据锁、显式上传门禁、恢复目录布局与
原 stock 保存步数规则。测试不构成真实 checkpoint 的推理/优化器恢复验收。
本次没有训练、真机动作、真实模型上传、正式数据下载、Colab 接入或真实 GPU 跨机恢复。
持久挂载准备、正式 100-episode revision、内存/batch sizing、实际模型离线验收、恢复演练仍待执行。

HF API 行为参见 [官方 HfApi 文档](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api)。
