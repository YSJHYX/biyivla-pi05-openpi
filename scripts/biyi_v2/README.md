# BiyiVLA 原生推理与 RTC v2

沿用旧版的两个进程：OpenPI 环境加载本地 Policy 并提供 WebSocket 服务；
现有 AVP-SG100 环境运行真机客户端，不导入 Torch/JAX 或加载模型权重。
旧版 `scripts/biyi/`、`scripts/biyi_real/` 入口继续保留。

## 启动

以下命令均从 OpenPI 仓库根目录执行，替换大写占位值。
完整命令也写在两个入口文件顶部的注释中。

模型环境，RTC 模式：

```bash
PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m scripts.biyi_v2.serve_policy \
  --checkpoint-dir CHECKPOINT --mode rtc --device cuda:0 \
  --num-steps 10 --max-guidance-weight 10 --host 0.0.0.0 --port 8000
```

原生模式使用同一命令，将 `--mode rtc` 改为 `--mode native`。
原生模式直接使用原生 Policy 与 sample_actions，不注入 RTC 前缀或采样回调。

真机环境（AVP_PYTHON 是原有 AVP-SG100 环境的解释器路径）：

```bash
PYTHONPATH=.:packages/openpi-client/src AVP_PYTHON -m scripts.biyi_v2.run_pi05_real \
  --avp-root AVP_ROOT --server-host 127.0.0.1 --server-port 8000 \
  --head-serial D455_SERIAL --wrist-serial D405_SERIAL --prompt "TASK" \
  --request-interval 10 --execution-horizon 10
```

例如 AVP_PYTHON 可为 `/home/khan/miniconda3/envs/avp_sg100/bin/python`。
客户端需要旧版 `scripts/biyi_real/requirements.txt` 中的通信依赖和原有硬件 SDK。
两端运行在不同电脑时，`--server-host` 使用模型电脑可达的地址。
客户端从服务端 metadata 自动选择 native/RTC，无需改客户端代码。

## 行为

- 复用原有相机、RM65、SG100 类、启动位姿和 MODE7 anchor。
- 用户确认后采集最新观测并获得首个原生 chunk，再启用实时动作发送。
- 每 `request_interval` 个控制 tick 异步请求下一块；此参数与 RTC 的
  `execution_horizon` 分开，后者决定旧前缀的引导范围。
- RTC 前缀为尚未取出的物理 17D 动作，在服务端使用 checkpoint 的 Normalize
  归一化并补到 32D。数学核与手写 rtc_adapter 保持原实现。
- 预测延迟由上一轮完整请求耗时估计；合并仍使用队列实际经过的 tick 差。
- arm 为弧度，hand 为 SG100 SDK-user 弧度；手部状态为已接受目标的
  `checkpoint_command_proxy`，不是实测关节反馈，不重复做符号变换。
- 推理动作走手写 `robot_io.send_action`，不经过旧的手部 submit/slew 包装。
  SDK 和 RM65 后端自身已有行为保留。
- 队列耗尽或机械臂后端停止时结束本次执行并关闭设备，不无限保留过期动作。
- 正常结束、后台异常、发送异常和 Ctrl-C 都先停止设备发送，再关闭连接并等待线程。

`--max-control-steps 0` 持续运行直到 Ctrl-C；正整数限制实际控制 tick 数。
本入口会执行原有启动位姿移动；不要与另一个拥有同一设备的 teleop/推理进程同时运行。
真实 checkpoint 的 GPU 延迟与真机轨迹效果仍需在目标部署环境验收。
