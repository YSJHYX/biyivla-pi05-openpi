# π0.5 → RM65 + SG100 真机执行

`run_pi05_real.py` 把 BIYIVLA 推理服务返回的 `actions[50,17]` 接到
AVP-SG100 已有的真机控制入口：

```text
D455 head RGB + D405 wrist RGB + state17
                  │
                  ▼
       standard / RTC WebSocket server
                  │ actions[50,17]，checkpoint 绝对弧度
        ┌─────────┴─────────┐
        ▼                   ▼
   [0:6] RM65          [6:17] SG100 SDK_user_rad
   send_joint_target   restricted_backend.submit
        │                   │
  已有 100 Hz CANFD     已有 MODE7 / F_cmd
  sender + limiter      + allowlist/slew 路径
```

客户端没有增加动作限幅、滤波、IK、retarget 或新的安全控制器。RM65 和 SG100
的参数、启动位姿、anchor/enable 与退出顺序取自
`AVP-SG100/sg100_teleop/sim/run_avp_arm_hand_sim.py` 当前真机路径。动作行由客户
端以 30 Hz 提交；RM65 后端仍由自己的 100 Hz sender 插值/发送。

不要同时启动 AVP teleop runner。两套进程不能同时拥有 RM65/SG100 command
authority。

## 一次性安装客户端依赖

推理服务和真机客户端分开运行。真机客户端使用已经包含 RealSense、RealMan、
SG100 SDK 的 `avp_sg100` 环境，只补充 OpenPI WebSocket 客户端的小依赖：

```bash
cd /home/khan/biyivla-pi05-openpi
/home/khan/miniconda3/envs/avp_sg100/bin/python -m pip install \
  -r scripts/biyi_real/requirements.txt
```

脚本直接从本仓库 `packages/openpi-client/src` 导入客户端，不要求在 AVP 环境中
安装完整 OpenPI/Torch/JAX。

## 标准推理

Terminal 1（模型环境）：

```bash
cd /home/khan/biyivla-pi05-openpi
python -m scripts.biyi.infer_pi05 \
  --checkpoint-dir /path/to/pi05_biyi_stage1/checkpoints/<run>/<step> \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8000
```

Terminal 2（AVP-SG100 真机环境；替换相机序列号和任务指令）：

```bash
cd /home/khan/biyivla-pi05-openpi
/home/khan/miniconda3/envs/avp_sg100/bin/python -m \
  scripts.biyi_real.run_pi05_real \
  --server-host 127.0.0.1 \
  --server-port 8000 \
  --arm-ip 192.168.1.18 \
  --hand-port /dev/ttyUSB0 \
  --head-serial 333422300281 \
  --wrist-serial <D405_SERIAL> \
  --prompt "<训练数据中对应的任务指令>"
```

启动流程沿用 AVP-SG100 双真机流程：

1. 相机和两个 SDK 连接；SG100 进入 realtime MODE_NULL。
2. SG100 到 `thumb_j2=-31°`、其余为 0；RM65 以 `rm_movej` 到已验证启动位姿。
3. SG100 用同一 checkpoint/SDK-user 目标做 MODE7 zero-output anchor。
4. 在 command authority 关闭时先获得第一块 `[50,17]`。
5. 终端按 Enter 后执行原来的 A-enable 顺序并开始 30 Hz 动作。
6. Ctrl-C 使用原来的 disable → passive/slow-stop → disconnect 方法退出。

默认每执行新块的 8 行就异步请求下一块。等待推理期间继续执行旧块；响应到达后
按实际经过的 30 Hz tick 丢弃新块头部对应行，避免把过期动作重新执行。

## RTC 推理

客户端不变，只把 Terminal 1 换成 RTC 服务：

```bash
python -m scripts.biyi.infer_pi05_rtc \
  --checkpoint-dir /path/to/pi05_biyi_stage1/checkpoints/<run>/<step> \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8000
```

客户端从 server metadata 自动识别 `rtc=true`。每次请求把当前未执行的 17D tail
作为 `action_prefix`，用已测 round-trip latency 的 EWMA 估算
`inference_delay`；收到响应后仍按实际消耗 tick 对齐。

标准和 RTC 客户端均不以跨 chunk 的模型目标差值拒绝 SG100 命令；新绝对目标
继续通过原有 `4 rad/s` 速度斜坡接近，不会触发 `PER_CYCLE_DELTA_EXCEEDED`。

## state17 的 SG100 状态来源

RM65 state 始终来自 `RM65RealBackend.joint_state()` 的真实反馈。SG100 state 严格
以 checkpoint 的 `SDK_user_rad` 为准：启动目标之后，使用最近一次已发送目标的
checkpoint-domain command proxy。客户端不再自动切换到 canonical feedback，也不
对 checkpoint 手部动作应用 `F_cmd` 符号变换。

未验证的 SDK callback 不会覆盖这个 checkpoint-domain command proxy。

## 离线检查

不连接真机的调度/维度测试：

```bash
python -m pytest scripts/biyi_real/runtime_test.py -q
```
