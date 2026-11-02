"""Serve native pi0.5 or the handwritten RTC v2 over official OpenPI WebSocket.

Run from the OpenPI checkout in the MODEL environment (replace CHECKPOINT):
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m scripts.biyi_v2.serve_policy \
    --checkpoint-dir CHECKPOINT --mode rtc --device cuda:0 --host 0.0.0.0 --port 8000
For native inference, replace --mode rtc with --mode native.
Robot startup is a separate command documented in run_pi05_real.py.
"""

import argparse


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--mode", choices=("native", "rtc"), default="rtc")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--max-guidance-weight", type=float, default=10.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.num_steps <= 0 or args.max_guidance_weight <= 0:
        raise ValueError("num_steps and max_guidance_weight must be positive")

    # Keep model imports out of the hardware environment and out of --help.
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from scripts.biyi_v2.native import load_native_policy

    policy = load_native_policy(args.checkpoint_dir, device=args.device, num_steps=args.num_steps)
    if args.mode == "rtc":
        from scripts.biyi_v2.service_policy import RTCServicePolicy

        # RTC differentiates the latent, not model parameters. Do not enclose
        # RTC sampling in torch.inference_mode(), which disables that gradient.
        policy._model.requires_grad_(requires_grad=False)  # noqa: SLF001 -- native Policy owns the loaded model
        served_policy = RTCServicePolicy(policy, args.max_guidance_weight)
    else:
        served_policy = policy
    metadata = {
        **policy.metadata,
        "observation_contract": "BIYIVLA-PI05-OPENPI",
        "action_horizon": 50,
        "model_action_dim": 32,
        "physical_action_dim": 17,
        "hand_state_source": "checkpoint_command_proxy",
        "rtc": args.mode == "rtc",
        "implementation": "biyi_v2",
        "rtc_max_guidance_weight": args.max_guidance_weight,
    }
    WebsocketPolicyServer(served_policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
