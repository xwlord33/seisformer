"""Measure inference latency/throughput for the final checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from scripts.evaluate_standardized import build_model
from scripts.rebuild_standardized import atomic_json


def load_checkpoint_model(checkpoint):
    config = checkpoint['model_config']
    config = asdict(config) if is_dataclass(config) else config
    if checkpoint.get('head') in ('concat', 'pqme', 'phasenet'):
        from scripts.train_multiphase import FORMAT_VERSION
        if checkpoint.get('format_version') != FORMAT_VERSION:
            raise ValueError('Legacy multi-phase checkpoints do not establish current inference speed')
        if checkpoint['head'] == 'phasenet':
            from seisformer.baseline_phasenet import OfficialPSMulti
            model = OfficialPSMulti(config['pretrained'])
        else:
            from seisformer.model import ModelConfig, SeisFormer
            from seisformer.multiphase import ConcatHeads, PhaseQueryExperts
            backbone = SeisFormer(ModelConfig(**config))
            cls = ConcatHeads if checkpoint['head'] == 'concat' else PhaseQueryExperts
            model = cls(backbone, ps_residual=checkpoint['ps_residual'])
    else:
        model = build_model(config)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.eval()


def bench(model, device, batch, length, iters=30, warmup=5):
    if min(batch, length, iters, warmup) < 1:
        raise ValueError('Positive batch size, input length, iterations and warmup required')
    device = torch.device(device)
    x = torch.randn(batch, 3, length, device=device)
    for _ in range(warmup):
        with torch.inference_mode():
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.inference_mode():
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = (time.perf_counter() - t0) / iters
    return {"device": str(device), "batch": batch, "length": length,
            "seconds_per_batch": dt,
            "ms_per_window": 1000 * dt / batch,
            "windows_per_second": batch / dt}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--length", type=int, default=6000)
    p.add_argument('--devices', nargs='+', choices=('cpu', 'cuda'), default=['cuda', 'cpu'])
    p.add_argument('--batches', nargs='+', type=int, default=[1, 16, 64])
    p.add_argument('--iters', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    loaded = load_checkpoint_model(ck)
    with Path(args.checkpoint).open('rb') as stream:
        checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    config = asdict(ck['model_config']) if is_dataclass(ck['model_config']) else ck['model_config']
    report = {"checkpoint": str(Path(args.checkpoint).resolve()),
              'checkpoint_sha256': checkpoint_hash, 'weights_loaded_strictly': True,
              "params": sum(v.numel() for v in loaded.parameters()),
              "config": config, 'head': ck.get('head', 'legacy-P/S'), "benchmarks": [],
              'torch_version': torch.__version__, 'cpu': platform.processor(), 'cpu_threads': args.threads,
              'scope': 'Warm synthetic-input forward passes; excludes disk I/O, preprocessing, transfer and alert assembly. Not causal streaming latency.',
              'precision_scope': 'FP32 is the primary summary. BF16 timing alone does not validate BF16 prediction accuracy.'}
    for device in args.devices:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        model = loaded.to(device).eval()
        if device == 'cuda':
            report['gpu'] = torch.cuda.get_device_name()
        for batch in args.batches:
            for use_bf16 in ([False, True] if device == "cuda" and torch.cuda.is_bf16_supported() else [False]):
                if use_bf16:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        r = bench(model, device, batch, args.length, args.iters, args.warmup)
                    r["autocast"] = "bf16"
                else:
                    r = bench(model, device, batch, args.length, args.iters, args.warmup)
                    r["autocast"] = "fp32"
                report["benchmarks"].append(r)
    # single-record deployment cost: 25 windows for a 300 s record at 10 s stride
    single = [b for b in report["benchmarks"] if b["device"] == "cuda" and b["batch"] == 1 and b['autocast'] == 'fp32']
    if single:
        best = min(single, key=lambda b: b["ms_per_window"])
        report["single_window_ms"] = best["ms_per_window"]
        report["single_record_300s_seconds"] = 25 * best["ms_per_window"] / 1000
    gpu = [b for b in report["benchmarks"] if b["device"] == "cuda" and b['autocast'] == 'fp32']
    if gpu:
        wps = max(b["windows_per_second"] for b in gpu)
        report["throughput_windows_per_second"] = wps
    if not report['benchmarks']:
        raise ValueError('No requested benchmark device is available')
    atomic_json(Path(args.output), report)
    print(json.dumps({k: v for k, v in report.items() if k != "benchmarks"}, ensure_ascii=False, indent=2))
    for b in report["benchmarks"]:
        print(b)


if __name__ == "__main__":
    main()
