# SPDX-License-Identifier: Apache-2.0
"""Validate GLM-Image outputs using server configs from a perf JSON file.

This is the merge-friendly companion to
``tests/dfx/perf/tests/test_glm_image_vllm_omni_focused_inline.json``. It reads
the same ``serve_args`` used by the perf runner, starts each server sequentially,
saves deterministic T2I/I2I outputs, and compares them against a baseline case.

The summary is written after every generated image so partial runs remain useful
if a later server cannot start because another job is occupying GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from validate_parallel_outputs import (  # type: ignore
    BASELINE_CASE,
    MODEL_PATH,
    _build_payload,
    _compare_images,
    _kill_process_tree,
    _make_i2i_input,
    _open_port,
    _send_request,
    _wait_for_port,
)


def _serve_args_to_cli(serve_args: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in serve_args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, dict):
            args.extend([flag, json.dumps(value, separators=(",", ":"))])
        else:
            args.extend([flag, str(value)])
    return args


def _load_cases(config_path: Path, selected: list[str] | None) -> dict[str, dict[str, Any]]:
    configs = json.loads(config_path.read_text())
    cases: dict[str, dict[str, Any]] = {}
    wanted = set(selected or [])
    for item in configs:
        name = item["test_name"]
        if wanted and name not in wanted:
            continue
        cases[name] = item["server_params"]["serve_args"]
    if wanted:
        missing = wanted - set(cases)
        if missing:
            raise ValueError(f"Requested case(s) not found in {config_path}: {sorted(missing)}")
    return cases


def _start_server(case: str, serve_args: dict[str, Any], log_path: Path, timeout_s: int) -> tuple[subprocess.Popen, str, int]:
    host = "127.0.0.1"
    port = _open_port()
    cmd = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        MODEL_PATH,
        "--omni",
        "--host",
        host,
        "--port",
        str(port),
    ] + _serve_args_to_cli(serve_args)

    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    proc = subprocess.Popen(
        cmd,
        cwd="/data/h/vllm-omni",
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_port(host, port, proc, timeout_s)
    except Exception:
        log_file.close()
        _kill_process_tree(proc)
        raise
    print(f"{case}: server ready on {host}:{port}", flush=True)
    return proc, host, port


def _write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, allow_nan=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-config-file",
        default="/data/h/vllm-omni/tests/dfx/perf/tests/test_glm_image_vllm_omni_focused_inline.json",
    )
    parser.add_argument("--output-dir", default="/data/h/test/logs/glm_output_validation_from_config")
    parser.add_argument("--baseline-case", default=BASELINE_CASE)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024, 512])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server-timeout-s", type=int, default=1200)
    parser.add_argument("--request-timeout-s", type=int, default=600)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    log_dir = output_dir / "server_logs"
    summary_path = output_dir / "summary.json"
    input_image = output_dir / "input" / "i2i_gradient.png"
    _make_i2i_input(input_image)

    cases = _load_cases(Path(args.test_config_file), args.cases)
    if args.baseline_case not in cases:
        raise ValueError(f"baseline case {args.baseline_case!r} must be included in the selected cases")

    ordered_names = [args.baseline_case] + [name for name in cases if name != args.baseline_case]
    records: list[dict[str, Any]] = []

    for case in ordered_names:
        proc: subprocess.Popen | None = None
        try:
            proc, host, port = _start_server(
                case,
                cases[case],
                log_dir / f"{case}.log",
                args.server_timeout_s,
            )
            for size in args.sizes:
                for mode in ("t2i", "i2i"):
                    payload = _build_payload(mode, size, args.steps, args.seed, input_image)
                    started = time.perf_counter()
                    image_bytes = _send_request(host, port, payload, args.request_timeout_s)
                    latency_s = time.perf_counter() - started

                    image_path = image_dir / case / f"{size}x{size}_{mode}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(image_bytes)

                    if case == args.baseline_case:
                        comparison = {
                            "byte_identical": True,
                            "pixel_identical": True,
                            "shape_match": True,
                            "max_abs_diff": 0,
                            "mean_abs_diff": 0.0,
                            "rmse": 0.0,
                            "psnr": float("inf"),
                        }
                    else:
                        baseline_path = image_dir / args.baseline_case / f"{size}x{size}_{mode}.png"
                        comparison = _compare_images(image_path, baseline_path)

                    record = {
                        "case": case,
                        "baseline_case": args.baseline_case,
                        "size": size,
                        "mode": mode,
                        "latency_s": latency_s,
                        "image_path": str(image_path),
                        "comparison_to_baseline": comparison,
                    }
                    records.append(record)
                    _write_summary(summary_path, records)
                    print(f"{case} {size} {mode}: latency={latency_s:.3f}s compare={comparison}", flush=True)
        finally:
            _kill_process_tree(proc)

    _write_summary(summary_path, records)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
