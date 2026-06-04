# SPDX-License-Identifier: Apache-2.0
"""Validate GLM-Image outputs across parallel deployment settings.

The script starts each online serving config sequentially, sends deterministic
T2I/I2I requests, saves the generated PNGs, and compares every setting against
the ``glm_image_1gpu_overlap`` baseline.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import socket
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import requests
from PIL import Image


MODEL_PATH = "/data/h/GLM-Image"

CASES: dict[str, str] = {
    "glm_image_1gpu_overlap": "/data/h/vllm-omni/vllm_omni/deploy/glm_image_perf_1gpu_overlap.yaml",
    "glm_image_1gpu_overlap_dit_cudagraph": "/data/h/vllm-omni/vllm_omni/deploy/glm_image_perf_1gpu_overlap_graph.yaml",
    "glm_image_2gpu_tp2_overlap_dit_cudagraph": (
        "/data/h/vllm-omni/vllm_omni/deploy/glm_image_perf_2gpu_tp2_overlap_graph.yaml"
    ),
    "glm_image_2gpu_artp2_sp2_dit_cudagraph": (
        "/data/h/vllm-omni/vllm_omni/deploy/glm_image_perf_2gpu_tp2_sp2_overlap_graph.yaml"
    ),
    "glm_image_4gpu_tp4_sp4_overlap_dit_cudagraph": (
        "/data/h/vllm-omni/vllm_omni/deploy/glm_image_perf_4gpu_tp4_sp4_overlap_graph.yaml"
    ),
    "glm_image_4gpu_tp4_tp4_overlap_dit_cudagraph": (
        "/data/h/vllm-omni/vllm_omni/deploy/glm_image_perf_4gpu_tp4_tp4_overlap_graph.yaml"
    ),
}

BASELINE_CASE = "glm_image_1gpu_overlap"
PROMPT = "Random prompt 0 for benchmarking diffusion models"


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _wait_for_port(host: str, port: int, proc: subprocess.Popen, timeout_s: int) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before ready, code={proc.returncode}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(2)
    raise TimeoutError(f"server did not open {host}:{port} within {timeout_s}s")


def _kill_process_tree(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        parent = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(children, timeout=10)
    for child in children:
        if child.is_running():
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    try:
        parent.terminate()
        parent.wait(timeout=10)
    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass


def _encode_image_as_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _make_i2i_input(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h = w = 512
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack(
        [
            (xx % 256).astype(np.uint8),
            (yy % 256).astype(np.uint8),
            (((xx + yy) // 2) % 256).astype(np.uint8),
        ],
        axis=-1,
    )
    Image.fromarray(arr, mode="RGB").save(path)


def _build_payload(mode: str, size: int, steps: int, seed: int, input_image: Path) -> dict[str, Any]:
    if mode == "i2i":
        content: Any = [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": _encode_image_as_data_url(input_image)}},
        ]
    else:
        content = PROMPT

    return {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": content}],
        "extra_body": {
            "height": size,
            "width": size,
            "num_inference_steps": steps,
            "guidance_scale": 1.5,
            "seed": seed,
        },
    }


def _extract_image_bytes(response: dict[str, Any]) -> bytes:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("response content does not contain an image list")
    item = content[0]
    url = item.get("image_url", {}).get("url", "")
    if not url.startswith("data:image"):
        raise ValueError("response image_url is not a data URL")
    return base64.b64decode(url.split(",", 1)[1])


def _send_request(host: str, port: int, payload: dict[str, Any], timeout_s: int) -> bytes:
    resp = requests.post(
        f"http://{host}:{port}/v1/chat/completions",
        json=payload,
        timeout=timeout_s,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    return _extract_image_bytes(resp.json())


def _compare_images(candidate_path: Path, baseline_path: Path) -> dict[str, Any]:
    candidate_bytes = candidate_path.read_bytes()
    baseline_bytes = baseline_path.read_bytes()
    byte_identical = candidate_bytes == baseline_bytes

    cand = np.asarray(Image.open(candidate_path).convert("RGB"), dtype=np.int16)
    base = np.asarray(Image.open(baseline_path).convert("RGB"), dtype=np.int16)
    if cand.shape != base.shape:
        return {
            "byte_identical": byte_identical,
            "pixel_identical": False,
            "shape_match": False,
            "candidate_shape": list(cand.shape),
            "baseline_shape": list(base.shape),
        }

    diff = cand - base
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    rmse = math.sqrt(mse)
    psnr = float("inf") if mse == 0 else 20.0 * math.log10(255.0 / rmse)

    return {
        "byte_identical": byte_identical,
        "pixel_identical": bool(np.array_equal(cand, base)),
        "shape_match": True,
        "max_abs_diff": int(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rmse": rmse,
        "psnr": psnr,
    }


def _start_server(case: str, deploy_config: str, log_path: Path, timeout_s: int) -> tuple[subprocess.Popen, str, int]:
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
        "--enable-diffusion-pipeline-profiler",
        "--deploy-config",
        deploy_config,
    ]
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
    print(f"{case}: server ready on {host}:{port}")
    return proc, host, port


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/data/h/test/logs/glm_output_validation")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024, 512])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server-timeout-s", type=int, default=1200)
    parser.add_argument("--request-timeout-s", type=int, default=600)
    parser.add_argument("--cases", nargs="+", default=list(CASES.keys()), choices=list(CASES.keys()))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    log_dir = output_dir / "server_logs"
    input_image = output_dir / "input" / "i2i_gradient.png"
    _make_i2i_input(input_image)

    results: list[dict[str, Any]] = []
    for case in args.cases:
        proc: subprocess.Popen | None = None
        try:
            proc, host, port = _start_server(
                case,
                CASES[case],
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

                    record: dict[str, Any] = {
                        "case": case,
                        "size": size,
                        "mode": mode,
                        "latency_s": latency_s,
                        "image_path": str(image_path),
                    }
                    if case == BASELINE_CASE:
                        record["comparison_to_baseline"] = {
                            "byte_identical": True,
                            "pixel_identical": True,
                            "shape_match": True,
                            "max_abs_diff": 0,
                            "mean_abs_diff": 0.0,
                            "rmse": 0.0,
                            "psnr": float("inf"),
                        }
                    else:
                        baseline_path = image_dir / BASELINE_CASE / f"{size}x{size}_{mode}.png"
                        if not baseline_path.exists():
                            raise FileNotFoundError(f"baseline image missing: {baseline_path}")
                        record["comparison_to_baseline"] = _compare_images(image_path, baseline_path)
                    results.append(record)
                    print(
                        f"{case} {size} {mode}: latency={latency_s:.3f}s "
                        f"compare={record['comparison_to_baseline']}"
                    )
        finally:
            _kill_process_tree(proc)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, allow_nan=True))
    print(f"summary written to {summary_path}")


if __name__ == "__main__":
    main()
