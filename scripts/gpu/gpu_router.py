#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GpuInfo:
    index: int
    name: str
    uuid: str
    memory_mb: int


def _run(cmd: List[str]) -> str:
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    return out.strip()


def detect_gpus() -> List[GpuInfo]:
    raw = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: List[GpuInfo] = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=str(parts[1]),
                    uuid=str(parts[2]),
                    memory_mb=int(float(parts[3])),
                )
            )
        except Exception:
            continue
    return gpus


def _gpu_desc(gpus: List[GpuInfo]) -> str:
    items = [f"{g.index}:{g.name},{g.memory_mb}MB" for g in gpus]
    return ";".join(items)


def _is_isaac_supported(gpu_name: str) -> bool:
    # Current machine policy: Isaac rendering is stable only on RTX 5090.
    return "5090" in gpu_name


def _find_by_id(gpus: List[GpuInfo], gpu_id: int) -> Optional[GpuInfo]:
    for g in gpus:
        if g.index == gpu_id:
            return g
    return None


def pick_gpu(
    role: str,
    gpus: List[GpuInfo],
    isaac_gpu_id_env: Optional[str],
    habitat_gpu_id_env: Optional[str],
    isaac_prefer: str,
) -> tuple[GpuInfo, str]:
    role = role.lower().strip()
    if role == "isaac":
        if isaac_gpu_id_env not in (None, ""):
            forced = int(isaac_gpu_id_env)
            g = _find_by_id(gpus, forced)
            if g is None:
                raise RuntimeError(f"forced_gpu_not_found:{forced}")
            if not _is_isaac_supported(g.name):
                raise ValueError("unsupported_or_failed_gpu")
            return g, "forced_env"

        prefer = isaac_prefer.lower().strip() or "auto"
        if prefer in {"5090", "auto"}:
            for g in gpus:
                if "5090" in g.name:
                    return g, "prefer_5090"
        # Fallback if 5090 is not present on machine.
        best = sorted(gpus, key=lambda x: (x.memory_mb, x.index), reverse=True)[0]
        return best, "fallback_max_memory"

    if role == "habitat":
        if habitat_gpu_id_env not in (None, ""):
            forced = int(habitat_gpu_id_env)
            g = _find_by_id(gpus, forced)
            if g is None:
                raise RuntimeError(f"forced_gpu_not_found:{forced}")
            return g, "forced_env"

        isaac_id = None
        if isaac_gpu_id_env not in (None, ""):
            try:
                isaac_id = int(isaac_gpu_id_env)
            except Exception:
                isaac_id = None

        non_isaac = [g for g in gpus if g.index != isaac_id]
        for g in non_isaac:
            if "2080" in g.name:
                return g, "prefer_2080_non_isaac"
        if len(non_isaac) > 0:
            return non_isaac[0], "first_non_isaac"
        return gpus[0], "single_gpu_fallback"

    raise RuntimeError(f"unsupported_role:{role}")


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU role router for Isaac/Habitat tasks.")
    ap.add_argument("--role", required=True, choices=["isaac", "habitat"])
    ap.add_argument("--output", choices=["anchors", "json", "id", "env"], default="anchors")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        gpus = detect_gpus()
    except Exception as e:
        if not args.quiet:
            print(f"[GPU_DETECT] count=0 list=\"\" error=\"{type(e).__name__}:{e}\"", flush=True)
        return 2

    if len(gpus) == 0:
        if not args.quiet:
            print('[GPU_DETECT] count=0 list=""', flush=True)
        return 2

    if not args.quiet:
        print(f"[GPU_DETECT] count={len(gpus)} list=\"{_gpu_desc(gpus)}\"", flush=True)

    isaac_gpu_id_env = os.environ.get("ISAAC_GPU_ID", "")
    habitat_gpu_id_env = os.environ.get("HABITAT_GPU_ID", "")
    isaac_prefer = os.environ.get("ISAAC_GPU_PREFER", "auto")

    try:
        picked, reason = pick_gpu(
            role=args.role,
            gpus=gpus,
            isaac_gpu_id_env=isaac_gpu_id_env,
            habitat_gpu_id_env=habitat_gpu_id_env,
            isaac_prefer=isaac_prefer,
        )
    except ValueError as e:
        if str(e) == "unsupported_or_failed_gpu":
            if not args.quiet:
                print(
                    '[GPU_PICK] role=isaac ok=0 reason="unsupported_or_failed_gpu" '
                    'hint="use ISAAC_GPU_ID=0 (5090)"',
                    flush=True,
                )
            return 3
        raise
    except Exception as e:
        if not args.quiet:
            print(
                f"[GPU_PICK] role={args.role} ok=0 reason=\"{type(e).__name__}:{e}\"",
                flush=True,
            )
        return 3

    if not args.quiet:
        print(
            f"[GPU_PICK] role={args.role} gpu_id={picked.index} gpu_name=\"{picked.name}\" reason={reason}",
            flush=True,
        )

    if args.output == "anchors":
        return 0
    if args.output == "id":
        print(str(picked.index))
        return 0
    if args.output == "env":
        key = "ISAAC_GPU_ID" if args.role == "isaac" else "HABITAT_GPU_ID"
        print(f"export {key}={picked.index}")
        print(f"export CUDA_VISIBLE_DEVICES={picked.index}")
        return 0
    if args.output == "json":
        print(
            json.dumps(
                {
                    "role": args.role,
                    "gpu_id": picked.index,
                    "gpu_name": picked.name,
                    "gpu_uuid": picked.uuid,
                    "gpu_mem_mb": picked.memory_mb,
                    "reason": reason,
                    "gpu_count": len(gpus),
                }
            )
        )
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

