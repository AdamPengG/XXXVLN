from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List


def _norm(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def _within(path: str, root: str) -> bool:
    try:
        Path(_norm(path)).relative_to(Path(_norm(root)))
        return True
    except Exception:
        return False


def run_localize(
    audit_report: str,
    stage: str,
    assets_root: str,
    out_root: str,
    allow_outside_root: bool,
) -> int:
    report_path = Path(audit_report)
    if not report_path.exists():
        print(f"[ISAAC_ASSET_LOCALIZE] ok=0 reason=audit_report_missing hint={report_path}", flush=True)
        return 2
    report = json.loads(report_path.read_text())
    rows: List[Dict[str, Any]] = list(report.get("dependencies", []))

    stage = _norm(stage)
    assets_root = _norm(assets_root)
    out_root = _norm(out_root)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    files: List[Dict[str, Any]] = []
    copied = 0
    copied_bytes = 0
    seen_src = set()
    outside = set(report.get("outside_root_paths", []) or [])

    def _copy_one(src: str, dep_type: str) -> None:
        nonlocal copied, copied_bytes
        src_n = _norm(src)
        if src_n in seen_src:
            return
        if not os.path.isfile(src_n):
            return
        if not _within(src_n, assets_root):
            outside.add(src_n)
            return
        rel = os.path.relpath(src_n, assets_root)
        dst = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src_n, dst)
        size = os.path.getsize(dst)
        copied += 1
        copied_bytes += int(size)
        seen_src.add(src_n)
        files.append(
            {
                "src": src_n,
                "dst": _norm(dst),
                "type": str(dep_type),
                "size": int(size),
            }
        )

    for row in sorted(rows, key=lambda x: (str(x.get("path", "")), str(x.get("type", "")))):
        _copy_one(str(row.get("path", "")), str(row.get("type", "other")))

    if os.path.isfile(stage):
        _copy_one(stage, "usd")

    stage_localized = ""
    if _within(stage, assets_root):
        stage_localized = _norm(os.path.join(out_root, os.path.relpath(stage, assets_root)))

    outside_sorted = sorted(set([_norm(x) if os.path.isabs(str(x)) else str(x) for x in outside]))
    if outside_sorted and not allow_outside_root:
        print(
            f"[ISAAC_ASSET_LOCALIZE] ok=0 reason=outside_root_deps hint=use_allow_outside_root_or_fix_assets",
            flush=True,
        )
        return 1

    manifest_json = Path(out_root) / "manifest_copied.json"
    manifest_txt = Path(out_root) / "manifest_copied.txt"
    payload = {
        "assets_root": assets_root,
        "stage_original": stage,
        "stage_localized": stage_localized,
        "out_root": out_root,
        "files_copied": int(copied),
        "bytes_copied": int(copied_bytes),
        "skipped_outside_root": len(outside_sorted),
        "outside_root_paths": outside_sorted,
        "files": files,
    }
    manifest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    lines = [
        f"stage_original\t{stage}",
        f"stage_localized\t{stage_localized}",
        f"files_copied\t{copied}",
        f"bytes_copied\t{copied_bytes}",
        f"skipped_outside_root\t{len(outside_sorted)}",
    ]
    for row in files:
        lines.append(f"{row['type']}\t{row['src']}\t{row['dst']}")
    if outside_sorted:
        lines.append("# outside_root")
        lines.extend([f"outside\t{p}" for p in outside_sorted])
    manifest_txt.write_text("\n".join(lines) + "\n")

    print(
        f"[ISAAC_ASSET_LOCALIZE] ok=1 out_root={out_root} files_copied={copied} "
        f"bytes={copied_bytes} skipped_outside_root={len(outside_sorted)}",
        flush=True,
    )
    print(
        f"[ISAAC_ASSET_LOCALIZE_OUT] manifest={manifest_json}",
        flush=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="v28 Isaac asset localizer.")
    ap.add_argument("--audit_report", type=str, required=True)
    ap.add_argument("--stage", type=str, required=True)
    ap.add_argument("--assets_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--allow_outside_root", type=int, default=0)
    args = ap.parse_args()
    return run_localize(
        audit_report=args.audit_report,
        stage=args.stage,
        assets_root=args.assets_root,
        out_root=args.out_root,
        allow_outside_root=bool(int(args.allow_outside_root)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
