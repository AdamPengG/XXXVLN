#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    return rows


def _rel(path: str, root: Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description="v31 failure cards generator")
    ap.add_argument("--failure_cases", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    failure_cases = Path(args.failure_cases)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(failure_cases)
    failures = [r for r in rows if int(r.get("success", 0)) == 0]
    by_fail = Counter(str(r.get("fail_type", "unknown")) for r in rows)
    by_bucket = Counter(str(r.get("goal_bucket", "unlabeled")) for r in rows)

    cards_json = out_dir / "cards.json"
    cards_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    table_rows = []
    for r in rows:
        run_id = str(r.get("run_id", ""))
        ok = int(r.get("success", 0))
        fail_type = str(r.get("fail_type", ""))
        reason = str(r.get("reason", ""))
        steps = int(r.get("steps_used", 0) or 0)
        dist = float(r.get("final_dist_to_goal_node", -1.0) or -1.0)
        scene = str(r.get("scene_id", ""))
        sid = str(r.get("start_id", ""))
        gid = str(r.get("goal_id", ""))
        gb = str(r.get("goal_bucket", ""))
        log_path = _rel(str(r.get("log_path", "")), out_dir.parent)
        dbg = _rel(str(r.get("debug_index", "")), out_dir.parent)
        table_rows.append(
            f"<tr><td>{run_id}</td><td>{scene}</td><td>{sid}</td><td>{gid}</td><td>{gb}</td>"
            f"<td>{ok}</td><td>{fail_type}</td><td>{reason}</td><td>{steps}</td><td>{dist:.3f}</td>"
            f"<td><a href='../{log_path}'>log</a></td><td>{('<a href=\'../'+dbg+'\'>debug</a>') if dbg and dbg!='.' else ''}</td></tr>"
        )

    top_fail = ", ".join([f"{k}:{v}" for k, v in sorted(by_fail.items())])
    top_bucket = ", ".join([f"{k}:{v}" for k, v in sorted(by_bucket.items())])

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'/><title>V31 Failure Cards</title>
<style>
body{{font-family:Arial,sans-serif;background:#f7f8fa;margin:0}}
.top{{background:#111827;color:#fff;padding:12px 16px}}
.wrap{{padding:12px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin-bottom:10px}}
table{{border-collapse:collapse;width:100%;background:#fff}}
th,td{{border:1px solid #ddd;padding:6px;font-size:12px}}th{{background:#f3f4f6;text-align:left}}
</style></head>
<body>
<div class='top'>V31 Failure Cards</div>
<div class='wrap'>
  <div class='card'><b>Cases</b>: {len(rows)} | <b>Failures</b>: {len(failures)}<br/>
  <b>by_fail_type</b>: {top_fail}<br/>
  <b>by_goal_bucket</b>: {top_bucket}</div>
  <table>
    <thead><tr><th>run_id</th><th>scene</th><th>start_id</th><th>goal_id</th><th>bucket</th><th>success</th><th>fail_type</th><th>reason</th><th>steps</th><th>final_dist</th><th>log</th><th>debug</th></tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
</div>
</body></html>
"""

    out_html = out_dir / "index.html"
    out_html.write_text(html, encoding="utf-8")

    print(f"[V31_FAILURE_CARDS_OK] path={out_html} cases={len(rows)} failures={len(failures)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
