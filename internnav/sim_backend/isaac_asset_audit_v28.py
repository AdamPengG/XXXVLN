from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

SIM_APP = None
try:
    from pxr import Sdf, Usd
except ModuleNotFoundError:
    from omni.isaac.kit import SimulationApp  # type: ignore

    SIM_APP = SimulationApp({"headless": True})
    from pxr import Sdf, Usd

USD_EXTS = {".usd", ".usda", ".usdc", ".usdz"}
TEX_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tga",
    ".exr",
    ".hdr",
    ".dds",
    ".ktx",
    ".ktx2",
    ".tif",
    ".tiff",
}
MDL_EXTS = {".mdl"}


@dataclass(frozen=True)
class DepEntry:
    path: str
    dep_type: str
    source: str
    raw: str


def _norm_abs(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def _strip_asset_token(path: str) -> str:
    s = str(path or "").strip()
    if s.startswith("@") and s.endswith("@") and len(s) >= 2:
        s = s[1:-1]
    if s.startswith("file://"):
        s = s[7:]
    return s.strip()


def _dep_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in USD_EXTS:
        return "usd"
    if ext in TEX_EXTS:
        return "texture"
    if ext in MDL_EXTS:
        return "mdl"
    return "other"


def _is_uri(path: str) -> bool:
    p = path.lower()
    return "://" in p and not p.startswith("file://")


def _is_within(path: str, root: str) -> bool:
    try:
        Path(_norm_abs(path)).relative_to(Path(_norm_abs(root)))
        return True
    except Exception:
        return False


def _extract_asset_paths(value: Any) -> List[str]:
    out: List[str] = []
    if value is None:
        return out
    if isinstance(value, Sdf.AssetPath):
        raw = value.path or value.resolvedPath or ""
        if raw:
            out.append(str(raw))
        return out
    if isinstance(value, str):
        s = value.strip()
        if s:
            out.append(s)
        return out
    if isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_extract_asset_paths(item))
        return out
    if hasattr(value, "path") and hasattr(value, "resolvedPath"):
        raw = getattr(value, "path", "") or getattr(value, "resolvedPath", "")
        if raw:
            out.append(str(raw))
        return out
    return out


def _resolve_candidate(raw: str, base_dir: str, search_paths: List[str]) -> str:
    token = _strip_asset_token(raw)
    if not token:
        return ""
    tnorm = token.lower()
    if tnorm.startswith("anon:") or tnorm in {"none", "guide"}:
        return ""
    if _is_uri(token):
        return token

    cand: List[str] = []
    if os.path.isabs(token):
        cand.append(token)
    else:
        cand.append(os.path.join(base_dir, token))
        for sp in search_paths:
            cand.append(os.path.join(sp, token))

    seen: Set[str] = set()
    normalized: List[str] = []
    for c in cand:
        n = _norm_abs(c)
        if n not in seen:
            seen.add(n)
            normalized.append(n)
    for n in normalized:
        if os.path.exists(n):
            return n
    return normalized[0] if normalized else token


def _iter_children(spec: Any) -> Iterable[Any]:
    obj = getattr(spec, "nameChildren", None)
    if obj is None:
        return []
    try:
        if isinstance(obj, dict):
            return list(obj.values())
        if hasattr(obj, "values"):
            return list(obj.values())
        return list(obj)
    except Exception:
        return []


def _iter_attrs(spec: Any) -> Iterable[Any]:
    obj = getattr(spec, "attributes", None)
    if obj is None:
        return []
    try:
        if isinstance(obj, dict):
            return list(obj.values())
        if hasattr(obj, "values"):
            return list(obj.values())
        return list(obj)
    except Exception:
        return []


def _listop_items(list_op: Any) -> Iterable[Any]:
    if list_op is None:
        return []
    fields = ("explicitItems", "prependedItems", "appendedItems")
    out: List[Any] = []
    for field in fields:
        try:
            items = list(getattr(list_op, field))
            out.extend(items)
        except Exception:
            continue
    return out


def _scan_layer_for_assets(layer: Sdf.Layer, search_paths: List[str]) -> List[DepEntry]:
    deps: List[DepEntry] = []
    layer_path = layer.realPath or layer.identifier
    layer_dir = os.path.dirname(layer_path) if layer_path else os.getcwd()

    for rel in list(layer.subLayerPaths or []):
        resolved = _resolve_candidate(str(rel), layer_dir, search_paths)
        if resolved:
            deps.append(DepEntry(path=resolved, dep_type=_dep_type(resolved), source=layer_path, raw=str(rel)))

    stack: List[Any] = []
    try:
        roots = list(layer.rootPrims or [])
        stack.extend(roots)
    except Exception:
        pass

    while stack:
        spec = stack.pop()
        for child in _iter_children(spec):
            stack.append(child)

        for ref in _listop_items(getattr(spec, "referenceList", None)):
            raw = str(getattr(ref, "assetPath", "") or "")
            if raw:
                resolved = _resolve_candidate(raw, layer_dir, search_paths)
                deps.append(DepEntry(path=resolved, dep_type=_dep_type(resolved), source=layer_path, raw=raw))

        for payload in _listop_items(getattr(spec, "payloadList", None)):
            raw = str(getattr(payload, "assetPath", "") or "")
            if raw:
                resolved = _resolve_candidate(raw, layer_dir, search_paths)
                deps.append(DepEntry(path=resolved, dep_type=_dep_type(resolved), source=layer_path, raw=raw))

        for attr in _iter_attrs(spec):
            try:
                val = getattr(attr, "default", None)
            except Exception:
                val = None
            if val is None:
                try:
                    val = attr.GetInfo("default")
                except Exception:
                    val = None
            for raw in _extract_asset_paths(val):
                resolved = _resolve_candidate(raw, layer_dir, search_paths)
                if resolved:
                    deps.append(DepEntry(path=resolved, dep_type=_dep_type(resolved), source=layer_path, raw=raw))

    return deps


def _scan_stage_asset_attrs(stage: Usd.Stage, search_paths: List[str], stage_path: str) -> List[DepEntry]:
    deps: List[DepEntry] = []
    for prim in stage.TraverseAll():
        for attr in prim.GetAttributes():
            try:
                type_name = str(attr.GetTypeName())
            except Exception:
                type_name = ""
            if "asset" not in type_name.lower():
                continue
            try:
                val = attr.Get()
            except Exception:
                val = None
            if val is None:
                continue
            source_layer = ""
            try:
                source_layer = attr.GetPropertyStack(0.0)[0].layer.realPath
            except Exception:
                source_layer = stage_path
            base_dir = os.path.dirname(source_layer or stage_path)
            for raw in _extract_asset_paths(val):
                resolved = _resolve_candidate(raw, base_dir, search_paths)
                if resolved:
                    deps.append(DepEntry(path=resolved, dep_type=_dep_type(resolved), source=source_layer, raw=raw))
    return deps


_MDL_IMPORT_RE = re.compile(r"^\s*import\s+([^\s;]+)\s*;", re.IGNORECASE)
_MDL_USING_RE = re.compile(r"^\s*using\s+([^\s;]+)\s+import\b", re.IGNORECASE)
_MDL_BUILTIN_MODULES = {
    "anno",
    "base",
    "df",
    "edf",
    "math",
    "scene",
    "state",
    "tex",
}


def _resolve_mdl_import_token(token: str, mdl_dir: str, search_paths: List[str]) -> str:
    t = str(token or "").strip().strip('"').strip("'")
    if not t:
        return ""
    if t.endswith("::*"):
        t = t[:-3]
    if not t:
        return ""
    base = t.lstrip(".:")
    head = base.split("::", 1)[0].strip().lower()
    if head in _MDL_BUILTIN_MODULES:
        return ""
    # Relative MDL module path.
    if t.startswith(".::"):
        rel = t[3:].replace("::", "/") + ".mdl"
        return _resolve_candidate(rel, mdl_dir, [mdl_dir] + list(search_paths))
    # Absolute MDL module path.
    if t.startswith("::"):
        rel = t[2:].replace("::", "/") + ".mdl"
        return _resolve_candidate(rel, mdl_dir, search_paths)
    # File path-like import.
    if t.endswith(".mdl") or "/" in t or "\\" in t:
        return _resolve_candidate(t, mdl_dir, [mdl_dir] + list(search_paths))
    # Plain module namespace import.
    rel = t.replace("::", "/") + ".mdl"
    return _resolve_candidate(rel, mdl_dir, [mdl_dir] + list(search_paths))


def _scan_mdl_imports(mdl_path: str, search_paths: List[str]) -> List[DepEntry]:
    deps: List[DepEntry] = []
    try:
        text = Path(mdl_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return deps
    mdl_dir = os.path.dirname(mdl_path)
    for line in text.splitlines():
        m = _MDL_IMPORT_RE.match(line) or _MDL_USING_RE.match(line)
        if not m:
            continue
        token = str(m.group(1) or "").strip()
        if not token:
            continue
        resolved = _resolve_mdl_import_token(token, mdl_dir=mdl_dir, search_paths=search_paths)
        if resolved:
            deps.append(
                DepEntry(path=resolved, dep_type=_dep_type(resolved), source=mdl_path, raw=token)
            )
    return deps


def _safe_open_check(path: str, safe_check: bool) -> Tuple[bool, str]:
    try:
        layer = Sdf.Layer.FindOrOpen(path)
        if layer is not None:
            return True, ""
    except Exception as e:
        if not safe_check:
            return False, f"{type(e).__name__}:{e}"
    if not safe_check:
        return False, "layer_open_failed"

    code = (
        "from pxr import Sdf\n"
        "import sys\n"
        "p=sys.argv[1]\n"
        "try:\n"
        "  l=Sdf.Layer.FindOrOpen(p)\n"
        "  rc=(0 if l is not None else 3)\n"
        "except Exception as e:\n"
        "  print(type(e).__name__ + ':' + str(e))\n"
        "  rc=4\n"
        "sys.exit(rc)\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as e:
        return False, f"{type(e).__name__}:{e}"
    if proc.returncode == 0:
        return True, ""
    msg = (proc.stderr or proc.stdout or f"exit={proc.returncode}").strip().replace("\n", " ")
    return False, msg[:300]


def run_audit(
    stage: str,
    assets_root: str,
    out_dir: str,
    allow_missing: int,
    allow_unreadable: int,
    safe_check: bool,
) -> int:
    stage = _norm_abs(stage)
    assets_root = _norm_abs(assets_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(stage):
        print(
            f"[ISAAC_ASSET_AUDIT] ok=0 reason=stage_not_found hint={stage}",
            flush=True,
        )
        return 2

    search_paths = [os.path.dirname(stage), assets_root]
    os.environ["PXR_AR_DEFAULT_SEARCH_PATH"] = os.pathsep.join(search_paths)

    stage_obj = Usd.Stage.Open(stage)
    if stage_obj is None:
        print(
            f"[ISAAC_ASSET_AUDIT] ok=0 reason=stage_open_failed hint={stage}",
            flush=True,
        )
        return 2

    deps: List[DepEntry] = []
    unique_keys: Set[Tuple[str, str, str]] = set()

    def _add(dep: DepEntry) -> None:
        key = (dep.path, dep.dep_type, dep.source)
        if key in unique_keys:
            return
        unique_keys.add(key)
        deps.append(dep)

    for layer in list(stage_obj.GetUsedLayers() or []):
        layer_path = layer.realPath or layer.identifier
        if layer_path and not str(layer_path).lower().startswith("anon:"):
            _add(DepEntry(path=_norm_abs(layer_path), dep_type="usd", source=stage, raw=layer_path))
        for dep in _scan_layer_for_assets(layer, search_paths=search_paths):
            _add(dep)

    for dep in _scan_stage_asset_attrs(stage_obj, search_paths=search_paths, stage_path=stage):
        _add(dep)

    # Expand MDL import graph recursively so localize captures material module closure.
    mdl_queue: List[str] = []
    mdl_seen: Set[str] = set()
    for dep in list(deps):
        if dep.dep_type == "mdl":
            mdl_path = _norm_abs(dep.path)
            if mdl_path not in mdl_seen:
                mdl_seen.add(mdl_path)
                mdl_queue.append(mdl_path)
    while mdl_queue:
        mdl = mdl_queue.pop(0)
        for dep in _scan_mdl_imports(mdl, search_paths=search_paths):
            _add(dep)
            if dep.dep_type == "mdl":
                mdl_path = _norm_abs(dep.path)
                if mdl_path not in mdl_seen:
                    mdl_seen.add(mdl_path)
                    mdl_queue.append(mdl_path)

    # Always include the root stage.
    _add(DepEntry(path=stage, dep_type="usd", source=stage, raw=stage))

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    unreadable: List[str] = []
    outside_root: List[str] = []

    for dep in sorted(deps, key=lambda d: (d.path, d.dep_type, d.source, d.raw)):
        path = str(dep.path)
        exists = os.path.isfile(path) if not _is_uri(path) else False
        in_root = _is_within(path, assets_root) if exists else False
        readable = True
        read_err = ""
        if dep.dep_type == "usd" and exists:
            readable, read_err = _safe_open_check(path=path, safe_check=safe_check)
        count_missing = dep.dep_type in {"usd", "texture", "mdl"}
        if (not exists) and count_missing:
            missing.append(path)
        if exists and dep.dep_type == "usd" and (not readable):
            unreadable.append(path)
        if exists and (not in_root):
            outside_root.append(path)
        if _is_uri(path):
            outside_root.append(path)
        rows.append(
            {
                "path": path,
                "type": dep.dep_type,
                "source_layer": dep.source,
                "raw": dep.raw,
                "exists": bool(exists),
                "readable": bool(readable),
                "read_error": str(read_err),
                "in_assets_root": bool(in_root),
            }
        )

    missing = sorted(set(missing))
    unreadable = sorted(set(unreadable))
    outside_root = sorted(set(outside_root))
    type_counter = Counter([r["type"] for r in rows])

    report = {
        "stage": stage,
        "assets_root": assets_root,
        "search_paths": search_paths,
        "total_dependencies": len(rows),
        "counts": {
            "usd": int(type_counter.get("usd", 0)),
            "textures": int(type_counter.get("texture", 0)),
            "mdl": int(type_counter.get("mdl", 0)),
            "other": int(type_counter.get("other", 0)),
            "missing": len(missing),
            "unreadable": len(unreadable),
            "outside_root": len(outside_root),
        },
        "allow_missing": int(allow_missing),
        "allow_unreadable": int(allow_unreadable),
        "safe_check": int(bool(safe_check)),
        "dependencies": rows,
        "missing_paths": missing,
        "unreadable_paths": unreadable,
        "outside_root_paths": outside_root,
    }

    report_json = out / "asset_audit_report.json"
    summary_md = out / "asset_audit_summary.md"
    dep_list = out / "deps_manifest.txt"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    dep_lines = [f"{r['type']}\t{r['path']}" for r in rows]
    dep_list.write_text("\n".join(dep_lines) + ("\n" if dep_lines else ""))

    md_lines = [
        "# v28 Isaac Asset Audit Summary",
        "",
        f"- stage: `{stage}`",
        f"- assets_root: `{assets_root}`",
        f"- total_dependencies: {len(rows)}",
        f"- usd: {type_counter.get('usd', 0)}",
        f"- textures: {type_counter.get('texture', 0)}",
        f"- mdl: {type_counter.get('mdl', 0)}",
        f"- other: {type_counter.get('other', 0)}",
        f"- missing: {len(missing)}",
        f"- unreadable: {len(unreadable)}",
        f"- outside_root: {len(outside_root)}",
        "",
        "## Missing",
    ]
    md_lines += [f"- `{p}`" for p in missing[:200]] or ["- (none)"]
    md_lines += ["", "## Unreadable USDs"]
    md_lines += [f"- `{p}`" for p in unreadable[:200]] or ["- (none)"]
    md_lines += ["", "## Outside Root"]
    md_lines += [f"- `{p}`" for p in outside_root[:200]] or ["- (none)"]
    summary_md.write_text("\n".join(md_lines) + "\n")

    ok = (
        len(missing) <= int(allow_missing)
        and len(unreadable) <= int(allow_unreadable)
        and len(outside_root) == 0
    )
    if ok:
        print(
            f"[ISAAC_ASSET_AUDIT] ok=1 stage={stage} assets_root={assets_root} "
            f"total={len(rows)} usd={int(type_counter.get('usd', 0))} "
            f"textures={int(type_counter.get('texture', 0))} missing={len(missing)} "
            f"unreadable={len(unreadable)} outside_root={len(outside_root)}",
            flush=True,
        )
        print(
            f"[ISAAC_ASSET_AUDIT_OUT] dir={out} report_json={report_json} "
            f"summary_md={summary_md} dep_list={dep_list}",
            flush=True,
        )
        return 0

    reason = []
    if len(missing) > int(allow_missing):
        reason.append("missing")
    if len(unreadable) > int(allow_unreadable):
        reason.append("unreadable")
    if len(outside_root) > 0:
        reason.append("outside_root")
    rs = ",".join(reason) if reason else "unknown"
    print(
        f"[ISAAC_ASSET_AUDIT] ok=0 reason={rs} hint=check_asset_paths_and_localization",
        flush=True,
    )
    print(
        f"[ISAAC_ASSET_AUDIT_OUT] dir={out} report_json={report_json} "
        f"summary_md={summary_md} dep_list={dep_list}",
        flush=True,
    )
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="v28 Isaac asset audit.")
    ap.add_argument("--stage", type=str, required=True)
    ap.add_argument("--assets_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--allow_missing", type=int, default=0)
    ap.add_argument("--allow_unreadable", type=int, default=0)
    ap.add_argument("--safe_check", type=int, default=1)
    args = ap.parse_args()

    return run_audit(
        stage=args.stage,
        assets_root=args.assets_root,
        out_dir=args.out_dir,
        allow_missing=int(args.allow_missing),
        allow_unreadable=int(args.allow_unreadable),
        safe_check=bool(int(args.safe_check)),
    )


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        if SIM_APP is not None:
            try:
                SIM_APP.close()
            except Exception:
                pass
    raise SystemExit(code)
