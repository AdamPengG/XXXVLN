#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def _resolve_stage(stage_arg: str) -> Tuple[str, str]:
    stage = str(stage_arg or os.environ.get("ISAAC_STAGE_USD", "")).strip()
    if stage:
        return stage, "override"
    roots = []
    env_root = str(os.environ.get("ISAAC_ASSETS_ROOT", "")).strip()
    if env_root:
        roots.append(env_root)
    roots.extend(["/home/peng/IsaacAssets", "/home/peng/isaacsim_assets"])
    for root in roots:
        for p in (
            Path(root) / "Assets/Isaac/5.1/Isaac/Environments/Office/office.usd",
            Path(root) / "Isaac/Environments/Office/office.usd",
            Path(root) / "Office/office.usd",
        ):
            if p.is_file():
                return str(p), "official_assets"
    return "", "missing"


def _has_collision(prim, UsdPhysics, PhysxSchema) -> bool:
    try:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            attr = prim.GetAttribute("physics:collisionEnabled")
            if attr.IsValid():
                v = attr.Get()
                if v is False:
                    return False
            return True
    except Exception:
        pass

    for attr_name in (
        "physics:collisionEnabled",
        "physxCollision:collisionEnabled",
        "collision:enabled",
    ):
        try:
            a = prim.GetAttribute(attr_name)
            if a.IsValid() and a.HasAuthoredValue():
                v = a.Get()
                if isinstance(v, bool):
                    return bool(v)
        except Exception:
            pass

    try:
        if hasattr(PhysxSchema, "PhysxCollisionAPI") and prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            return True
    except Exception:
        pass

    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="")
    ap.add_argument("--out_dir", default="runs/topo_mvp/v34b_nav2_demo/collision_audit")
    ap.add_argument("--allow_sparse_colliders", type=int, default=0)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--min_ratio", type=float, default=0.5)
    args = ap.parse_args()

    stage_path, stage_source = _resolve_stage(args.stage)
    exists = int(bool(stage_path) and Path(stage_path).is_file())
    print(f"[V34B_ISAAC_STAGE] usd={stage_path} exists={exists} source={stage_source}", flush=True)
    if exists != 1:
        print('[ISAAC_COLLISION_AUDIT] ok=0 total_prims=0 meshes=0 colliders=0 collider_ratio=0.0000 missing_topN="" reason=stage_missing', flush=True)
        return 2

    from omni.isaac.kit import SimulationApp  # type: ignore

    sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    try:
        import omni.usd  # type: ignore
        from omni.isaac.core.utils.stage import open_stage  # type: ignore
        from pxr import PhysxSchema, UsdGeom, UsdPhysics  # type: ignore

        if not open_stage(stage_path):
            print('[ISAAC_COLLISION_AUDIT] ok=0 total_prims=0 meshes=0 colliders=0 collider_ratio=0.0000 missing_topN="" reason=open_stage_failed', flush=True)
            return 2

        for _ in range(20):
            sim_app.update()

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print('[ISAAC_COLLISION_AUDIT] ok=0 total_prims=0 meshes=0 colliders=0 collider_ratio=0.0000 missing_topN="" reason=stage_none', flush=True)
            return 2

        total_prims = 0
        meshes = 0
        colliders = 0
        missing: List[str] = []

        for prim in stage.TraverseAll():
            total_prims += 1
            try:
                is_mesh = prim.IsA(UsdGeom.Mesh)
            except Exception:
                is_mesh = prim.GetTypeName() == "Mesh"
            if not is_mesh:
                continue
            meshes += 1
            if _has_collision(prim, UsdPhysics, PhysxSchema):
                colliders += 1
            else:
                missing.append(str(prim.GetPath()))

        collider_ratio = float(colliders) / float(meshes) if meshes > 0 else 0.0
        missing_top = missing[: max(1, int(args.topk))]
        missing_top_str = ";".join(missing_top)
        sparse = collider_ratio < float(args.min_ratio)
        allow_sparse = bool(int(args.allow_sparse_colliders))
        ok = 1 if (not sparse or allow_sparse) else 0

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "ok": int(ok),
            "stage": stage_path,
            "stage_source": stage_source,
            "total_prims": int(total_prims),
            "meshes": int(meshes),
            "colliders": int(colliders),
            "collider_ratio": float(collider_ratio),
            "min_ratio": float(args.min_ratio),
            "sparse": bool(sparse),
            "allow_sparse_colliders": int(allow_sparse),
            "missing_top": missing_top,
            "missing_count": int(len(missing)),
        }
        (out_dir / "collision_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        md = [
            "# Collision Audit v34b2",
            "",
            f"- stage: `{stage_path}`",
            f"- stage_source: `{stage_source}`",
            f"- total_prims: {total_prims}",
            f"- meshes: {meshes}",
            f"- colliders: {colliders}",
            f"- collider_ratio: {collider_ratio:.4f}",
            f"- sparse: {sparse}",
            f"- allow_sparse_colliders: {allow_sparse}",
            "",
            "## Missing top paths",
        ]
        md.extend([f"- `{p}`" for p in missing_top])
        (out_dir / "collision_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

        if ok:
            print(
                f"[ISAAC_COLLISION_AUDIT] ok=1 total_prims={total_prims} meshes={meshes} colliders={colliders} collider_ratio={collider_ratio:.4f} missing_topN=\"{missing_top_str}\"",
                flush=True,
            )
            if sparse:
                print(
                    f"[ISAAC_COLLISION_AUDIT_WARN] reason=sparse_colliders ratio={collider_ratio:.4f} min_ratio={float(args.min_ratio):.4f} allow_sparse=1",
                    flush=True,
                )
            return 0

        print(
            f"[ISAAC_COLLISION_AUDIT] ok=0 total_prims={total_prims} meshes={meshes} colliders={colliders} collider_ratio={collider_ratio:.4f} missing_topN=\"{missing_top_str}\" reason=sparse_colliders",
            flush=True,
        )
        return 3
    finally:
        try:
            sim_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
