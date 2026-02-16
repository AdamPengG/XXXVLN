#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple


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
                val = attr.Get()
                if val is False:
                    return False
            return True
    except Exception:
        pass
    try:
        if hasattr(PhysxSchema, "PhysxCollisionAPI") and prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            return True
    except Exception:
        pass
    return False


def _apply_collision(prim, UsdPhysics, PhysxSchema, method: str) -> bool:
    changed = False
    try:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
            changed = True
        col = UsdPhysics.CollisionAPI(prim)
        if col:
            attr = col.GetCollisionEnabledAttr()
            if not attr.IsValid():
                attr = col.CreateCollisionEnabledAttr()
            if attr.Get() is not True:
                attr.Set(True)
                changed = True
    except Exception:
        pass

    try:
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        if mesh_api:
            approx = mesh_api.GetApproximationAttr()
            if not approx.IsValid():
                approx = mesh_api.CreateApproximationAttr()
            current = approx.Get()
            target = "convexHull" if method == "convexHull" else "meshSimplification"
            if current != target:
                approx.Set(target)
                changed = True
    except Exception:
        pass

    try:
        if hasattr(PhysxSchema, "PhysxCollisionAPI"):
            if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
                changed = True
    except Exception:
        pass
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="")
    ap.add_argument("--out_usd", default="runs/topo_mvp/v34b_nav2_demo/office_phys/office_phys.usd")
    ap.add_argument("--method", default=os.environ.get("V34B_COLLIDER_METHOD", "convexHull"))
    args = ap.parse_args()

    src_stage, src_kind = _resolve_stage(args.stage)
    src_exists = int(bool(src_stage) and Path(src_stage).is_file())
    print(f"[V34B_ISAAC_STAGE] usd={src_stage} exists={src_exists} source={src_kind}", flush=True)
    if src_exists != 1:
        print("[ISAAC_COLLISION_BUILD] ok=0 out_usd= meshes_processed=0 colliders_added=0 method=unknown reason=stage_missing", flush=True)
        return 2

    from omni.isaac.kit import SimulationApp  # type: ignore

    sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    try:
        import omni.usd  # type: ignore
        from omni.isaac.core.utils.stage import open_stage  # type: ignore
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # type: ignore

        if not open_stage(src_stage):
            print("[ISAAC_COLLISION_BUILD] ok=0 out_usd= meshes_processed=0 colliders_added=0 method=unknown reason=open_stage_failed", flush=True)
            return 2
        for _ in range(20):
            sim_app.update()
        source_stage = omni.usd.get_context().get_stage()
        if source_stage is None:
            print("[ISAAC_COLLISION_BUILD] ok=0 out_usd= meshes_processed=0 colliders_added=0 method=unknown reason=stage_none", flush=True)
            return 2

        missing_paths: List[str] = []
        mesh_count = 0
        for prim in source_stage.TraverseAll():
            if not prim.IsActive() or prim.IsAbstract():
                continue
            try:
                is_mesh = prim.IsA(UsdGeom.Mesh)
            except Exception:
                is_mesh = prim.GetTypeName() == "Mesh"
            if not is_mesh:
                continue
            mesh_count += 1
            if not _has_collision(prim, UsdPhysics, PhysxSchema):
                missing_paths.append(str(prim.GetPath()))

        out_usd = Path(args.out_usd)
        out_usd.parent.mkdir(parents=True, exist_ok=True)
        if out_usd.exists():
            out_usd.unlink()

        overlay_stage = Usd.Stage.CreateNew(str(out_usd))
        root = overlay_stage.GetRootLayer()
        root.subLayerPaths.append(str(Path(src_stage).resolve()))

        processed = 0
        added = 0
        for path in missing_paths:
            prim = overlay_stage.OverridePrim(path)
            processed += 1
            if _apply_collision(prim, UsdPhysics, PhysxSchema, str(args.method)):
                added += 1

        overlay_stage.GetRootLayer().Save()

        meta = {
            "ok": 1,
            "source_stage": src_stage,
            "source_kind": src_kind,
            "out_usd": str(out_usd),
            "meshes_processed": int(mesh_count),
            "missing_meshes": int(len(missing_paths)),
            "colliders_added": int(added),
            "method": str(args.method),
            "overlay_sublayer": str(Path(src_stage).resolve()),
        }
        (out_usd.parent / "collision_build.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(
            f"[ISAAC_COLLISION_BUILD] ok=1 out_usd={out_usd} meshes_processed={mesh_count} colliders_added={added} method={args.method}",
            flush=True,
        )
        return 0
    finally:
        try:
            sim_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
