#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

OFFICE_DEFAULT = "/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"


def _resolve_stage(stage_arg: str) -> Tuple[str, str]:
    stage = str(stage_arg or os.environ.get("ISAAC_STAGE_USD", "")).strip()
    if stage:
        return stage, "override"
    if Path(OFFICE_DEFAULT).is_file():
        return OFFICE_DEFAULT, "official_assets"
    return "", "missing"


def _has_collision(prim, UsdPhysics, PhysxSchema) -> bool:
    try:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            attr = prim.GetAttribute("physics:collisionEnabled")
            if attr.IsValid() and attr.HasAuthoredValue() and attr.Get() is False:
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
            target = "convexHull" if str(method).lower() == "convexhull" else "meshSimplification"
            if approx.Get() != target:
                approx.Set(target)
                changed = True
    except Exception:
        pass

    try:
        if hasattr(PhysxSchema, "PhysxCollisionAPI") and not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            PhysxSchema.PhysxCollisionAPI.Apply(prim)
            changed = True
    except Exception:
        pass

    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="")
    ap.add_argument("--out_usd", default="runs/topo_mvp/v34e_nav2_office_phys/office_phys/office_phys.usd")
    ap.add_argument("--method", default=os.environ.get("V34E_COLLIDER_METHOD", "convexHull"))
    args = ap.parse_args()

    src_stage, src_kind = _resolve_stage(args.stage)
    if not src_stage or not Path(src_stage).is_file():
        print("[V34E_COLLISION_BUILD] ok=0 out_usd= colliders_added=0 method=unknown reason=stage_missing", flush=True)
        return 2

    from omni.isaac.kit import SimulationApp  # type: ignore

    sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    try:
        import omni.usd  # type: ignore
        from omni.isaac.core.utils.stage import open_stage  # type: ignore
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # type: ignore

        if not open_stage(src_stage):
            print("[V34E_COLLISION_BUILD] ok=0 out_usd= colliders_added=0 method=unknown reason=open_stage_failed", flush=True)
            return 2
        for _ in range(20):
            sim_app.update()
        source_stage = omni.usd.get_context().get_stage()
        if source_stage is None:
            print("[V34E_COLLISION_BUILD] ok=0 out_usd= colliders_added=0 method=unknown reason=stage_none", flush=True)
            return 2

        out_usd = Path(args.out_usd)
        out_usd.parent.mkdir(parents=True, exist_ok=True)
        if out_usd.exists():
            out_usd.unlink()

        overlay_stage = Usd.Stage.CreateNew(str(out_usd))
        overlay_stage.SetMetadata("metersPerUnit", 1.0)
        try:
            src_up = UsdGeom.GetStageUpAxis(source_stage)
            UsdGeom.SetStageUpAxis(overlay_stage, src_up)
        except Exception:
            src_up = UsdGeom.Tokens.y
        overlay_stage.GetRootLayer().subLayerPaths.append(str(Path(src_stage).resolve()))

        mesh_total = 0
        added = 0
        processed = 0
        missing_paths: List[str] = []

        for prim in source_stage.TraverseAll():
            if not prim.IsActive() or prim.IsAbstract():
                continue
            try:
                is_mesh = prim.IsA(UsdGeom.Mesh)
            except Exception:
                is_mesh = prim.GetTypeName() == "Mesh"
            if not is_mesh:
                continue
            mesh_total += 1
            p = str(prim.GetPath())
            missing_paths.append(p)
            prim_ovr = overlay_stage.OverridePrim(p)
            processed += 1
            if _apply_collision(prim_ovr, UsdPhysics, PhysxSchema, str(args.method)):
                added += 1

        # Ensure a simple ground collider exists for safety.
        ground_prim = overlay_stage.GetPrimAtPath("/World/V34EGround")
        if not ground_prim.IsValid():
            plane = UsdGeom.Cube.Define(overlay_stage, Sdf.Path("/World/V34EGround"))
            plane.AddScaleOp().Set(Gf.Vec3f(100.0, 0.1, 100.0))
            if str(src_up).upper().startswith("Z"):
                plane.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -0.05))
            else:
                plane.AddTranslateOp().Set(Gf.Vec3f(0.0, -0.05, 0.0))
            pprim = plane.GetPrim()
            UsdPhysics.CollisionAPI.Apply(pprim)
            UsdPhysics.MeshCollisionAPI.Apply(pprim)
            added += 1

        overlay_stage.GetRootLayer().Save()

        report = {
            "ok": 1,
            "source_stage": src_stage,
            "source_kind": src_kind,
            "out_usd": str(out_usd),
            "meshes_total": mesh_total,
            "meshes_processed": processed,
            "colliders_added": added,
            "method": str(args.method),
        }
        (out_usd.parent / "collision_build_v34e.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(
            f"[V34E_COLLISION_BUILD] ok=1 out_usd={out_usd} colliders_added={added} method={args.method}",
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
