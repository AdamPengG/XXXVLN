from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from internnav.sim_backend.isaac_camera_config_v29 import (
    IsaacCameraConfig,
    apply_camera_config,
    build_camera_config,
    depth_stats,
    rgb_fidelity_stats,
)
from internnav.sim_backend.base import Obs, Pose, SimBackend
from internnav.topo.local_planner import plan_grid_path


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _wrap_pi(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


class IsaacSimBackend(SimBackend):
    """
    Isaac backend with optional fallback to Habitat.

    v22 default is strict real-isaac: fallback is disabled unless
    ISAAC_BACKEND_ALLOW_FALLBACK=1 is explicitly set by caller.
    """

    def __init__(
        self,
        scene_id: str,
        scene_cfg: Dict[str, object],
        config_path: str,
        out_dir: str,
        fallback_to_habitat: bool = False,
        dt_action: float = 0.5,
    ) -> None:
        self.scene_id = str(scene_id)
        self.scene_cfg = dict(scene_cfg or {})
        self.dt_action = float(dt_action)
        self.forward_speed = float(self.scene_cfg.get("forward_speed_mps", 0.25))
        self.turn_rate_deg = float(self.scene_cfg.get("turn_rate_degps", 15.0))
        self.turn_rate = math.radians(self.turn_rate_deg)
        self._allow_fallback = bool(fallback_to_habitat)

        self._fallback = None
        self._isaac_ready = False
        self._world = None
        self._sim_app = None
        self._camera = None

        self._pose_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._yaw = 0.0
        self._last_obs: Optional[Obs] = None
        self._last_collision: Optional[bool] = None
        self._goal_geodesic: Optional[float] = None
        self._backend_mode = "real_isaac"
        self._renderer_requested = str(os.environ.get("ISAAC_RENDERER", "rtx")).strip().lower() or "rtx"
        self._renderer_fallback = "storm" if self._renderer_requested == "rtx" else "rtx"
        self._renderer_used = "unknown"
        self._enable_depth = bool(int(os.environ.get("ISAAC_ENABLE_DEPTH", "0")))
        self._rgb_capture = bool(int(os.environ.get("ISAAC_RGB_CAPTURE", "0")))
        self._skip_world_step = bool(int(os.environ.get("ISAAC_SKIP_WORLD_STEP", "0")))
        self._render = bool(int(os.environ.get("ISAAC_RENDER", "0")))
        # v33c: always render sensor frames on step by default so RGB/depth
        # are refreshed with pose updates during PlanB rollouts.
        self._step_render = bool(int(os.environ.get("ISAAC_STEP_RENDER", "1")))
        self._step_render_every_n = max(1, int(os.environ.get("ISAAC_STEP_RENDER_EVERY_N", "1")))
        self._obs_step_idx = 0
        self._minimal = bool(int(os.environ.get("ISAAC_MINIMAL", "0")))
        self._camera_cfg: IsaacCameraConfig = build_camera_config(self.scene_cfg)
        self._camera_cfg_info: Dict[str, object] = {}
        self._camera_fidelity_info: Dict[str, object] = {}
        self._camera_probe_done = False
        self._stage_up_axis = "Y"
        self._camera_height_m = float(os.environ.get("ISAAC_CAMERA_HEIGHT_M", "1.5"))
        # v26: RGB capture forces world init + render
        if self._rgb_capture:
            self._minimal = False
            self._render = True
            self._skip_world_step = False
            self._step_render = True
            self._step_render_every_n = 1

        bounds = self.scene_cfg.get("bounds", [-6.0, 6.0, -6.0, 6.0])
        if not isinstance(bounds, list) or len(bounds) != 4:
            bounds = [-6.0, 6.0, -6.0, 6.0]
        self._bounds = [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]

        self._obstacles = self._parse_obstacles(self.scene_cfg.get("obstacles", []))
        if len(self._obstacles) == 0:
            self._obstacles = [
                {"x": 0.0, "z": 1.5, "r": 0.6},
                {"x": -1.5, "z": -1.0, "r": 0.5},
                {"x": 2.0, "z": -0.5, "r": 0.7},
            ]

        usd_hint = str(self.scene_cfg.get("usd_hint", "")).strip()
        usd_path = str(self.scene_cfg.get("usd_path", "")).strip()
        resolved = ""
        if usd_path and os.path.isfile(usd_path):
            resolved = usd_path
        elif usd_hint:
            root = os.environ.get("ISAAC_SIM_ROOT", "")
            search_roots = [root, os.path.join(root, "assets"), os.path.join(root, "Isaac", "Assets")]
            for base in search_roots:
                if not base or not os.path.isdir(base):
                    continue
                for dirpath, _, files in os.walk(base):
                    for f in files:
                        if usd_hint in f and f.lower().endswith(".usd"):
                            resolved = os.path.join(dirpath, f)
                            break
                    if resolved:
                        break
                if resolved:
                    break
        if usd_hint or usd_path:
            print(f"[ISAAC_ASSET] usd_hint={usd_hint} resolved={resolved}", flush=True)
        self._usd_resolved = str(resolved)

        try:
            self._boot_sim_app()
            if not self._minimal:
                self._init_isaac_world()
            self._isaac_ready = True
            root = os.environ.get("ISAAC_SIM_ROOT", "")
            print(
                f"[ISAAC_BACKEND] mode=real_isaac scene={self.scene_id} root={root}",
                flush=True,
            )
        except Exception as e:
            err = f"{type(e).__name__}:{str(e).replace(chr(10), ' ')[:240]}"
            if not self._allow_fallback:
                print(f"[ISAAC_BACKEND_ERROR] reason={err}", flush=True)
                raise
            try:
                from internnav.sim_backend.habitat_backend import HabitatSimBackend

                self._fallback = HabitatSimBackend(config_path=config_path, out_dir=out_dir)
                self._backend_mode = "fallback_habitat"
                print(
                    f"[ISAAC_BACKEND] mode=fallback_habitat scene={self.scene_id} reason={err}",
                    flush=True,
                )
            except Exception as e2:
                err2 = f"{type(e2).__name__}:{str(e2).replace(chr(10), ' ')[:240]}"
                print(f"[ISAAC_BACKEND_ERROR] reason={err2}", flush=True)
                raise

    def _boot_sim_app(self) -> None:
        from omni.isaac.kit import SimulationApp  # type: ignore

        def _cfg(mode: str) -> Dict[str, object]:
            mode = (mode or "rtx").strip().lower()
            if mode == "storm":
                return {"headless": True, "renderer": "Mesa"}
            return {"headless": True, "renderer": "RayTracedLighting"}

        requested = self._renderer_requested
        fallback = self._renderer_fallback
        first_err = None
        try:
            self._sim_app = SimulationApp(_cfg(requested))
            self._renderer_used = requested
            os.environ["ISAAC_RENDERER_USED"] = self._renderer_used
            print(
                f"[ISAAC_RENDERER] requested={requested} fallback={fallback} ok=1 used={self._renderer_used}",
                flush=True,
            )
            return
        except Exception as e:
            first_err = e
            try:
                if self._sim_app is not None:
                    self._sim_app.close()
            except Exception:
                pass
            self._sim_app = None

        try:
            self._sim_app = SimulationApp(_cfg(fallback))
            self._renderer_used = fallback
            os.environ["ISAAC_RENDERER_USED"] = self._renderer_used
            print(
                f"[ISAAC_RENDERER] requested={requested} fallback={fallback} ok=1 used={self._renderer_used}",
                flush=True,
            )
            return
        except Exception as e2:
            reason = (
                f"{type(first_err).__name__}:{str(first_err).replace(chr(10), ' ')[:160]} | "
                f"{type(e2).__name__}:{str(e2).replace(chr(10), ' ')[:160]}"
            )
            print(
                f"[ISAAC_RENDERER] requested={requested} fallback={fallback} ok=0 reason={reason}",
                flush=True,
            )
            self._sim_app = None
            self._renderer_used = "none"
            os.environ["ISAAC_RENDERER_USED"] = self._renderer_used
            # Keep pipeline alive with synthetic RGB fallback.
            self._minimal = True
            self._render = False
            self._step_render = False
            self._skip_world_step = True
            return

    @staticmethod
    def _parse_obstacles(raw: object) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                out.append(
                    {
                        "x": float(item.get("x", 0.0)),
                        "z": float(item.get("z", 0.0)),
                        "r": float(item.get("r", item.get("radius", 0.6))),
                    }
                )
            except Exception:
                continue
        return out

    @property
    def using_fallback_habitat(self) -> bool:
        return self._fallback is not None

    @property
    def backend_mode(self) -> str:
        return str(self._backend_mode)

    @property
    def renderer_used(self) -> str:
        return str(self._renderer_used)

    def _init_isaac_world(self) -> None:
        from omni.isaac.core import World  # type: ignore
        from pxr import UsdGeom  # type: ignore

        if self._usd_resolved and os.path.isfile(self._usd_resolved):
            try:
                from omni.isaac.core.utils.stage import open_stage  # type: ignore

                open_stage(self._usd_resolved)
            except Exception:
                pass

        self._world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
        try:
            stage = self._world.stage
            if stage is not None:
                self._stage_up_axis = str(UsdGeom.GetStageUpAxis(stage)).upper()
        except Exception:
            self._stage_up_axis = "Y"
        try:
            self._world.scene.add_default_ground_plane()
        except Exception:
            # Avoid hard dependency on Isaac asset root.
            pass

        try:
            from pxr import Sdf, UsdLux  # type: ignore

            stage = self._world.stage
            if stage is not None:
                if stage.GetPrimAtPath("/World/TopoDomeLight").IsValid():
                    light = UsdLux.DistantLight(stage.GetPrimAtPath("/World/TopoDomeLight"))
                else:
                    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/TopoDomeLight"))
                light.CreateIntensityAttr(3000.0)
        except Exception:
            pass

        try:
            from omni.isaac.core.objects import VisualCuboid  # type: ignore

            for i, o in enumerate(self._obstacles):
                size = float(max(0.3, o["r"] * 2.0))
                cuboid = VisualCuboid(
                    prim_path=f"/World/Obs_{i}",
                    name=f"Obs_{i}",
                    position=np.array([o["x"], 0.6, o["z"]], dtype=np.float32),
                    scale=np.array([size, 1.2, size], dtype=np.float32),
                    color=np.array([0.7, 0.2, 0.2], dtype=np.float32),
                )
                self._world.scene.add(cuboid)
        except Exception:
            # Obstacles are also enforced in the kinematic collision check below.
            pass

        try:
            from omni.isaac.sensor import Camera  # type: ignore

            w = int(self._camera_cfg.width)
            h = int(self._camera_cfg.height)
            self._camera = Camera(
                prim_path="/World/TopoCamera",
                frequency=20,
                resolution=(w, h),
                position=np.array([0.0, float(self._camera_height_m), 0.0], dtype=np.float32),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
            self._camera.initialize()
            try:
                self._camera.add_rgba_to_frame()
            except Exception:
                pass
            try:
                self._camera.add_distance_to_image_plane_to_frame()
            except Exception:
                pass
            self._camera_cfg_info = apply_camera_config(self._camera, self._camera_cfg)
            if len(self._camera_cfg_info) > 0:
                print(
                    f"[ISAAC_CAMERA_CFG] w={int(self._camera_cfg_info.get('camera_w', w))} "
                    f"h={int(self._camera_cfg_info.get('camera_h', h))} "
                    f"fov_deg={float(self._camera_cfg_info.get('camera_fov_deg', self._camera_cfg.fov_deg)):.2f} "
                    f"near={float(self._camera_cfg_info.get('camera_near', self._camera_cfg.near_m)):.4f} "
                    f"far={float(self._camera_cfg_info.get('camera_far', self._camera_cfg.far_m)):.2f} "
                    f"auto_exposure={int(self._camera_cfg_info.get('auto_exposure', self._camera_cfg.auto_exposure))} "
                    f"exposure={float(self._camera_cfg_info.get('exposure', self._camera_cfg.exposure)):.4f} "
                    f"aspect={self._camera_cfg_info.get('aspect_policy', self._camera_cfg.aspect_policy)}",
                    flush=True,
                )
        except Exception:
            self._camera = None
            self._camera_cfg_info = {
                "camera_w": int(self._camera_cfg.width),
                "camera_h": int(self._camera_cfg.height),
                "camera_fov_deg": float(self._camera_cfg.fov_deg),
                "camera_near": float(self._camera_cfg.near_m),
                "camera_far": float(self._camera_cfg.far_m),
                "auto_exposure": int(self._camera_cfg.auto_exposure),
                "exposure": float(self._camera_cfg.exposure),
                "aspect_policy": str(self._camera_cfg.aspect_policy),
            }
            print(
                f"[ISAAC_CAMERA_CFG] w={self._camera_cfg.width} h={self._camera_cfg.height} "
                f"fov_deg={self._camera_cfg.fov_deg:.2f} near={self._camera_cfg.near_m:.4f} "
                f"far={self._camera_cfg.far_m:.2f} auto_exposure={self._camera_cfg.auto_exposure} "
                f"exposure={self._camera_cfg.exposure:.4f} aspect={self._camera_cfg.aspect_policy}",
                flush=True,
            )

        self._world.reset()
        if not self._skip_world_step:
            warm_render = bool(self._render or self._step_render)
            for _ in range(3):
                self._world.step(render=warm_render)
        print(
            f"[ISAAC_STEP_CFG] render_legacy={int(self._render)} step_render={int(self._step_render)} "
            f"step_render_every_n={int(self._step_render_every_n)} skip_world_step={int(self._skip_world_step)}",
            flush=True,
        )

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        # Quaternion multiply in wxyz convention.
        w1, x1, y1, z1 = [float(v) for v in q1]
        w2, x2, y2, z2 = [float(v) for v in q2]
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float32,
        )

    def _set_pose(self, xyz: np.ndarray, yaw: float) -> None:
        self._pose_xyz = np.asarray(xyz, dtype=np.float32)
        self._yaw = _wrap_pi(float(yaw))
        if self._camera is None:
            return
        if str(self._stage_up_axis).upper().startswith("Z"):
            pos = np.array(
                [
                    float(self._pose_xyz[0]),
                    float(self._pose_xyz[2]),
                    float(self._camera_height_m),
                ],
                dtype=np.float32,
            )
            # USD camera looks along local -Z; for Z-up stages rotate camera so
            # forward lies in XY plane, then apply yaw about world Z axis.
            q_base = np.array(
                [float(math.cos(-math.pi / 4.0)), float(math.sin(-math.pi / 4.0)), 0.0, 0.0],
                dtype=np.float32,
            )
            q_yaw = np.array(
                [float(math.cos(self._yaw / 2.0)), 0.0, 0.0, float(math.sin(self._yaw / 2.0))],
                dtype=np.float32,
            )
            quat_wxyz = self._quat_mul(q_yaw, q_base)
        else:
            pos = np.array(
                [float(self._pose_xyz[0]), float(self._camera_height_m), float(self._pose_xyz[2])],
                dtype=np.float32,
            )
            qw = float(math.cos(self._yaw / 2.0))
            qy = float(math.sin(self._yaw / 2.0))
            quat_wxyz = np.array([qw, 0.0, qy, 0.0], dtype=np.float32)
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        try:
            self._camera.set_world_pose(position=pos, orientation=quat_wxyz)
        except Exception:
            try:
                self._camera.set_world_pose(position=pos, orientation=quat_xyzw)
            except Exception:
                pass

    def _in_collision(self, pos: np.ndarray) -> bool:
        x = float(pos[0])
        z = float(pos[2])
        if x < self._bounds[0] or x > self._bounds[1] or z < self._bounds[2] or z > self._bounds[3]:
            return True
        for o in self._obstacles:
            dx = x - float(o["x"])
            dz = z - float(o["z"])
            if (dx * dx + dz * dz) <= float(o["r"] ** 2):
                return True
        return False

    def get_bounds(self) -> List[float]:
        return list(self._bounds)

    def get_obstacles(self) -> List[Dict[str, float]]:
        return list(self._obstacles)

    def get_motion_params(self) -> Dict[str, float]:
        return {
            "dt_action": float(self.dt_action),
            "forward_speed_mps": float(self.forward_speed),
            "turn_rate_radps": float(self.turn_rate),
        }

    def _simulate_step(
        self, pose_xyz: np.ndarray, yaw: float, action: int
    ) -> Tuple[np.ndarray, float, bool]:
        v = 0.0
        w = 0.0
        if action == 1:
            v = float(self.forward_speed)
        elif action == 2:
            w = float(self.turn_rate)
        elif action == 3:
            w = -float(self.turn_rate)
        dt = float(self.dt_action)
        new_yaw = _wrap_pi(float(yaw) + w * dt)
        new_xyz = np.array(pose_xyz, dtype=np.float32)
        if abs(v) > 1e-8:
            new_xyz[0] += float(math.sin(new_yaw) * v * dt)
            new_xyz[2] += float(-math.cos(new_yaw) * v * dt)
        collision = False
        if abs(v) > 1e-8 and self._in_collision(new_xyz):
            collision = True
            new_xyz = np.array(pose_xyz, dtype=np.float32)
        if self._in_collision(new_xyz):
            collision = True
            new_xyz = np.array(pose_xyz, dtype=np.float32)
        return new_xyz, new_yaw, bool(collision)

    def _simulate_waypoint_follow(
        self,
        start_xyz: np.ndarray,
        start_yaw: float,
        waypoints: List[Tuple[float, float]],
        max_steps: int,
        stop_thresh: float,
        turn_thresh: float,
    ) -> Tuple[bool, int]:
        wp_idx = 0
        cur = np.array(start_xyz, dtype=np.float32)
        yaw = float(start_yaw)
        steps = 0
        while steps < max_steps and wp_idx < len(waypoints):
            wx, wz = waypoints[wp_idx]
            delta = np.array([wx, 0.0, wz], dtype=np.float32) - cur
            dist = float(np.linalg.norm(delta[[0, 2]]))
            if dist <= stop_thresh:
                wp_idx += 1
                continue
            forward = np.array([math.sin(yaw), 0.0, -math.cos(yaw)], dtype=np.float32)
            right = np.array([math.cos(yaw), 0.0, math.sin(yaw)], dtype=np.float32)
            f_comp = float(np.dot(delta, forward))
            r_comp = float(np.dot(delta, right))
            bearing = _wrap_pi(math.atan2(r_comp, f_comp))
            if abs(bearing) > turn_thresh:
                action = 2 if bearing < 0 else 3
            else:
                action = 1
            cur, yaw, _ = self._simulate_step(cur, yaw, action)
            steps += 1
        success = wp_idx >= len(waypoints)
        return bool(success), int(steps)

    def _grid_astar(self, start: np.ndarray, goal: np.ndarray) -> Optional[float]:
        step = float(self.scene_cfg.get("oracle_grid_step", 0.25))
        if step <= 1e-6:
            return None
        xmin, xmax, zmin, zmax = self._bounds
        def to_idx(x: float, z: float) -> Tuple[int, int]:
            ix = int(round((x - xmin) / step))
            iz = int(round((z - zmin) / step))
            return ix, iz

        def to_pos(ix: int, iz: int) -> Tuple[float, float]:
            x = xmin + ix * step
            z = zmin + iz * step
            return x, z

        sx, sz = float(start[0]), float(start[2])
        gx, gz = float(goal[0]), float(goal[2])
        s_idx = to_idx(sx, sz)
        g_idx = to_idx(gx, gz)
        max_ix = int(round((xmax - xmin) / step))
        max_iz = int(round((zmax - zmin) / step))
        if max_ix <= 0 or max_iz <= 0:
            return None
        if s_idx == g_idx:
            return 0.0

        import heapq

        def h(ix: int, iz: int) -> float:
            x, z = to_pos(ix, iz)
            return float(math.sqrt((x - gx) ** 2 + (z - gz) ** 2))

        open_heap = []
        heapq.heappush(open_heap, (h(*s_idx), 0.0, s_idx))
        gscore = {s_idx: 0.0}
        visited = set()
        neigh = [
            (-1, 0, step),
            (1, 0, step),
            (0, -1, step),
            (0, 1, step),
            (-1, -1, step * math.sqrt(2.0)),
            (-1, 1, step * math.sqrt(2.0)),
            (1, -1, step * math.sqrt(2.0)),
            (1, 1, step * math.sqrt(2.0)),
        ]
        while open_heap:
            _, gcur, (ix, iz) = heapq.heappop(open_heap)
            if (ix, iz) in visited:
                continue
            visited.add((ix, iz))
            if (ix, iz) == g_idx:
                return float(gcur)
            for dx, dz, cost in neigh:
                nix = ix + dx
                niz = iz + dz
                if nix < 0 or nix > max_ix or niz < 0 or niz > max_iz:
                    continue
                x, z = to_pos(nix, niz)
                pos = np.array([x, 0.0, z], dtype=np.float32)
                if self._in_collision(pos):
                    continue
                ng = gcur + cost
                if ng < gscore.get((nix, niz), 1e12):
                    gscore[(nix, niz)] = ng
                    heapq.heappush(open_heap, (ng + h(nix, niz), ng, (nix, niz)))
        return float("inf")

    def _synthetic_rgbd(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        h = int(self._camera_cfg.height)
        w = int(self._camera_cfg.width)
        yy, xx = np.mgrid[0:h, 0:w]
        x01 = xx.astype(np.float32) / max(1.0, float(w - 1))
        y01 = yy.astype(np.float32) / max(1.0, float(h - 1))
        yaw01 = (_wrap_pi(self._yaw) + math.pi) / (2.0 * math.pi)
        px = (float(self._pose_xyz[0]) - self._bounds[0]) / max(1e-6, self._bounds[1] - self._bounds[0])
        pz = (float(self._pose_xyz[2]) - self._bounds[2]) / max(1e-6, self._bounds[3] - self._bounds[2])
        r = np.clip(0.55 * x01 + 0.45 * yaw01, 0.0, 1.0)
        g = np.clip(0.55 * y01 + 0.45 * px, 0.0, 1.0)
        b = np.clip(0.55 * (1.0 - x01) + 0.45 * pz, 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
        rgb = (255.0 * rgb).astype(np.uint8)
        return rgb, None

    def _apply_camera_tonemap(self, rgb: np.ndarray) -> np.ndarray:
        arr = np.asarray(rgb, dtype=np.uint8)
        if int(self._camera_cfg.auto_exposure) != 0:
            return arr
        exp_scale = float(max(1e-4, self._camera_cfg.exposure))
        if abs(exp_scale - 1.0) <= 1e-4:
            return arr
        out = np.clip(arr.astype(np.float32) * exp_scale, 0.0, 255.0).astype(np.uint8)
        return out

    def _read_camera_rgbd(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self._camera is None:
            if self._rgb_capture:
                print("[ISAAC_RGB] ok=0 reason=camera_none", flush=True)
            return self._synthetic_rgbd()
        rgb = None
        depth = None
        try:
            rgba = self._camera.get_rgba()
            if rgba is not None:
                arr = np.asarray(rgba)
                if arr.ndim == 3 and arr.shape[-1] >= 3:
                    rgb = arr[..., :3].astype(np.uint8)
        except Exception:
            rgb = None
        if rgb is None:
            try:
                frame = self._camera.get_current_frame()
                if isinstance(frame, dict):
                    rgba = frame.get("rgba", None)
                    if rgba is not None:
                        arr = np.asarray(rgba)
                        if arr.ndim == 3 and arr.shape[-1] >= 3:
                            rgb = arr[..., :3].astype(np.uint8)
            except Exception:
                rgb = None
        try:
            d = self._camera.get_depth()
            if d is not None:
                depth = np.asarray(d, dtype=np.float32)
                if depth.ndim == 3:
                    depth = depth[..., 0]
        except Exception:
            depth = None
        if not self._enable_depth:
            depth = None
        if rgb is not None:
            rgb = self._apply_camera_tonemap(rgb)
        if rgb is not None and self._rgb_capture:
            _var = float(np.var(rgb.astype(np.float32)))
            _mean = float(np.mean(rgb.astype(np.float32)))
            print(
                f"[ISAAC_RGB] ok=1 shape=({rgb.shape[0]},{rgb.shape[1]},3) "
                f"mean={_mean:.1f} var={_var:.1f}",
                flush=True,
            )
        if rgb is None:
            if self._rgb_capture:
                print("[ISAAC_RGB] ok=0 reason=get_rgba_returned_none", flush=True)
            rgb, syn_depth = self._synthetic_rgbd()
            if depth is None:
                depth = syn_depth
        return rgb, depth

    def _build_obs(self, info: Optional[Dict[str, object]] = None) -> Obs:
        world_stepped = 0
        sensor_rendered = 0
        if self._world is not None and not self._skip_world_step:
            self._obs_step_idx += 1
            periodic_render = bool(
                self._step_render and ((self._obs_step_idx % int(self._step_render_every_n)) == 0)
            )
            do_render = bool(self._render or periodic_render)
            self._world.step(render=do_render)
            world_stepped = 1
            sensor_rendered = int(do_render)
        elif self._world is not None and (self._rgb_capture or self._step_render):
            # Even if world stepping is skipped, a render pass is needed to avoid
            # stale camera pixels in debug/fidelity flows.
            self._world.step(render=True)
            world_stepped = 1
            sensor_rendered = 1
        rgb, depth = self._read_camera_rgbd()
        obs = Obs(
            rgb=np.asarray(rgb, dtype=np.uint8),
            depth=None if depth is None else np.asarray(depth, dtype=np.float32),
            pose=Pose(
                x=float(self._pose_xyz[0]),
                y=float(self._pose_xyz[1]),
                z=float(self._pose_xyz[2]),
                yaw=float(self._yaw),
            ),
            collision=self._last_collision,
            timestamp=time.time(),
            info=dict(info or {}),
        )
        obs.info["renderer_used"] = str(self._renderer_used)
        obs.info["sensor_rendered"] = int(sensor_rendered)
        obs.info["world_stepped"] = int(world_stepped)
        obs.info["step_render_every_n"] = int(self._step_render_every_n)
        if len(self._camera_cfg_info) > 0:
            obs.info["camera"] = dict(self._camera_cfg_info)
        if len(self._camera_fidelity_info) > 0:
            obs.info["camera_fidelity"] = dict(self._camera_fidelity_info)
        self._last_obs = obs
        return obs

    def run_camera_fidelity_probe(self, frames: int = 10) -> Dict[str, object]:
        num = max(3, int(frames))
        if self._camera is None:
            out = {
                "ok": 0,
                "frames": 0,
                "placeholder_ratio": 1.0,
                "mean_luma": 0.0,
                "luma_std": 0.0,
                "mean_frame_delta": 0.0,
                "reason": "camera_none",
            }
            self._camera_fidelity_info = dict(out)
            print("[ISAAC_CAMERA_FIDELITY] ok=0 reason=camera_none hint=check_camera_init", flush=True)
            return out

        orig_xyz = np.array(self._pose_xyz, dtype=np.float32)
        orig_yaw = float(self._yaw)
        rgbs: List[np.ndarray] = []
        depths: List[np.ndarray] = []
        for i in range(num):
            yaw_off = (float(i) - 0.5 * float(num - 1)) * 0.01
            self._set_pose(orig_xyz, _wrap_pi(orig_yaw + yaw_off))
            if self._world is not None:
                self._world.step(render=True)
            rgb, depth = self._read_camera_rgbd()
            rgbs.append(np.asarray(rgb, dtype=np.uint8))
            if self._enable_depth and depth is not None:
                depths.append(np.asarray(depth, dtype=np.float32))
        self._set_pose(orig_xyz, orig_yaw)
        if self._world is not None:
            self._world.step(render=True)

        stats = rgb_fidelity_stats(rgbs)
        ok = (
            float(stats["placeholder_ratio"]) < 0.95
            and float(stats["mean_luma"]) > 2.0
            and float(stats["mean_luma"]) < 252.0
            and float(stats["mean_frame_delta"]) > 0.0
        )
        out = {
            "ok": int(bool(ok)),
            "frames": int(num),
            "placeholder_ratio": float(stats["placeholder_ratio"]),
            "mean_luma": float(stats["mean_luma"]),
            "luma_std": float(stats["luma_std"]),
            "mean_frame_delta": float(stats["mean_frame_delta"]),
        }
        self._camera_fidelity_info = dict(out)
        if bool(ok):
            print(
                f"[ISAAC_CAMERA_FIDELITY] ok=1 frames={num} "
                f"placeholder_ratio={float(stats['placeholder_ratio']):.3f} "
                f"mean_luma={float(stats['mean_luma']):.3f} "
                f"luma_std={float(stats['luma_std']):.3f} "
                f"mean_frame_delta={float(stats['mean_frame_delta']):.3f}",
                flush=True,
            )
        else:
            print(
                f"[ISAAC_CAMERA_FIDELITY] ok=0 reason=degenerate_stream "
                f"frames={num} placeholder_ratio={float(stats['placeholder_ratio']):.3f} "
                f"mean_luma={float(stats['mean_luma']):.3f} "
                f"luma_std={float(stats['luma_std']):.3f} "
                f"mean_frame_delta={float(stats['mean_frame_delta']):.3f} "
                f"hint=check_renderer_or_camera_cfg",
                flush=True,
            )
        if self._enable_depth:
            dst = depth_stats(depths)
            out["depth"] = dict(dst)
            print(
                f"[ISAAC_DEPTH] ok={int(dst.get('ok', 0.0) > 0.5)} frames={len(depths)} "
                f"min={float(dst.get('min', 0.0)):.4f} max={float(dst.get('max', 0.0)):.4f} "
                f"invalid_ratio={float(dst.get('invalid_ratio', 1.0)):.4f}",
                flush=True,
            )
        return out

    def get_camera_info(self) -> Dict[str, object]:
        out: Dict[str, object] = dict(self._camera_cfg_info)
        if len(self._camera_fidelity_info) > 0:
            out["fidelity"] = dict(self._camera_fidelity_info)
        out["renderer_used"] = str(self._renderer_used)
        return out

    def reset(self, scene_id: str, start_spec: Dict[str, object]) -> Obs:
        if self._fallback is not None:
            obs = self._fallback.reset(scene_id=scene_id, start_spec=start_spec)
            obs.info["backend"] = "isaac_fallback_habitat"
            self._last_obs = obs
            return obs

        starts = self.scene_cfg.get("starts", [])
        idx = int(start_spec.get("start_offset", 0))
        if isinstance(starts, list) and len(starts) > 0:
            s = starts[idx % len(starts)]
            if isinstance(s, dict):
                x = float(s.get("x", 0.0))
                z = float(s.get("z", 0.0))
                y = float(s.get("y", 0.0))
                yaw = float(s.get("yaw", s.get("yaw_deg", 0.0)))
                if abs(yaw) > 2.0 * math.pi:
                    yaw = math.radians(yaw)
                self._set_pose(np.array([x, y, z], dtype=np.float32), yaw)
            else:
                self._set_pose(np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0)
        else:
            # Deterministic fallback start ring.
            ang = float((idx % 12) / 12.0 * 2.0 * math.pi)
            rad = 2.8
            x = rad * math.cos(ang)
            z = rad * math.sin(ang)
            self._set_pose(np.array([x, 0.0, z], dtype=np.float32), -ang)

        if not self._camera_probe_done and bool(int(os.environ.get("ISAAC_CAMERA_PROBE_ON_RESET", "1"))):
            self.run_camera_fidelity_probe(frames=int(os.environ.get("ISAAC_CAMERA_PROBE_FRAMES", "10")))
            self._camera_probe_done = True
        self._last_collision = False
        return self._build_obs(
            info={
                "backend": "real_isaac",
                "scene_id": self.scene_id,
                "renderer_used": str(self._renderer_used),
            }
        )

    def step(self, action_discrete: int) -> Tuple[Obs, bool, Dict[str, object]]:
        if self._fallback is not None:
            obs, done, info = self._fallback.step(action_discrete)
            info = dict(info or {})
            info["backend"] = "isaac_fallback_habitat"
            obs.info.update(info)
            self._last_obs = obs
            self._last_collision = obs.collision
            self._goal_geodesic = self._fallback.get_goal_geodesic()
            return obs, bool(done), info

        act = int(action_discrete)
        dt = float(self.dt_action)
        old_xyz = self._pose_xyz.copy()
        old_yaw = float(self._yaw)

        v = 0.0
        w = 0.0
        if act == 1:
            v = float(self.forward_speed)
        elif act == 2:
            w = float(self.turn_rate)
        elif act == 3:
            w = -float(self.turn_rate)

        new_yaw = _wrap_pi(old_yaw + w * dt)
        new_xyz = old_xyz.copy()
        if abs(v) > 1e-8:
            new_xyz[0] += float(math.sin(new_yaw) * v * dt)
            new_xyz[2] += float(-math.cos(new_yaw) * v * dt)

        collision = False
        if abs(v) > 1e-8 and self._in_collision(new_xyz):
            collision = True
            new_xyz = old_xyz
        if self._in_collision(new_xyz):
            collision = True
            new_xyz = old_xyz

        self._set_pose(new_xyz, new_yaw)
        self._last_collision = bool(collision)
        info = {
            "backend": "real_isaac",
            "renderer_used": str(self._renderer_used),
            "collision": int(bool(collision)),
            "delta_pose": {
                "dx": float(new_xyz[0] - old_xyz[0]),
                "dy": float(new_xyz[1] - old_xyz[1]),
                "dz": float(new_xyz[2] - old_xyz[2]),
                "dyaw": float(_wrap_pi(new_yaw - old_yaw)),
            },
        }
        obs = self._build_obs(info=info)
        return obs, False, info

    def close(self) -> None:
        if self._fallback is not None:
            self._fallback.close()
            return
        try:
            if self._sim_app is not None:
                self._sim_app.close()
        except Exception:
            pass

    def get_pose(self) -> Pose:
        if self._fallback is not None:
            return self._fallback.get_pose()
        if self._last_obs is None:
            return Pose(0.0, 0.0, 0.0, 0.0)
        return self._last_obs.pose

    def get_rgb(self) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback.get_rgb()
        if self._last_obs is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)
        return np.asarray(self._last_obs.rgb, dtype=np.uint8)

    def get_depth(self) -> Optional[np.ndarray]:
        if self._fallback is not None:
            return self._fallback.get_depth()
        if self._last_obs is None or self._last_obs.depth is None:
            return None
        return np.asarray(self._last_obs.depth, dtype=np.float32)

    def get_collision_flag(self) -> Optional[bool]:
        if self._fallback is not None:
            return self._fallback.get_collision_flag()
        return self._last_collision

    def get_goal_geodesic(self) -> Optional[float]:
        if self._fallback is not None:
            return self._fallback.get_goal_geodesic()
        return self._goal_geodesic

    def oracle_probe(
        self,
        goal_position: np.ndarray,
        max_steps: int,
        success_dist: float,
    ) -> Dict[str, object]:
        if self._fallback is not None:
            out = self._fallback.oracle_probe(
                goal_position=goal_position,
                max_steps=max_steps,
                success_dist=success_dist,
            )
            out["oracle_backend"] = "habitat_fallback"
            return out
        pose = self.get_pose()
        cur = np.array([pose.x, pose.y, pose.z], dtype=np.float32)
        goal = np.asarray(goal_position, dtype=np.float32)
        euclid = float(np.linalg.norm(cur - goal))
        ok, waypoints, _ = plan_grid_path(
            bounds=self.get_bounds(),
            obstacles=self.get_obstacles(),
            start=cur,
            goal=goal,
            meters_per_cell=float(self.scene_cfg.get("oracle_grid_step", 0.25)),
        )
        if not ok or len(waypoints) == 0:
            print("[ISAAC_ORACLE] mode=grid_astar available=0 reason=no_path", flush=True)
            return {
                "oracle_available": 0,
                "oracle_success": 0,
                "oracle_geodesic_start": None,
                "oracle_geodesic_final": None,
                "oracle_steps_used": 0,
                "oracle_euclidean": euclid,
                "oracle_backend": "real_isaac_grid_astar",
                "timestamp": float(time.time()),
            }
        geodesic = 0.0
        for i in range(1, len(waypoints)):
            x0, z0 = waypoints[i - 1]
            x1, z1 = waypoints[i]
            geodesic += float(math.hypot(x1 - x0, z1 - z0))
        stop_thresh = float(self.scene_cfg.get("oracle_stop_thresh", 0.35))
        if float(success_dist) > stop_thresh:
            stop_thresh = float(success_dist)
        turn_thresh = math.radians(float(self.scene_cfg.get("oracle_turn_thresh_deg", 26.0)))
        succ, used = self._simulate_waypoint_follow(
            start_xyz=cur,
            start_yaw=float(pose.yaw),
            waypoints=waypoints,
            max_steps=int(max_steps),
            stop_thresh=stop_thresh,
            turn_thresh=turn_thresh,
        )
        print(
            f"[ISAAC_ORACLE] mode=grid_astar available=1 success={int(succ)} steps={int(used)}",
            flush=True,
        )
        return {
            "oracle_available": 1,
            "oracle_success": int(bool(succ)),
            "oracle_geodesic_start": float(geodesic),
            "oracle_geodesic_final": float(geodesic),
            "oracle_steps_used": int(used),
            "oracle_euclidean": euclid,
            "oracle_backend": "real_isaac_grid_astar",
            "timestamp": float(time.time()),
        }
