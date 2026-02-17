#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _tokenize_pnm(buf: bytes) -> List[bytes]:
    toks: List[bytes] = []
    i = 0
    n = len(buf)
    while i < n:
        c = buf[i]
        if c in b" \t\r\n":
            i += 1
            continue
        if c == ord("#"):
            while i < n and buf[i] not in b"\r\n":
                i += 1
            continue
        j = i
        while j < n and buf[j] not in b" \t\r\n#":
            j += 1
        toks.append(buf[i:j])
        i = j
    return toks


def _load_pgm_u8(path: Path) -> Tuple[int, int, List[int]]:
    raw = path.read_bytes()
    toks = _tokenize_pnm(raw)
    if len(toks) < 4:
        raise RuntimeError("pgm_header_too_short")
    magic = toks[0].decode("ascii", errors="ignore")
    if magic not in {"P5", "P2"}:
        raise RuntimeError(f"unsupported_pgm_magic:{magic}")
    w = int(toks[1])
    h = int(toks[2])
    maxv = int(toks[3])
    if w <= 0 or h <= 0 or maxv <= 0:
        raise RuntimeError("invalid_pgm_dims_or_max")
    if magic == "P2":
        vals: List[int] = []
        for t in toks[4:]:
            try:
                vals.append(int(t))
            except Exception:
                continue
        if len(vals) < w * h:
            raise RuntimeError("pgm_ascii_data_short")
        if maxv != 255:
            vals = [int(round(v * 255.0 / float(maxv))) for v in vals[: w * h]]
        else:
            vals = vals[: w * h]
        return w, h, vals

    # P5 binary: locate start of data by scanning first 4 tokens in raw buffer.
    # This avoids fragile byte offsets when comments/whitespace vary.
    consumed = 0
    found = 0
    n = len(raw)
    while consumed < n and found < 4:
        c = raw[consumed]
        if c in b" \t\r\n":
            consumed += 1
            continue
        if c == ord("#"):
            while consumed < n and raw[consumed] not in b"\r\n":
                consumed += 1
            continue
        while consumed < n and raw[consumed] not in b" \t\r\n#":
            consumed += 1
        found += 1
    while consumed < n and raw[consumed] in b" \t\r\n":
        consumed += 1
    pix = raw[consumed:]
    if len(pix) < w * h:
        raise RuntimeError("pgm_binary_data_short")
    if maxv == 255:
        vals = list(pix[: w * h])
    else:
        vals = [int(round(v * 255.0 / float(maxv))) for v in pix[: w * h]]
    return w, h, vals


def _parse_yaml_minimal(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        k, v = s.split(":", 1)
        key = k.strip()
        val = v.strip()
        if key == "image":
            out["image"] = val.strip("'\"")
        elif key in {"resolution", "occupied_thresh", "free_thresh"}:
            try:
                out[key] = float(val)
            except Exception:
                pass
        elif key == "origin":
            try:
                origin = ast.literal_eval(val)
                if isinstance(origin, Sequence) and len(origin) >= 2:
                    out["origin"] = [float(origin[0]), float(origin[1]), float(origin[2] if len(origin) >= 3 else 0.0)]
            except Exception:
                pass
    return out


def _load_map_meta(map_yaml: Path) -> Dict[str, Any]:
    obj: Dict[str, Any] = {}
    txt = map_yaml.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(txt)
        if isinstance(loaded, dict):
            obj = dict(loaded)
    except Exception:
        obj = _parse_yaml_minimal(txt)

    image_name = str(obj.get("image", "")).strip()
    if not image_name:
        raise RuntimeError("map_yaml_missing_image")
    image_path = (map_yaml.parent / image_name).resolve()
    if not image_path.is_file():
        raise RuntimeError(f"map_image_missing:{image_path}")

    w: int
    h: int
    grid_u8: List[int]
    if image_path.suffix.lower() == ".pgm":
        w, h, grid_u8 = _load_pgm_u8(image_path)
    else:
        try:
            from PIL import Image  # type: ignore

            im = Image.open(image_path).convert("L")
            w, h = int(im.size[0]), int(im.size[1])
            grid_u8 = list(im.tobytes())
        except Exception as e:
            raise RuntimeError(f"unsupported_map_image:{image_path}:{type(e).__name__}") from e

    origin = obj.get("origin", [0.0, 0.0, 0.0])
    if not isinstance(origin, Sequence) or len(origin) < 2:
        origin = [0.0, 0.0, 0.0]

    return {
        "grid_u8": grid_u8,
        "w": int(w),
        "h": int(h),
        "resolution": float(obj.get("resolution", 0.05)),
        "origin_x": float(origin[0]),
        "origin_y": float(origin[1]),
        "occupied_thresh": float(obj.get("occupied_thresh", 0.65)),
        "free_thresh": float(obj.get("free_thresh", 0.196)),
    }


def _world_from_cell(row: int, col: int, meta: Dict[str, Any]) -> Tuple[float, float]:
    res = float(meta["resolution"])
    ox = float(meta["origin_x"])
    oy = float(meta["origin_y"])
    h = int(meta["h"])
    x = ox + (float(col) + 0.5) * res
    y = oy + (float(h - 1 - row) + 0.5) * res
    return float(x), float(y)


def _yaw_from_quat(z: float, w: float) -> float:
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


def _make_pose_stamped(frame: str, x: float, y: float, yaw: float, now_msg: Any, PoseStamped: Any) -> Any:
    ps = PoseStamped()
    ps.header.frame_id = str(frame)
    ps.header.stamp = now_msg
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.position.z = 0.0
    ps.pose.orientation.z = math.sin(float(yaw) * 0.5)
    ps.pose.orientation.w = math.cos(float(yaw) * 0.5)
    return ps


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample reachable Nav2 goals using planner service")
    ap.add_argument("--map_yaml", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--radius_min", type=float, default=0.8)
    ap.add_argument("--radius_max", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planner_action", type=str, default="/compute_path_to_pose")
    ap.add_argument("--timeout_s", type=float, default=4.0)
    args = ap.parse_args()

    random.seed(int(args.seed))

    map_yaml = Path(args.map_yaml).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    meta = _load_map_meta(map_yaml)
    w = int(meta["w"])
    h = int(meta["h"])
    grid = list(meta["grid_u8"])
    free_cells: List[Tuple[int, int]] = []
    for r in range(h):
        row_off = r * w
        for c in range(w):
            if int(grid[row_off + c]) >= 245:
                free_cells.append((r, c))
    if len(free_cells) == 0:
        print("[V34J_PLANNER_SERVICE] ok=0 reason=map_no_free_cells", flush=True)
        print(f"[V34J_GOAL_SAMPLE] candidates=0 reachable=0 chosen=0 saved={out_json} ok=0", flush=True)
        return 2

    try:
        import rclpy  # type: ignore
        from geometry_msgs.msg import PoseStamped  # type: ignore
        from nav2_msgs.action import ComputePathToPose  # type: ignore
        from nav_msgs.msg import Odometry  # type: ignore
        from rclpy.action import ActionClient  # type: ignore
        from rclpy.node import Node  # type: ignore
    except Exception as e:
        print(f"[V34J_PLANNER_SERVICE] ok=0 reason=rclpy_missing:{type(e).__name__}", flush=True)
        print(f"[V34J_GOAL_SAMPLE] candidates=0 reachable=0 chosen=0 saved={out_json} ok=0", flush=True)
        return 3

    class SamplerNode(Node):
        def __init__(self) -> None:
            super().__init__("v34j_goal_sampler")
            self.last_odom: Optional[Odometry] = None
            self.create_subscription(Odometry, "/odom", self._on_odom, 10)

        def _on_odom(self, msg: Odometry) -> None:
            self.last_odom = msg

    rclpy.init(args=None)
    node: Optional[SamplerNode] = None
    try:
        node = SamplerNode()
        # Wait for odom.
        t_odom = time.time() + 8.0
        while time.time() < t_odom and node.last_odom is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.last_odom is None:
            print("[V34J_PLANNER_SERVICE] ok=0 reason=odom_unavailable", flush=True)
            print(f"[V34J_GOAL_SAMPLE] candidates=0 reachable=0 chosen=0 saved={out_json} ok=0", flush=True)
            return 4

        action_name = str(args.planner_action).strip() or "/compute_path_to_pose"
        planner_ac = ActionClient(node, ComputePathToPose, action_name)
        if not planner_ac.wait_for_server(timeout_sec=8.0):
            print(f"[V34J_PLANNER_SERVICE] ok=0 reason=action_unavailable name={action_name}", flush=True)
            print(f"[V34J_GOAL_SAMPLE] candidates=0 reachable=0 chosen=0 saved={out_json} ok=0", flush=True)
            return 5
        print(f"[V34J_PLANNER_SERVICE] ok=1 name={action_name} mode=action", flush=True)

        od = node.last_odom
        start_x = float(od.pose.pose.position.x)
        start_y = float(od.pose.pose.position.y)
        qz = float(od.pose.pose.orientation.z)
        qw = float(od.pose.pose.orientation.w)
        start_yaw = float(_yaw_from_quat(qz, qw))

        now_msg = node.get_clock().now().to_msg()
        start_pose = _make_pose_stamped("map", start_x, start_y, start_yaw, now_msg, PoseStamped)

        chosen: List[Dict[str, float]] = []
        seen_cells = set()
        reachable = 0
        attempts = 0
        max_attempts = min(1200, max(150, int(args.count) * 300))

        while attempts < max_attempts and len(chosen) < int(args.count):
            attempts += 1
            r, c = free_cells[random.randrange(0, len(free_cells))]
            if (r, c) in seen_cells:
                continue
            seen_cells.add((r, c))
            gx, gy = _world_from_cell(r, c, meta)
            dist = float(math.hypot(gx - start_x, gy - start_y))
            if dist < float(args.radius_min) or dist > float(args.radius_max):
                continue
            gyaw = float(math.atan2(gy - start_y, gx - start_x))
            goal_pose = _make_pose_stamped("map", gx, gy, gyaw, now_msg, PoseStamped)

            req = ComputePathToPose.Goal()
            req.start = start_pose
            req.goal = goal_pose
            if hasattr(req, "planner_id"):
                req.planner_id = ""
            if hasattr(req, "use_start"):
                req.use_start = True

            send_fut = planner_ac.send_goal_async(req)
            t_send = time.time() + float(args.timeout_s)
            while time.time() < t_send and not send_fut.done():
                rclpy.spin_once(node, timeout_sec=0.05)
            if not send_fut.done():
                continue
            gh = send_fut.result()
            if gh is None or not gh.accepted:
                continue
            fut = gh.get_result_async()
            t_end = time.time() + float(args.timeout_s)
            while time.time() < t_end and not fut.done():
                rclpy.spin_once(node, timeout_sec=0.05)
            if not fut.done():
                continue
            try:
                wrapped = fut.result()
            except Exception:
                continue
            resp = getattr(wrapped, "result", None) if wrapped is not None else None
            path = getattr(resp, "path", None) if resp is not None else None
            poses = getattr(path, "poses", []) if path is not None else []
            if len(poses) < 2:
                continue
            reachable += 1
            chosen.append(
                {
                    "id": f"goal_{len(chosen) + 1:03d}",
                    "x": float(gx),
                    "y": float(gy),
                    "yaw_deg": float(math.degrees(gyaw)),
                    "dist_from_start_m": float(dist),
                }
            )

        out_obj = {
            "ok": int(len(chosen) >= int(args.count)),
            "map_yaml": str(map_yaml),
            "planner_service": str(action_name),
            "start": {"x": float(start_x), "y": float(start_y), "yaw_deg": float(math.degrees(start_yaw))},
            "goals": chosen,
        }
        out_json.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
        ok = 1 if len(chosen) >= int(args.count) else 0
        print(
            f"[V34J_GOAL_SAMPLE] candidates={attempts} reachable={reachable} chosen={len(chosen)} saved={out_json} ok={ok}",
            flush=True,
        )
        return 0 if ok == 1 else 7
    finally:
        try:
            if node is not None:
                node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
