import io
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np
from PIL import Image
from PIL import ImageDraw


@dataclass
class _BufferedStep:
    record: Dict
    rgb_png: Optional[bytes]
    depth_png: Optional[bytes]


class DebugRingBuffer:
    def __init__(
        self,
        enabled: bool = False,
        ring_steps: int = 220,
        frame_stride: int = 3,
        save_depth: bool = False,
        placeholder: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.ring_steps = max(1, int(ring_steps))
        self.frame_stride = max(1, int(frame_stride))
        self.save_depth = bool(save_depth)
        self.placeholder = bool(placeholder)
        self._buf: Deque[_BufferedStep] = deque(maxlen=self.ring_steps)
        self._rgb_ok_count = 0
        self._rgb_placeholder_count = 0

    def _make_placeholder(self, step_idx: int, reason: str, size: Optional[tuple] = None) -> Optional[bytes]:
        try:
            if size is None:
                size = (320, 240)
            w, h = int(size[0]), int(size[1])
            img = Image.new("RGB", (w, h), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            msg = f"step {step_idx}\n{reason}"
            draw.text((10, 10), msg, fill=(255, 255, 255))
            bio = io.BytesIO()
            img.save(bio, format="PNG", optimize=True)
            return bio.getvalue()
        except Exception:
            return None

    def _encode_rgb(self, rgb: np.ndarray) -> Optional[bytes]:
        if rgb is None:
            return None
        arr = np.asarray(rgb)
        if arr.ndim != 3:
            return None
        try:
            img = Image.fromarray(arr.astype(np.uint8))
            bio = io.BytesIO()
            img.save(bio, format="PNG", optimize=True)
            return bio.getvalue()
        except Exception:
            return None

    def _encode_depth(self, depth: np.ndarray) -> Optional[bytes]:
        if depth is None:
            return None
        arr = np.asarray(depth, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim != 2:
            return None
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 10.0)
        arr16 = np.round(arr / 10.0 * 65535.0).astype(np.uint16)
        try:
            img = Image.fromarray(arr16, mode="I;16")
            bio = io.BytesIO()
            img.save(bio, format="PNG", optimize=True)
            return bio.getvalue()
        except Exception:
            return None

    def push(
        self,
        record: Dict,
        rgb: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        event_frame: bool = False,
    ) -> None:
        if not self.enabled:
            return
        step_idx = int(record.get("step_idx", -1))
        keep_frame = bool(event_frame or (step_idx >= 0 and step_idx % self.frame_stride == 0))
        rgb_png = self._encode_rgb(rgb) if keep_frame else None
        if keep_frame and rgb_png is not None:
            self._rgb_ok_count += 1
        if keep_frame and rgb_png is None and self.placeholder:
            reason = "rgb_none" if rgb is None else "rgb_encode_fail"
            size = None
            try:
                if rgb is not None and hasattr(rgb, "shape") and len(rgb.shape) >= 2:
                    size = (int(rgb.shape[1]), int(rgb.shape[0]))
            except Exception:
                size = None
            rgb_png = self._make_placeholder(step_idx, reason, size=size)
            self._rgb_placeholder_count += 1
            record = dict(record)
            record["frame_placeholder"] = int(rgb_png is not None)
            record["frame_placeholder_reason"] = reason
        depth_png = self._encode_depth(depth) if (keep_frame and self.save_depth) else None
        self._buf.append(
            _BufferedStep(
                record=dict(record),
                rgb_png=rgb_png,
                depth_png=depth_png,
            )
        )

    def flush(self, out_dir: str, meta: Dict) -> Dict[str, int]:
        if not self.enabled:
            return {"saved_steps": 0, "saved_frames": 0, "saved_depth": 0}
        base = Path(out_dir)
        rgb_dir = base / "rgb"
        depth_dir = base / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        if self.save_depth:
            depth_dir.mkdir(parents=True, exist_ok=True)
        trace_path = base / "trace.jsonl"
        saved_steps = 0
        saved_frames = 0
        saved_depth = 0
        with trace_path.open("w") as f:
            for row in list(self._buf):
                rec = dict(row.record)
                step = int(rec.get("step_idx", -1))
                if row.rgb_png is not None:
                    rel = f"rgb/rgb_{step:06d}.png"
                    (base / rel).write_bytes(row.rgb_png)
                    rec["frame_rgb"] = rel
                    saved_frames += 1
                if row.depth_png is not None:
                    rel_d = f"depth/depth_{step:06d}.png"
                    (base / rel_d).write_bytes(row.depth_png)
                    rec["frame_depth"] = rel_d
                    saved_depth += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                saved_steps += 1
        stats = {
            "saved_steps": int(saved_steps),
            "saved_frames": int(saved_frames),
            "saved_depth": int(saved_depth),
            "rgb_ok_steps": int(self._rgb_ok_count),
            "rgb_placeholder_steps": int(self._rgb_placeholder_count),
        }
        # Write capture quality into meta
        meta_out = dict(meta)
        meta_out["rgb_capture_stats"] = {
            "rgb_ok": int(self._rgb_ok_count),
            "rgb_placeholder": int(self._rgb_placeholder_count),
        }
        (base / "meta.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False))
        return stats
