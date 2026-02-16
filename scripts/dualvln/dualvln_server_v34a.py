#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import socket
import sys
import time
import zlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image


def _server_fail(reason: str) -> None:
    msg = str(reason).replace("\n", " ")
    print(f"[DUALVLN_SERVER] ok=0 reason={msg}", flush=True)
    print(f"[DUALVLN_SERVER_ERR] ok=0 reason={msg}", flush=True)


def _decode_rgb(rgb_jpeg_b64: str) -> np.ndarray:
    raw = base64.b64decode(rgb_jpeg_b64.encode("utf-8"))
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _decode_depth(payload: Dict[str, Any]) -> Optional[np.ndarray]:
    depth_b64 = str(payload.get("depth_f16_zlib_b64", "") or "").strip()
    if not depth_b64:
        return None
    raw = zlib.decompress(base64.b64decode(depth_b64.encode("utf-8")))
    depth_f16 = np.frombuffer(raw, dtype=np.float16)
    depth_shape = payload.get("depth_shape")
    if isinstance(depth_shape, (list, tuple)) and len(depth_shape) == 2:
        h, w = int(depth_shape[0]), int(depth_shape[1])
    else:
        intr = payload.get("intrinsics", {}) if isinstance(payload.get("intrinsics", {}), dict) else {}
        h = int(intr.get("h", 0) or 0)
        w = int(intr.get("w", 0) or 0)
    if h <= 0 or w <= 0:
        return None
    if depth_f16.size != h * w:
        return None
    return depth_f16.reshape((h, w)).astype(np.float32)


def _pick_action_from_text(text: str) -> str:
    t = (text or "").strip().upper()
    for token in ("FORWARD", "LEFT", "RIGHT", "STOP"):
        if token in t:
            return token
    if "FWD" in t:
        return "FORWARD"
    return "FORWARD"


def _heuristic_action(depth_m: Optional[np.ndarray], instruction: str) -> str:
    ins = str(instruction or "").lower()
    if "stop" in ins:
        return "STOP"
    if depth_m is None or depth_m.size == 0:
        return "FORWARD"
    h, w = depth_m.shape
    y0 = int(0.30 * h)
    y1 = int(0.85 * h)
    x0 = int(0.45 * w)
    x1 = int(0.55 * w)
    patch = depth_m[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 1e-3)
    if not valid.any():
        return "LEFT"
    p20 = float(np.percentile(patch[valid], 20))
    if p20 < 0.7:
        return "LEFT"
    return "FORWARD"


class DualVlnEngine:
    def __init__(self, model_path: str, max_new_tokens: int = 8) -> None:
        self.model_path = str(model_path)
        self.max_new_tokens = int(max_new_tokens)
        self.torch = None
        self.transformers = None
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.dtype_name = "unknown"
        self.quant_name = "none"
        self.device_name = "cpu"
        self._load()

    def _load(self) -> None:
        try:
            import torch  # type: ignore
            import transformers  # type: ignore

            self.torch = torch
            self.transformers = transformers
        except Exception as e:
            _server_fail(f"import_failed:{type(e).__name__}:{e}")
            raise

        if not os.path.isdir(self.model_path):
            msg = f"model_path_not_found:{self.model_path}"
            _server_fail(msg)
            raise FileNotFoundError(msg)

        if not self.torch.cuda.is_available():
            msg = "cuda_not_available"
            _server_fail(msg)
            raise RuntimeError(msg)

        self.device_name = str(self.torch.cuda.get_device_name(0))

        if not self._try_load_fp16():
            if not self._try_load_8bit():
                msg = "model_load_failed_fp16_and_8bit"
                _server_fail(msg)
                raise RuntimeError(msg)

        print(
            f"[DUALVLN_SERVER] ok=1 gpu_id=0 gpu_name=\"{self.device_name}\" "
            f"dtype={self.dtype_name} quant={self.quant_name} model_path={self.model_path}",
            flush=True,
        )

    def _try_load_fp16(self) -> bool:
        try:
            tfm = self.transformers
            kwargs = dict(
                trust_remote_code=True,
                local_files_only=True,
            )
            try:
                self.processor = tfm.AutoProcessor.from_pretrained(self.model_path, **kwargs)
            except Exception:
                self.processor = None
            try:
                self.tokenizer = tfm.AutoTokenizer.from_pretrained(self.model_path, **kwargs)
            except Exception:
                self.tokenizer = None

            load_kwargs = dict(
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=self.torch.float16,
            )
            model = None
            last_err: Optional[Exception] = None
            for cls_name in ("AutoModelForVision2Seq", "AutoModelForCausalLM", "AutoModel"):
                cls = getattr(tfm, cls_name, None)
                if cls is None:
                    continue
                try:
                    model = cls.from_pretrained(self.model_path, **load_kwargs)
                    break
                except Exception as e:
                    last_err = e
                    continue
            if model is None:
                if last_err is not None:
                    raise last_err
                raise RuntimeError("no_transformers_auto_model_class")
            model.eval()
            model.to("cuda:0")
            self.model = model
            self.dtype_name = "fp16"
            self.quant_name = "none"
            return True
        except Exception as e:
            _server_fail(f"fp16_load_failed:{type(e).__name__}:{e}")
            return False

    def _try_load_8bit(self) -> bool:
        try:
            import bitsandbytes  # type: ignore  # noqa: F401
        except Exception as e:
            print(f"[DUALVLN_SERVER_ERR] ok=0 reason=8bit_missing_bitsandbytes:{type(e).__name__}:{e}", flush=True)
            return False
        try:
            tfm = self.transformers
            kwargs = dict(
                trust_remote_code=True,
                local_files_only=True,
            )
            if self.processor is None:
                try:
                    self.processor = tfm.AutoProcessor.from_pretrained(self.model_path, **kwargs)
                except Exception:
                    self.processor = None
            if self.tokenizer is None:
                try:
                    self.tokenizer = tfm.AutoTokenizer.from_pretrained(self.model_path, **kwargs)
                except Exception:
                    self.tokenizer = None
            load_kwargs = dict(
                trust_remote_code=True,
                local_files_only=True,
                load_in_8bit=True,
                device_map="auto",
            )
            model = None
            last_err: Optional[Exception] = None
            for cls_name in ("AutoModelForVision2Seq", "AutoModelForCausalLM", "AutoModel"):
                cls = getattr(tfm, cls_name, None)
                if cls is None:
                    continue
                try:
                    model = cls.from_pretrained(self.model_path, **load_kwargs)
                    break
                except Exception as e:
                    last_err = e
                    continue
            if model is None:
                if last_err is not None:
                    raise last_err
                raise RuntimeError("no_transformers_auto_model_class_8bit")
            model.eval()
            self.model = model
            self.dtype_name = "int8"
            self.quant_name = "8bit"
            return True
        except Exception as e:
            _server_fail(f"8bit_load_failed:{type(e).__name__}:{e}")
            return False

    def infer_action(self, payload: Dict[str, Any], rgb: np.ndarray, depth_m: Optional[np.ndarray]) -> str:
        instruction = str(payload.get("instruction", "") or "")
        # Primary path: generic text generation -> parse action token.
        if self.model is not None and self.tokenizer is not None:
            prompt = (
                "You are a robot navigation policy. "
                "Given observation summary, output one token in {FORWARD, LEFT, RIGHT, STOP}.\n"
                f"Instruction: {instruction}\n"
            )
            if depth_m is not None and depth_m.size > 0:
                valid = np.isfinite(depth_m) & (depth_m > 1e-3)
                if valid.any():
                    d20 = float(np.percentile(depth_m[valid], 20))
                    d50 = float(np.percentile(depth_m[valid], 50))
                    prompt += f"Depth p20={d20:.3f} p50={d50:.3f}\n"
            try:
                toks = self.tokenizer(prompt, return_tensors="pt")
                toks = {k: v.to("cuda:0") for k, v in toks.items()}
                with self.torch.no_grad():
                    out = self.model.generate(**toks, max_new_tokens=self.max_new_tokens)
                text = self.tokenizer.decode(out[0], skip_special_tokens=True)
                return _pick_action_from_text(text)
            except Exception as e:
                _server_fail(f"infer_generate_failed:{type(e).__name__}:{e}")
        # Fallback deterministic heuristic.
        return _heuristic_action(depth_m, instruction)


class _Handler(BaseHTTPRequestHandler):
    server_version = "DualVlnServerV34a/1.0"

    def _write_json(self, code: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        # Quiet default http.server access logs.
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            self._write_json(HTTPStatus.OK, {"ok": 1, "service": "dualvln_v34a"})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"ok": 0, "reason": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/infer":
            self._write_json(HTTPStatus.NOT_FOUND, {"ok": 0, "reason": "not_found"})
            return
        t0 = time.time()
        try:
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n) if n > 0 else b""
            payload = json.loads(raw.decode("utf-8"))
            rgb = _decode_rgb(str(payload.get("rgb_jpeg_b64", "")))
            depth = _decode_depth(payload if isinstance(payload, dict) else {})
            action = self.server.engine.infer_action(payload=payload, rgb=rgb, depth_m=depth)  # type: ignore[attr-defined]
            latency_ms = float((time.time() - t0) * 1000.0)
            rgb_bytes = int(len(str(payload.get("rgb_jpeg_b64", ""))))
            depth_bytes = int(len(str(payload.get("depth_f16_zlib_b64", ""))))
            print(
                f"[DUALVLN_SERVER_REQ] ok=1 latency_ms={latency_ms:.2f} action={action} "
                f"req_bytes={len(raw)} rgb_bytes={rgb_bytes} depth_bytes={depth_bytes}",
                flush=True,
            )
            self._write_json(
                HTTPStatus.OK,
                {"ok": 1, "action": str(action), "extra": {"latency_ms": latency_ms}},
            )
        except Exception as e:
            msg = f"{type(e).__name__}:{e}"
            _server_fail(msg)
            self._write_json(HTTPStatus.OK, {"ok": 0, "reason": msg, "action": "STOP"})


def _wait_port_free(host: str, port: int, timeout_s: float = 2.0) -> bool:
    t0 = time.time()
    while time.time() - t0 <= timeout_s:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex((host, port)) != 0:
                return True
        time.sleep(0.05)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="DualVLN RPC server for v34a")
    ap.add_argument("--host", type=str, default=os.environ.get("DUALVLN_SERVER_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("DUALVLN_SERVER_PORT", "18080")))
    ap.add_argument("--model_path", type=str, default=os.environ.get("DUALVLN_MODEL_PATH", ""))
    ap.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("DUALVLN_MAX_NEW_TOKENS", "8")))
    args = ap.parse_args()

    if not args.model_path:
        _server_fail("missing_DUALVLN_MODEL_PATH")
        return 2
    if not _wait_port_free(args.host, int(args.port), timeout_s=2.0):
        _server_fail(f"port_in_use:{args.host}:{args.port}")
        return 2

    try:
        engine = DualVlnEngine(model_path=args.model_path, max_new_tokens=int(args.max_new_tokens))
    except Exception:
        return 3

    httpd = HTTPServer((args.host, int(args.port)), _Handler)
    httpd.engine = engine  # type: ignore[attr-defined]
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
