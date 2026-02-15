#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
cd "${ROOT_DIR}"

if [ "${1:-}" != "--" ]; then
  echo "Usage: bash scripts/isaac/run_with_isaac_python.sh -- <python args>" >&2
  exit 2
fi
shift

if [ "$#" -eq 0 ]; then
  echo "No python args provided." >&2
  exit 2
fi

mapfile -t cmd < <(printf '%s\n' "$@")
if [ "${cmd[0]}" = "python" ] || [ "${cmd[0]}" = "python3" ]; then
  cmd=("${cmd[@]:1}")
fi
if [ "${#cmd[@]}" -eq 0 ]; then
  echo "No script/module passed after python token." >&2
  exit 2
fi

root_spec="$(bash scripts/isaac/find_isaac_root.sh || true)"
if [ -z "${root_spec}" ]; then
  echo "[ISAAC_ENV_FAIL] root= err=Isaac root not found" >&2
  exit 2
fi
echo "[ISAAC_ENV_RESOLVED] root=${root_spec}"

preflight_code='import omni; import sys; sys.stdout.write("OMNI_OK\\n"); sys.stdout.flush()'

run_local() {
  local root="$1"
  if ! "${root}/python.sh" -c "${preflight_code}" >/tmp/isaac_preflight.out 2>/tmp/isaac_preflight.err; then
    echo "[ISAAC_ENV_FAIL] root=${root} err=$(tr '\n' ' ' </tmp/isaac_preflight.err | head -c 400)" >&2
    return 2
  fi
  if ! (grep -q "OMNI_OK" /tmp/isaac_preflight.out || grep -q "OMNI_OK" /tmp/isaac_preflight.err); then
    echo "[ISAAC_ENV_FAIL] root=${root} err=OMNI_OK marker missing in preflight output" >&2
    return 2
  fi
  echo "[ISAAC_ENV_OK] root=${root} msg=OMNI_OK"
  ISAAC_SKIP_WORLD_STEP="${ISAAC_SKIP_WORLD_STEP:-0}" \
  ISAAC_RENDER="${ISAAC_RENDER:-0}" \
  ISAAC_RGB_CAPTURE="${ISAAC_RGB_CAPTURE:-0}" \
  PYTHONPATH="/home/peng/DualVLN/repo/InternNav:${PYTHONPATH:-}" \
    "${root}/python.sh" "${cmd[@]}"
}

run_docker_exec() {
  local cname="$1"
  local root="$2"
  local translated=()
  local perm_targets=()
  local arg
  for arg in "${cmd[@]}"; do
    translated+=("${arg//\/home\/peng\/DualVLN/\/workspace\/DualVLN}")
  done
  for arg in "${translated[@]}"; do
    if [[ "${arg}" == /workspace/DualVLN/runs/* ]]; then
      perm_targets+=("${arg}")
    fi
  done
  if ! docker exec "${cname}" "${root}/python.sh" -c "${preflight_code}" >/tmp/isaac_preflight.out 2>/tmp/isaac_preflight.err; then
    echo "[ISAAC_ENV_FAIL] root=docker-container:${cname}:${root} err=$(tr '\n' ' ' </tmp/isaac_preflight.err | head -c 400)" >&2
    return 2
  fi
  if ! (grep -q "OMNI_OK" /tmp/isaac_preflight.out || grep -q "OMNI_OK" /tmp/isaac_preflight.err); then
    echo "[ISAAC_ENV_FAIL] root=docker-container:${cname}:${root} err=OMNI_OK marker missing in preflight output" >&2
    return 2
  fi
  echo "[ISAAC_ENV_OK] root=docker-container:${cname}:${root} msg=OMNI_OK"
  docker exec \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    -e PYTHONPATH="/workspace/DualVLN/repo/InternNav:${PYTHONPATH:-}" \
    -e ISAAC_SIM_ROOT="${root}" \
    -e ISAAC_SKIP_WORLD_STEP="${ISAAC_SKIP_WORLD_STEP:-0}" \
    -e ISAAC_RENDER="${ISAAC_RENDER:-0}" \
    -e ISAAC_MINIMAL="${ISAAC_MINIMAL:-0}" \
    -e ISAAC_RGB_CAPTURE="${ISAAC_RGB_CAPTURE:-0}" \
    "${cname}" \
    "${root}/python.sh" "${translated[@]}"
  if [ "${#perm_targets[@]}" -gt 0 ]; then
    local fix_cmd=""
    for arg in "${perm_targets[@]}"; do
      fix_cmd="${fix_cmd} chmod -R a+rwx '${arg}' 2>/dev/null || true;"
    done
    docker exec "${cname}" /bin/bash -lc "${fix_cmd}" >/dev/null 2>&1 || true
  fi
}

run_docker_image() {
  local image="$1"
  local root="$2"
  local cache_root="${ISAAC_CACHE_ROOT:-${HOME}/docker/isaac-sim}"
  local translated=()
  local perm_targets=()
  local arg
  for arg in "${cmd[@]}"; do
    translated+=("${arg//\/home\/peng\/DualVLN/\/workspace\/DualVLN}")
  done
  for arg in "${translated[@]}"; do
    if [[ "${arg}" == /workspace/DualVLN/runs/* ]]; then
      perm_targets+=("${arg}")
    fi
  done
  mkdir -p "${cache_root}/cache" "${cache_root}/logs" "${cache_root}/data" "${cache_root}/documents"

  if ! docker run --rm --gpus all \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e OMNI_ENV_PRIVACY_CONSENT=1 \
    -e ISAAC_SKIP_WORLD_STEP="${ISAAC_SKIP_WORLD_STEP:-0}" \
    -e ISAAC_RENDER="${ISAAC_RENDER:-0}" \
    -v "${cache_root}/cache:/isaac-sim/kit/cache" \
    -v "${cache_root}/logs:/root/.nvidia-omniverse/logs" \
    -v "${cache_root}/data:/root/.local/share/ov/data" \
    -v "${cache_root}/documents:/root/Documents" \
    -v "/home/peng/DualVLN:/workspace/DualVLN" \
    -w "/workspace/DualVLN" \
    --entrypoint "${root}/python.sh" \
    "${image}" \
    -c "${preflight_code}" >/tmp/isaac_preflight.out 2>/tmp/isaac_preflight.err; then
    echo "[ISAAC_ENV_FAIL] root=docker-image:${image}:${root} err=$(tr '\n' ' ' </tmp/isaac_preflight.err | head -c 400)" >&2
    return 2
  fi
  if ! (grep -q "OMNI_OK" /tmp/isaac_preflight.out || grep -q "OMNI_OK" /tmp/isaac_preflight.err); then
    echo "[ISAAC_ENV_FAIL] root=docker-image:${image}:${root} err=OMNI_OK marker missing in preflight output" >&2
    return 2
  fi
  echo "[ISAAC_ENV_OK] root=docker-image:${image}:${root} msg=OMNI_OK"
  docker run --rm --gpus all \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e OMNI_ENV_PRIVACY_CONSENT=1 \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    -e PYTHONPATH="/workspace/DualVLN/repo/InternNav:${PYTHONPATH:-}" \
    -e ISAAC_SIM_ROOT="${root}" \
    -e ISAAC_SKIP_WORLD_STEP="${ISAAC_SKIP_WORLD_STEP:-0}" \
    -e ISAAC_RENDER="${ISAAC_RENDER:-0}" \
    -e ISAAC_MINIMAL="${ISAAC_MINIMAL:-0}" \
    -e ISAAC_RGB_CAPTURE="${ISAAC_RGB_CAPTURE:-0}" \
    -v "${cache_root}/cache:/isaac-sim/kit/cache" \
    -v "${cache_root}/logs:/root/.nvidia-omniverse/logs" \
    -v "${cache_root}/data:/root/.local/share/ov/data" \
    -v "${cache_root}/documents:/root/Documents" \
    -v "/home/peng/DualVLN:/workspace/DualVLN" \
    -w "/workspace/DualVLN" \
    --entrypoint "${root}/python.sh" \
    "${image}" "${translated[@]}"
  if [ "${#perm_targets[@]}" -gt 0 ]; then
    local fix_cmd=""
    for arg in "${perm_targets[@]}"; do
      fix_cmd="${fix_cmd} chmod -R a+rwx '${arg}' 2>/dev/null || true;"
    done
    docker run --rm \
      --entrypoint /bin/bash \
      -v "/home/peng/DualVLN:/workspace/DualVLN" \
      "${image}" -lc "${fix_cmd}" >/dev/null 2>&1 || true
  fi
}

if [[ "${root_spec}" == docker-container:* ]]; then
  cname="$(printf '%s' "${root_spec}" | cut -d: -f2)"
  root="$(printf '%s' "${root_spec}" | cut -d: -f3-)"
  run_docker_exec "${cname}" "${root}"
elif [[ "${root_spec}" == docker-image:* ]]; then
  image="$(printf '%s' "${root_spec}" | cut -d: -f2- | sed 's#:/isaac-sim##')"
  root="/isaac-sim"
  run_docker_image "${image}" "${root}"
else
  run_local "${root_spec}"
fi
