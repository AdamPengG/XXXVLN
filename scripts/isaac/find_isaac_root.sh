#!/usr/bin/env bash
set -euo pipefail

has_python_sh() {
  local root="$1"
  [ -n "${root}" ] && [ -d "${root}" ] && [ -x "${root}/python.sh" ]
}

if has_python_sh "${ISAAC_SIM_ROOT:-}"; then
  printf '%s\n' "${ISAAC_SIM_ROOT}"
  exit 0
fi

declare -a roots=(
  "/home/peng/FirstVLN"
  "/home/peng"
  "/opt"
  "/usr/local"
)

for r in "${roots[@]}"; do
  [ -d "${r}" ] || continue
  while IFS= read -r py; do
    d="$(dirname "${py}")"
    if has_python_sh "${d}"; then
      printf '%s\n' "${d}"
      exit 0
    fi
  done < <(
    find "${r}" -maxdepth 7 -type f -name "python.sh" 2>/dev/null \
      | grep -Ei 'isaac|ov/pkg|omniverse' \
      | sort -u
  )
done

if command -v docker >/dev/null 2>&1; then
  while IFS= read -r cname; do
    [ -n "${cname}" ] || continue
    if docker exec "${cname}" sh -lc 'test -x /isaac-sim/python.sh' >/dev/null 2>&1; then
      printf 'docker-container:%s:/isaac-sim\n' "${cname}"
      exit 0
    fi
  done < <(docker ps --format '{{.Names}}')

  if docker image inspect nvcr.io/nvidia/isaac-sim:5.1.0 >/dev/null 2>&1; then
    printf 'docker-image:nvcr.io/nvidia/isaac-sim:5.1.0:/isaac-sim\n'
    exit 0
  fi
  img="$(docker image ls --format '{{.Repository}}:{{.Tag}}' | grep -Ei 'isaac-sim' | head -n 1 || true)"
  if [ -n "${img}" ]; then
    printf 'docker-image:%s:/isaac-sim\n' "${img}"
    exit 0
  fi
fi

exit 1
