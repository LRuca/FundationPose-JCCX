#!/usr/bin/env bash
set -euo pipefail

WHEELHOUSE="/mnt/c/Users/lenovo/Desktop/JXCX/wheelhouse"

show_one() {
  local name="$1"
  local total="$2"
  local path="${WHEELHOUSE}/${name}"
  local size=0
  [[ -f "${path}" ]] && size="$(stat -c%s "${path}" 2>/dev/null || echo 0)"
  awk -v n="${name}" -v s="${size}" -v t="${total}" 'BEGIN {
    printf "[%s] %s %.2fMB/%.2fMB %.2f%%\n", strftime("%F %T"), n, s/1024/1024, t/1024/1024, s*100/t
  }'
}

show_one "torch-2.0.0+cu118-cp39-cp39-linux_x86_64.whl" 2267301959
show_one "torchvision-0.15.1+cu118-cp39-cp39-linux_x86_64.whl" 6071585
show_one "torchaudio-2.0.1+cu118-cp39-cp39-linux_x86_64.whl" 4406043
ps -ef | grep wget | grep -v grep || true
