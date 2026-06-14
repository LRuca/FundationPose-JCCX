#!/usr/bin/env bash
set -euo pipefail

WHEELHOUSE="/mnt/c/Users/lenovo/Desktop/JXCX/wheelhouse"
LOG="${WHEELHOUSE}/download.log"
mkdir -p "${WHEELHOUSE}"

urls=(
  "https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/torch-2.0.0%2Bcu118-cp39-cp39-linux_x86_64.whl"
  "https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/torchvision-0.15.1%2Bcu118-cp39-cp39-linux_x86_64.whl"
  "https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/torchaudio-2.0.1%2Bcu118-cp39-cp39-linux_x86_64.whl"
)

names=(
  "torch-2.0.0+cu118-cp39-cp39-linux_x86_64.whl"
  "torchvision-0.15.1+cu118-cp39-cp39-linux_x86_64.whl"
  "torchaudio-2.0.1+cu118-cp39-cp39-linux_x86_64.whl"
)

sizes=(
  2267301959
  6071585
  4406043
)

human_bytes() {
  local bytes="$1"
  awk -v b="${bytes}" 'BEGIN {
    split("B KB MB GB TB", u, " ");
    i=1;
    while (b >= 1024 && i < 5) { b/=1024; i++ }
    printf "%.2f%s", b, u[i]
  }'
}

remote_size() {
  local url="$1"
  curl -L -sI "${url}" \
    | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {size=$2} END {gsub("\r","",size); print size+0}'
}

log_progress() {
  local file="$1"
  local total="$2"
  local last_size=0
  local last_time
  last_time="$(date +%s)"

  while true; do
    sleep 10
    local now size elapsed delta speed percent remain eta
    now="$(date +%s)"
    size=0
    [[ -f "${file}" ]] && size="$(stat -c%s "${file}" 2>/dev/null || echo 0)"
    elapsed=$(( now - last_time ))
    delta=$(( size - last_size ))
    if [[ "${elapsed}" -gt 0 ]]; then
      speed=$(( delta / elapsed ))
    else
      speed=0
    fi
    if [[ "${total}" -gt 0 ]]; then
      percent="$(awk -v s="${size}" -v t="${total}" 'BEGIN { printf "%.2f", s*100/t }')"
      remain=$(( total - size ))
      if [[ "${speed}" -gt 0 ]]; then
        eta=$(( remain / speed ))
      else
        eta=-1
      fi
      printf '[%s] %s %s/%s %s%% speed=%s/s eta=%ss\n' \
        "$(date '+%F %T')" "$(basename "${file}")" "$(human_bytes "${size}")" \
        "$(human_bytes "${total}")" "${percent}" "$(human_bytes "${speed}")" "${eta}" >> "${LOG}"
    else
      printf '[%s] %s %s speed=%s/s total=unknown\n' \
        "$(date '+%F %T')" "$(basename "${file}")" "$(human_bytes "${size}")" \
        "$(human_bytes "${speed}")" >> "${LOG}"
    fi
    last_size="${size}"
    last_time="${now}"
  done
}

{
  echo "==== download started $(date '+%F %T') ===="
  for i in "${!urls[@]}"; do
    url="${urls[$i]}"
    name="${names[$i]}"
    out="${WHEELHOUSE}/${name}"
    total="${sizes[$i]}"
    echo "---- ${name} total=$(human_bytes "${total}") ----"
    log_progress "${out}" "${total}" &
    monitor_pid="$!"
    wget -c -q -O "${out}" "${url}"
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    final_size="$(stat -c%s "${out}" 2>/dev/null || echo 0)"
    echo "DONE ${name} size=$(human_bytes "${final_size}")"
  done
  echo "==== download finished $(date '+%F %T') ===="
} >> "${LOG}" 2>&1
