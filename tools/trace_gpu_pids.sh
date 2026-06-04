#!/usr/bin/env bash
set -euo pipefail

# Trace GPU compute PIDs from nvidia-smi back to process metadata.
# Read-only: this script never kills or modifies processes.
#
# Usage:
#   tools/trace_gpu_pids.sh
#   tools/trace_gpu_pids.sh --parents 3 --children
#   tools/trace_gpu_pids.sh --filter GLM-Image

parents=2
show_children=0
filter=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parents)
      parents="${2:?missing value for --parents}"
      shift 2
      ;;
    --children)
      show_children=1
      shift
      ;;
    --filter)
      filter="${2:?missing value for --filter}"
      shift 2
      ;;
    -h|--help)
      sed -n '1,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 1
fi

tmp_gpu="$(mktemp)"
tmp_proc="$(mktemp)"
trap 'rm -f "$tmp_gpu" "$tmp_proc"' EXIT

nvidia-smi --query-gpu=index,uuid,memory.used,memory.total \
  --format=csv,noheader,nounits > "$tmp_gpu"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits > "$tmp_proc" || true

echo "GPU UUID mapping:"
awk -F', *' '{ printf "  GPU %s  %s  used=%s/%s MiB\n", $1, $2, $3, $4 }' "$tmp_gpu"

if [[ ! -s "$tmp_proc" ]]; then
  echo
  echo "No GPU compute processes found."
  exit 0
fi

gpu_index_for_uuid() {
  local uuid="$1"
  awk -F', *' -v uuid="$uuid" '$2 == uuid { print $1; exit }' "$tmp_gpu"
}

gpu_mem_for_uuid() {
  local uuid="$1"
  awk -F', *' -v uuid="$uuid" '$2 == uuid { print $3 "/" $4 " MiB"; exit }' "$tmp_gpu"
}

print_ps() {
  local pid="$1"
  if [[ -z "$pid" ]] || ! ps -p "$pid" >/dev/null 2>&1; then
    echo "  PID $pid no longer exists"
    return
  fi
  ps -o pid,ppid,pgid,user,etimes,stat,cmd -p "$pid" --no-headers | sed 's/^/  /'
}

print_parent_chain() {
  local pid="$1"
  local depth=0
  local current="$pid"
  while [[ "$depth" -lt "$parents" ]]; do
    current="$(ps -o ppid= -p "$current" 2>/dev/null | tr -d ' ')"
    if [[ -z "$current" || "$current" == "0" ]]; then
      break
    fi
    echo "  parent[$((depth + 1))]:"
    print_ps "$current"
    depth=$((depth + 1))
  done
}

print_process_group() {
  local pgid="$1"
  if [[ -z "$pgid" ]]; then
    return
  fi
  echo "  process_group[$pgid]:"
  ps -eo pid,ppid,pgid,user,etimes,stat,cmd | awk -v pgid="$pgid" '$3 == pgid { print "  " $0 }'
}

print_children() {
  local root="$1"
  echo "  children:"
  ps -eo pid,ppid,pgid,user,etimes,stat,cmd | awk -v root="$root" '$2 == root { print "  " $0 }'
}

echo
echo "GPU compute processes:"
while IFS=',' read -r raw_uuid raw_pid raw_name raw_mem; do
  uuid="$(echo "$raw_uuid" | xargs)"
  pid="$(echo "$raw_pid" | xargs)"
  name="$(echo "$raw_name" | xargs)"
  mem="$(echo "$raw_mem" | xargs)"
  index="$(gpu_index_for_uuid "$uuid")"
  total_mem="$(gpu_mem_for_uuid "$uuid")"
  ps_line="$(ps -o pid,ppid,pgid,user,etimes,stat,cmd -p "$pid" --no-headers 2>/dev/null || true)"

  if [[ -n "$filter" ]] && [[ "$name $ps_line" != *"$filter"* ]]; then
    continue
  fi

  echo "--------------------------------------------------------------------------------"
  echo "GPU ${index:-?}  uuid=$uuid  gpu_mem=$mem MiB  gpu_total_used=$total_mem"
  echo "process_name=$name"
  echo "process:"
  if [[ -n "$ps_line" ]]; then
    echo "$ps_line" | sed 's/^/  /'
  else
    echo "  PID $pid no longer exists"
  fi
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
  print_process_group "$pgid"
  print_parent_chain "$pid"
  if [[ "$show_children" -eq 1 ]]; then
    print_children "$pid"
  fi
done < "$tmp_proc"
#!/usr/bin/env bash
set -euo pipefail

# Trace GPU compute PIDs from nvidia-smi back to their process tree metadata.
#
# Usage:
#   bash tools/trace_gpu_pids.sh
#   bash tools/trace_gpu_pids.sh --parents 3 --children
#   bash tools/trace_gpu_pids.sh --filter GLM-Image
#
# This script is read-only. It does not kill or modify processes.

parents=2
show_children=0
filter=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parents)
      parents="${2:?missing value for --parents}"
      shift 2
      ;;
    --children)
      show_children=1
      shift
      ;;
    --filter)
      filter="${2:?missing value for --filter}"
      shift 2
      ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 1
fi

tmp_gpu="$(mktemp)"
tmp_proc="$(mktemp)"
trap 'rm -f "$tmp_gpu" "$tmp_proc"' EXIT

nvidia-smi --query-gpu=index,uuid,memory.used,memory.total \
  --format=csv,noheader,nounits > "$tmp_gpu"

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits > "$tmp_proc" || true

if [[ ! -s "$tmp_proc" ]]; then
  echo "No GPU compute processes found."
  exit 0
fi

gpu_index_for_uuid() {
  local uuid="$1"
  awk -F', *' -v uuid="$uuid" '$2 == uuid { print $1; exit }' "$tmp_gpu"
}

gpu_mem_for_uuid() {
  local uuid="$1"
  awk -F', *' -v uuid="$uuid" '$2 == uuid { print $3 "/" $4 " MiB"; exit }' "$tmp_gpu"
}

print_ps() {
  local pid="$1"
  if [[ -z "$pid" ]] || ! ps -p "$pid" >/dev/null 2>&1; then
    echo "  PID $pid no longer exists"
    return
  fi
  ps -o pid,ppid,pgid,user,etimes,stat,cmd -p "$pid" --no-headers \
    | sed 's/^/  /'
}

print_parent_chain() {
  local pid="$1"
  local depth=0
  local current="$pid"
  while [[ "$depth" -lt "$parents" ]]; do
    current="$(ps -o ppid= -p "$current" 2>/dev/null | tr -d ' ')"
    if [[ -z "$current" || "$current" == "0" ]]; then
      break
    fi
    echo "  parent[$((depth + 1))]:"
    print_ps "$current"
    depth=$((depth + 1))
  done
}

print_children() {
  local root="$1"
  echo "  children:"
  ps -eo pid,ppid,pgid,user,etimes,stat,cmd \
    | awk -v root="$root" '$2 == root { print "  " $0 }'
}

print_process_group() {
  local pgid="$1"
  if [[ -z "$pgid" ]]; then
    return
  fi
  echo "  process_group[$pgid]:"
  ps -eo pid,ppid,pgid,user,etimes,stat,cmd \
    | awk -v pgid="$pgid" '$3 == pgid { print "  " $0 }'
}

printf '%s\n' "GPU UUID mapping:"
awk -F', *' '{ printf "  GPU %s  %s  used=%s/%s MiB\n", $1, $2, $3, $4 }' "$tmp_gpu"
printf '\n%s\n' "GPU compute processes:"

while IFS=',' read -r raw_uuid raw_pid raw_name raw_mem; do
  uuid="$(echo "$raw_uuid" | xargs)"
  pid="$(echo "$raw_pid" | xargs)"
  name="$(echo "$raw_name" | xargs)"
  mem="$(echo "$raw_mem" | xargs)"
  index="$(gpu_index_for_uuid "$uuid")"
  total_mem="$(gpu_mem_for_uuid "$uuid")"

  ps_line="$(ps -o pid,ppid,pgid,user,etimes,stat,cmd -p "$pid" --no-headers 2>/dev/null || true)"
  if [[ -n "$filter" ]] && [[ "$name $ps_line" != *"$filter"* ]]; then
    continue
  fi

  echo "--------------------------------------------------------------------------------"
  echo "GPU ${index:-?}  uuid=$uuid  gpu_mem=$mem MiB  gpu_total_used=$total_mem"
  echo "process_name=$name"
  echo "process:"
  if [[ -n "$ps_line" ]]; then
    echo "$ps_line" | sed 's/^/  /'
  else
    echo "  PID $pid no longer exists"
  fi
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
  print_process_group "$pgid"
  print_parent_chain "$pid"
  if [[ "$show_children" -eq 1 ]]; then
    print_children "$pid"
  fi
done < "$tmp_proc"
