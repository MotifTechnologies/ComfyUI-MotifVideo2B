#!/bin/bash
# session-init.sh — SessionStart hook
# 안내 메시지 + machine-profile 자동 생성

set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"

# --- cgroup 기반 리소스 감지 함수 ---

get_cpu_cores() {
  # cgroup v2
  if [[ -f /sys/fs/cgroup/cpu.max ]]; then
    local quota period
    read -r quota period < /sys/fs/cgroup/cpu.max
    if [[ "$quota" != "max" && "$quota" -gt 0 ]]; then
      awk "BEGIN {v=$quota/$period; if(v==int(v)) printf \"%d\", v; else printf \"%.1f\", v}"
      return
    fi
  fi
  # cgroup v1
  if [[ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
    local quota period
    quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo "-1")
    period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo "100000")
    if [[ "$quota" -gt 0 ]]; then
      awk "BEGIN {v=$quota/$period; if(v==int(v)) printf \"%d\", v; else printf \"%.1f\", v}"
      return
    fi
  fi
  # fallback
  nproc 2>/dev/null || echo "?"
}

get_memory_gb() {
  # cgroup v2
  if [[ -f /sys/fs/cgroup/memory.max ]]; then
    local mem_bytes
    mem_bytes=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo "max")
    if [[ "$mem_bytes" != "max" && "$mem_bytes" -gt 0 ]]; then
      awk "BEGIN {printf \"%.1f\", $mem_bytes/1024/1024/1024}"
      return
    fi
  fi
  # cgroup v1
  if [[ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    local mem_bytes
    mem_bytes=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo "0")
    # 9223372036854771712 = 무제한 (커널 기본값)
    if [[ "$mem_bytes" -lt 9223372036854771712 ]]; then
      awk "BEGIN {printf \"%.1f\", $mem_bytes/1024/1024/1024}"
      return
    fi
  fi
  # fallback
  if [[ -f /proc/meminfo ]]; then
    local total_kb
    total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo "0")
    awk "BEGIN {printf \"%.1f\", $total_kb/1024/1024}"
  else
    echo "?"
  fi
}

is_cgroup_limited() {
  # CPU 또는 메모리가 cgroup 제한 하에 있는지 확인
  # cgroup v2
  if [[ -f /sys/fs/cgroup/cpu.max ]]; then
    local quota
    read -r quota _ < /sys/fs/cgroup/cpu.max
    if [[ "$quota" != "max" && "$quota" -gt 0 ]]; then
      echo "true"
      return
    fi
  fi
  if [[ -f /sys/fs/cgroup/memory.max ]]; then
    local mem_bytes
    mem_bytes=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo "max")
    if [[ "$mem_bytes" != "max" ]]; then
      echo "true"
      return
    fi
  fi
  # cgroup v1
  if [[ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
    local quota
    quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo "-1")
    if [[ "$quota" -gt 0 ]]; then
      echo "true"
      return
    fi
  fi
  if [[ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    local mem_bytes
    mem_bytes=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo "0")
    if [[ "$mem_bytes" -gt 0 && "$mem_bytes" -lt 9223372036854771712 ]]; then
      echo "true"
      return
    fi
  fi
  echo "false"
}

detect_environment() {
  # K8s
  if [[ -n "${KUBERNETES_SERVICE_HOST:-}" ]]; then
    echo "Kubernetes"
    return
  fi
  # SkyPilot
  if command -v sky &>/dev/null || [[ -n "${SKYPILOT_TASK_ID:-}" ]] || [[ -n "${SKYPILOT_CLUSTER_NAME:-}" ]]; then
    echo "SkyPilot"
    return
  fi
  # Container (Docker/containerd)
  if [[ -f /.dockerenv ]]; then
    echo "Docker"
    return
  fi
  if [[ -f /proc/1/cgroup ]] && grep -qE 'docker|containerd' /proc/1/cgroup 2>/dev/null; then
    echo "Container"
    return
  fi
  echo "Bare Metal"
}

get_gpu_info() {
  # nvidia-smi 없으면 즉시 반환
  if ! command -v nvidia-smi &>/dev/null; then
    echo "NONE"
    return
  fi

  # CUDA_VISIBLE_DEVICES 처리
  local cuda_var="${CUDA_VISIBLE_DEVICES-__unset__}"
  local nvidia_var="${NVIDIA_VISIBLE_DEVICES-__unset__}"

  # 빈 문자열이면 GPU 없음 취급
  if [[ "$cuda_var" == "" ]] || [[ "$nvidia_var" == "" ]]; then
    echo "BLOCKED"
    return
  fi

  local gpu_info
  gpu_info=$(timeout 5 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null)
  local exit_code=$?
  if [[ $exit_code -eq 124 ]]; then
    echo "TIMEOUT"
    return
  fi
  if [[ $exit_code -ne 0 || -z "$gpu_info" ]]; then
    echo "FAIL"
    return
  fi

  local total_count
  total_count=$(echo "$gpu_info" | wc -l)
  local gpu_name
  gpu_name=$(echo "$gpu_info" | head -1 | cut -d',' -f1 | xargs)
  local gpu_mem
  gpu_mem=$(echo "$gpu_info" | head -1 | cut -d',' -f2 | xargs)

  # CUDA_VISIBLE_DEVICES가 설정되어 있으면 해당 개수만 카운트
  local visible_count="$total_count"
  if [[ "$cuda_var" != "__unset__" && -n "$cuda_var" ]]; then
    # 쉼표로 구분된 ID 개수
    visible_count=$(echo "$cuda_var" | tr ',' '\n' | wc -l)
    echo "VISIBLE|${gpu_name}|${gpu_mem}|${visible_count}|${total_count}"
  elif [[ "$nvidia_var" != "__unset__" && -n "$nvidia_var" && "$nvidia_var" != "all" ]]; then
    visible_count=$(echo "$nvidia_var" | tr ',' '\n' | wc -l)
    echo "VISIBLE|${gpu_name}|${gpu_mem}|${visible_count}|${total_count}"
  else
    echo "ALL|${gpu_name}|${gpu_mem}|${total_count}"
  fi
}

# 안내 메시지 (stdout → Claude context에 전달)
echo "📌 CLAUDE.md 로드됨. /task-plan으로 작업 시작, /restore으로 이전 작업 복구."

# _old_claude_files/ 존재 시 /migrate 안내
if [[ -d "$PROJECT_DIR/_old_claude_files" ]]; then
  echo "📦 _old_claude_files/ 감지됨. /migrate를 실행하여 기존 설정을 이식하세요."
fi

# machine-profile 생성 (없거나 24시간 경과 시 재생성)
PROFILE="$PROJECT_DIR/.manuals/env/machine-profile.md"
REGEN=false
if [[ ! -f "$PROFILE" ]]; then
  REGEN=true
else
  PROFILE_AGE=$(( $(date +%s) - $(stat -c %Y "$PROFILE" 2>/dev/null || echo 0) ))
  if [[ "$PROFILE_AGE" -ge 86400 ]]; then
    REGEN=true
  fi
fi

if [[ "$REGEN" == "true" ]]; then
  mkdir -p "$(dirname "$PROFILE")"

  {
    echo "# Machine Profile"
    echo ""
    echo "자동 생성: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # 실행 환경
    echo "## 실행 환경"
    ENVIRONMENT=$(detect_environment)
    CGROUP_LIMITED=$(is_cgroup_limited)
    echo "- 환경: ${ENVIRONMENT}"
    if [[ "$CGROUP_LIMITED" == "true" ]]; then
      echo "- cgroup 리소스 제한 적용됨"
    fi
    echo ""

    # CPU
    echo "## CPU"
    if [[ -f /proc/cpuinfo ]]; then
      CPU_MODEL=$(awk -F: '/model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null || echo "알 수 없음")
      echo "- 모델: ${CPU_MODEL}"
    fi
    CPU_CORES=$(get_cpu_cores)
    if [[ "$CGROUP_LIMITED" == "true" ]]; then
      HOST_CORES=$(nproc 2>/dev/null || echo "?")
      echo "- 코어: ${CPU_CORES} (cgroup 제한, 호스트: ${HOST_CORES})"
    else
      echo "- 코어: ${CPU_CORES}"
    fi
    echo ""

    # 메모리
    echo "## 메모리"
    MEM_GB=$(get_memory_gb)
    if [[ "$MEM_GB" != "?" ]]; then
      if [[ "$CGROUP_LIMITED" == "true" ]]; then
        echo "- 전체: ${MEM_GB}GB (cgroup 제한)"
      else
        echo "- 전체: ${MEM_GB}GB"
      fi
    else
      echo "- 정보 없음"
    fi
    echo ""

    # GPU
    echo "## GPU"
    GPU_RESULT=$(get_gpu_info) || true
    case "${GPU_RESULT%%|*}" in
      NONE)
        echo "- 없음"
        ;;
      BLOCKED)
        echo "- 없음 (CUDA_VISIBLE_DEVICES/NVIDIA_VISIBLE_DEVICES 비어있음)"
        ;;
      FAIL)
        echo "- nvidia-smi 실행 실패"
        ;;
      TIMEOUT)
        echo "- GPU: 감지 타임아웃 (nvidia-smi 응답 없음)"
        ;;
      VISIBLE)
        IFS='|' read -r _ gpu_name gpu_mem visible_count total_count <<< "$GPU_RESULT"
        echo "- 모델: ${gpu_name}"
        echo "- VRAM: ${gpu_mem}MB"
        echo "- 개수: ${visible_count}/${total_count} (CUDA_VISIBLE_DEVICES 제한)"
        ;;
      ALL)
        IFS='|' read -r _ gpu_name gpu_mem gpu_count <<< "$GPU_RESULT"
        echo "- 모델: ${gpu_name}"
        echo "- VRAM: ${gpu_mem}MB"
        echo "- 개수: ${gpu_count}"
        ;;
    esac
    echo ""

    # 디스크
    echo "## 디스크"
    DISK_INFO=$(timeout 3 df -h "$PROJECT_DIR" 2>/dev/null | awk 'NR==2 {print $4 " 여유 / " $2 " 전체"}') || DISK_INFO=""
    echo "- ${DISK_INFO:-정보 없음}"
    echo ""

    # OS
    echo "## OS"
    if [[ -f /etc/os-release ]]; then
      OS_NAME=$(. /etc/os-release && echo "${PRETTY_NAME:-$NAME $VERSION}" 2>/dev/null || echo "알 수 없음")
    else
      OS_NAME=$(uname -sr 2>/dev/null || echo "알 수 없음")
    fi
    echo "- ${OS_NAME}"
    echo "- 커널: $(uname -r 2>/dev/null || echo '?')"

    # Python/PyTorch
    echo ""
    echo "## Python"
    PYVER=$(timeout 3 python3 --version 2>/dev/null || timeout 3 python --version 2>/dev/null || echo "없음")
    echo "- ${PYVER}"
    TORCH=$(timeout 5 python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>/dev/null || echo "없음")
    echo "- ${TORCH}"
  } > "$PROFILE"

  echo "[machine-profile 생성 완료: $PROFILE]"
else
  # machine-profile 이미 존재 → 간결 요약 출력
  CPU_CORES=$(get_cpu_cores)
  MEM_GB=$(get_memory_gb)
  # 정수로 변환 (요약용)
  MEM_GB_INT=$(awk "BEGIN {printf \"%.0f\", ${MEM_GB:-0}}" 2>/dev/null || echo "?")
  GPU_SUMMARY="없음"
  GPU_RESULT=$(get_gpu_info) || true
  case "${GPU_RESULT%%|*}" in
    NONE)
      GPU_SUMMARY="없음"
      ;;
    BLOCKED)
      GPU_SUMMARY="없음 (CUDA blocked)"
      ;;
    FAIL)
      GPU_SUMMARY="감지 실패"
      ;;
    TIMEOUT)
      GPU_SUMMARY="감지 타임아웃"
      ;;
    VISIBLE)
      IFS='|' read -r _ gpu_name _ visible_count total_count <<< "$GPU_RESULT"
      GPU_SUMMARY="${gpu_name} x${visible_count}/${total_count}"
      ;;
    ALL)
      IFS='|' read -r _ gpu_name _ gpu_count <<< "$GPU_RESULT"
      GPU_SUMMARY="${gpu_name} x${gpu_count}"
      ;;
  esac
  echo "🖥️ CPU ${CPU_CORES}코어 | 메모리 ${MEM_GB_INT}GB | GPU: ${GPU_SUMMARY}"
fi

# 원격 template 업데이트 감지
MANIFEST="$PROJECT_DIR/.claude/.template-manifest"
if [[ -f "$MANIFEST" ]]; then
  LOCAL_VERSION=$(grep '^VERSION=' "$MANIFEST" | cut -d= -f2-)
  SOURCE=$(grep '^SOURCE=' "$MANIFEST" | cut -d= -f2-)

  if [[ -n "$LOCAL_VERSION" && -n "$SOURCE" ]]; then
    # 24시간 캐시 체크
    CACHE_KEY=$(echo "$SOURCE" | md5sum | cut -d' ' -f1)
    CACHE_FILE="/tmp/.claude-version-check-${CACHE_KEY}"

    RUN_CHECK=true
    if [[ -f "$CACHE_FILE" ]]; then
      CACHE_AGE=$(( $(date +%s) - $(stat -c %Y "$CACHE_FILE" 2>/dev/null || echo 0) ))
      if [[ "$CACHE_AGE" -lt 86400 ]]; then
        RUN_CHECK=false
        # 캐시된 REMOTE_HEAD로 비교
        CACHED_REMOTE=$(cat "$CACHE_FILE" 2>/dev/null)
        if [[ -n "$CACHED_REMOTE" && "$LOCAL_VERSION" != "$CACHED_REMOTE" ]]; then
          echo "📦 template 업데이트 감지 — /migrate --update 로 적용하세요." >&2
        fi
      fi
    fi

    if [[ "$RUN_CHECK" == "true" ]]; then
      REMOTE_HEAD=$(timeout 3 git ls-remote "$SOURCE" HEAD 2>/dev/null | cut -f1) || true
      if [[ -n "$REMOTE_HEAD" ]]; then
        echo "$REMOTE_HEAD" > "$CACHE_FILE"
        if [[ "$LOCAL_VERSION" != "$REMOTE_HEAD" ]]; then
          echo "📦 template 업데이트 감지 — /migrate --update 로 적용하세요." >&2
        fi
      fi
    fi
  fi
fi

exit 0
