#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"

service_root="$PWD"
service_port="${FF_PORT:-10010}"
service_host="${FF_HOST:-127.0.0.1}"
log_dir="$service_root/var/log"
log_file="$log_dir/factorfactory-${service_port}.log"
service_label="com.factorfactory.${service_port}"
service_domain="gui/$(id -u)/${service_label}"

if [[ "$service_port" != <-> ]]; then
  print -u2 "FF_PORT 必须是整数。"
  exit 2
fi

listener_pid() {
  lsof -nP -t -iTCP:"$service_port" -sTCP:LISTEN 2>/dev/null | head -n 1
}

job_snapshot() {
  launchctl print "$service_domain" 2>/dev/null
}

job_pid() {
  job_snapshot | awk '/^[[:space:]]*pid = / { print $3; exit }'
}

wait_for_job_removal() {
  local attempts=0
  while job_snapshot >/dev/null && (( attempts < 80 )); do
    sleep 0.25
    attempts=$((attempts + 1))
  done
  ! job_snapshot >/dev/null
}

start_service() {
  mkdir -p "$log_dir"
  local process_id="$(job_pid || true)"
  local occupied_by="$(listener_pid || true)"
  if [[ -n "$process_id" ]] && [[ "$occupied_by" == "$process_id" ]]; then
    print "FactorFactory 已运行：PID $process_id · $service_host:$service_port"
    return 0
  fi
  if job_snapshot >/dev/null; then
    launchctl remove "$service_label"
    if ! wait_for_job_removal; then
      print -u2 "旧 launchd job 在 20 秒内未完成注销；未提交新实例。"
      return 1
    fi
  fi
  if [[ -n "$occupied_by" ]]; then
    print -u2 "端口 $service_port 已被 PID $occupied_by 占用；为避免双实例，未启动。"
    return 1
  fi

  launchctl submit \
    -l "$service_label" \
    -o "$log_file" \
    -e "$log_file" \
    -- /usr/bin/env \
    "FF_HOST=$service_host" \
    "FF_PORT=$service_port" \
    "$service_root/run.sh"

  local attempts=0
  while (( attempts < 160 )); do
    process_id="$(job_pid || true)"
    occupied_by="$(listener_pid || true)"
    if [[ -n "$process_id" ]] && [[ "$occupied_by" == "$process_id" ]]; then
      print "FactorFactory 已启动：PID $process_id · $service_host:$service_port"
      return 0
    fi
    sleep 0.25
    attempts=$((attempts + 1))
  done
  launchctl remove "$service_label" 2>/dev/null || true
  print -u2 "服务在 40 秒内没有就绪；已撤销该 launchd job。最近日志："
  tail -n 40 "$log_file" >&2
  return 1
}

stop_service() {
  local process_id="$(job_pid || true)"
  local occupied_by="$(listener_pid || true)"
  if [[ -z "$process_id" ]]; then
    if [[ -n "$occupied_by" ]]; then
      print -u2 "端口由未受 service.sh 管理的 PID $occupied_by 占用；未擅自终止。"
      return 1
    fi
    print "FactorFactory 未运行。"
    return 0
  fi
  if [[ -n "$occupied_by" ]] && [[ "$occupied_by" != "$process_id" ]]; then
    print -u2 "launchd job 与端口监听者不一致；为避免误杀，未执行停止。"
    return 1
  fi

  launchctl remove "$service_label"
  local attempts=0
  while [[ "$(listener_pid || true)" == "$process_id" ]] && (( attempts < 80 )); do
    sleep 0.25
    attempts=$((attempts + 1))
  done
  if [[ "$(listener_pid || true)" == "$process_id" ]]; then
    print -u2 "PID $process_id 在 20 秒内未退出；未执行强制终止。"
    return 1
  fi
  if ! wait_for_job_removal; then
    print -u2 "PID 已退出，但 launchd job 在 20 秒内未完成注销。"
    return 1
  fi
  print "FactorFactory 已停止：PID $process_id"
}

status_service() {
  local process_id="$(job_pid || true)"
  local occupied_by="$(listener_pid || true)"
  if [[ -n "$process_id" ]] && [[ "$occupied_by" == "$process_id" ]]; then
    print "running · launchd · PID $process_id · $service_host:$service_port"
    curl --noproxy "*" -fsS \
      "http://localhost:${service_port}/api/health/ready" || true
    print
    return 0
  fi
  if [[ -n "$process_id" ]]; then
    print "starting-or-failed · launchd · PID $process_id · $service_host:$service_port"
    return 2
  fi
  if [[ -n "$occupied_by" ]]; then
    print "unmanaged-listener · PID $occupied_by · *:$service_port"
    return 2
  fi
  print "stopped · $service_host:$service_port"
  return 1
}

case "${1:-status}" in
  start)
    start_service
    ;;
  stop)
    stop_service
    ;;
  restart)
    stop_service
    start_service
    ;;
  status)
    status_service
    ;;
  health)
    curl --noproxy "*" -fsS \
      "http://localhost:${service_port}/api/health/ready"
    print
    ;;
  logs)
    if [[ "${2:-}" == "-f" ]]; then
      tail -n 80 -f "$log_file"
    else
      tail -n 80 "$log_file"
    fi
    ;;
  *)
    print -u2 "用法：./service.sh {start|stop|restart|status|health|logs [-f]}"
    exit 2
    ;;
esac
