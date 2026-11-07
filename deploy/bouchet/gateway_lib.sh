# Start and stop the record-and-replay gateway (harness/gateway.py) inside a Slurm job.
# Source after env.sh and common.sh:
#   gateway_start <record|replay|eval> <namespace> <port> <log.jsonl> [--upstream <url>] ...
#   gateway_stop
# With GATEWAY_SOCKET set (keep it short, under /tmp), the gateway also listens on that Unix
# socket; pass the same path to harness.runner --gateway-socket so stage containers run with no
# network and reach the gateway only through it.
# The content-addressed store lives in $REPO/cas. Keys use the pinned model revision, gpt-oss's
# pinned system date and the pipeline image's sha256, so recording and replay must agree on all three.

gateway_start() {
  local mode="$1" ns="$2" port="$3" log="$4"
  shift 4
  local socket=()
  if [ -n "${GATEWAY_SOCKET:-}" ]; then
    mkdir -p "$(dirname "$GATEWAY_SOCKET")" && chmod 700 "$(dirname "$GATEWAY_SOCKET")"
    socket=(--socket "$GATEWAY_SOCKET")
  fi
  local cmd=(.venv/bin/python -m harness.gateway serve --cas "$REPO/cas" --namespace "$ns" --mode "$mode"
             --port "$port" --log "$log" --model-rev "$PIPELINE_MODEL_REV" "${socket[@]}"
             --system-start-date "$SYSTEM_START_DATE"
             --container-digest "$(cut -c1-64 "$SIF_DIR/pipeline.sif.sha256")" "$@")
  if [ -n "${GATEWAY_KEY_FILE:-}" ]; then  # an API key (backend_lib.sh): in this process's environment only
    ILAAS_API_KEY="$(< "$GATEWAY_KEY_FILE")" "${cmd[@]}" > "${log%.jsonl}.out" 2>&1 &
  else
    "${cmd[@]}" > "${log%.jsonl}.out" 2>&1 &
  fi
  GATEWAY_PID=$!
  local waited=0
  while (( waited < 60 )); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && { echo "gateway ($mode, $ns) ready"; return 0; }
    kill -0 "$GATEWAY_PID" 2>/dev/null || { echo "gateway exited during startup"; cat "${log%.jsonl}.out"; return 1; }
    sleep 1
    waited=$((waited + 1))
  done
  echo "gateway not ready after 60s"
  return 1
}

gateway_stop() {
  kill "$GATEWAY_PID" 2>/dev/null
  wait "$GATEWAY_PID" 2>/dev/null
  [ -n "${GATEWAY_SOCKET:-}" ] && rmdir "$(dirname "$GATEWAY_SOCKET")" 2>/dev/null
  return 0
}
