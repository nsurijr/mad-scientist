# Where the pipeline's model is served. Source after env.sh, common.sh, vllm_lib.sh and
# gateway_lib.sh:
#   backend_start <batch dir> <port>    then pass "${UPSTREAM_ARGS[@]}" to gateway_start
#   backend_stop
# BACKEND=vllm (default): our own vLLM on this node's GPU, on <port>.
# BACKEND=ilaas: the hosted event API, no GPU. Every call goes
# through the gateway, the only process given the key (ILAAS_API_KEY, read from $ILAAS_KEY_FILE by
# gateway_start); the stage containers never see it. BACKEND_JOBS is how many background jobs
# backend_start left running (the vLLM server, or none).

backend_start() {
  local batch="$1" port="$2"
  case "${BACKEND:-vllm}" in
    vllm)
      vllm_start "$batch/vllm.log" "$PIPELINE_MODEL" "$PIPELINE_MODEL_REV" gpt-oss-120b "$port"
      vllm_wait_ready "$port" || { tail -20 "$batch/vllm.log"; return 1; }
      UPSTREAM_ARGS=(--upstream "http://127.0.0.1:$port/v1")
      BACKEND_JOBS=1
      ;;
    ilaas)
      [ -s "$ILAAS_KEY_FILE" ] || { echo "BACKEND=ilaas needs the API key in $ILAAS_KEY_FILE"; return 1; }
      .venv/bin/python - "$batch/upstreams.json" <<'EOF' || return 1
import json, os, sys
spec = {"base_url": os.environ["ILAAS_BASE_URL"], "model_rev": os.environ["ILAAS_MODEL_REV"],
        "server": {"backend": "ilaas", "base_url": os.environ["ILAAS_BASE_URL"]},
        "api_key_env": "ILAAS_API_KEY", "retries": int(os.environ["ILAAS_RETRIES"]),
        "max_concurrency": int(os.environ["ILAAS_MAX_CONCURRENCY"]),
        "drop_params": os.environ.get("ILAAS_DROP_PARAMS", "").replace(",", " ").split()}
if os.environ.get("ILAAS_SERVED_NAME"):
    spec["served_name"] = spec["server"]["served_name"] = os.environ["ILAAS_SERVED_NAME"]
open(sys.argv[1], "w").write(json.dumps({"*": spec}, indent=2) + "\n")
EOF
      UPSTREAM_ARGS=(--upstreams "$batch/upstreams.json")
      GATEWAY_KEY_FILE="$ILAAS_KEY_FILE"
      BACKEND_JOBS=0
      echo "backend: iLaaS $ILAAS_BASE_URL, model ${ILAAS_SERVED_NAME:-as requested}, model_rev $ILAAS_MODEL_REV," \
           "retries $ILAAS_RETRIES, at most $ILAAS_MAX_CONCURRENCY in flight, dropped params [${ILAAS_DROP_PARAMS:-}]"
      ;;
    *)
      echo "BACKEND must be vllm or ilaas, not $BACKEND"
      return 1
      ;;
  esac
}

backend_stop() {
  [ "${BACKEND:-vllm}" = vllm ] && vllm_stop
  return 0
}
