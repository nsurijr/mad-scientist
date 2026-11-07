#!/bin/bash
#SBATCH --job-name=ms-vllm-concurrency
#SBATCH --account=pi_smd26
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Determinism with long concurrent prompts: replay real agent requests (first Researcher request of
# six runs plus the longest request of one run) alone, all at once, and staggered, against one
# server started with EXTRA_ARGS. Every reply must equal its "alone" reply.
#   EXTRA_ARGS="--no-enable-chunked-prefill" sbatch -J ms-conc-nochunk deploy/bouchet/vllm_concurrency_check.sbatch
# MODEL / MODEL_REV pick another model (default: the pipeline model); the recorded requests are
# replayed unchanged except for the model name.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/bouchet/common.sh
source deploy/bouchet/vllm_lib.sh

MODEL="${MODEL:-$PIPELINE_MODEL}"
REV="${MODEL_REV:-$PIPELINE_MODEL_REV}"
SERVED="${MODEL##*/}"
SOURCE="${SOURCE_BATCH:-runs/batch-30142957}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
PORT=$((20000 + SLURM_JOB_ID % 20000))
OUT="logs/vllm-concurrency-$SLURM_JOB_ID"
mkdir -p "$OUT"
echo "model=$MODEL@${REV:0:12} extra_args=[${EXTRA_ARGS:-}] vllm=$VLLM_TAG source=$SOURCE"

python3 - "$SOURCE" "$OUT" "$SERVED" <<'EOF'
import json, sys
src, out, served = sys.argv[1], sys.argv[2], sys.argv[3]
probes = {}
for run in ("s6-base", "s7-base", "s8-base", "s9-base", "s10-base", "s11-base"):
    with open(f"{src}/{run}/trace/trace.jsonl") as f:
        probes[f"{run}-first"] = json.loads(f.readline())["request"]
with open(f"{src}/s8-base/trace/trace.jsonl") as f:
    longest = max((json.loads(line) for line in f), key=lambda r: r["response"]["usage"]["prompt_tokens"])
probes["s8-base-longest"] = longest["request"]
for name, body in probes.items():
    body["model"] = served
    with open(f"{out}/{name}.json", "w") as f:
        json.dump(body, f)
print(f"{len(probes)} probes written; longest prompt {longest['response']['usage']['prompt_tokens']} tokens (gpt-oss tokenizer)")
EOF

# (VLLM_ATTENTION_BACKEND is ignored by vLLM 0.32; choose the backend with --attention-backend.)
vllm_start "$OUT/server.log" "$MODEL" "$REV" "$SERVED" "$PORT" "${EXTRA[@]}"
vllm_wait_ready "$PORT" || { grep -E "Error|error" "$OUT/server.log" | tail -5 | cut -c1-400; exit 1; }
grep -o -E "Chunked prefill is (enabled|disabled)[^.]*|max_num_batched_tokens=[0-9]+|Using [A-Z_]+ attention backend|AttentionBackendEnum\.[A-Z_]+" "$OUT/server.log" | sort -u

python3 deploy/bouchet/determinism_check.py replay --port "$PORT" --requests "$OUT"/s*.json --out "$OUT/replay.json"
status=$?
vllm_stop
exit $status
