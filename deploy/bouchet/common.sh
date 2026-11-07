# Shared paths and version pins for Bouchet jobs. Source from the repo root:
#   source deploy/bouchet/common.sh
export REPO="$HOME/project_pi_smd26/nts36/mad-scientist"
export SIF_DIR="$REPO/containers"
export APPTAINER_CACHEDIR="$HOME/scratch_pi_smd26/nts36/apptainer_cache"
export HF_HOME="$HOME/project_pi_smd26/nts36/hf_cache"
# Triton / torch.compile caches for vLLM; scratch is fine since they are rebuilt on a miss.
export VLLM_CACHE_DIR="$HOME/scratch_pi_smd26/nts36/vllm_cache"

# vLLM release image, pinned by its amd64 digest. v0.32.0 (published 2026-10-18) is the first
# release with a batch-invariant MXFP4 MoE backend for gpt-oss on H200.
# Override both together to try another release, e.g. v0.31.0:
#   VLLM_TAG=v0.31.0 VLLM_DIGEST=sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
export VLLM_TAG="${VLLM_TAG:-v0.32.0}"
export VLLM_DIGEST="${VLLM_DIGEST:-sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9}"
export VLLM_SIF="$SIF_DIR/vllm-openai_${VLLM_TAG}.sif"

# Throwaway image for sandbox checks; the real sandbox image is built from containers/*.def.
export SANDBOX_SIF="$SIF_DIR/python-3.12-slim.sif"

# Model revisions are pinned to exact Hugging Face commits.
export PIPELINE_MODEL="openai/gpt-oss-120b"
export PIPELINE_MODEL_REV="b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
export MONITOR_MODEL="Qwen/Qwen3-32B"
export MONITOR_MODEL_REV="9216db5781bf21249d130ec9da846c4624c16137"
# Monitor candidates, none gated on Hugging Face.
export GEMMA_MODEL="google/gemma-4-31B-it"
export GEMMA_MODEL_REV="842da3794eaa0b77d5f08bae87a17459d91ff475"
export MISTRAL_MODEL="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
export MISTRAL_MODEL_REV="95a6d26c4bfb886c58daf9d3f7332c857cb27b43"

# gpt-oss's system header states the conversation start date. vLLM uses the server's clock unless
# VLLM_SYSTEM_START_DATE is set, so identical requests got different replies on different days.
# Pinned for every run, recording and replay: the first event day.
export SYSTEM_START_DATE="${SYSTEM_START_DATE:-2026-11-23}"

# iLaaS, the hosted event API: BACKEND=ilaas (backend_lib.sh) sends the
# pipeline's model calls there through the gateway. The key stays in secrets/ (gitignored).
export BACKEND="${BACKEND:-vllm}"
export ILAAS_BASE_URL="${ILAAS_BASE_URL:-https://llm.aristote.education/v1}"
export ILAAS_KEY_FILE="${ILAAS_KEY_FILE:-secrets/ilaas.key}"
# Goes into every cache key in place of the vLLM revision, so iLaaS and vLLM recordings never mix.
# Set it from the conformance check (api_conformance.py: the version iLaaS reports) and keep it
# fixed from the golden recording through the event.
export ILAAS_MODEL_REV="${ILAAS_MODEL_REV:-ilaas-unverified}"
export ILAAS_RETRIES="${ILAAS_RETRIES:-6}"
export ILAAS_MAX_CONCURRENCY="${ILAAS_MAX_CONCURRENCY:-16}"
export ILAAS_DROP_PARAMS="${ILAAS_DROP_PARAMS:-}"  # request fields iLaaS refuses, e.g. "seed"
# The iLaaS model that answers the agents' gpt-oss-120b calls, if another (our key has no
# gpt-oss-120b). It goes into the cache key, so each model's recordings stay apart.
export ILAAS_SERVED_NAME="${ILAAS_SERVED_NAME:-}"
