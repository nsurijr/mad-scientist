#!/bin/bash
#SBATCH --job-name=ms-sandbox-check
#SBATCH --account=pi_smd26
#SBATCH --partition=devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
# Checks the sandbox profile and the network setup on a compute node.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/bouchet/common.sh

status=0
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; status=1; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
SANDBOX=(apptainer exec --containall --net --network none)

echo "node: $(hostname)  apptainer: $(apptainer --version)"

# Eval workers must reach the public front VM over outbound HTTPS.
code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 https://github.com)
[ "$code" = "200" ] && pass "compute node has outbound HTTPS (github.com -> $code)" \
                    || fail "compute node outbound HTTPS (github.com -> $code)"

# An unprivileged network namespace with loopback only.
ifaces=$("${SANDBOX[@]}" "$SANDBOX_SIF" sh -c "tail -n +3 /proc/net/dev | cut -d: -f1 | tr -d ' ' | tr '\n' ' '" 2>&1)
[ "$ifaces" = "lo " ] && pass "sandbox sees only loopback" || fail "sandbox interfaces: $ifaces"

if "${SANDBOX[@]}" "$SANDBOX_SIF" python3 -c \
    "import socket; socket.create_connection(('140.82.114.4', 443), timeout=5)" 2>/dev/null; then
  fail "sandbox opened an outbound TCP connection"
else
  pass "sandbox cannot open outbound TCP"
fi

# The gateway reaches sandboxes only through a bind-mounted Unix socket.
mkdir "$work/gw"
python3 deploy/bouchet/uds_echo.py "$work/gw/gw.sock" &
srv=$!
sleep 1
reply=$("${SANDBOX[@]}" --bind "$work/gw:/run/gw" "$SANDBOX_SIF" python3 -c \
  "import socket; s = socket.socket(socket.AF_UNIX); s.connect('/run/gw/gw.sock'); s.sendall(b'ping'); print(s.recv(16).decode())" 2>&1)
kill "$srv" 2>/dev/null
[ "$reply" = "pong" ] && pass "sandbox reaches a bind-mounted Unix socket" || fail "Unix socket reply: $reply"

# Inputs are mounted read-only.
mkdir "$work/data" && echo original > "$work/data/c_data.csv"
"${SANDBOX[@]}" --bind "$work/data:/data:ro" "$SANDBOX_SIF" sh -c "echo changed > /data/c_data.csv" 2>/dev/null
[ "$(cat "$work/data/c_data.csv")" = "original" ] && pass "/data is read-only in the sandbox" \
                                                   || fail "sandbox modified /data"

# Project and home storage must not be visible. System bind paths may still mount /nfs.
for flags in "" "--no-mount bind-paths"; do
  seen=$("${SANDBOX[@]}" $flags "$SANDBOX_SIF" sh -c "ls -d /nfs/roberts/project/pi_smd26 $HOME/project_pi_smd26 2>/dev/null" 2>&1)
  label="project storage hidden (extra flags: ${flags:-none})"
  [ -z "$seen" ] && pass "$label" || fail "$label -> visible: $seen"
done

echo "overall: $([ $status -eq 0 ] && echo PASS || echo FAIL)"
exit $status
