#!/bin/bash
set -e

cd "$(dirname "$0")"

# -----------------------------------------------------------------------------
# Step 1: Verify torch + CUDA.
# Evidence from the 2026-04-21 real-machine log: the platform has torch + CUDA
# pre-installed at /usr/local/lib/python3.12/dist-packages (system-wide), and
# vLLM imports fine there. The bootstrap branch below is only for a hypothetical
# greenfield machine — the real competition platform never takes that path.
# -----------------------------------------------------------------------------
echo "[setup] Step 1: Verifying torch + CUDA..."
if python -c "
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print(f'[setup] torch {torch.__version__}  cuda {torch.version.cuda}  '
      f'devices={torch.cuda.device_count()}')
" 2>/dev/null; then
    :
else
    echo "[setup] No working torch+CUDA found; bootstrapping torch 2.7.0 + cu128 family..."
    pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 \
        || { echo "[setup] FATAL: torch cu128 bootstrap failed (no network?)"; exit 1; }
    python -c "
import torch
assert torch.cuda.is_available(), 'CUDA unavailable even after install'
print(f'[setup] Bootstrapped torch {torch.__version__}  cuda {torch.version.cuda}')
" || { echo "[setup] FATAL: torch installed but CUDA still unavailable"; exit 1; }
fi

# -----------------------------------------------------------------------------
# Step 2: Ensure a 0.9+ vllm is available (0.9+ is when --no-enable-log-requests
# and --speculative-config JSON replaced the old flags we hit in the first upload).
#
# DO NOT downgrade to 0.9.2 if the platform already has a newer version — the
# 2026-04-21 log shows the platform has vLLM 0.11.x-or-newer (evidenced by
# flags like --moe-backend / --kv-offloading-backend / --gdn-prefill-backend
# that don't exist before 0.11). Downgrading would break binary/dep integrity.
#
# Use a semver comparison in Python rather than a bash glob — a glob like
# `0.9.*|0.10.*|0.11.*` silently misses 0.12.x / 1.x / pre-release dev tags.
# -----------------------------------------------------------------------------
echo "[setup] Step 2: Checking platform vllm..."
VLLM_STATUS=$(python - <<'PY'
try:
    import vllm
    v = vllm.__version__
    parts = v.split('.')
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    # 0.9+ or any 1.x+ has the new CLI flag syntax we need.
    compat = (major > 0) or (major == 0 and minor >= 9)
    print(f"{'compat' if compat else 'old'}|{v}")
except ImportError:
    print("missing|")
except Exception as e:
    print(f"error|{type(e).__name__}:{e}")
PY
)
VLLM_STATE="${VLLM_STATUS%%|*}"
VLLM_VER="${VLLM_STATUS#*|}"
echo "[setup] vllm state=$VLLM_STATE  version=$VLLM_VER"

case "$VLLM_STATE" in
    compat)
        echo "[setup] Keeping platform vllm $VLLM_VER (supports new CLI flags)."
        ;;
    old)
        echo "[setup] WARNING: platform vllm $VLLM_VER predates 0.9. Installing 0.9.2 --no-deps."
        pip install --no-deps vllm==0.9.2
        ;;
    missing)
        echo "[setup] No vllm found; installing 0.9.2 with deps (greenfield path)."
        # PEP 440 local-version matching: installed torch X.Y.Z+cu128 satisfies
        # vLLM's hard `torch==X.Y.Z` spec, so pip won't replace it.
        pip install vllm==0.9.2
        ;;
    error)
        echo "[setup] FATAL: vllm import raised $VLLM_VER"; exit 1
        ;;
esac

# -----------------------------------------------------------------------------
# Step 3: Ensure httpx (used by client.py + warmup.py).
# Platform likely already has it (fastapi pulls it transitively); skip if so.
# -----------------------------------------------------------------------------
echo "[setup] Step 3: Ensuring httpx..."
if python -c "import httpx" 2>/dev/null; then
    python -c "import httpx; print(f'[setup] httpx {httpx.__version__} already present')"
else
    pip install httpx==0.27.2
fi

# -----------------------------------------------------------------------------
# Step 4: Import sanity check — all critical packages load before run.sh bothers
# to spin up vLLM. Cheaper to fail here than after the server eats 30s of GPU mem.
# -----------------------------------------------------------------------------
echo "[setup] Step 4: Import sanity check..."
python -c "
import vllm, httpx, transformers, torch
print(f'[setup] vllm={vllm.__version__}  torch={torch.__version__}  '
      f'transformers={transformers.__version__}  httpx={httpx.__version__}')
" || { echo "[setup] FATAL: import check failed"; exit 1; }

# -----------------------------------------------------------------------------
# Step 5 (optional): download draft model for speculative decoding.
# Disabled by default — draft model is typically bundled inside
# submission/draft_model/ at build time. Enable via:
#   DOWNLOAD_SPEC_MODEL=1 bash setup.sh
# -----------------------------------------------------------------------------
if [ "${DOWNLOAD_SPEC_MODEL:-0}" = "1" ]; then
    DRAFT_DIR=${SPEC_MODEL:-/tmp/spec_draft}
    DRAFT_REPO=${SPEC_REPO:-Qwen/Qwen3-1.5B}
    echo "[setup] Step 5: Downloading draft model ${DRAFT_REPO} → ${DRAFT_DIR}"
    pip install --quiet huggingface_hub
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${DRAFT_REPO}', local_dir='${DRAFT_DIR}', local_dir_use_symlinks=False)
print('[setup] draft model ready at ${DRAFT_DIR}')
" || echo "[setup] WARN: draft download failed; speculative decoding will be disabled"
fi

echo "[setup] Done."
