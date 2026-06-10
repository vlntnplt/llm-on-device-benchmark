# Source this (from the repo root) to put the CUDA-12 pip-wheel libs on the
# loader path so onnxruntime-node's CUDA EP can resolve libcublas/libcufft.
# Usage:  source cuda-env.sh
NVIDIA="$PWD/harness/.venv/lib/python3.13/site-packages/nvidia"
export LD_LIBRARY_PATH="$(printf '%s:' "$NVIDIA"/*/lib)$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH primed with: $(printf '%s ' "$NVIDIA"/*/lib)"
