#!/usr/bin/env bash
# One-shot contributor entry point: check the exe runs here, set up a private
# Python, fetch models, measure, and pack a submission tarball. Safe to re-run —
# every step resumes or is idempotent. Nothing is installed outside this folder
# and uv's cache.
set -euo pipefail
cd "$(dirname "$0")"

UV="bin/uv"
say()  { printf '\n== %s\n' "$*"; }
fail() { printf '\n!! %s\n' "$*" >&2; exit 1; }

[ -x "$UV" ] || fail "bin/uv is missing or not executable — the zip may be
   incomplete; please re-download it."

say "Checking the benchmark exe starts on this machine"
if ! backends/ggml/build/bench-ggml version >/dev/null 2>exe-error.log; then
  fail "the benchmark exe failed to start (details in exe-error.log).
   Common causes: a Linux distribution older than ~2022 (glibc), or an unusual
   GPU driver stack. Please open an issue and attach exe-error.log:
   https://github.com/vlntnplt/llm-on-device-benchmark/issues/new"
fi

say "Setting up Python (self-contained — nothing touches your system Python)"
"$UV" sync --project harness ||
  fail "Python setup failed — check your network connection and re-run ./run.sh
   (it picks up where it left off)."

say "Fetching models (tens of GB on first run; safe to interrupt and re-run)"
"$UV" run --project harness bench fetch ||
  fail "model download failed or was interrupted — re-run ./run.sh to resume.
   Note: downloads can look stalled for minutes and then jump; that is normal."

say "Conformance-checking the exe against the contract"
"$UV" run --project harness bench check --backend ggml ||
  fail "the exe runs but failed its conformance check — please open an issue
   with the output above."

say "What will be measured on this machine"
"$UV" run --project harness bench plan --backend ggml
echo
echo "   If a GPU you expected is missing above: on headless Linux boxes your"
echo "   user usually needs the 'render' group (sudo usermod -aG render \$USER,"
echo "   then log out and back in) for the GPU to be visible."

say "Running the benchmark (typically 1–3 hours; keep the machine plugged in and idle)"
"$UV" run --project harness bench run --backend ggml --out results/local ||
  fail "the benchmark run failed — please open an issue with the output above."

say "Packing your submission"
"$UV" run --project harness bench bundle results/local --out . ||
  fail "bundling failed — please open an issue with the output above."

say "All done — attach the submission-*.tar.gz above to a new issue (link above)."
