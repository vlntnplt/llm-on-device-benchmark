# One-shot contributor entry point (Windows): check the exe runs here, set up a
# private Python, fetch models, measure, and pack a submission tarball. Safe to
# re-run — every step resumes or is idempotent. Nothing is installed outside
# this folder and uv's cache.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Say($msg)  { Write-Host "`n== $msg" }
function Fail($msg) { Write-Host "`n!! $msg" -ForegroundColor Red; exit 1 }

$uv = Join-Path $PSScriptRoot "bin\uv.exe"
if (-not (Test-Path $uv)) { Fail "bin\uv.exe is missing - the zip may be incomplete; please re-download it." }

Say "Checking the benchmark exe starts on this machine"
& backends\ggml\build\bench-ggml.exe version *> exe-error.log
if ($LASTEXITCODE -ne 0) {
  Fail ("the benchmark exe failed to start (details in exe-error.log). " +
    "Please open an issue and attach exe-error.log: " +
    "https://github.com/vlntnplt/llm-on-device-benchmark/issues/new")
}

Say "Setting up Python (self-contained - nothing touches your system Python)"
& $uv sync --project harness
if ($LASTEXITCODE -ne 0) { Fail "Python setup failed - check your network connection and re-run run.bat (it resumes)." }

Say "Fetching models (tens of GB on first run; safe to interrupt and re-run)"
& $uv run --project harness bench fetch
if ($LASTEXITCODE -ne 0) { Fail "model download failed or was interrupted - re-run run.bat to resume. Downloads can look stalled for minutes and then jump; that is normal." }

Say "Conformance-checking the exe against the contract"
& $uv run --project harness bench check --backend ggml
if ($LASTEXITCODE -ne 0) { Fail "the exe runs but failed its conformance check - please open an issue with the output above." }

Say "What will be measured on this machine"
& $uv run --project harness bench plan --backend ggml
if ($LASTEXITCODE -ne 0) { Fail "planning failed - please open an issue with the output above." }

Say "Running the benchmark (typically 1-3 hours; keep the machine plugged in and idle)"
& $uv run --project harness bench run --backend ggml --out results/local
if ($LASTEXITCODE -ne 0) { Fail "the benchmark run failed - please open an issue with the output above." }

Say "Packing your submission"
& $uv run --project harness bench bundle results/local --out .
if ($LASTEXITCODE -ne 0) { Fail "bundling failed - please open an issue with the output above." }

Say "All done - attach the submission-*.tar.gz above to a new issue (link above)."
