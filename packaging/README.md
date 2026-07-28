# packaging/ — the contributor artifact

One zip per platform turns "clone + toolchain + uv" into "unzip + run". The
zip is a pruned snapshot of the committed tree (`git archive HEAD`) with the
build products staged exactly where `backend.toml` already points
(`backends/ggml/build/bench-ggml`) plus a bundled `uv` — so the harness has no
packaged-layout special case, and the contributor needs nothing preinstalled.

```
bench-<tag>-<target>/
├── run.sh | run.bat + run.ps1    the only thing a contributor touches
├── README.txt                     ← contributor-readme.txt
├── bin/uv[.exe]                   pinned, checksum-verified at package time
├── backends/ggml/{backend.toml, build/bench-ggml + shared libs/modules}
├── harness/  schema/  tasks/  models.yaml
├── licenses/  MANIFEST.txt
```

- **`package.sh <target> [tag]`** — configure → build → stage → verify → zip
  (+ `.sha256`). Targets: `linux-x64`, `macos-arm64`, `windows-x64`. The
  toolchain comes from the environment (CI workflow or your shell); the
  script installs nothing. It hard-fails if the CPU-variant / Vulkan module
  set is short (a missing dlopen'd module is a silent capability downgrade on
  someone's box, i.e. wrong data) and smoke-runs the staged exe.
- **Build shape**: linux/windows use `GGML_BACKEND_DL` + `GGML_CPU_ALL_VARIANTS`
  (+ Vulkan, shared libs, `$ORIGIN` rpath on linux) so one x86 binary is
  *correct* on every microarch; macOS is a static Metal build with embedded
  shaders (one arm64 microarch — no variant machinery).
- **`run.sh` / `run.ps1`** — the contributor flow: exe smoke test → `uv sync`
  → `bench fetch` → `bench check` → `bench plan` → `bench run` →
  `bench bundle`. Every failure message says what to do next; every step
  resumes on re-run.
- Release builds run in CI (`.github/workflows/release.yml`), one job per
  target, on a tag push. `git archive HEAD` means only *committed* files ship.
