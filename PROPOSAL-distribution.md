# Proposal: distributable benchmark + submission pipeline

Status: **proposal** — delete this file once rolled out (tracking lives in the
rollout checklist at the bottom).

## Goal

A person with idle home machines and no dev tooling can contribute a
submission: download one zip, run one script, attach one small file to a
GitHub issue. Identical pinned binaries on every machine; the maintainer
reviews and merges; the published report regenerates automatically.

Principles:

- **Zero prerequisites** on the contributor's box: no Python, no compiler, no
  git, no package manager. Network only (models must be fetched anyway).
- **Byte-identical measurement stack**: everyone runs the same exe built at
  the same llama.cpp pin with the same flags. "Compiled differently" stops
  existing as a confound.
- **The harness stays Python** (portable via a bundled `uv`); **analysis never
  ships** — it runs in CI and on the maintainer's machine only.
- **Humans gate publishing**: automation does the mechanical work; nothing
  lands on `main` (→ Pages) without maintainer review.

Four workstreams. A (device lanes) changes the wire format and must land
before any release binaries are frozen; B–D are independent of each other.

---

## A. Device-indexed provider lanes

### Problem

`providers` collapses devices to families (`provider_of`: "Vulkan0" →
"vulkan", deduped) and `select_device` takes the **first** device in a
family. A box exposing two GPUs under one family (iGPU + dGPU on Vulkan,
dual NVIDIA on CUDA) silently measures only the first. For a fleet
benchmark, unmeasured silicon is missing data.

### Design

The provider axis becomes a **device lane**: `<family>:<index>` —
`vulkan:0`, `vulkan:1`, `cuda:0`, `cpu:0`. The index is the device's
position within its family in ggml's registry order.

- **Backend** — `providers` emits one entry per compute device:
  `[{"id": "vulkan:0", "description": "AMD Radeon 780M"}, …]`.
  `--ep` accepts a lane id; `select_device` resolves (family, index) instead
  of first-match. `probe` runs per lane, not per family. Bare families are
  rejected (no silent index-0 fallback: ambiguity is the bug being fixed).
- **Schema** — events `schema_version` → `"3"`: `provider` is documented as a
  lane id; `device` (human label) is unchanged and remains the display name.
  `results.schema.json` bumps in step: `probes[]` and cells key on the lane.
  Both `provider` fields are already free-form strings, so the change is
  semantic; the version bump is what makes it loud.
- **Harness** — treats lane ids as opaque strings throughout (it already
  does); `plan` prints each lane with its device description so a
  two-GPU box shows two Vulkan rows before a multi-hour run.
- **Analysis loader** — accepts v2 traces by mapping a bare family to
  `<family>:0` (true for every published run: no multi-device box has been
  measured). Lanes are already silicon-named from `device`, so the report
  needs no display changes.

Same-silicon-different-family lanes (RTX 5080 as both `cuda:0` and
`vulkan:0`) are intentional and already the case today: they measure
different stacks on the same device.

### Notes

- Registry order is stable per machine per driver set — good enough, since a
  submission is one run. The lane's identity in *results* is
  (lane id + device description + stack versions), not the index alone.
- ggml at the pin drops software rasterizers (llvmpipe) from the registry;
  `bench check` should assert no lane's description looks like one, as a
  regression tripwire.

---

## B. The contributor artifact

One zip per platform: `bench-<tag>-linux-x64.tar.gz`,
`bench-<tag>-macos-arm64.tar.gz`, `bench-<tag>-windows-x64.zip`.

**Layout = pruned repo snapshot + prebuilt exe + bundled uv.** The build
products sit exactly where `backend.toml` already points
(`{dir}/build/bench-ggml`), so the harness needs **no layout changes**:

```
bench-<tag>-<platform>/
├── run.sh                     # or run.bat → run.ps1 on Windows
├── bin/uv                     # pinned single static binary, sha256-verified at package time
├── backends/ggml/
│   ├── backend.toml
│   └── build/
│       ├── bench-ggml[.exe]
│       ├── ggml-cpu-*.{so,dylib,dll}     # one per microarch (GGML_BACKEND_DL + CPU_ALL_VARIANTS)
│       └── ggml-vulkan.{so,dll}          # Metal compiled into the macOS exe (embedded shaders)
├── harness/                   # as-is, uv.lock + .python-version pinned
├── schema/  tasks/  models.yaml
├── licenses/                  # llama.cpp (MIT), uv, CLI11
└── README.txt                 # run this → wait → attach the output file to an issue
```

**`run.sh`** is non-interactive and chains existing commands:

1. `bin/uv run --project harness bench …` — uv fetches its managed CPython
   and the locked deps into a local cache; nothing touches the system.
2. `bench fetch` (prints disk cost first) → `bench check --backend ggml`
   (conformance-gates the shipped exe on *this* machine) → `bench plan`
   (echoed so the contributor sees their lanes) → the full run →
   `bench bundle` → `submission-<name>.tar.gz` next to `run.sh`
   (~300 KB: results + gzipped raw traces).
3. Prints the prefilled submission URL (workstream D) and stops.

`<name>` derives from machine fields via the existing "<CPU> (<GPU>)"
slugging — never free text.

---

## C. Build & release

### `packaging/package.sh <target>`

Assumes the toolchain is present (providing it is the workflow's job — the
script never installs anything). Does: configure → build → stage → verify →
zip + sha256. Platform awareness is one `case "$target"` block: cmake flags,
which module files to stage, which uv asset to fetch, tar vs zip.

Configure flags per target:

| target | flags |
|---|---|
| all | `Release`, `GGML_NATIVE=OFF` |
| linux-x64 | `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`, `GGML_VULKAN=ON`, shared libs + `$ORIGIN` rpath |
| windows-x64 | same variant/Vulkan set; cmake's default generator picks the installed VS (no vcvars, no pinned version) |
| macos-arm64 | **static** Metal build with `GGML_METAL_EMBED_LIBRARY=ON` — one arm64 microarch, no variant machinery needed; ad-hoc codesign as today |

**Module manifest check**: the staged `ggml-*` module set is compared
against an expected per-target manifest and the build **fails on mismatch**.
A missing dlopen'd module is a silent capability downgrade on some
contributor's box — the wrong-data failure mode, not a packaging nit.

### CI workflow (`release.yml`)

Matrix: `ubuntu-22.04` / `macos-15` / `windows-2025` — all GitHub-hosted, no
self-hosted hardware, no cross-compilation. Per-OS setup steps (Vulkan SDK
via LunarG apt / SDK installer) are adaptable from llama.cpp's own
`release.yml`, which builds this exact dependency set every release.
Building *on* ubuntu-22.04 sets the glibc floor at 2.35 (distros from
~2022). Trigger: pushed tag → build matrix → attach zips + sha256s to a
GitHub Release.

The release tag is the compatibility anchor: it pins the llama.cpp commit,
the schema versions, and the harness revision together. Submission
validation (D) checks a tarball's embedded stack versions against known
release tags.

Local reproduction of the Linux artifact (only needed when testing packaging
itself) is a documented `docker run --rm -v … ubuntu:22.04 …` one-liner in
the script header — dev builds on the workstation are unchanged.

### Known limitations (accepted)

- **macOS Gatekeeper**: ad-hoc signing ≠ notarization; browser-downloaded
  zips get quarantined. README ships the `xattr -d com.apple.quarantine`
  line. Proper fix is an Apple Developer ID ($99/yr) — deferred.
- **Intel Macs**: out of scope (hosted runners are arm64-only).
- **Windows arm64**: out of scope until someone asks.

---

## D. Submission pipeline (issue-ops)

Contributor: `run.sh` prints
`…/issues/new?template=submission.yml&title=submission%3A+<name>` → they
drag `submission-<name>.tar.gz` into the form → done.

**Issue form** (`.github/ISSUE_TEMPLATE/submission.yml`): the tarball drop
plus only what the tarball can't know — power state (plugged/battery),
anything unusual about the box.

**Workflow** — triggers on `issues: labeled`, on the `submission` label the
issue form applies at creation. The pipeline runs unattended; the PR review
is the one human gate, and merging is acceptance. Maintainer workload per
submission ≈ one click. Worst-case abuse of the unattended run is spam PRs
and CI minutes (the validator is built for hostile input); if spam ever
materialises, re-gate the workflow condition on a maintainer-applied label.

On `submission`:

1. Fetch the attachment (public URL on a public repo — plain `curl`).
2. **Hardened extraction** — the tarball is untrusted input: size cap,
   reject absolute paths and `..`, then copy out only the expected filenames
   rather than trusting archive structure.
3. **Validate** via `bench ingest <tarball>` (new harness command, also
   runnable locally on an emailed tarball): schema-validates results and
   traces, checks `schema_version` and stack pins against a known release
   tag, rejects control characters / absurd string lengths in free-text
   fields, confirms completions are present in the traces.
4. Fail → bot comments on the issue with the reason and stops.
   Pass → bot commits `results/published/<name>/` on a branch, regenerates
   `report.html` (analysis runs here, in CI), opens a PR titled
   `submission(<name>): …` with `Closes #N`.
5. Maintainer reviews the PR — including eyeballing `turn-end.completion`
   texts, which is what they're in the traces for — and merges. Pages
   republishes from `main` as usual.

Permissions: default `GITHUB_TOKEN` with `contents/pull-requests/issues:
write`. No secrets, no PATs. Trigger is `issues`, so workflow code always
comes from `main` — no fork-PR code-execution surface.

**Security invariants**:

- The **report is the attack surface**: submission strings (`completion`,
  `device`, versions) render into the published Pages site. Verify Jinja
  autoescaping covers every trace-derived string — including values passed
  into Vega specs — before the first external submission. One surviving
  `<script>` is stored XSS on the project domain.
- Directory and commit names derive from parsed machine fields through the
  existing slugger, never from contributor-typed text.
- **No auto-merge**, even on green: review is also editorial (throttled box?
  duplicate? battery?). Schema-valid numbers can still be bad data.

Fallback channel: email the tarball (it's ~300 KB); maintainer runs the same
`bench ingest`. One README line, no infrastructure.

Deferred: upload endpoint (infra + spam surface, unneeded at this volume);
auto-comment validation on *unlabeled* issues (runs stranger input without
human review — revisit only with a sandboxed validate-only job).

---

## Rollout

Each phase has an exit criterion; later phases depend on earlier ones except
where noted. Fleet re-runs are pending anyway post-v2-protocol, so phase 2
doubles as the fresh data collection.

**Phase 0 — device lanes (A).**
Backend + schemas + harness + loader. Exit: `bench check` passes; on the
workstation, `bench plan` lists `cpu:0`, `cuda:0`, `vulkan:0` distinctly; a
box with iGPU+dGPU under Vulkan lists two vulkan lanes.

**Phase 1 — packaging (B, C) + ingest (D's validator).**
`package.sh`, `run.sh`/`run.ps1`, `bench bundle`, `bench ingest`,
`release.yml`. Exit: a tagged pre-release produces three zips; sha256s
verify; `bench ingest` round-trips a locally-produced bundle.

**Phase 2 — dogfood on the fleet.**
Run the *packaged zips* (not dev builds) on every owned machine as if a
contributor: laptop, homeci box, workstation, the Mac. Exit: each produces a
valid submission bundle with correct lanes; the packaged-run numbers are
consistent with dev-build runs on the same box; re-run data committed.

**Phase 3 — submission automation (D).**
Issue form + workflow. Exit: a self-submitted issue (from the phase-2
laptop bundle) flows label → validate → PR → merge → Pages rebuild,
end-to-end. XSS escaping audit of templates + Vega done. A hostile-tarball
test (path traversal, oversized, garbage JSON) is rejected with a readable
bot comment.

**Phase 4 — open the doors.**
README quickstart (download → run → attach), submission-contents disclosure
(what's in the tarball, so people know what they publish), first real
release tag, invite the first outside contributors. Exit: one submission
from a machine we've never touched lands on Pages with two maintainer
clicks.

## Open questions

- **Multi-GPU submission naming**: `bench bundle` slugs CPU + first GPU; a
  two-GPU box probably wants both in the name. Decide when the first such box
  submits (phase 2 covers the workstation, whose lanes now include the
  Raphael iGPU — a first test case).
- **Notarization**: revisit if mac contributor friction proves real.

Resolved: Windows validation — the maintainer's Windows laptop plus friendly
testers cover phase 2; the macOS work machine checks the mac artifact.
