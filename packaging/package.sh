#!/usr/bin/env bash
# package.sh <linux-x64|macos-arm64|windows-x64> [tag] — build + stage + zip one
# contributor artifact.
#
# The environment provides the toolchain (cmake, a compiler, the Vulkan SDK on
# linux/windows); this script never installs anything. CI invokes it per-OS
# (see .github/workflows/release.yml); the same invocation works on any
# matching machine. The Linux artifact should be built on the oldest supported
# base (ubuntu-22.04 → glibc 2.35); reproduce locally with:
#
#   docker run --rm -v "$PWD":/src -w /src ubuntu:22.04 bash -c \
#     'apt-get update && apt-get install -y build-essential cmake git curl \
#        libvulkan-dev glslc zip && packaging/package.sh linux-x64'
#
# Staging is a pruned snapshot of the *committed* tree (git archive HEAD) with
# the build products placed exactly where backend.toml already points
# ({dir}/build/bench-ggml) — the harness needs no packaged-layout special case.
set -euo pipefail

TARGET="${1:?usage: package.sh <linux-x64|macos-arm64|windows-x64> [tag]}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${2:-$(git -C "$REPO" describe --tags --always)}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

UV_VERSION="0.9.5" # bundled uv release; bump deliberately, the .sha256 asset verifies it

BUILD="$REPO/backends/ggml/build-dist/$TARGET"
STAGE_NAME="bench-$TAG-$TARGET"
DIST="$REPO/dist"
STAGE="$DIST/$STAGE_NAME"

say()  { printf '\n== %s\n' "$*"; }
fail() { printf '\n!! package.sh: %s\n' "$*" >&2; exit 1; }
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

# ---------------------------------------------------------------- configure + build
# One binary must be right on every contributor box: GGML_BACKEND_DL +
# CPU_ALL_VARIANTS build one dlopen'd CPU module per x86 microarch, picked at
# runtime; GGML_NATIVE=OFF keeps the host's ISA out of the shared code. Vulkan
# is the universal GPU lane on linux/windows; macOS is a static Metal build
# (one arm64 microarch — no variant machinery needed) with embedded shaders.
COMMON_FLAGS=(-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF)
case "$TARGET" in
linux-x64)
  FLAGS=("${COMMON_FLAGS[@]}" -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON
    -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=ON
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON "-DCMAKE_INSTALL_RPATH=\$ORIGIN")
  ;;
windows-x64)
  FLAGS=("${COMMON_FLAGS[@]}" -G "Visual Studio 17 2022" -DGGML_BACKEND_DL=ON
    -DGGML_CPU_ALL_VARIANTS=ON -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=ON)
  ;;
macos-arm64)
  FLAGS=("${COMMON_FLAGS[@]}" -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
    -DBUILD_SHARED_LIBS=OFF)
  ;;
*) fail "unknown target $TARGET" ;;
esac

say "configure + build ($TARGET, -j$JOBS)"
cmake -B "$BUILD" -S "$REPO/backends/ggml" "${FLAGS[@]}"
cmake --build "$BUILD" --config Release -j "$JOBS"

# ---------------------------------------------------------------- stage
say "stage → $STAGE"
rm -rf "$STAGE" && mkdir -p "$STAGE"
git -C "$REPO" archive HEAD harness schema tasks models.yaml backends/ggml/backend.toml \
  | tar -x -C "$STAGE"

EXE_DIR="$STAGE/backends/ggml/build"
mkdir -p "$EXE_DIR"
EXE="$(find "$BUILD" -type f \( -name bench-ggml -o -name bench-ggml.exe \) | head -1)"
[ -n "$EXE" ] || fail "no bench-ggml produced under $BUILD"
cp "$EXE" "$EXE_DIR/"
# Shared libs + dlopen'd backend modules, flat next to the exe: the exe's
# $ORIGIN rpath (linux) / same-dir DLL lookup (windows) / static build (macos)
# all resolve there, and ggml_backend_load_all searches the exe's directory.
find "$BUILD" -type f \( -name 'libggml*.so*' -o -name 'libllama*.so*' \
  -o -name 'ggml*.dll' -o -name 'llama*.dll' -o -name '*.dylib' \) \
  -exec cp -P {} "$EXE_DIR/" \;

case "$TARGET" in
linux-x64 | windows-x64)
  # The whole point of the variant build: a missing module is a silent
  # capability downgrade on someone's machine, so count them.
  CPU_MODULES=$(find "$EXE_DIR" \( -name '*ggml-cpu-*.so*' -o -name 'ggml-cpu-*.dll' \) | wc -l)
  [ "$CPU_MODULES" -ge 4 ] || fail "only $CPU_MODULES CPU variant modules staged (expected ≥4)"
  find "$EXE_DIR" \( -name '*ggml-vulkan*' \) | grep -q . || fail "vulkan module missing"
  ;;
esac

say "smoke-test the staged exe"
case "$TARGET" in
windows-x64) "$EXE_DIR/bench-ggml.exe" version >/dev/null ;;
*) "$EXE_DIR/bench-ggml" version >/dev/null ;;
esac

# ---------------------------------------------------------------- bundle uv
say "bundle uv $UV_VERSION"
mkdir -p "$STAGE/bin"
UV_BASE="https://github.com/astral-sh/uv/releases/download/$UV_VERSION"
fetch_verified() { # <asset> — download + verify against its published .sha256
  curl -fsSL -o "$DIST/$1" "$UV_BASE/$1"
  curl -fsSL -o "$DIST/$1.sha256" "$UV_BASE/$1.sha256"
  (cd "$DIST" && sha256 -c "$1.sha256" >/dev/null) || fail "uv checksum mismatch for $1"
}
case "$TARGET" in
linux-x64)
  fetch_verified uv-x86_64-unknown-linux-gnu.tar.gz
  tar -xzf "$DIST/uv-x86_64-unknown-linux-gnu.tar.gz" -C "$STAGE/bin" \
    --strip-components=1 uv-x86_64-unknown-linux-gnu/uv
  ;;
macos-arm64)
  fetch_verified uv-aarch64-apple-darwin.tar.gz
  tar -xzf "$DIST/uv-aarch64-apple-darwin.tar.gz" -C "$STAGE/bin" \
    --strip-components=1 uv-aarch64-apple-darwin/uv
  ;;
windows-x64)
  fetch_verified uv-x86_64-pc-windows-msvc.zip
  (cd "$STAGE/bin" && unzip -oq "$DIST/uv-x86_64-pc-windows-msvc.zip" uv.exe)
  ;;
esac

# ---------------------------------------------------------------- entry point + docs
cp "$REPO/packaging/contributor-readme.txt" "$STAGE/README.txt"
case "$TARGET" in
windows-x64) cp "$REPO/packaging/run.ps1" "$REPO/packaging/run.bat" "$STAGE/" ;;
*) cp "$REPO/packaging/run.sh" "$STAGE/" && chmod +x "$STAGE/run.sh" "$STAGE/bin/uv" ;;
esac

mkdir -p "$STAGE/licenses"
LLAMA_LICENSE="$(find "$BUILD/_deps" -maxdepth 2 -name LICENSE -path '*llamacpp*' | head -1)"
[ -n "$LLAMA_LICENSE" ] && cp "$LLAMA_LICENSE" "$STAGE/licenses/llama.cpp-LICENSE"
cat >"$STAGE/licenses/THIRD_PARTY.md" <<'EOF'
Bundled third-party components:
- llama.cpp (MIT) — https://github.com/ggml-org/llama.cpp (LICENSE alongside)
- CLI11 (BSD-3-Clause) — https://github.com/CLIUtils/CLI11 (license header embedded in the built exe's source)
- uv (MIT OR Apache-2.0) — https://github.com/astral-sh/uv
EOF

(cd "$STAGE" && find . -type f | sort | xargs -I{} sh -c 'echo {}') >"$STAGE/MANIFEST.txt"

# ---------------------------------------------------------------- archive
say "archive"
case "$TARGET" in
windows-x64)
  ARCHIVE="$DIST/$STAGE_NAME.zip"
  (cd "$DIST" && powershell.exe -NoProfile -Command \
    "Compress-Archive -Force -Path '$STAGE_NAME' -DestinationPath '$STAGE_NAME.zip'")
  ;;
*)
  ARCHIVE="$DIST/$STAGE_NAME.tar.gz"
  tar -czf "$ARCHIVE" -C "$DIST" "$STAGE_NAME"
  ;;
esac
(cd "$DIST" && sha256 "$(basename "$ARCHIVE")" >"$ARCHIVE.sha256")
say "done: $ARCHIVE"
cat "$ARCHIVE.sha256"
