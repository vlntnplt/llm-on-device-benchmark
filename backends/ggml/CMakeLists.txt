# bench-ggml — the llama.cpp / GGML backend exe.
#
# Dependencies are self-contained: both llama.cpp and CLI11 are fetched at
# configure time, pinned for reproducibility — no top-level third_party/ checkout
# is required. To build against an existing llama.cpp instead (e.g. offline, or a
# local working copy), point LLAMACPP_SOURCE_DIR at it:
#   cmake -B build -S . -DLLAMACPP_SOURCE_DIR=../../third_party/llama.cpp
#
# Produces ./build/bench-ggml — the path declared in backend.toml.
# Metal is auto-enabled on macOS; CUDA/Vulkan are opt-in:
#   cmake -B build -S .                          # CPU + Metal on macOS, CPU-only elsewhere
#   cmake -B build -S . -DGGML_METAL=OFF         # force CPU-only on macOS
#   cmake -B build -S . -DGGML_CUDA=ON
cmake_minimum_required(VERSION 3.18)  # DOWNLOAD_NO_EXTRACT (CLI11 single header)
project(bench-ggml CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Metal is always available on macOS (10.11+); default ON so a plain
# `cmake -B build -S .` gives both cpu and metal providers on Mac.
if(APPLE)
  option(GGML_METAL "Metal GPU support" ON)
endif()

include(FetchContent)

# --- llama.cpp ---------------------------------------------------------------
# Pinned to the commit this backend is developed against. Override with
# -DLLAMACPP_SOURCE_DIR=<path> to consume a local checkout instead of fetching.
set(LLAMACPP_GIT_TAG "e8023568d05ffdb9d0ac65695c521ea7e72c6b75" CACHE STRING
    "llama.cpp commit to fetch when LLAMACPP_SOURCE_DIR is unset")
set(LLAMACPP_SOURCE_DIR "" CACHE PATH
    "Path to an existing llama.cpp checkout; if empty, the pinned commit is fetched")

# We embed llama.cpp as a subproject. It defaults most of these to its
# LLAMA_STANDALONE value (OFF when embedded), so force the ones we need BEFORE it
# is added to the build:
#   - COMMON ON   : we use common_tokenize / build-info / vendored nlohmann json.
#   - the rest OFF: we don't want its tests/tools/examples/server, and we don't
#                   want the OpenSSL/cURL deps that the server/common HTTP bits pull.
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI       OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)

if(LLAMACPP_SOURCE_DIR)
  message(STATUS "bench-ggml: using local llama.cpp at ${LLAMACPP_SOURCE_DIR}")
  add_subdirectory(${LLAMACPP_SOURCE_DIR} ${CMAKE_BINARY_DIR}/llama.cpp)
  set(LLAMACPP_SRC_INTERNAL "${LLAMACPP_SOURCE_DIR}/src")
else()
  message(STATUS "bench-ggml: fetching llama.cpp @ ${LLAMACPP_GIT_TAG}")
  FetchContent_Declare(llamacpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        ${LLAMACPP_GIT_TAG})
  FetchContent_MakeAvailable(llamacpp)
  set(LLAMACPP_SRC_INTERNAL "${llamacpp_SOURCE_DIR}/src")
endif()

# --- CLI11 (single header) ---------------------------------------------------
# Pinned by content hash; DOWNLOAD_NO_EXTRACT leaves the bare header in the dir.
FetchContent_Declare(cli11
  URL      https://github.com/CLIUtils/CLI11/releases/download/v2.6.2/CLI11.hpp
  URL_HASH SHA256=227a16fe5f9f8ada80c3c409492475536f597e7bd83a6c26eacc3c8c149a9295
  DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(cli11)

# --- the executable ----------------------------------------------------------
add_executable(bench-ggml main.cpp)

# Linking llama-common transitively brings llama (→ ggml), its include dirs, the
# common helpers, build-info, and the vendored nlohmann/json.hpp. cli11_SOURCE_DIR
# holds the downloaded CLI11.hpp. llama.cpp's src/ is on the path for
# llama-ext.h (llama_get_memory_breakdown — LLAMA_API-exported but declared
# outside the public include dir; same consumption pattern as common/fit.cpp).
target_link_libraries(bench-ggml PRIVATE llama llama-common)
target_include_directories(bench-ggml PRIVATE ${cli11_SOURCE_DIR} ${LLAMACPP_SRC_INTERNAL})

# Ad-hoc sign with get-task-allow so the harness can call task_for_pid and read
# task_vm_info (full memory decomposition including Metal/IOKit allocations).
if(APPLE)
  add_custom_command(TARGET bench-ggml POST_BUILD
    COMMAND codesign --entitlements ${CMAKE_CURRENT_SOURCE_DIR}/entitlements.plist
                     -fs - $<TARGET_FILE:bench-ggml>
    COMMENT "ad-hoc signing bench-ggml with get-task-allow entitlement")
endif()
