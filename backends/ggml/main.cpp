// bench-ggml — the llama.cpp / GGML backend for the on-device benchmark.
//
// Implements the CLI contract over the low-level llama.h API (not generate()):
//
//   bench-ggml version
//   bench-ggml providers --model <path.gguf>
//   bench-ggml run   --model <path.gguf> --quant <fp16|q8|q4|q2> --ep <lane>
//                    --task <task.json> --iters <K> --out <events.json|->
//   bench-ggml sweep --model <path.gguf> --quant <fp16|q8|q4|q2> --ep <lane>
//                    --out <events.json|->
//   bench-ggml probe --ep <lane> --out <events.json|->
//
// A provider is a device lane, "<family>:<index>" ("vulkan:0", "cpu:0") — one
// per compute device, so a machine with an iGPU and a dGPU under the same
// family measures both.
//
// `run` executes a chat task (brain-check or the validation job). `sweep`
// measures the cost function in one instrumented pass: a full-context prefill
// timed per ubatch-sized chunk (the chunk timings ARE the curve), then decode
// rates walking the primed cache downward — no chat semantics, nothing
// measured twice. `probe` measures bare device ceilings (GEMM, buffer copies)
// with no model loaded, on the same device-selection logic inference uses.
//
// stdout carries ONLY the JSON value for the subcommand; everything else → stderr.
// One process = one measurement unit; the harness spawns per cell.
#include "llama.h"
#include "llama-cpp.h" // RAII: llama_model_ptr / llama_context_ptr / llama_sampler_ptr
#include "llama-ext.h" // llama_get_memory_breakdown — exported, declared in src/
#include "common.h"
#include "chat.h" // common_chat_templates_* — derive the thinking-off block from the template
#include "build-info.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h" // ggml_backend_cpu_set_n_threads for the probe
#include "gguf.h"     // typed KV + tensor inventory for the geometry block

#include "nlohmann/json.hpp"
#include "CLI11.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::ordered_json;
using namespace std::chrono;

namespace {

// An expected failure with a human-readable reason (mapped to a nonzero exit in main).
struct BenchError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------- clocks
// Durations use a monotonic clock; one wall anchor lets the harness map them to wall time.
int64_t monotonic_ns() {
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}
int64_t wall_clock_ns() {
    return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
}

struct TimeSpan {
    int64_t start_ns;
    int64_t end_ns;

    double seconds() const { return static_cast<double>(end_ns - start_ns) / 1e9; }
};
template <class Body>
TimeSpan time_span(Body && body) {
    const int64_t start_ns = monotonic_ns();
    body();
    return {start_ns, monotonic_ns()};
}
json span_json(TimeSpan span) {
    return {{"start_ns", span.start_ns}, {"end_ns", span.end_ns}};
}
json load_event(std::string_view phase, TimeSpan span) {
    return {{"type", phase}, {"start_ns", span.start_ns}, {"end_ns", span.end_ns}};
}

// ---------------------------------------------------------------- adaptive repetition
// Repeat a measurement until its spread is characterized or it is expensive:
// a point that already took ≥20 s is precise on its own; otherwise repeat until
// (max−min)/median ≤ 5% (judged from 3 repeats) or the repeat cap.
constexpr int    kMinCheckedRepeats = 3;
constexpr int    kMaxRepeats        = 5;
constexpr double kSpreadTarget      = 0.05;
constexpr double kSingleShotSeconds = 20.0;

// Measure: () -> std::pair<double /*seconds*/, json /*repeat entry*/>
template <class Measure>
json adaptive_repeats(Measure && measure) {
    json                repeats = json::array();
    std::vector<double> secs;
    for (;;) {
        auto [s, entry] = measure();
        secs.push_back(s);
        repeats.push_back(std::move(entry));
        if (secs.front() >= kSingleShotSeconds) break;
        if (static_cast<int>(secs.size()) >= kMaxRepeats) break;
        if (static_cast<int>(secs.size()) >= kMinCheckedRepeats) {
            std::vector<double> sorted = secs;
            std::sort(sorted.begin(), sorted.end());
            const double median = sorted[sorted.size() / 2];
            if (median > 0 && (sorted.back() - sorted.front()) / median <= kSpreadTarget) break;
        }
    }
    return repeats;
}

// ---------------------------------------------------------------- small string helpers
std::string to_lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}
std::string trimmed(std::string_view text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return std::string{text.substr(first, last - first + 1)};
}
// "/path/Qwen3-0.6B-Q8_0.gguf" → "Qwen3-0.6B": drop dir + ".gguf" + the quant tag.
std::string model_name_from_path(const fs::path & artifact_path) {
    static const std::regex quant_suffix{"-(ud-)?(f16|bf16|q|iq)[0-9].*$", std::regex::icase};
    return std::regex_replace(artifact_path.stem().string(), quant_suffix, "");
}
// ---------------------------------------------------------------- devices / providers
// Family of a device: drop a trailing index, lowercase ("CUDA0"→"cuda", "CPU"→"cpu").
std::string family_of(ggml_backend_dev_t device) {
    std::string name = ggml_backend_dev_name(device);
    while (!name.empty() && std::isdigit(static_cast<unsigned char>(name.back()))) name.pop_back();
    return to_lower(name);
}
// A GGUF runs on any compiled compute device; skip pure accelerators / meta devices.
std::vector<ggml_backend_dev_t> compute_devices() {
    std::vector<ggml_backend_dev_t> devices;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        const auto         type   = ggml_backend_dev_type(device);
        if (type == GGML_BACKEND_DEVICE_TYPE_ACCEL || type == GGML_BACKEND_DEVICE_TYPE_META)
            continue;
        devices.push_back(device);
    }
    return devices;
}

// A provider is a device LANE — "<family>:<index>", the index being the device's
// position within its family in ggml's registry order ("vulkan:0", "vulkan:1").
// A machine with an iGPU and a dGPU under one family exposes two lanes; a bare
// family name is ambiguous there, so lanes are the only accepted --ep values.
struct Lane {
    std::string        id;
    ggml_backend_dev_t handle;
};
std::vector<Lane> device_lanes() {
    std::vector<Lane>          lanes;
    std::map<std::string, int> family_counts;
    for (ggml_backend_dev_t device : compute_devices()) {
        const std::string family = family_of(device);
        lanes.push_back({family + ":" + std::to_string(family_counts[family]++), device});
    }
    return lanes;
}
json available_providers() {
    json out = json::array();
    for (const Lane & lane : device_lanes())
        out.push_back(
            {{"id", lane.id}, {"description", ggml_backend_dev_description(lane.handle)}});
    return out;
}

struct Device {
    ggml_backend_dev_t handle = nullptr;
    bool               is_cpu = false;
    std::string        description; // human label, e.g. "NVIDIA GeForce RTX 5080"
};
Device select_device(std::string_view provider) {
    std::string available;
    for (const Lane & lane : device_lanes()) {
        if (lane.id == provider)
            return {lane.handle,
                    ggml_backend_dev_type(lane.handle) == GGML_BACKEND_DEVICE_TYPE_CPU,
                    ggml_backend_dev_description(lane.handle)};
        available += (available.empty() ? "" : ", ") + lane.id;
    }
    throw BenchError("--ep " + std::string{provider} +
                     " is not a device lane on this machine; lanes here: [" + available +
                     "] (a lane is <family>:<index> as listed by `providers`)");
}

// ---------------------------------------------------------------- versions
// Exact stack identity; embedded verbatim as the events `versions` object.
json versions_json() {
    return {
        {"backend", "ggml"},
        {"llama_cpp_commit", llama_commit()},
        {"llama_cpp_build", llama_build_number()},
        {"compiler", llama_compiler()},
        {"target", llama_build_target()},
        {"system_info", llama_print_system_info()},
        {"use_mmap", false}, // the shipped configuration (see Session::open)
        {"threads", common_cpu_get_num_physical_cores()},
    };
}

// ---------------------------------------------------------------- task model
struct Message {
    std::string              role;
    std::string              content;             // system/user text (inlined by the harness)
    int                      generate_tokens = 0; // assistant: tokens to generate
    std::vector<std::string> expect;              // assistant: plumbing check (substring match)

    bool is_assistant() const { return role == "assistant"; }
};
struct Task {
    std::string          name;
    int                  context_length = 512; // max_context_length → n_ctx
    std::vector<Message> messages;
};
Task load_task(const fs::path & task_path) {
    std::ifstream file{task_path};
    if (!file) throw BenchError("cannot open task file: " + task_path.string());
    json parsed;
    try {
        file >> parsed;
    } catch (const std::exception & error) {
        throw BenchError(std::string{"bad task json: "} + error.what());
    }
    Task task;
    task.name           = parsed.value("name", "task");
    task.context_length = parsed.value("max_context_length", 512);
    for (const auto & entry : parsed.at("messages")) {
        Message message;
        message.role            = entry.at("role");
        message.content         = entry.value("content", std::string{});
        message.generate_tokens = entry.value("nb_tokens", 0);
        if (entry.contains("expect"))
            message.expect = entry.at("expect").get<std::vector<std::string>>();
        task.messages.push_back(std::move(message));
    }
    return task;
}

// Vacuously true when the expect list is empty.
bool passes_expect(const std::string & completion, const std::vector<std::string> & expect) {
    if (expect.empty()) return true;
    const std::string haystack = to_lower(trimmed(completion));
    return std::any_of(expect.begin(), expect.end(), [&](const std::string & needle) {
        return haystack.find(to_lower(trimmed(needle))) != std::string::npos;
    });
}

// ---------------------------------------------------------------- chat templating
class Conversation {
  public:
    void add(std::string role, std::string content) {
        roles_.push_back(std::move(role));
        contents_.push_back(std::move(content));
    }
    std::vector<common_chat_msg> messages() const {
        std::vector<common_chat_msg> out;
        out.reserve(roles_.size());
        for (size_t i = 0; i < roles_.size(); ++i) {
            common_chat_msg m;
            m.role    = roles_[i];
            m.content = contents_[i];
            out.push_back(std::move(m));
        }
        return out;
    }

  private:
    std::vector<std::string> roles_;
    std::vector<std::string> contents_;
};

// ---------------------------------------------------------------- geometry
// What the runtime knows about the model, reported from the artifact and the
// live context — never hand-maintained. The GGUF is reopened without
// allocating tensor data (header-only), which costs nothing at inference time.

struct GgufGuard {
    gguf_context * ctx = nullptr;
    ~GgufGuard() {
        if (ctx) gguf_free(ctx);
    }
};

std::string meta_str(const llama_model * model, const char * key) {
    std::array<char, 256> buf{};
    if (llama_model_meta_val_str(model, key, buf.data(), buf.size()) < 0) return {};
    return buf.data();
}

// Per-layer attention typing from the arch's own metadata:
//   1. `<arch>.attention.sliding_window_pattern` (bool per layer; true → swa)
//   2. `<arch>.attention.recurrent_layers`       (bool per layer; true → recurrent)
//   3. `<arch>.full_attention_interval`          (layer i is full iff (i+1) % I == 0,
//      the rest recurrent — llama.cpp's own convention for hybrid archs)
//   4. none of the above → every layer is full attention.
json layers_json(gguf_context * gguf, const std::string & arch, int layer_count, int window) {
    json layers = json::array();
    auto emit   = [&](const char * kind, int w) {
        layers.push_back({{"kind", kind}, {"window", w}});
    };

    const int64_t pattern_key =
        gguf_find_key(gguf, (arch + ".attention.sliding_window_pattern").c_str());
    if (pattern_key >= 0 && gguf_get_arr_type(gguf, pattern_key) == GGUF_TYPE_BOOL) {
        const auto * swa = static_cast<const int8_t *>(gguf_get_arr_data(gguf, pattern_key));
        const int    n   = static_cast<int>(gguf_get_arr_n(gguf, pattern_key));
        for (int i = 0; i < layer_count; ++i)
            (i < n && swa[i]) ? emit("swa", window) : emit("full", 0);
        return layers;
    }

    const int64_t recurrent_key =
        gguf_find_key(gguf, (arch + ".attention.recurrent_layers").c_str());
    if (recurrent_key >= 0 && gguf_get_arr_type(gguf, recurrent_key) == GGUF_TYPE_BOOL) {
        const auto * rec = static_cast<const int8_t *>(gguf_get_arr_data(gguf, recurrent_key));
        const int    n   = static_cast<int>(gguf_get_arr_n(gguf, recurrent_key));
        for (int i = 0; i < layer_count; ++i)
            (i < n && rec[i]) ? emit("recurrent", 0) : emit("full", 0);
        return layers;
    }

    const int64_t interval_key = gguf_find_key(gguf, (arch + ".full_attention_interval").c_str());
    if (interval_key >= 0) {
        const int interval = static_cast<int>(gguf_get_val_u32(gguf, interval_key));
        for (int i = 0; i < layer_count; ++i)
            ((i + 1) % interval == 0) ? emit("full", 0) : emit("recurrent", 0);
        return layers;
    }

    for (int i = 0; i < layer_count; ++i) emit("full", 0);
    return layers;
}

// Tensor inventory split by role, bytes as landed at this quant. body is what a
// prefill token multiplies; embedding is a lookup; head runs once per decoded
// token — and lives under embedding when tied (streamed in full every step).
json tensors_json(gguf_context * gguf, bool & tied_head) {
    struct Group {
        uint64_t params = 0, bytes = 0;
    };
    Group embedding, body, head;
    bool  has_output_weight = false;

    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        const std::string   name  = gguf_get_tensor_name(gguf, i);
        const size_t        bytes = gguf_get_tensor_size(gguf, i);
        const enum ggml_type type = gguf_get_tensor_type(gguf, i);
        const uint64_t params = bytes / ggml_type_size(type) * ggml_blck_size(type);

        Group * group = &body;
        if (name.find("token_embd") != std::string::npos) group = &embedding;
        else if (name == "output.weight" || name == "output_norm.weight") group = &head;
        if (name == "output.weight") has_output_weight = true;

        group->params += params;
        group->bytes += bytes;
    }
    tied_head = !has_output_weight;
    auto group_json = [](const Group & g) {
        return json{{"params", g.params}, {"bytes", g.bytes}};
    };
    return {{"embedding", group_json(embedding)},
            {"body", group_json(body)},
            {"head", group_json(head)},
            {"tied_head", tied_head}};
}

json buffers_json(llama_context * context) {
    json buffers = json::array();
    for (const auto & [buffer_type, data] : llama_get_memory_breakdown(context)) {
        if (data.total() == 0) continue;
        buffers.push_back({{"name", ggml_backend_buft_name(buffer_type)},
                           {"model_bytes", data.model},
                           {"context_bytes", data.context},
                           {"compute_bytes", data.compute}});
    }
    return buffers;
}

json geometry_json(const llama_model * model, llama_context * context,
                   const std::string & model_path, int n_ctx, int n_batch, int n_ubatch) {
    GgufGuard gguf{gguf_init_from_file(model_path.c_str(), {/*no_alloc=*/true, nullptr})};
    if (!gguf.ctx) throw BenchError("cannot reopen gguf header: " + model_path);

    const std::string arch        = meta_str(model, "general.architecture");
    const int         layer_count = llama_model_n_layer(model);
    const int         window      = llama_model_n_swa(model);

    int           shared_kv  = 0;
    const int64_t shared_key = gguf_find_key(gguf.ctx, (arch + ".attention.shared_kv_layers").c_str());
    if (shared_key >= 0) shared_kv = static_cast<int>(gguf_get_val_u32(gguf.ctx, shared_key));

    bool tied_head = false;
    json tensors   = tensors_json(gguf.ctx, tied_head);

    return {
        {"n_layer", layer_count},
        {"n_embd", llama_model_n_embd(model)},
        {"n_head", llama_model_n_head(model)},
        {"n_head_kv", llama_model_n_head_kv(model)},
        {"n_swa", window},
        {"shared_kv_layers", shared_kv},
        {"n_ctx_train", llama_model_n_ctx_train(model)},
        {"n_params", llama_model_n_params(model)},
        {"file_bytes", fs::file_size(model_path)},
        {"context", {{"n_ctx", n_ctx}, {"n_batch", n_batch}, {"n_ubatch", n_ubatch}}},
        {"layers", layers_json(gguf.ctx, arch, layer_count, window)},
        {"tensors", std::move(tensors)},
        {"buffers", buffers_json(context)},
    };
}

// ---------------------------------------------------------------- synthetic tokens
// Plain-text token ids for sweeps: a fixed sentence tokenized once and tiled to
// length. Deterministic, no specials, varied enough to avoid degenerate paths.
std::vector<llama_token> synthetic_tokens(const llama_vocab * vocab, int count) {
    static const char * kText =
        "The quick brown fox jumps over the lazy dog while seventeen sailors watch from the "
        "harbor wall, counting waves against the stones as evening settles over the bay. ";
    const std::vector<llama_token> base =
        common_tokenize(vocab, kText, /*add_special=*/false, /*parse_special=*/false);
    if (base.empty()) throw BenchError("tokenizer produced no synthetic tokens");
    std::vector<llama_token> out;
    out.reserve(count);
    for (int i = 0; i < count; ++i) out.push_back(base[i % base.size()]);
    return out;
}

// ---------------------------------------------------------------- the loaded session
// Owns the llama resources (RAII) and runs the canonical loop.
class Session {
  public:
    // Load model + context, timing each phase into `load_phases`. The context's
    // n_batch covers the largest single prefill call; n_ubatch stays at the
    // deployment default (512) — micro-batch is NEVER tied to context size, so
    // measured rates and compute buffers describe a real operating point.
    static Session open(const std::string & model_path, const Device & device, int context_length,
                        int thread_count, json & load_phases) {
        llama_model_params model_params = llama_model_default_params();
        model_params.use_mmap           = false; // ship-as-is: weights read into allocations
        // Pin to exactly the selected device (NULL-terminated list). For the cpu EP this
        // is essential on a GPU-enabled build: without it, llama keeps a GPU in play and
        // offloads the KV cache there (offload_kqv), so "cpu" silently uses VRAM and isn't
        // a clean CPU measurement.
        std::array<ggml_backend_dev_t, 2> device_list{device.handle, nullptr};
        model_params.devices = device_list.data();
        if (device.is_cpu) {
            model_params.n_gpu_layers = 0;
        } else {
            model_params.n_gpu_layers = -1; // offload all layers
            model_params.main_gpu     = 0;
            model_params.split_mode   = LLAMA_SPLIT_MODE_NONE;
        }

        llama_model_ptr model;
        load_phases.push_back(load_event("model-load", time_span([&] {
                                             model.reset(llama_model_load_from_file(
                                                 model_path.c_str(), model_params));
                                         })));
        if (!model) throw BenchError("failed to load model: " + model_path);

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx                = context_length;
        context_params.n_batch              = context_length; // one llama_decode per prefill
        context_params.n_ubatch = std::min(512, context_length); // deployment default
        context_params.n_threads = context_params.n_threads_batch = thread_count;
        if (device.is_cpu) context_params.offload_kqv = false; // keep the KV cache off any GPU

        llama_context_ptr context;
        load_phases.push_back(load_event(
            "context-init",
            time_span([&] { context.reset(llama_init_from_model(model.get(), context_params)); })));
        if (!context) throw BenchError("failed to create context");

        return Session{std::move(model), std::move(context), context_params, model_path};
    }

    // warmup is exactly one token in, one token out — the minimal pass that pays
    // the one-time kernel/shader JIT so everything after runs warm. Timed into a
    // `load` span and dropped from aggregation by the harness.
    void warmup(json & load_phases) {
        load_phases.push_back(load_event(
            "warmup", time_span([&] {
                std::vector<llama_token> probe =
                    common_tokenize(vocab_, "A", /*add_special=*/false, false);
                llama_token prompt_token = probe.empty() ? 0 : probe.front();
                if (llama_decode(context_.get(), llama_batch_get_one(&prompt_token, 1)) != 0)
                    throw BenchError("warmup prefill failed");
                llama_token generated_token =
                    llama_sampler_sample(greedy_.get(), context_.get(), -1);
                if (llama_decode(context_.get(), llama_batch_get_one(&generated_token, 1)) != 0)
                    throw BenchError("warmup decode failed");
            })));
        clear_kv_cache();
    }

    json geometry() const {
        return geometry_json(model_.get(), context_.get(), model_path_,
                             static_cast<int>(context_params_.n_ctx),
                             static_cast<int>(context_params_.n_batch),
                             static_cast<int>(context_params_.n_ubatch));
    }

    // The allocator breakdown at another context size: free the current
    // context (headroom on tight devices — only one context ever lives at a
    // time), reopen at `context_length`, read the same per-buffer-type
    // breakdown. Called across a ladder of sizes this measures the memory
    // cost curve: weights constant, KV/state and compute workspace as actual
    // functions of context. Ends the session's measuring life — call last.
    json context_point(int context_length) {
        context_.reset();
        llama_context_params params = context_params_;
        params.n_ctx                = context_length;
        params.n_batch              = context_length;
        params.n_ubatch             = std::min(512, context_length);
        context_.reset(llama_init_from_model(model_.get(), params));
        if (!context_)
            throw BenchError("failed to reopen context at n_ctx " +
                             std::to_string(context_length));
        kv_cache_ = llama_get_memory(context_.get());
        return {{"n_ctx", context_length},
                {"n_batch", context_length},
                {"n_ubatch", std::min(512, context_length)},
                {"buffers", buffers_json(context_.get())}};
    }

    // --- the chat path (mode run) -------------------------------------------

    // One timed iteration: walk the messages in order. Each assistant turn
    // prefills the full re-rendered conversation from a cleared cache, decodes
    // its budget, and joins the conversation — later turns see real history.
    // Multi-turn without cross-turn KV reuse: every turn pays its whole
    // prompt, so each turn's events stand alone (context_before stays 0).
    json run_iteration(const Task & task, bool & healthy) {
        Conversation conversation;
        json         events = json::array();
        for (const Message & message : task.messages) {
            if (message.is_assistant()) {
                std::string completion;
                healthy = run_turn(conversation, message, events, completion) && healthy;
                conversation.add("assistant", std::move(completion));
            } else {
                conversation.add(message.role, message.content);
            }
        }
        return {{"events", std::move(events)}};
    }

    // --- the sweep path (mode sweep) ----------------------------------------

    // Append one synthetic chunk at the current cache depth (the caller tracks
    // depth; chunks are contiguous). Returns the chunk's prefill event.
    json append_chunk(int context_before, int token_count) {
        std::vector<llama_token> tokens = synthetic_tokens(vocab_, token_count);
        return prefill(tokens, context_before);
    }

    // Trim the cache down to `fill` tokens. Recurrent/hybrid caches refuse
    // partial truncation — false then; 0 always succeeds (clear).
    bool trim_cache(int fill) {
        if (fill == 0) {
            clear_kv_cache();
            return true;
        }
        return llama_memory_seq_rm(kv_cache_, 0, fill, -1);
    }

    // Decode `count` tokens at the current fill (caller ensures cache == fill);
    // returns the sweep repeat entry.
    json decode_point(int fill, int count) {
        int  context_size    = fill;
        auto [event, unused] = decode_tokens(count, context_size, /*capture_text=*/false);
        (void) unused;
        TimeSpan span{event["start_ns"], event["end_ns"]};
        return {{"token_ns", std::move(event["token_ns"])},
                {"start_ns", span.start_ns},
                {"end_ns", span.end_ns}};
    }

    int n_ctx() const { return static_cast<int>(context_params_.n_ctx); }

  private:
    Session(llama_model_ptr model, llama_context_ptr context, llama_context_params context_params,
            std::string model_path)
        : model_{std::move(model)}, context_{std::move(context)},
          greedy_{llama_sampler_init_greedy()}, vocab_{llama_model_get_vocab(model_.get())},
          kv_cache_{llama_get_memory(context_.get())},
          templates_{common_chat_templates_init(model_.get(), "")},
          model_path_{std::move(model_path)}, context_params_{context_params} {}

    void clear_kv_cache() { llama_memory_clear(kv_cache_, /*data=*/true); }

    // Render the conversation + generation prompt through the model's own template with thinking
    // DISABLED: `enable_thinking=false` makes the template emit its own thinking-off
    // prompt inline (an empty-think block, if it uses one); templates without the knob ignore it.
    std::string render_prompt(const Conversation & conversation) const {
        common_chat_templates_inputs in;
        in.messages              = conversation.messages();
        in.add_generation_prompt = true;
        in.enable_thinking       = false;
        return common_chat_templates_apply(templates_.get(), in).prompt;
    }

    // Ingest `tokens` in one llama_decode call (llama.cpp micro-batches at
    // n_ubatch internally) and emit the prefill event. The explicit sync keeps
    // the span honest on GPU backends, where llama_decode returns before the
    // device finishes — without it a chunk's cost lands in the next chunk's
    // timing.
    json prefill(std::vector<llama_token> & tokens, int context_before) {
        const TimeSpan span = time_span([&] {
            if (llama_decode(context_.get(), llama_batch_get_one(
                                                 tokens.data(),
                                                 static_cast<int32_t>(tokens.size()))) != 0)
                throw BenchError("prefill llama_decode failed (context too small?)");
            llama_synchronize(context_.get());
        });
        return {{"type", "prefill"},
                {"context_size", context_before},
                {"tokens_count", tokens.size()},
                {"start_ns", span.start_ns},
                {"end_ns", span.end_ns}};
    }

    // Prefill the rendered prompt, decode the token budget greedily, emit prefill/decode/turn-end.
    bool run_turn(const Conversation & conversation, const Message & message, json & events,
                  std::string & completion_out) {
        clear_kv_cache();
        // The template owns its special tokens (incl. any BOS it emits), so tokenize with
        // add_special=false and parse the specials already in the string.
        const std::string        prompt = render_prompt(conversation);
        std::vector<llama_token> tokens =
            common_tokenize(vocab_, prompt, /*add_special=*/false, /*parse_special=*/true);
        if (tokens.empty()) throw BenchError("empty prompt for the assistant turn");

        events.push_back(prefill(tokens, /*context_before=*/0));
        int context_size = static_cast<int>(tokens.size());

        auto [decode_event, completion] =
            decode_tokens(message.generate_tokens, context_size, /*capture_text=*/true);
        events.push_back(std::move(decode_event));

        const bool     expect_ok = passes_expect(completion, message.expect);
        const TimeSpan turn_end  = time_span([] {});
        events.push_back({{"type", "turn-end"},
                          {"completion", completion},
                          {"expect_pass", expect_ok},
                          {"start_ns", turn_end.start_ns},
                          {"end_ns", turn_end.end_ns}});
        completion_out = std::move(completion);
        return expect_ok;
    }

    // Greedy/argmax decode of exactly `count` tokens (EOS ignored); stamp each as it lands.
    std::pair<json, std::string> decode_tokens(int count, int & context_size, bool capture_text) {
        const int            context_at_start = context_size;
        std::vector<int64_t> token_times;
        token_times.reserve(count);
        std::string   completion;
        const int64_t start_ns = monotonic_ns();
        for (int i = 0; i < count; ++i) {
            llama_token token = llama_sampler_sample(greedy_.get(), context_.get(), -1);
            token_times.push_back(monotonic_ns());
            if (capture_text)
                completion += common_token_to_piece(context_.get(), token, /*special=*/false);
            if (llama_decode(context_.get(), llama_batch_get_one(&token, 1)) != 0)
                throw BenchError("decode llama_decode failed");
            ++context_size;
        }
        json event = {{"type", "decode"},      {"context_size", context_at_start},
                      {"tokens_count", count}, {"token_ns", token_times},
                      {"start_ns", start_ns},  {"end_ns", monotonic_ns()}};
        return {std::move(event), std::move(completion)};
    }

    llama_model_ptr           model_;
    llama_context_ptr         context_;
    llama_sampler_ptr         greedy_;
    const llama_vocab *       vocab_    = nullptr; // borrowed from model_
    llama_memory_t            kv_cache_ = nullptr; // borrowed from context_
    common_chat_templates_ptr templates_;          // the model's own chat template (jinja)
    std::string               model_path_;
    llama_context_params      context_params_; // the operating point context_ opened at
};

// ---------------------------------------------------------------- argument parsing
enum class Subcommand { Version, Providers, Run, Sweep, Probe };
struct Arguments {
    Subcommand  subcommand = Subcommand::Version;
    std::string model, quant, provider, task;
    std::string out         = "-";
    int         iters       = 1;
    int         deadline_ms = 0; // 0 = no soft cap
};
struct Cli {
    CLI::App   app{"bench-ggml — llama.cpp / GGML backend"};
    Arguments  args;
    CLI::App * version_cmd   = nullptr;
    CLI::App * providers_cmd = nullptr;
    CLI::App * run_cmd       = nullptr;
    CLI::App * sweep_cmd     = nullptr;
    CLI::App * probe_cmd     = nullptr;

    Cli() {
        app.require_subcommand(1);

        version_cmd = app.add_subcommand("version", "Print exact library/build versions as JSON");

        providers_cmd = app.add_subcommand(
            "providers",
            "List device lanes this artifact runs here (JSON array of {id, description})");
        providers_cmd->add_option("--model", args.model, "Resolved .gguf artifact path")
            ->required();

        run_cmd = app.add_subcommand("run", "Run one task on one provider; emit one events object");
        run_cmd->add_option("--model", args.model, "Resolved .gguf artifact path")->required();
        run_cmd->add_option("--quant", args.quant, "Quant label echoed into events (fp16|q8|q4|q2)")
            ->required();
        run_cmd->add_option("--ep", args.provider, "Device lane to run, as listed by `providers` (e.g. vulkan:0)")->required();
        run_cmd->add_option("--task", args.task, "Resolved task JSON path")->required();
        run_cmd->add_option("--iters", args.iters, "Timed iterations after one load+warmup")
            ->capture_default_str();
        run_cmd
            ->add_option(
                "--deadline-ms", args.deadline_ms,
                "Soft time-box: stop after the current iteration once elapsed ≥ this (0 = off)")
            ->capture_default_str();
        run_cmd->add_option("--out", args.out, "Events output path, or '-' for stdout")
            ->capture_default_str();

        sweep_cmd = app.add_subcommand(
            "sweep", "Measure prefill vs prompt length and decode vs KV fill (synthetic tokens)");
        sweep_cmd->add_option("--model", args.model, "Resolved .gguf artifact path")->required();
        sweep_cmd->add_option("--quant", args.quant, "Quant label echoed into events (fp16|q8|q4|q2)")
            ->required();
        sweep_cmd->add_option("--ep", args.provider, "Device lane to run, as listed by `providers` (e.g. vulkan:0)")->required();
        sweep_cmd
            ->add_option("--deadline-ms", args.deadline_ms,
                         "Soft budget: stop the chunk/fill ladders once elapsed ≥ this (0 = off); "
                         "the first chunk and first decode point always complete")
            ->capture_default_str();
        sweep_cmd->add_option("--out", args.out, "Events output path, or '-' for stdout")
            ->capture_default_str();

        probe_cmd = app.add_subcommand(
            "probe", "Measure bare device ceilings (GEMM, buffer copies); no model");
        probe_cmd
            ->add_option("--ep", args.provider,
                         "Device lane to probe, as listed by `providers` (e.g. vulkan:0)")
            ->required();
        probe_cmd->add_option("--out", args.out, "Events output path, or '-' for stdout")
            ->capture_default_str();
    }

    Subcommand which() const {
        if (providers_cmd->parsed()) return Subcommand::Providers;
        if (run_cmd->parsed()) return Subcommand::Run;
        if (sweep_cmd->parsed()) return Subcommand::Sweep;
        if (probe_cmd->parsed()) return Subcommand::Probe;
        return Subcommand::Version;
    }
};

// ---------------------------------------------------------------- output
void write_json(const std::string & destination, const json & value) {
    if (destination == "-") {
        std::cout << value.dump() << '\n';
    } else {
        std::ofstream file{destination};
        if (!file) throw BenchError("cannot write --out " + destination);
        file << value.dump() << '\n';
    }
}

json event_header(const char * mode, const Arguments & args, const Device & device,
                  const json & anchor) {
    json header = {{"schema_version", "3"},
                   {"backend", "ggml"},
                   {"mode", mode},
                   {"provider", args.provider},
                   {"device", device.description}};
    if (!args.model.empty()) {
        header["model"] = model_name_from_path(args.model);
        header["quant"] = args.quant;
    }
    header["versions"] = versions_json();
    header["anchor"]   = anchor;
    return header;
}

// ---------------------------------------------------------------- run subcommand
int cmd_run(const Arguments & args) {
    const Task   task         = load_task(args.task);
    const Device device       = select_device(args.provider);
    const int    thread_count = common_cpu_get_num_physical_cores();
    const json   anchor       = {{"wall_unix_ns", wall_clock_ns()}, {"mono_ns", monotonic_ns()}};

    json    load_phases = json::array();
    Session session =
        Session::open(args.model, device, task.context_length, thread_count, load_phases);
    session.warmup(load_phases);

    // Timed iterations (≤K). Iteration 1 always completes; later ones are skipped
    // once the soft deadline is hit — every emitted iteration is a
    // whole N-token decode, so the events shape is unchanged, just shorter.
    json          iterations     = json::array();
    bool          healthy        = true;
    const int64_t timed_start_ns = monotonic_ns();
    const int64_t deadline_ns    = static_cast<int64_t>(args.deadline_ms) * 1'000'000;
    for (int i = 0; i < args.iters; ++i) {
        if (i > 0 && deadline_ns > 0 && monotonic_ns() - timed_start_ns >= deadline_ns) {
            std::cerr << "ggml: deadline hit — ran " << i << "/" << args.iters << " iters\n";
            break;
        }
        iterations.push_back(session.run_iteration(task, healthy));
    }

    json out       = event_header("run", args, device, anchor);
    out["task"]    = task.name;
    out["healthy"] = healthy;
    out["load"]    = std::move(load_phases);
    out["geometry"]   = session.geometry();
    out["iterations"] = std::move(iterations);
    write_json(args.out, out);
    return healthy ? 0 : 2; // nonzero when an expect failed
}

// ---------------------------------------------------------------- sweep subcommand
// One instrumented pass, nothing measured twice. A prompt is ingested
// ubatch-by-ubatch anyway, so a single full-context prefill, timed per chunk,
// IS the prefill cost curve — a 4k prefill is literally the first half of an
// 8k one. The pass leaves the cache primed, and the decode ladder walks DOWN
// from the depth reached by trimming (free), so priming is never paid either.
// The soft budget stops the ladder between items: on slow silicon the
// measured envelope shrinks instead of the time growing; past it is
// extrapolation, and the data says so. 8k is the envelope cap.
constexpr int                kSweepChunk   = 512; // == n_ubatch, the deployment micro-batch
constexpr int                kSweepDepth   = 8192;
constexpr std::array<int, 2> kDecodeFills{2048, 0}; // walked below the reached depth
constexpr int                kDecodeTokens = 64;
constexpr int                kSweepContext = 8192 + 128; // top depth + decode headroom
// Context sizes for the allocator's memory cost curve (log ladder to the
// envelope cap; includes the job's 2048). Exact numbers, no repeats needed.
constexpr std::array<int, 5> kMemoryContexts{512, 1024, 2048, 4096, 8192};

int cmd_sweep(const Arguments & args) {
    const Device device       = select_device(args.provider);
    const int    thread_count = common_cpu_get_num_physical_cores();
    const json   anchor       = {{"wall_unix_ns", wall_clock_ns()}, {"mono_ns", monotonic_ns()}};

    json    load_phases = json::array();
    Session session = Session::open(args.model, device, kSweepContext, thread_count, load_phases);
    session.warmup(load_phases);

    const int64_t timed_start_ns = monotonic_ns();
    const int64_t budget_ns      = static_cast<int64_t>(args.deadline_ms) * 1'000'000;
    auto          past_budget    = [&](bool first) {
        return !first && budget_ns > 0 && monotonic_ns() - timed_start_ns >= budget_ns;
    };

    // The instrumented pass: chunks append until the envelope cap or the
    // budget; the first chunk always completes. Warmup left the cache empty.
    json prefill_chunks = json::array();
    int  depth          = 0;
    while (depth + kSweepChunk <= std::min(kSweepDepth, session.n_ctx())) {
        if (past_budget(prefill_chunks.empty())) {
            std::cerr << "ggml: sweep budget reached — envelope stops at " << depth
                      << " tokens\n";
            break;
        }
        prefill_chunks.push_back(session.append_chunk(depth, kSweepChunk));
        depth += kSweepChunk;
    }

    // Decode at the reached depth, then at the standard fills below it — the
    // cache only ever gets trimmed, never re-primed.
    json decode_points = json::array();
    auto measure_fill  = [&](int fill) {
        decode_points.push_back(
            {{"kv_fill", fill},
             {"tokens", kDecodeTokens},
             {"repeats", json::array({session.decode_point(fill, kDecodeTokens)})}});
    };
    if (depth + kDecodeTokens <= session.n_ctx()) measure_fill(depth);
    for (const int fill : kDecodeFills) {
        if (fill >= depth) continue; // beyond (or equal to) what the pass reached
        if (past_budget(decode_points.empty())) break;
        if (!session.trim_cache(fill)) { // hybrid/recurrent: only 0 is reachable
            std::cerr << "ggml: cache refuses partial trim — skipping fill " << fill << "\n";
            continue;
        }
        measure_fill(fill);
    }

    json geometry = session.geometry();
    json memory_points = json::array();
    for (const int ctx : kMemoryContexts) memory_points.push_back(session.context_point(ctx));
    geometry["memory_points"] = std::move(memory_points);

    json out              = event_header("sweep", args, device, anchor);
    out["load"]           = std::move(load_phases);
    out["geometry"]       = std::move(geometry);
    out["prefill_chunks"] = std::move(prefill_chunks);
    out["decode_points"]  = std::move(decode_points);
    write_json(args.out, out);
    return 0;
}

// ---------------------------------------------------------------- probe subcommand
// Bare device ceilings on the exact device inference selects. GEMM shapes mimic
// a prefill micro-batch (n = the 512-token ubatch against square weights);
// copies use one large buffer. Everything is warmed once before timing.

struct GgmlCtxGuard {
    ggml_context * ctx = nullptr;
    ~GgmlCtxGuard() {
        if (ctx) ggml_free(ctx);
    }
};
struct BackendBufferGuard {
    ggml_backend_buffer_t buffer = nullptr;
    ~BackendBufferGuard() {
        if (buffer) ggml_backend_buffer_free(buffer);
    }
};

constexpr std::array<std::array<int, 2>, 2> kGemmShapes{{{4096, 4096}, {8192, 8192}}}; // {m, k}
constexpr int                               kGemmBatch = 512;                          // n
constexpr size_t                            kCopyBytes = 256ull << 20;

json gemm_point(ggml_backend_t backend, ggml_backend_buffer_type_t buffer_type, int m, int n,
                int k) {
    GgmlCtxGuard guard{ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr,
                                  /*no_alloc=*/true})};
    ggml_context * ctx = guard.ctx;
    if (!ctx) throw BenchError("probe: ggml_init failed");

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);

    BackendBufferGuard buffer{ggml_backend_alloc_ctx_tensors_from_buft(ctx, buffer_type)};
    if (!buffer.buffer) throw BenchError("probe: buffer allocation failed for gemm");

    // Benign non-zero data: uninitialized device memory can hold NaNs/denormals
    // that throw off timing on some hardware.
    {
        std::vector<float> row(static_cast<size_t>(k), 0.0f);
        for (int i = 0; i < k; ++i) row[i] = 0.001f * static_cast<float>(i % 97);
        std::vector<ggml_fp16_t> half_row(static_cast<size_t>(k));
        ggml_fp32_to_fp16_row(row.data(), half_row.data(), k);
        const size_t a_row_bytes = ggml_row_size(GGML_TYPE_F16, k);
        for (int r = 0; r < m; ++r)
            ggml_backend_tensor_set(a, half_row.data(), static_cast<size_t>(r) * a_row_bytes,
                                    a_row_bytes);
        const size_t b_row_bytes = static_cast<size_t>(k) * sizeof(float);
        for (int r = 0; r < n; ++r)
            ggml_backend_tensor_set(b, row.data(), static_cast<size_t>(r) * b_row_bytes,
                                    b_row_bytes);
    }

    auto compute = [&] {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw BenchError("probe: gemm compute failed");
        ggml_backend_synchronize(backend);
    };
    compute(); // warm

    return {{"m", m},
            {"n", n},
            {"k", k},
            {"dtype", "f16"},
            {"repeats", adaptive_repeats([&] {
                 const TimeSpan span = time_span(compute);
                 return std::pair<double, json>{span.seconds(), span_json(span)};
             })}};
}

json copy_points(ggml_backend_t backend, ggml_backend_buffer_type_t buffer_type) {
    const size_t elements = kCopyBytes / sizeof(float);
    GgmlCtxGuard guard{ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr,
                                  /*no_alloc=*/true})};
    ggml_context * ctx = guard.ctx;
    if (!ctx) throw BenchError("probe: ggml_init failed");

    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, static_cast<int64_t>(elements));
    ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, static_cast<int64_t>(elements));
    ggml_tensor * copied = ggml_cpy(ctx, x, y); // device-side copy kernel
    ggml_cgraph * graph  = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, copied);

    BackendBufferGuard buffer{ggml_backend_alloc_ctx_tensors_from_buft(ctx, buffer_type)};
    if (!buffer.buffer) throw BenchError("probe: buffer allocation failed for copy");

    std::vector<float> host(elements);
    for (size_t i = 0; i < elements; ++i) host[i] = static_cast<float>(i % 251);

    auto h2d = [&] {
        ggml_backend_tensor_set(x, host.data(), 0, kCopyBytes);
        ggml_backend_synchronize(backend);
    };
    auto d2h = [&] {
        ggml_backend_tensor_get(x, host.data(), 0, kCopyBytes);
        ggml_backend_synchronize(backend);
    };
    auto d2d = [&] {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw BenchError("probe: copy compute failed");
        ggml_backend_synchronize(backend);
    };

    json points = json::array();
    auto add    = [&](const char * kind, auto && op) {
        op(); // warm
        points.push_back({{"kind", kind},
                          {"bytes", kCopyBytes},
                          {"repeats", adaptive_repeats([&] {
                               const TimeSpan span = time_span(op);
                               return std::pair<double, json>{span.seconds(), span_json(span)};
                           })}});
    };
    add("h2d", h2d);
    add("d2h", d2h);
    add("d2d", d2d);
    return points;
}

int cmd_probe(const Arguments & args) {
    const Device device       = select_device(args.provider);
    const int    thread_count = common_cpu_get_num_physical_cores();
    const json   anchor       = {{"wall_unix_ns", wall_clock_ns()}, {"mono_ns", monotonic_ns()}};

    ggml_backend_t backend = ggml_backend_dev_init(device.handle, nullptr);
    if (!backend) throw BenchError("probe: cannot init backend for --ep " + args.provider);
    struct BackendFree {
        ggml_backend_t b;
        ~BackendFree() { ggml_backend_free(b); }
    } backend_free{backend};
    if (device.is_cpu) ggml_backend_cpu_set_n_threads(backend, thread_count);
    ggml_backend_buffer_type_t buffer_type = ggml_backend_dev_buffer_type(device.handle);

    json gemm = json::array();
    for (const auto & [m, k] : kGemmShapes)
        gemm.push_back(gemm_point(backend, buffer_type, m, kGemmBatch, k));

    json out    = event_header("probe", args, device, anchor);
    out["gemm"] = std::move(gemm);
    out["copy"] = copy_points(backend, buffer_type);
    write_json(args.out, out);
    return 0;
}

// route llama/ggml logs to stderr so stdout stays JSON-only.
void log_to_stderr(ggml_log_level, const char * text, void *) { std::cerr << text; }

} // namespace

int main(int argc, char ** argv) {
    Cli cli;
    CLI11_PARSE(cli.app, argc, argv); // handles --help and usage errors with proper exit codes
    cli.args.subcommand    = cli.which();
    const Arguments & args = cli.args;

    try {
        llama_log_set(log_to_stderr, nullptr);
        ggml_log_set(log_to_stderr, nullptr);
        llama_backend_init(); // also runs ggml_backend_load_all() → populates device registry
        struct BackendGuard {
            ~BackendGuard() { llama_backend_free(); }
        } backend_guard;

        switch (args.subcommand) {
        case Subcommand::Version:
            std::cout << versions_json().dump() << '\n';
            return 0;
        case Subcommand::Providers: // a GGUF runs on any compiled device; --model isn't loaded
            std::cout << json(available_providers()).dump() << '\n';
            return 0;
        case Subcommand::Run:
            return cmd_run(args);
        case Subcommand::Sweep:
            return cmd_sweep(args);
        case Subcommand::Probe:
            return cmd_probe(args);
        }
    } catch (const std::exception & error) {
        std::cerr << "bench-ggml: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
