// bench-ggml — the llama.cpp / GGML backend for the on-device benchmark.
//
// Implements the three-subcommand CLI contract over the low-level
// llama.h API (not generate()):
//
//   bench-ggml version
//   bench-ggml providers --model <path.gguf>
//   bench-ggml run --model <path.gguf> --quant <fp16|q8|q4|q4f16> --ep <provider>
//                  --task <task.json> --iters <K> --out <events.json|->
//
// stdout carries ONLY the JSON value for the subcommand; everything else → stderr.
// One process = one (model, provider, task); the harness spawns per cell.
#include "llama.h"
#include "llama-cpp.h" // RAII: llama_model_ptr / llama_context_ptr / llama_sampler_ptr
#include "common.h"
#include "chat.h" // common_chat_templates_* — derive the thinking-off block from the template
#include "build-info.h"
#include "ggml.h"
#include "ggml-backend.h"

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
};
template <class Body>
TimeSpan time_span(Body && body) {
    const int64_t start_ns = monotonic_ns();
    body();
    return {start_ns, monotonic_ns()};
}
json load_event(std::string_view phase, TimeSpan span) {
    return {{"type", phase}, {"start_ns", span.start_ns}, {"end_ns", span.end_ns}};
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
    static const std::regex quant_suffix{"-(f16|bf16|q|iq)[0-9].*$", std::regex::icase};
    return std::regex_replace(artifact_path.stem().string(), quant_suffix, "");
}
// ---------------------------------------------------------------- devices / providers
// Provider family of a device: drop a trailing index, lowercase ("CUDA0"→"cuda", "CPU"→"cpu").
std::string provider_of(ggml_backend_dev_t device) {
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
std::vector<std::string> available_providers() {
    std::vector<std::string> providers;
    for (ggml_backend_dev_t device : compute_devices()) {
        std::string provider = provider_of(device);
        if (std::find(providers.begin(), providers.end(), provider) == providers.end())
            providers.push_back(std::move(provider));
    }
    return providers;
}

struct Device {
    ggml_backend_dev_t handle = nullptr;
    bool               is_cpu = false;
    std::string        description; // human label, e.g. "NVIDIA GeForce RTX 5080"
};
Device select_device(std::string_view provider) {
    for (ggml_backend_dev_t device : compute_devices()) {
        if (provider_of(device) == provider)
            return {device, ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU,
                    ggml_backend_dev_description(device)};
    }
    throw BenchError("provider --ep " + std::string{provider} + " not available on this build");
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
        {"use_mmap", llama_supports_mmap()},
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
    int                  context_length = 512; // max_context_length → n_ctx / n_batch
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

// ---------------------------------------------------------------- the loaded session
// Owns the llama resources (RAII) and runs the canonical loop.
class Session {
  public:
    // Load model + context, timing each phase into `load_phases`.
    static Session open(const std::string & model_path, const Device & device, int context_length,
                        int thread_count, json & load_phases) {
        llama_model_params model_params = llama_model_default_params();
        model_params.use_mmap           = true; // ship-as-is: weights stay page-cache-backed
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
        // Single-batch prefill keeps prefill timing cleanly isolated.
        context_params.n_ctx = context_params.n_batch = context_params.n_ubatch = context_length;
        context_params.n_threads = context_params.n_threads_batch = thread_count;
        if (device.is_cpu) context_params.offload_kqv = false; // keep the KV cache off any GPU

        llama_context_ptr context;
        load_phases.push_back(load_event(
            "context-init",
            time_span([&] { context.reset(llama_init_from_model(model.get(), context_params)); })));
        if (!context) throw BenchError("failed to create context");

        return Session{std::move(model), std::move(context)};
    }

    // warmup is exactly one token in, one token out — the minimal pass that pays
    // the one-time kernel/shader JIT so iterations 1..K all run warm. Timed into a `load` span,
    // OUTSIDE the K loop, and dropped from aggregation by the harness.
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

    // One timed iteration: collect the system/user turns, then prefill+decode the assistant
    // turn. Tasks are single-turn (one assistant message), so there is no KV reuse
    // across turns to manage.
    json run_iteration(const Task & task, bool & healthy) {
        clear_kv_cache();
        Conversation conversation;
        json         events = json::array();
        for (const Message & message : task.messages) {
            if (message.is_assistant())
                healthy = run_turn(conversation, message, events) && healthy;
            else
                conversation.add(message.role, message.content);
        }
        return {{"events", std::move(events)}};
    }

  private:
    Session(llama_model_ptr model, llama_context_ptr context)
        : model_{std::move(model)}, context_{std::move(context)},
          greedy_{llama_sampler_init_greedy()}, vocab_{llama_model_get_vocab(model_.get())},
          kv_cache_{llama_get_memory(context_.get())},
          templates_{common_chat_templates_init(model_.get(), "")} {}

    void clear_kv_cache() { llama_memory_clear(kv_cache_, /*data=*/true); }

    // Render the conversation + generation prompt through the model's own template with thinking
    // DISABLED: `enable_thinking=false` makes the template emit its own thinking-off
    // prompt inline (an empty-think block, if it uses one); templates without the knob ignore it.
    // No hand-assembled role text or <think> blocks — the bytes match the other stacks.
    std::string render_prompt(const Conversation & conversation) const {
        common_chat_templates_inputs in;
        in.messages              = conversation.messages();
        in.add_generation_prompt = true;
        in.enable_thinking       = false;
        return common_chat_templates_apply(templates_.get(), in).prompt;
    }

    // Prefill the rendered prompt, decode the token budget greedily, emit prefill/decode/turn-end.
    bool run_turn(const Conversation & conversation, const Message & message, json & events) {
        // The template owns its special tokens (incl. any BOS it emits), so tokenize with
        // add_special=false and parse the specials already in the string.
        const std::string        prompt = render_prompt(conversation);
        std::vector<llama_token> tokens =
            common_tokenize(vocab_, prompt, /*add_special=*/false, /*parse_special=*/true);
        if (tokens.empty()) throw BenchError("empty prompt for the assistant turn");

        const TimeSpan prefill      = time_span([&] {
            if (llama_decode(context_.get(), llama_batch_get_one(tokens.data(), tokens.size())) !=
                0)
                throw BenchError("prefill llama_decode failed (context too small?)");
        });
        int            context_size = static_cast<int>(tokens.size());
        events.push_back({{"type", "prefill"},
                          {"context_size", 0},
                          {"tokens_count", tokens.size()},
                          {"start_ns", prefill.start_ns},
                          {"end_ns", prefill.end_ns}});

        auto [decode_event, completion] = decode_tokens(message.generate_tokens, context_size);
        events.push_back(std::move(decode_event));

        const bool     expect_ok = passes_expect(completion, message.expect);
        const TimeSpan turn_end  = time_span([] {});
        events.push_back({{"type", "turn-end"},
                          {"completion", completion},
                          {"expect_pass", expect_ok},
                          {"start_ns", turn_end.start_ns},
                          {"end_ns", turn_end.end_ns}});
        return expect_ok;
    }

    // Greedy/argmax decode of exactly `count` tokens (EOS ignored); stamp each as it lands.
    std::pair<json, std::string> decode_tokens(int count, int & context_size) {
        const int            context_at_start = context_size;
        std::vector<int64_t> token_times;
        token_times.reserve(count);
        std::string   completion;
        const int64_t start_ns = monotonic_ns();
        for (int i = 0; i < count; ++i) {
            llama_token token = llama_sampler_sample(greedy_.get(), context_.get(), -1);
            token_times.push_back(monotonic_ns());
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
};

// ---------------------------------------------------------------- argument parsing
enum class Subcommand { Version, Providers, Run };
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

    Cli() {
        app.require_subcommand(1);

        version_cmd = app.add_subcommand("version", "Print exact library/build versions as JSON");

        providers_cmd =
            app.add_subcommand("providers", "List providers this artifact runs here (JSON array)");
        providers_cmd->add_option("--model", args.model, "Resolved .gguf artifact path")
            ->required();

        run_cmd = app.add_subcommand("run", "Run one task on one provider; emit one events object");
        run_cmd->add_option("--model", args.model, "Resolved .gguf artifact path")->required();
        run_cmd
            ->add_option("--quant", args.quant, "Quant label echoed into events (fp16|q8|q4|q4f16)")
            ->required();
        run_cmd->add_option("--ep", args.provider, "Single provider/EP to run")->required();
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
    }

    Subcommand which() const {
        if (providers_cmd->parsed()) return Subcommand::Providers;
        if (run_cmd->parsed()) return Subcommand::Run;
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

    write_json(args.out, {
                             {"schema_version", "1"},
                             {"backend", "ggml"},
                             {"provider", args.provider},
                             {"device", device.description},
                             {"model", model_name_from_path(args.model)},
                             {"quant", args.quant},
                             {"task", task.name},
                             {"versions", versions_json()},
                             {"anchor", anchor},
                             {"healthy", healthy},
                             {"load", std::move(load_phases)},
                             {"iterations", std::move(iterations)},
                         });
    return healthy ? 0 : 2; // nonzero when an expect failed
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
        }
    } catch (const std::exception & error) {
        std::cerr << "bench-ggml: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
