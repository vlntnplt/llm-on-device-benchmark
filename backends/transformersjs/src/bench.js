/**
 * `tjs` backend — Transformers.js over onnxruntime-node.
 *
 * Implements the CLI + events contract (see ARCHITECTURE.md) over a MANUAL
 * forward loop (not `generate()`), so prefill is isolated from the first
 * decode step and both `ttft_ms` and `prefill_tps` are measurable.
 * stdout carries the JSON object only; every log line → stderr.
 *
 *   bench-tjs providers --model <tjs-dir|onnx-file>
 *   bench-tjs run --model <…> --quant <fp16|q8|q4|q4f16> --ep <provider> \
 *                 --task <task.json> --iters <K> --out <events.json|->
 *   bench-tjs version
 */
import { AutoModelForCausalLM, AutoTokenizer, Tensor, env } from '@huggingface/transformers';
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync, statSync, existsSync } from 'node:fs';
import { cpus, platform, arch } from 'node:os';
import { execFileSync } from 'node:child_process';
import path from 'node:path';

const require = createRequire(import.meta.url);

// ── Offline only: never touch the network, always resolve from the local path.
env.allowRemoteModels = false;
env.allowLocalModels = true;

// ── Clock & anchor. Stamps are ns since process start (small enough for a JS
// double over hours); anchor.mono_ns = 0 lets the harness map them to wall time.
const T0 = process.hrtime.bigint();
const WALL0_NS = BigInt(Date.now()) * 1_000_000n;
const mono = () => Number(process.hrtime.bigint() - T0);
const ANCHOR = { wall_unix_ns: Number(WALL0_NS), mono_ns: 0 };

const log = (...a) => process.stderr.write(a.join(' ') + '\n');

function die(msg) {
  log(`✗ ${msg}`);
  process.exit(1);
}

// ── arg parsing ────────────────────────────────────────────────────────────
function parseArgs(argv) {
  const cmd = argv[0];
  const opts = {};
  for (let i = 1; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) {
      opts[a.slice(2)] = true; // valueless flag
    } else {
      opts[a.slice(2)] = next;
      i++;
    }
  }
  return { cmd, opts };
}

// ── physical core count (pin intra-op threads to physical cores) ─
function physicalCores() {
  try {
    if (platform() === 'linux') {
      const txt = readFileSync('/proc/cpuinfo', 'utf8');
      const pairs = new Set();
      let phys = '0';
      let core = null;
      for (const line of txt.split('\n')) {
        const [kRaw, vRaw] = line.split(':');
        if (!vRaw) {
          if (line.trim() === '' && core !== null) {
            pairs.add(`${phys}:${core}`);
            core = null;
          }
          continue;
        }
        const k = kRaw.trim();
        const v = vRaw.trim();
        if (k === 'physical id') phys = v;
        else if (k === 'core id') core = v;
      }
      if (core !== null) pairs.add(`${phys}:${core}`);
      if (pairs.size > 0) return pairs.size;
    } else if (platform() === 'darwin') {
      const n = parseInt(
        execFileSync('sysctl', ['-n', 'hw.physicalcpu'], { encoding: 'utf8' }).trim(),
        10,
      );
      if (n > 0) return n;
    }
  } catch {
    /* fall through */
  }
  // Fallback: assume SMT ×2 when logical is even, else use logical as-is.
  const logical = cpus().length || 1;
  return logical % 2 === 0 ? logical / 2 : logical;
}

// ── the per-platform device list onnxruntime-node supports, `cpu` first.
// A GPU EP that can't actually load is caught by the harness's brain-check gate.
function supportedProviders() {
  const out = ['cpu'];
  if (platform() === 'linux' && arch() === 'x64') out.push('cuda');
  else if (platform() === 'darwin') out.push('coreml');
  out.push('webgpu'); // experimental
  return out;
}

// ── human device label for the events object — the device the EP actually
// used, matching ggml's bare description ("NVIDIA GeForce RTX 5080"); never
// the CPU brand for a GPU run.
function cpuName() {
  return cpus()[0]?.model?.trim() || 'CPU';
}
function deviceLabel(ep) {
  // onnxruntime-node exposes no device-name API, so name the GPU out-of-band.
  if (ep === 'cuda') {
    // Local query, no download. EP defaults to device 0 → first nvidia-smi row.
    try {
      const out = execFileSync('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'], {
        encoding: 'utf8',
      });
      const name = out
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean)[0];
      if (name) return name;
    } catch {
      /* no nvidia-smi → fall through to the best-effort label below */
    }
    return `cuda (${cpuName()})`; // couldn't name the GPU; at least say it was a cuda run
  }
  // Apple's SoC is both CPU and GPU under unified memory, so the brand string is
  // the right label for coreml and webgpu too (mirrors ggml's Metal description).
  if (ep === 'coreml') return cpuName();
  if (ep === 'webgpu' && platform() === 'darwin') return cpuName();
  if (ep === 'cpu') return cpuName();
  return `${ep} (${cpuName()})`; // webgpu (non-Mac) / unknown: best-effort
}

// ── version identity ─────────────────────────────────────────────
function pkgVersion(name) {
  // Some packages (e.g. @huggingface/transformers) don't expose ./package.json
  // via their `exports` map, so resolve the entry point and walk up to it.
  try {
    return require(`${name}/package.json`).version;
  } catch {
    /* fall through */
  }
  try {
    let dir = path.dirname(require.resolve(name));
    for (let i = 0; i < 8; i++) {
      try {
        return JSON.parse(readFileSync(path.join(dir, 'package.json'), 'utf8')).version;
      } catch {
        const parent = path.dirname(dir);
        if (parent === dir) break;
        dir = parent;
      }
    }
  } catch {
    /* fall through */
  }
  return 'unknown';
}
function versionInfo() {
  return {
    threads: physicalCores(),
    transformers_js: pkgVersion('@huggingface/transformers'),
    onnxruntime_node: pkgVersion('onnxruntime-node'),
    node: process.versions.node,
    v8: process.versions.v8,
    platform: platform(),
    arch: arch(),
  };
}

// ── --model may be the fetched ONNX repo root OR a resolved onnx file inside
// its onnx/ subdir. Derive the dir from_pretrained wants by its marker files,
// not its *name* — the repo root itself is named `onnx` in the
// models/<name>/onnx/ layout, so a name test misfires.
function resolveModelDir(p) {
  const abs = path.resolve(p);
  let st;
  try {
    st = statSync(abs);
  } catch {
    die(`--model path does not exist: ${abs}`);
  }
  const isModelDir = (d) =>
    existsSync(path.join(d, 'tokenizer.json')) || existsSync(path.join(d, 'config.json'));
  let dir = st.isFile() ? path.dirname(abs) : abs;
  // climb out of an onnx/ weights subdir to the repo root that holds the markers
  if (!isModelDir(dir) && path.basename(dir) === 'onnx') dir = path.dirname(dir);
  return dir;
}

const QUANT_TO_DTYPE = { fp16: 'fp16', q8: 'q8', q4: 'q4', q4f16: 'q4f16' };

// Thinking is disabled across ALL backends (contract decision): every prompt
// renders with `enable_thinking: false` and the model's OWN template emits its
// thinking-off block — nothing hardcoded, so token ids match across stacks.
// `templateHasThinking` only logs that a reasoning model was detected.
const templateHasThinking = (tokenizer) =>
  typeof tokenizer.chat_template === 'string' &&
  tokenizer.chat_template.includes('enable_thinking');

// ── tensor helpers (int64, matching the lib's `ones`/cat dtype) ──────────────
function idTensor(ids) {
  return new Tensor(
    'int64',
    BigInt64Array.from(ids, (x) => BigInt(x)),
    [1, ids.length],
  );
}
function onesMask(len) {
  return new Tensor('int64', new BigInt64Array(len).fill(1n), [1, len]);
}

// greedy argmax over the LAST position's logits (handles fp16 via .to float32)
function argmaxLast(logits) {
  const V = logits.dims.at(-1);
  const S = logits.dims.at(-2);
  const data = logits.to('float32').data;
  const off = (S - 1) * V;
  let best = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < V; i++) {
    const v = data[off + i];
    if (v > bestVal) {
      bestVal = v;
      best = i;
    }
  }
  return best;
}

// Render the conversation through the model's own template, thinking disabled.
// `enable_thinking: false` makes the template emit its own thinking-off prompt
// (an empty-think block, if it uses one); templates without the toggle ignore it.
const renderRaw = (tokenizer, convo, addGen) =>
  tokenizer.apply_chat_template(convo, {
    add_generation_prompt: addGen,
    enable_thinking: false,
    tokenize: false,
  });
// apply_chat_template(tokenize:true) is just render → encode(rendered,
// {add_special_tokens:false}), so tokenizing this string the same way is faithful.
const renderPrompt = (tokenizer, convo) => renderRaw(tokenizer, convo, true);
const encodeNoSpecial = (tokenizer, str) => tokenizer.encode(str, { add_special_tokens: false });

/**
 * Prefill `newTokens` in a single batch, then decode exactly `nbTokens` greedy
 * tokens, one forward each (prefill isolated from the first decode
 * step). Returns the turn's events, the generated ids, and the decode end stamp.
 */
async function runTurn(model, newTokens, nbTokens) {
  let mi = {
    input_ids: idTensor(newTokens),
    attention_mask: onesMask(newTokens.length),
  };
  const allInputIds = [newTokens.map((x) => BigInt(x))]; // vestigial for the decoder path

  // ---- prefill (one batch over newTokens; its logits give the first token) ---
  mi = model.prepare_inputs_for_generation(allInputIds, mi, null);
  const prefillTokens = mi.input_ids.dims.at(1);
  const prefillStart = mono();
  let outputs = await model.forward(mi);
  const prefillEnd = mono();

  const tokenNs = [];
  const genIds = [];
  let id = argmaxLast(outputs.logits);
  tokenNs.push(mono()); // first token available
  genIds.push(id);
  allInputIds[0].push(BigInt(id));
  mi = model._update_model_kwargs_for_generation({
    generated_input_ids: [[BigInt(id)]],
    outputs,
    model_inputs: mi,
    is_encoder_decoder: false,
  });

  // ---- decode the remaining N-1 tokens (one forward each) -------------------
  for (let k = 1; k < nbTokens; k++) {
    mi = model.prepare_inputs_for_generation(allInputIds, mi, null);
    outputs = await model.forward(mi);
    id = argmaxLast(outputs.logits);
    tokenNs.push(mono());
    genIds.push(id);
    allInputIds[0].push(BigInt(id));
    mi = model._update_model_kwargs_for_generation({
      generated_input_ids: [[BigInt(id)]],
      outputs,
      model_inputs: mi,
      is_encoder_decoder: false,
    });
  }
  const decodeEnd = mono();

  const events = [
    {
      type: 'prefill',
      context_size: 0,
      tokens_count: prefillTokens,
      start_ns: prefillStart,
      end_ns: prefillEnd,
    },
    {
      type: 'decode',
      context_size: prefillTokens,
      tokens_count: nbTokens,
      token_ns: tokenNs,
      start_ns: prefillEnd,
      end_ns: decodeEnd,
    },
  ];
  return { events, genIds, decodeEnd };
}

// expect: completion must contain one of the strings (case-insensitive, trimmed)
function checkExpect(completion, expect) {
  if (!expect || expect.length === 0) return true;
  const hay = completion.toLowerCase().trim();
  return expect.some((s) => hay.includes(String(s).toLowerCase().trim()));
}

/**
 * Run the task once. Tasks are single-turn (one assistant message): render the whole
 * conversation through the model's own template (thinking disabled), prefill it, and
 * decode the token budget. The generated text is never re-tokenized.
 */
async function runIteration(model, tokenizer, task) {
  const convo = [];
  const events = [];
  let allExpectPass = true;

  for (const msg of task.messages) {
    if (msg.role !== 'assistant') {
      convo.push({ role: msg.role, content: msg.content });
      continue;
    }
    const newTokens = encodeNoSpecial(tokenizer, renderRaw(tokenizer, convo, true));
    const turn = await runTurn(model, newTokens, msg.nb_tokens);

    const completion = tokenizer.decode(turn.genIds, { skip_special_tokens: true });
    const expectPass = checkExpect(completion, msg.expect);
    allExpectPass = allExpectPass && expectPass;
    events.push(...turn.events);
    events.push({
      type: 'turn-end',
      completion,
      expect_pass: expectPass,
      start_ns: turn.decodeEnd,
      end_ns: mono(),
    });

    convo.push({ role: 'assistant', content: completion });
  }
  return { events, allExpectPass };
}

// ── warmup: 1 token in → 1 token out, once per process ─────────
async function warmup(model, tokenizer) {
  const ids = encodeNoSpecial(
    tokenizer,
    renderPrompt(tokenizer, [{ role: 'user', content: 'Hi' }]),
  );
  await runTurn(model, ids, 1);
}

// ── subcommands ──────────────────────────────────────────────────────────────
function cmdVersion() {
  process.stdout.write(JSON.stringify(versionInfo()) + '\n');
}

function cmdProviders(opts) {
  if (!opts.model) die('providers: --model is required');
  resolveModelDir(opts.model); // validate the path exists
  process.stdout.write(JSON.stringify(supportedProviders()) + '\n');
}

async function cmdRun(opts) {
  for (const req of ['model', 'quant', 'ep', 'task']) {
    if (!opts[req]) die(`run: --${req} is required`);
  }
  const dtype = QUANT_TO_DTYPE[opts.quant];
  if (!dtype) die(`run: unknown --quant ${opts.quant} (expected fp16|q8|q4|q4f16)`);
  const ep = opts.ep;
  if (!supportedProviders().includes(ep))
    die(`run: --ep ${ep} not in providers ${JSON.stringify(supportedProviders())}`);
  const iters = Math.max(1, parseInt(opts.iters ?? '1', 10));
  // Optional soft time-box: iteration 1 always completes; later
  // iterations are skipped once elapsed-since-the-first-timed-iteration ≥ deadline.
  const deadlineNs = opts['deadline-ms'] ? Math.max(0, parseInt(opts['deadline-ms'], 10)) * 1e6 : 0;
  const outPath = opts.out ?? '-';

  const modelDir = resolveModelDir(opts.model);
  const task = JSON.parse(readFileSync(opts.task, 'utf8'));
  const threads = physicalCores();

  log(
    `tjs: load ${modelDir} dtype=${dtype} ep=${ep} threads=${threads} iters=${iters} task=${task.name}`,
  );

  // ---- load (measured once per process) ------------------------
  const session_options = {
    intraOpNumThreads: threads,
    interOpNumThreads: 1,
    executionMode: 'sequential',
  };
  const modelLoadStart = mono();
  let model;
  try {
    model = await AutoModelForCausalLM.from_pretrained(modelDir, {
      dtype,
      device: ep,
      session_options,
      local_files_only: true,
    });
  } catch (e) {
    die(`model load failed on ep=${ep}: ${e?.message ?? e}`);
  }
  const modelLoadEnd = mono();

  // context-init: tokenizer (the inference session is created during load for tjs)
  const ctxStart = mono();
  const tokenizer = await AutoTokenizer.from_pretrained(modelDir, { local_files_only: true });
  const ctxEnd = mono();

  if (templateHasThinking(tokenizer))
    log('tjs: reasoning template detected → thinking disabled via enable_thinking=false');

  const warmStart = mono();
  await warmup(model, tokenizer);
  const warmEnd = mono();

  const load = [
    { type: 'model-load', start_ns: modelLoadStart, end_ns: modelLoadEnd },
    { type: 'context-init', start_ns: ctxStart, end_ns: ctxEnd },
    { type: 'warmup', start_ns: warmStart, end_ns: warmEnd },
  ];

  // ---- timed iterations (≤K; soft deadline) --------------------
  const iterations = [];
  let healthy = true;
  const timedStart = mono();
  for (let i = 0; i < iters; i++) {
    // Always run iteration 1; stop before any later one once the deadline is hit.
    if (i > 0 && deadlineNs && mono() - timedStart >= deadlineNs) {
      log(
        `tjs: deadline hit — ran ${i}/${iters} iters (${((mono() - timedStart) / 1e9).toFixed(1)}s)`,
      );
      break;
    }
    const { events, allExpectPass } = await runIteration(model, tokenizer, task);
    iterations.push({ events });
    healthy = healthy && allExpectPass;
  }

  const events = {
    schema_version: '1',
    backend: 'tjs',
    provider: ep,
    device: deviceLabel(ep),
    model: task.model ?? path.basename(path.dirname(modelDir)) ?? 'unknown',
    quant: opts.quant,
    task: task.name,
    versions: versionInfo(),
    anchor: ANCHOR,
    healthy,
    load,
    iterations,
  };

  const json = JSON.stringify(events);
  if (outPath === '-') process.stdout.write(json + '\n');
  else writeFileSync(outPath, json);

  // exit 0 iff every expect passed (the token budget holds by construction).
  if (!healthy) {
    log('✗ one or more `expect` checks failed → unhealthy run');
    process.exit(1);
  }
}

// ── entry ────────────────────────────────────────────────────────────────────
async function main() {
  const { cmd, opts } = parseArgs(process.argv.slice(2));
  switch (cmd) {
    case 'version':
      cmdVersion();
      break;
    case 'providers':
      cmdProviders(opts);
      break;
    case 'run':
      await cmdRun(opts);
      break;
    default:
      die(`unknown subcommand ${cmd ?? '(none)'} — expected providers | run | version`);
  }
}

main().catch((e) => die(`unhandled: ${e?.stack ?? e}`));
