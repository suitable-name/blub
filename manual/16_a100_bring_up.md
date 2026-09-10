# Chapter 16: A100 Bring-Up Sheet

This chapter is a checklist, not a design document. It exists because a large GPU work programme — column cache (F1b-1/2/2b/3), Agg1/Agg2 timeframe indicators (O09), the full-history GPU stage (Task I), the compact per-individual gather (Task H'), pinned memory + streams (Task J), and the throughput benchmark harness (`apps/alpha-bench`) — has **never executed on a GPU**. The development host has no CUDA/HIP device. Every line of it has been verified only by compilation, by CPU twins (`kernel_reference.rs`, `device_gather_reference.rs`), and by host-side tests. The first run on the production NVIDIA A100 80GB is the first time any of it actually executes.

Use this sheet to tell, in a few minutes, whether that first run is healthy — and if not, to map a symptom to a cause without re-deriving the reasoning from source. Every command, log string, and error message below is quoted verbatim from the live code with a `file:line` citation; anywhere that could not be verified against the code is called out explicitly rather than guessed at.

---

## 1. Pre-flight

Do these three checks before running anything else.

### 1.1 Build targets the right architecture

`crates/alpha-simulation/build.rs` bakes a specific `-gencode arch=compute_XX,code=sm_XX` into the compiled kernel binary. For an A100 (compute capability 8.0) the correct target is `sm_80`, and that is the default:

```rust
// build.rs:104-107
let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());
let compute_arch = cuda_arch.replacen("sm_", "compute_", 1);
build.flag(format!("-gencode=arch={compute_arch},code={cuda_arch}"));
build.flag(format!("-gencode=arch={compute_arch},code={compute_arch}"));
```

No action needed for a stock A100 build. To target different hardware, set `CUDA_ARCH=sm_XX` before building (e.g. `CUDA_ARCH=sm_90` for Hopper) — `build.rs:17` registers `cargo:rerun-if-env-changed=CUDA_ARCH` so changing it triggers a rebuild. If you skip the target flag entirely, `nvcc` falls back to its own old default virtual architecture and the kernel runs as PTX JIT-compiled from scratch at every process load — it still produces correct results, just with a load-time compile tax and no SM-8.0-specific SASS (`build.rs:64-70`).

**How to confirm it actually matches the hardware**: the GPU/CPU parity tool (§2.1) prints the compute capability it queried from the driver, right next to a reminder of what `build.rs` targeted (`crates/alpha-simulation/src/gpu.rs:777-784`). A mismatch there means step 2.1's device-identification block below.

### 1.2 The ABI guard

`extern "C"` declarations on the Rust side and the C++ definitions in `backtest_kernel.hip.cpp` are compiled **independently** — nothing at build time checks that a parameter count, order, or type actually agree. This project has already hit this once:

> "C++ once had 21 parameters while Rust passed 20, with zero warnings anywhere." — `crates/alpha-simulation/src/gpu.rs:485-486`

`LAUNCH_BACKTEST_KERNEL_ABI_VERSION` (`gpu.rs:521`, currently `5`) and the matching C++ function `launch_backtest_kernel_abi_version()` exist to turn a recurrence of that class of bug into an immediate, legible startup failure instead. `GpuEvaluator::new` checks them against each other as the very first thing it does, before any other GPU allocation or upload (`crates/alpha-simulation/src/gpu_evaluator.rs:475-499`). On mismatch:

```
launch_backtest_kernel ABI version mismatch: the compiled backtest_kernel.hip.cpp binary
reports abi_version={device_abi_version}, but this Rust build expects
LAUNCH_BACKTEST_KERNEL_ABI_VERSION={LAUNCH_BACKTEST_KERNEL_ABI_VERSION} -- the kernel binary
and this crate were built from mismatched sources for launch_backtest_kernel's signature (or
GpSimulationParamsGpu/OhlcGpu's layout). Rebuild both from the same commit before running
anything on GPU: a silent continuation here risks corrupting memory rather than merely
producing a wrong number.
```
(`gpu_evaluator.rs:487-497`, exact text)

**What it means**: a stale build artifact or a partial rebuild — the compiled `.o`/binary for `backtest_kernel.hip.cpp` and the Rust crate it's linked into came from different commits. **Remedy**: a full rebuild (`cargo clean -p alpha-simulation` or equivalent) from one checkout, so both sides compile from the same source at the same time.

### 1.3 Config keys that gate GPU use

| Key | Default | Source |
|---|---|---|
| `gpu.enabled` | `true` | `crates/alpha-config/src/gpu.rs:76-78` |
| `gpu.device_id` | `0` | `crates/alpha-config/src/gpu.rs:79-81` — **verified unused**: `query_gpu_device_info` always queries device 0 (`gpu.rs:777`, "Queries device 0's name..."), and nothing in this workspace reads `config.gpu.device_id` anywhere. Harmless on the single-A100 production target, but do not expect setting it to select a different device. |
| `gpu.max_snapshot_candles` | `1_500_000` | `crates/alpha-config/src/gpu.rs:82-86`, `26-28` |
| `gpu.column_cache_budget_mb` | `49_152` MiB = 48 GiB | `crates/alpha-config/src/gpu.rs:87-119`, `30-71` |
| `GPU_EVALUATOR_CANDLE_CAPACITY` | `1_500_000` (a Rust constant, not a config key) | `crates/alpha-simulation/src/gpu_evaluator.rs:386` |

None of the shipped configs (`config.toml`, `config.harvest.toml`, `config.bolt.toml`, `config.pgo.toml`) contain a `[gpu]` section at all (verified: no match for `[gpu]` in any of them), so **every shipped config runs on these defaults unmodified**. These are not currently documented in `manual/10_configuration_reference.md`.

**Combinations that silently produce a CPU-only run**:
- `gpu.enabled = false` — every stage (tier-1, tier-2, full-history) falls back to CPU. No crash, no distinguishing log beyond the absence of the GPU init line (§2.3).
- `gpu.column_cache_budget_mb == 0` — disables the GPU column cache entirely. Tier-1/tier-2 fall back to the old per-`IndicatorPeriods`-tuple `build_snapshot_matrix` replay (still GPU, just without the cache); the full-history stage falls back to CPU **entirely**, because that per-tuple replay is infeasible at ~1,000,000 candles for anything but a single shared periods tuple (`alpha-config/src/gpu.rs:109-117`; `run_full_history_filter_gpu`'s own gate, `crates/alpha-simulation/src/engine/tiering.rs:1013-1015`).
- A dataset window whose real per-ticker candle count exceeds `min(gpu.max_snapshot_candles, GpuEvaluator's actual max_candles)` — that one stage silently reverts to CPU for that call. See §2.3 for the log line that makes this visible for full-history (tier-1/tier-2 have no equivalent log — see the note in §2.3).

---

## 2. Step-by-step bring-up

Run these in order, on the A100 box.

### 2.1 GPU/CPU backtest-kernel parity self-check

This is covered in full by `manual/13_gpu_cpu_parity_self_check.md` — do not skip it, and do not duplicate its work by reading this section alone. In short:

```bash
cargo build --release --features gpu -p alpha-worker
./target/release/alpha-worker --gpu-parity-check
```

(flag defined at `apps/alpha-worker/src/args.rs:109-119`, short-circuits `main` before any master/QuestDB/network setup at `apps/alpha-worker/src/main.rs:44-47`). Also run `cargo test --features gpu -p alpha-simulation` alongside it — see `manual/13` §2 for why both are needed. A passing run ends in `RESULT: PASS` with exit code 0, and prints the device name/compute-capability/memory block described in §2.2 first. See `manual/13` §3 for the full healthy-output example and §4 for its own symptom table — several rows of §3 below point back to it rather than repeating it.

### 2.2 Device identification

`query_gpu_device_info()` (`crates/alpha-simulation/src/gpu.rs:792-824`) queries device 0's name, compute capability, and live free/total memory via `hipGetDeviceProperties`/`hipMemGetInfo` (or the CUDA equivalents), returning a `GpuDeviceInfo { name, compute_capability: (i32, i32), total_bytes, free_bytes }`. Its own doc comment states plainly why this is the first thing to check:

> "This is the first thing the GPU/CPU parity self-check tool prints: `build.rs` bakes a specific `-gencode arch=compute_XX,code=sm_XX` into this binary (`sm_80` for the production A100 by default), and the cheapest way to confirm that target actually matches the hardware a given run executes on is to ask the driver directly, before trusting any number the kernel itself produces." — `gpu.rs:779-784`

`--gpu-parity-check` prints this block automatically (§2.1). `apps/alpha-bench` also calls it in `try_init` before constructing a `GpuEvaluator` (`apps/alpha-bench/src/gpu_support.rs:70-78`), formatting it as `"{name} (compute {major}.{minor}, {total_gib:.1} GiB)"` for its report header. Confirm the printed compute capability reads `8.0` — if it reads anything else, `build.rs`'s default `sm_80` target does not match this card; see §3's compute-capability row.

### 2.3 A short real GA/worker run — what proves the GPU path was taken

Start `alpha-worker --features gpu` (or run a short GA generation via `ga-runner` against a worker) and watch the logs for these, in the order they'd appear:

**Startup — GPU evaluator construction** (`apps/alpha-worker/src/main.rs:126-141`):
- Healthy: `"[GPU] CUDA/HIP device detected and GpuEvaluator initialized (capacity: {} individuals)."` (`main.rs:132-135`)
- Failed: `"[GPU] Failed to initialize GPU evaluator: {e}. Falling back to CPU."` (`main.rs:139`) — the worker keeps running, CPU-only, for its entire lifetime. `{e}` is the actual `GpuEvaluator::new` error (§1.2's ABI message, or a `DeviceBuffer::alloc` failure — see §3).

**Full-history stage, per generation** (`crates/alpha-simulation/src/engine/tiering.rs`):
- Healthy: `tracing::info!` `"GPU full-history stage: column-cache groups planned"` (`tiering.rs:507-513`), with `distinct_tuples`/`groups`/`survivors`/`budget_mb` fields — this is the strongest positive proof the GPU actually took the full-history stage.
- Entirely skipped: `tracing::warn!` `"GPU full-history stage skipped: ticker length exceeds the evaluator's candle capacity..."` (`tiering.rs:465-473` — full text in §3).
- Partially skipped: `tracing::warn!` `"GPU full-history stage: some individuals fall back to CPU..."` (`tiering.rs:515-524` — full text in §3).

**Tier-1/tier-2 — no equivalent log line exists.** The `use_gpu_tier1`/`use_gpu_tier2` decisions (`tiering.rs:826-845`, `926-938`) are silent booleans with no `tracing` call on either branch — unlike full-history, there is currently no log line that proves or disproves whether a given generation's tier-1/tier-2 filtering actually ran on GPU. The only indirect signal is the startup line above (if the evaluator failed to construct, tier-1/2 never attempt GPU for the rest of the process) combined with wall-clock time or the benchmark harness (§2.4). This is a real gap, not a documentation omission — flagged here rather than worked around, per this task's brief.

### 2.4 Throughput benchmark harness

Covered in full by `manual/15_throughput_benchmark_harness.md` — do not duplicate it here. For the A100:

```bash
cargo run --release --features gpu -p alpha-bench -- \
  --config config.toml \
  --mode both \
  --output-dir benchmark_results
```

This is also the first tool that will produce a *real, measured* CPU-vs-GPU throughput number for this system (see §4). `manual/15` §6 documents its GPU-skip behavior (feature not compiled, config disables GPU, no device, or a specific row's window too large) — every case names the reason in the report rather than fabricating a number.

### 2.5 Population-level GPU vs CPU tiering parity (F27/2.2.2)

```bash
cargo test -p alpha-simulation --features gpu \
  gpu_cpu_tiering_survivor_sets_match_on_seeded_population -- --nocapture
```

(`crates/alpha-simulation/tests/gpu_cpu_parity_tests.rs`, subtask `d50d`.) Everything through §2.4 above exercises the kernel directly, at the granularity of a single hand-picked strategy or a small FFI-level fixture; this gate is the first one that runs the REAL GPU tiering path (`run_tier_filter_gpu` / `launch_backtest_kernel`, not a bespoke test harness) end-to-end at population scale. It generates 200 individuals via `alpha_ga::MultiObjectiveGA::create_random_individual_stateless`, each seeded from `seeded_rng(42, &[tags::TESTING, item_idx])` so the population is fully reproducible, and runs `engine::evaluate_batch_tiered` once with a real `GpuEvaluator` (`config.gpu.enabled = true`) and once CPU-only (`config.gpu.enabled = false`) over the identical population and `common::two_ticker_unequal_dataset()`, asserting the two runs' survivor index sets (`rejected_tier.is_none()`) are equal.

**The 1% tolerance rule**: `manual/14` §3.10 documents that a strategy whose tier-gating score lands within noise of the tier-1/tier-2 threshold can flip sides purely from the kernel's `f32` vs `run_simulation`'s `f64` arithmetic — a documented structural boundary case, not a bug. Rather than assert bit-identical survivor sets (which would make this gate permanently, spuriously red on exactly the boundary case §3.10 describes), the test allows **up to 1% of the population** to differ, and prints every differing individual's index so a real regression (more than the documented noise) is still caught and immediately diagnosable.

**Healthy**: test passes, with at most 1% of the 200 individuals printed as differing (0-2 individuals). **Unhealthy**: the test fails outright (more than 1% differ) — treat this as a real GPU/CPU tiering divergence, not a boundary-noise artifact, and investigate via `manual/14` before assuming it is safe to raise the tolerance.

---

## 3. Symptom → cause → check table

| Symptom | Meaning | Check / remedy |
|---|---|---|
| ABI version mismatch at startup (§1.2's exact text) | Stale build artifact / partial rebuild — Rust and the compiled `backtest_kernel.hip.cpp` disagree on `launch_backtest_kernel`'s signature or a `#[repr(C)]` struct's layout. | Full rebuild from one checkout. See `LAUNCH_BACKTEST_KERNEL_ABI_VERSION`'s own bump history for what changes require it (`gpu.rs:505-519`). |
| `GpuEvaluator::new` fails (not the ABI message) | No device, driver mismatch, or insufficient GPU memory. Surfaces as a `DeviceBuffer::alloc` failure: `"GPU allocation failed: requested {count} element(s) ({size} bytes)"` (`gpu_evaluator.rs:69-71`), or as `query_gpu_device_info` failing first if no device answers at all. | Check `nvidia-smi` for other processes holding VRAM. See `manual/13` §4's own row for this exact case. |
| `"GPU full-history stage skipped: ticker length exceeds the evaluator's candle capacity; running every survivor on CPU. Raise \`gpu.max_snapshot_candles\` AND the capacity \`GpuEvaluator::new\` is constructed with (see GPU_EVALUATOR_CANDLE_CAPACITY) to enable it."` | The dataset's real per-ticker candle count exceeds the *smaller* of `gpu.max_snapshot_candles` and the evaluator's actual allocated capacity (`GPU_EVALUATOR_CANDLE_CAPACITY = 1_500_000`, `gpu_evaluator.rs:386`). The entire full-history stage runs on CPU that generation. | Raise **both**: `gpu.max_snapshot_candles` in config, and the capacity constant `GpuEvaluator::new` is built with (`GPU_EVALUATOR_CANDLE_CAPACITY`, wired at `apps/alpha-worker/src/main.rs:126-129`) — raising only one has no effect, since the smaller of the two always wins (`alpha-config/src/gpu.rs:7-25`). |
| `"GPU full-history stage: some individuals fall back to CPU -- their \`IndicatorPeriods\` tuple's own column footprint alone exceeds the column-cache budget, so it can never be made resident by splitting further. Raise \`gpu.column_cache_budget_mb\` or narrow the widest period search-space bounds to bring these onto the GPU."` | A subset of survivors' own period combination needs more column-cache bytes than the *entire* configured budget — `plan_column_cache_groups` cannot fit them even alone, so they're routed to CPU rather than risking a `NO_COLUMN` eviction (`tiering.rs:371-378`). | Raise `gpu.column_cache_budget_mb`, or narrow the GA's period search-space bounds. |
| `NO_COLUMN` residency hard-fail: `"GPU column cache: at least one requested (ticker, family, period) column was not resident when the pool was built..."` | A column an individual actually needs was evicted under budget, or was never requested when the cache was built. This is a **hard error, not a fallback**, deliberately: silently reverting to the base snapshot's default-period value would produce a *wrong* number for a period-dependent slot, not graceful degradation (`gpu_evaluator.rs:1040-1057`). | Increase `gpu.column_cache_budget_mb`, or reduce batch size (fewer individuals/tickers per generation). If this fires from `run_full_history_filter_gpu` despite its own pre-planning, that means the planning estimate under-counted the real cost — worth a closer look, not just a budget bump (`tiering.rs:399-408`). |
| `validate_periods_for_device_gather` rejects a config | One of three combine-step period scalars is at or past `WARMUP_PERIOD` (2000 candles), which the on-device gather assumes is already closed by the first bar it visits: `stochastic_k_period + stochastic_d_period - 2 >= WARMUP_PERIOD`, `vwap_period > WARMUP_PERIOD + 1`, or `lcp_period > WARMUP_PERIOD + 1` (`device_gather_reference.rs:630-687`, called from `gpu_evaluator.rs:778` before any device upload). Each error names the offending field(s) and value(s). | Lower the relevant period's search-space upper bound (`IndicatorPeriodsSearchSpaceConfig`, `alpha-config/src/ga.rs`) below the stated threshold. No shipped config's search space currently gets close (`config.harvest.toml`'s widest tops out at 63, per the module's own doc comment) — this should only fire from a deliberately widened or hand-built config. |
| A kernel launch failure | `gpu_status_to_result(launch_status, "launch_backtest_kernel")` (`gpu_evaluator.rs:1128`) turns a nonzero status into `"{context} failed with GPU status {status}: {msg}"` (`gpu.rs:878-892`), where `{msg}` comes directly from the driver's own `gpu_get_error_string`, i.e. `hipGetErrorString`/`cudaGetErrorString` (`backtest_kernel.hip.cpp:2607-2608`) — not a generic message. | Read `{msg}` directly; per `manual/13` §4, a common one is an out-of-bounds access, which would indicate a real kernel bug rather than an environment problem. |
| Results that are wrong, not absent (no error at all) — two known-fixed classes | **(1)** The historical 21-vs-20 FFI parameter mismatch (§1.2) — silently compiled on both sides, corrupted memory at runtime, discovered only by manual audit. Now guarded by `LAUNCH_BACKTEST_KERNEL_ABI_VERSION`. **(2)** Task H's discovery that production never called `compact_individual_for_device` before upload: every individual carried `referenced_slot_count == -1`, the kernel's defensive clamp set `ref_count = 0`, and the per-bar `snapshot_row` — a local stack array, never zero-initialized — was read via `get_terminal_value` with *no writes at all* first: uninitialized stack memory for a raw terminal id `< 64`, a defined `0.0` for one `>= 64`. Silently wrong, invisible on a host with no GPU (`manual/14` §10.2, full detail). Fixed by adding a compaction pass in `prepare_and_launch`, immediately after `compile_batch` and before upload — the call site is `gpu_evaluator.rs:956`. | A recurrence of either class looks the same: a run that completes with no error, no warning, no crash, but plausible-looking numbers that are actually wrong. First things to check: (1) did any recent change to `launch_backtest_kernel`'s signature or a `#[repr(C)]` struct bump `LAUNCH_BACKTEST_KERNEL_ABI_VERSION` on **both** sides (`gpu.rs:495-521`, "The contract")? (2) is `compact_individual_for_device` still called on the actual device-launch path (`gpu_evaluator.rs:956`, inside `prepare_and_launch`) rather than only in `compile_batch` (which deliberately still produces uncompacted output, `gpu_evaluator.rs:916-918`)? Both corruption classes are also exactly the kind of thing `--gpu-parity-check`'s exact-match `trades`/`wins` comparison is positioned to catch — see `manual/13` §4's note that an integer-field FAIL is "the strongest possible signal of a real kernel bug." Run it after any change that touches either path. |

---

## 4. What "healthy" looks like numerically

### 4.0 Pending hardware execution (2.2.1 / 2.4 — operator action required)

Neither of these has been run yet — both are hardware-only steps that cannot be executed from this (non-A100) host, and this documentation pass cannot fill them in on the operator's behalf. Leave these rows blank until an operator with real A100 access runs them; do not fabricate placeholder numbers.

**2.2.1 (`6e55`) — parity self-check device-identification block, and per-field PASS/FAIL** (`manual/13`, `manual/16` §2.1):

| Field | Value | Date |
|---|---|---|
| Device name | *(pending — run `--gpu-parity-check` on the A100)* | |
| Compute capability | *(pending — expect `8.0`, see §1.1)* | |
| Free memory (at check time) | *(pending)* | |
| Total memory | *(pending)* | |

| Checked field | PASS / FAIL |
|---|---|
| `trades` (exact match) | *(pending)* |
| `wins` (exact match) | *(pending)* |
| `final_equity` (tolerance) | *(pending)* |
| `max_drawdown_percent` (tolerance) | *(pending)* |
| `min_trades_across_tickers` (exact match) | *(pending)* |

Per the subtask's own gate: **any FAIL here is a blocker** — do not proceed to run 2.2.2's population-level parity test (§2.5 above) or any later A100 step until every field above reads PASS. On a FAIL, open a subtask under 2.1 naming the failing field, fix it, and re-run before filling in this table for real.

**2.4 (`40b2`) — kernel throughput/occupancy measurement** (`manual/15` harness, run after 2.2 is fully green):

| Metric | Value | Date |
|---|---|---|
| Achieved occupancy (Nsight or `-Xptxas -v` spill report) | *(pending)* | |
| Local-memory spill bytes/thread | *(pending)* | |
| ms/generation, tier-1, at 1000 / 4000 / 16384 individuals | *(pending)* | |
| ms/generation, tier-2, at 1000 / 4000 / 16384 individuals | *(pending)* | |
| ms/generation, full-history, at 1000 / 4000 / 16384 individuals | *(pending)* | |

Per the subtask's own trigger: if spills are found to be `> 0` bytes or occupancy `< 25%`, a design note comparing warp-per-individual and bar-parallel layouts is required in a new `manual/17_gpu_layout_options.md` (no kernel rewrite in this wave regardless of the outcome) — §8 of `manual/14` already anticipates and pre-empts most of that analysis for the warp-per-individual option specifically. The real, measured numbers this row asks for belong in `manual/15` itself once collected, per that subtask's own instruction — this row is only a tracking placeholder pointing at the still-outstanding work.

**No number in this section has ever been measured on a GPU.** `apps/alpha-bench` (§2.4) is the tool that will produce the first real ones — run it and compare against these analytical estimates, treating a mismatch as *information to investigate*, not a failure of the estimate.

- **Order-of-magnitude CPU-vs-GPU speedup**: not stated anywhere in this codebase's manuals or source comments as a numeric expectation — searched and not found. Do not treat any speedup figure as a target; `apps/alpha-bench`'s `candle_evaluations_per_sec` (`manual/15` §2) is the number to actually compare, CPU row vs GPU row, from a real run.
- **F1b-3's upload reduction**: originally flagged here as uncitable — the "~987×" figure existed only in the F1b-3 implementation report, not in any manual or source comment. It has since been written up with its full arithmetic in `manual/12` **§7.1** ("What the column cache actually saved, per generation"): ≈ 1,648 GiB → ≈ 1.67 GiB per generation at `config.harvest.toml`'s 16,000 individuals × 5 tickers × 43,200 tier-1 candles, and ≈ 990× for tier-2. Cite §7.1, not this line. Note it is an **analytical** figure derived from config arithmetic, never measured — Task K's harness (`manual/15`) is what turns it into a measurement, and a disagreement on the A100 is information about the model, not a failure.
- **Task H' local-memory footprint** (verified, `manual/14` §10.4, `manual/14_gpu_cpu_kernel_divergence_audit.md:1428-1437`): known dynamically-indexed local memory per thread drops from **768 B** to **512 B** (-33%) — `snapshot_row`'s allocated capacity shrinks from `float[128]` (512 B) to `float[64]` (256 B); `DeviceStack<float,32>` (128 B) and `per_state_pnl[32]` (128 B) are unchanged.
- **Task H' per-bar slot computations** (verified, `manual/14` §10.5, lines 1455-1461): unconditional gathers of all 128 slots/bar drop to **~25** (mean 24.7, median 25, steady state) — **~80% fewer** — or **~42** (generation-0, uncapped) — **~67% fewer**. Both are per-individual averages from a standalone probe, not a hard bound.

**Both H' figures are explicitly flagged in their own source section (`manual/14` §10.7, lines 1519-1552) as static, size-of-the-declared-array / call-site-counting arithmetic — never a register/spill measurement, never a Nsight Compute run, never a wall-clock or memory-traffic measurement.** Treat them the same way here: a real profiler run producing a smaller local-memory reduction, or a smaller "fewer slot computations" benefit than the wall-clock improvement suggests, is exactly the kind of gap §5's checklist exists to close — not evidence the code is broken.

---

## 5. Profiling checklist

`manual/14` §8.6 is the authoritative, itemized profiling checklist (register count, achieved occupancy, local-memory spill counters, a real `SlotGatherKind` histogram, plus Task J's three Nsight Systems/bandwidth items) — do not duplicate its content here, only its priority order for a first A100 session:

1. **Items 1-4 first** (`manual/14` lines 1117-1120): `nvcc -Xptxas -v` register count, Nsight Compute achieved occupancy + warp execution efficiency, local-memory spill counters, and a `SlotGatherKind` histogram from a real production `d_slot_gather_plan`. These are foundational — cheapest to gather (one profiled run), and directly confirm or correct §4's static arithmetic before anything else is worth trusting.
2. **Item 5** (line 1121) is a conditional trigger from items 1-4's results, not an independent check — only relevant if the gather dominates kernel time (>70%) and occupancy is low specifically from local-memory pressure.
3. **Item 6 is already closed** — implemented as Task H' (§10); nothing left to do there.
4. **Items 7-9 next** (Task J, lines 1123-1125): confirm the host/device overlap in `run_full_history_filter_gpu`'s group-to-group boundaries is real (Nsight Systems timeline), confirm the assumed ~2× pinned-vs-pageable bandwidth gain actually holds on this host/driver, and confirm `kernel_stream` isn't implicitly serializing against the default stream. These validate Task J's design assumptions specifically and are next in priority since Task J is the newest, least-exercised piece.

---

## 6. Known-approximate results

The full-history GPU stage's per-individual result is a **known approximation**, not a bug: `BacktestResultGpu` carries no true per-ticker breakdown, only a cross-ticker aggregate plus the real `min_trades_across_tickers` scalar (O06). `approximate_result_from_gpu` (`crates/alpha-simulation/src/engine/tiering.rs:108-158`) synthesizes a per-ticker `BacktestResult` by:

- Splitting `gpu_res.trades`/`gpu_res.wins` **evenly** across every ticker (integer division) — not the real per-ticker counts.
- Reporting the **same** whole-batch `max_drawdown_percent` for every ticker.
- Hardcoding `sharpe_ratio`, `sortino_ratio`, and `calmar_ratio` to `0.0` for every ticker (`tiering.rs:144-146`) — this is divergence **O10** (`manual/14`'s summary table: "Sharpe/Sortino/Calmar, daily returns, per-ticker/per-state breakdown entirely absent from the GPU result path" — status **Accepted** as of F32/2.3.3, since the GPU tier gate never reads these fields on any path).

**What is NOT approximated**: aggregate fitness. `total_pnl` (the sum across every ticker) and `min_trades_across_tickers` (the real per-ticker minimum, tracked on-device since O06 — not derived from the average) are both real, exact values used for tier pass/fail decisions and downstream fitness scoring (`tiering.rs:316-333`). Only **per-ticker attribution** — which ticker traded how much, individual Sharpe/Sortino/Calmar — is an average-based approximation for anything the GPU full-history stage or a GPU-rejected tier-1/tier-2 individual produced.

**Practical consequence**: trust the aggregate score (fitness, pass/fail, total PnL) from a GPU run. Do not trust a specific ticker's reported trade count, win count, or risk ratios from a GPU-evaluated individual for forensic/attribution purposes — re-run that specific strategy through the CPU path (`run_simulation`) if per-ticker accuracy is needed.
