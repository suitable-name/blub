# Chapter 15: Throughput Benchmark Harness (`alpha-bench`)

Task K. A CLI binary (`apps/alpha-bench`) that sweeps
`{1, 2 years} x {1, 3, 5 tickers} x {CPU, GPU}` and reports both **raw**
and **effective** throughput for each configuration, so the operator has
a real, hardware-comparable number instead of the misleading estimate
that early stopping and tiering produce.

No code in `alpha-simulation`'s kernel, `build_snapshot_matrix`, or the
ABI was touched to build this — the harness is a new, additive consumer
of `alpha_simulation::engine::evaluate_batch_tiered`, the exact function
`apps/alpha-worker`'s `executor::execute_batches` and the master's
local-fallback path already call. It measures the real dispatch/tiering/
GPU logic the production pipeline runs, not a re-implementation of it.

---

## 0. Shape: a plain CLI, not `criterion`

`criterion` and `clap` are both already workspace dependencies, so using
`criterion` was a real option. It was not used, deliberately:
`criterion`'s whole design is built around statistical resampling of a
short, cheap, repeatable operation (it re-runs a benchmark body tens to
hundreds of times to build a confidence interval) — appropriate for a
microbenchmark, not for a 12-to-24-row sweep where a single row's
"operation" is itself a multi-GB data load, an HMM state precompute, and
a full-population, full-history simulation batch that can legitimately
take minutes on its own. Re-running each row enough times for
`criterion`'s statistics to mean anything would multiply an
already-expensive sweep by `criterion`'s sample count for no benefit
this harness needs. A plain `clap`-driven binary that times ONE
representative run per row (the same population, so rows are directly
comparable to each other without needing resampling to average out
noise) is the right shape for this task.

---

## 1. Why "raw" and "effective" are both reported

An earlier throughput estimate for this system counted individuals
evaluated per second under the *production* config — tiering enabled,
early stopping on. That number is not comparable to other backtesting
software, or even to a different Alpha Suite config, because most
individuals in a real run are culled after a cheap 30-day tier-1 slice
and never reach the expensive full-history stage at all. A config that
culls more aggressively looks "faster" by this metric while doing *less*
work, not more.

- **Raw** forces `ga.evaluation_tiers.enabled = false` for the row, so
  every individual in the population runs the full-history stage — the
  same amount of work for every individual, every row, every engine.
  This is the number to compare against other backtesting software.
- **Effective** uses the config's real `ga.evaluation_tiers` settings
  (tier-1/tier-2 culling, whatever `config.toml` ships), and additionally
  reports how many individuals were rejected at tier 1, tier 2, or
  survived to the full-history stage — so the culling that makes
  effective throughput lower (or higher, in individuals/sec terms) than
  raw is visible in the report, not hidden inside one opaque wall-clock
  number.

Both are measured for every `(years, ticker count, engine)` point by
default (`--measure both`).

---

## 2. Metric definitions

Per row (one `(years, ticker_count, compute, measure)` combination):

| Column | Meaning |
|---|---|
| `wall_seconds` | Wall-clock time of the ONE `evaluate_batch_tiered` call covering the whole fixed population, for this row. |
| `individuals_per_sec` | `population_size / wall_seconds`. |
| `bars_per_ticker` | Mean candle count per ticker actually handed to the evaluator for this row — the real, measured length of the sliced window (see §3), **including** the leading `WARMUP_PERIOD` (2000 candles) lookback every slice carries. Not a nominal `years * 365 * 1440` estimate. |
| `total_bars` | The exact SUM of every ticker's real sliced candle count for this row (not `ticker_count * bars_per_ticker`, which assumes every ticker has exactly the mean length — this is the precise sum, correct even when tickers' lengths differ slightly due to data gaps). |
| `candle_evaluations_per_sec` | `(population_size * ticker_count * bars_per_ticker) / wall_seconds`. **This is the number most comparable to other backtesting software** — it normalizes away population size and dataset shape, so two configurations processing the same number of (individual, ticker, bar) triples per second are doing the same amount of work, however that work was split between population size and history length. It is a conservative (slightly understated) estimate of pure post-warmup strategy-evaluation throughput, since it also counts the shared indicator-warmup replay as "evaluated" work — which is real CPU/GPU time spent regardless of whether it is cached across individuals. |
| `total_candle_evaluations` | `population_size * ticker_count * bars_per_ticker` — the numerator of the rate above, useful for sanity-checking or re-deriving the rate from the CSV directly. |
| `tier1_rejected` / `tier2_rejected` / `survived_to_full` | Effective-mode only (always `0`/`0`/`population_size` for raw, since raw never runs the tier ladder). Counted from `evaluate_batch_tiered`'s own per-individual `rejected_tier` tag — not re-derived or estimated. |
| `skipped_reason` | Non-empty only for a row that could not run (GPU unavailable, dataset window too large for the GPU ceiling, insufficient data would have aborted the whole run earlier — see §5). A skipped row's timing columns are empty, never zero or fabricated. |

Population size, seed, and the exact ticker list are also recorded per
row so a report file is self-describing.

---

## 3. How a row's dataset window is built

For ticker count `T` and years `Y`:

1. Tickers are the first `T` entries of `config.tickers` (in the order
   they appear in `config.toml`) — nested by construction, so the
   1-ticker row's ticker is also in the 3-ticker row, which is also in
   the 5-ticker row. Ticker COUNT is isolated from WHICH tickers.
2. The full history for those tickers (loaded once, up front, for the
   whole run) is sliced to its trailing `Y * 365` calendar days via
   `alpha_core::build_tier_slices` — the exact same trailing-window
   arithmetic the production tier-1/tier-2 ladder uses (extended
   `WARMUP_PERIOD` candles further back so the first evaluated bar's
   indicators have real history to warm up against).
3. For effective mode, tier-1/tier-2 sub-windows are carved out of THAT
   row's own window (via the same slicing function, using the config's
   `tier_1_days`/`tier_2_days`), not out of the full unsliced dataset —
   tiers are always relative to whatever "full" window a row is using.

Every row builds **fresh** indicator warmup caches (never reused across
rows, and never shared between a row's raw and effective measurement).
Sharing them would let whichever row happens to run second benefit from
the first row's already-warmed indicator state, biasing the comparison
by execution order instead of measuring each configuration's own
throughput. This means the harness measures one cold-cache generation's
worth of throughput, consistently, on every row — not the warmer,
amortized throughput a real multi-generation run settles into after its
first generation.

The population is built **once**, from a fixed RNG seed
(`--seed`, default `42`), and reused unmodified across every row — CPU
and GPU rows, every years/ticker-count combination, evaluate the
IDENTICAL individuals for a given seed. This is what makes the
comparison like-for-like rather than an artifact of which individuals
happened to be cheaper or more expensive to simulate.

---

## 4. Prerequisites

- A `config.toml` (or equivalent) with:
  - `tickers` listing at least as many tickers as the largest
    `--ticker-counts` value.
  - A valid, already-trained HMM model at `paths.hmm_model` — the
    harness calls `alpha_orchestrator::hmm_setup::precompute_states_systemic`,
    the SAME state-precompute path a real GA run uses, and that function
    requires an existing model file (it does not train one). If you can
    already run a normal GA discovery run against this config, this
    requirement is already satisfied.
  - `questdb_pg_uri` reachable, with historical OHLCV data for the
    requested tickers spanning at least the largest `--years` value
    (checked up front — see §5).
- For a GPU run: `gpu.enabled = true` and `gpu.column_cache_budget_mb >
  0` in the config (both are checked before any GPU row runs — see §6),
  a `gpu`-capable toolchain, and a real device.

---

## 5. Insufficient-data handling

Before running anything, the harness checks every requested ticker has
at least `years * 365` calendar days of history (for the LARGEST
requested `--years` value). A ticker that falls short aborts the whole
run with a message naming every short ticker and its found-vs-needed
span, e.g.:

```
insufficient historical data for 2 ticker(s):
  - BONK: found 210.3 day(s) of history, need 365.0
  - TURBO: found 340.1 day(s) of history, need 365.0
```

This never silently benchmarks a shorter-than-requested series.

---

## 6. GPU handling — never a fabricated number

If `--mode gpu`/`--mode both` is requested:

1. **`gpu` cargo feature not compiled in**: every GPU row is skipped with
   `"this binary was built without the gpu cargo feature..."`. All CPU
   rows still run and are reported normally.
2. **Feature compiled in, but `config.gpu.enabled = false` or
   `config.gpu.column_cache_budget_mb == 0`**: GPU rows are skipped with
   a message naming which config field is the problem. (Both of these
   also gate the GPU full-history stage inside `evaluate_batch_tiered`
   itself — skipping here up front prevents every "gpu" row from
   silently running the identical CPU fallback path a "cpu" row does,
   mislabeled as a GPU measurement.)
3. **Feature compiled in, config permits it, but no device answers**
   (`query_gpu_device_info` fails, or `GpuEvaluator::new` fails — e.g. no
   GPU present, or out of device memory): every GPU row is skipped with
   the underlying error message. All CPU rows still run.
4. **A specific row's dataset window exceeds the GPU ceiling**
   (`min(config.gpu.max_snapshot_candles, evaluator capacity)`): that ONE
   row is skipped (not the whole GPU mode) with a message naming the
   candle count and the ceiling it exceeded.

In every case: the process never crashes, a GPU row is never silently
answered by the CPU fallback path and reported as if it were a GPU
number, and every CPU row that *can* run still does.

---

## 7. Exact commands

### 7a. No-GPU dev host (this box)

Builds and runs the CPU half of the matrix only — this binary does not
need, and does not require, the `gpu` cargo feature to build or run.

```
cargo run --release -p alpha-bench -- \
  --config config.toml \
  --mode cpu \
  --output-dir benchmark_results
```

This reproduces the full `{1, 2 years} x {1, 3, 5 tickers}` sweep for
CPU only (both raw and effective), using the defaults (`--population
500 --seed 42`). Output lands in
`benchmark_results/throughput_<UTC timestamp>.csv` and the matching
`.json`, plus a human-readable table on stdout.

**What this run tells you**: the CPU half of the matrix, and a baseline
to compare the A100 run against. **What it does NOT tell you**: anything
about GPU throughput — there are no GPU rows in this report at all (not
even skipped placeholders framed as zeros; `--mode cpu` never attempts a
GPU row in the first place).

If you want to run the full `--mode both` sweep on this host anyway
(harmless — GPU rows will just report `skipped_reason: "this binary was
built without the gpu cargo feature..."` and every CPU row still runs),
drop `--mode cpu`.

### 7b. A100 target

Requires building with the `gpu` cargo feature, and `config.toml`'s
`gpu.enabled = true` and `gpu.column_cache_budget_mb > 0` (both are the
production defaults — see `alpha_config::gpu::GpuConfig`'s
`Default` impl — so an unmodified production config already satisfies
this; only change it if you've deliberately turned GPU off).

```
cargo run --release --features gpu -p alpha-bench -- \
  --config config.toml \
  --mode both \
  --output-dir benchmark_results
```

This produces the FULL `{1, 2 years} x {1, 3, 5 tickers} x {CPU, GPU}`
sweep, both raw and effective, in one run. Output lands in the same
`benchmark_results/throughput_<UTC timestamp>.csv` / `.json` shape as
the dev-host run, so the two report files can be diffed directly.

### 7c. Comparing the two runs

Both report files carry a `#`-prefixed metadata header (host, whether
the `gpu` feature was compiled in, GPU device name if any, config path,
git commit if available, generation timestamp) before the CSV column
header, so a report file is self-describing even without the command
that produced it. Load both CSVs (skipping `#` lines) into your
comparison tool of choice, or diff the JSON files directly — every field
name and row shape is identical between the two runs.

---

## 8. CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `config.toml` | Path to the Alpha Suite config to load tickers, GPU, and evaluation-tier settings from. |
| `--years` | `1,2` | Repeatable or comma-separated. Years of trailing history per row. |
| `--ticker-counts` | `1,3,5` | Repeatable or comma-separated. Ticker counts per row (nested prefixes of `config.tickers`). |
| `--mode` | `both` | `cpu`, `gpu`, or `both`. |
| `--measure` | `both` | `raw`, `effective`, or `both`. |
| `--population` | `500` | Fixed population size every row evaluates. |
| `--seed` | `42` | RNG seed for the fixed population. |
| `--output-dir` | `benchmark_results` | Directory the CSV/JSON report is written to (created if missing). |

---

## 9. What this harness deliberately does NOT do

- It never queries or mutates the database beyond the one read
  (`load_historical_data`) the production pipeline already does for a
  normal run — no new query layer.
- It never changes `build_snapshot_matrix`, the kernel, or the ABI.
- It never trains an HMM model — a valid, already-trained model must
  already exist at `config.paths.hmm_model` (see §4).
- It does not attempt to reproduce the GA's own multi-generation warm-
  cache amortization (see §3) — every row is a cold-cache measurement,
  by design, so rows are comparable to each other regardless of
  execution order.
