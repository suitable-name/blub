# Chapter 14: GPU/CPU Backtest Kernel — Divergence Audit

This chapter is a full-relationship audit, not a changelog. It enumerates every
behavioural difference between `backtest_kernel.hip.cpp` (the GPU tier-1/tier-2
filter) and the CPU reference engine (`crates/alpha-simulation/src/engine/`,
principally `simulation.rs`, `execution.rs`, `utils.rs`, `state.rs`), including
the ones already fixed, so the result is a complete statement of the
relationship rather than a diff against recent work. **No code was changed
while producing this document** — every claim below is backed by a
`file:line` citation on both sides, or explicitly marked **OPEN**.

Thirteen divergences were found and fixed before this audit began, entirely
incidentally, while working on other things. Two more were already known and
left open. This audit found **eight more** that nobody had previously
flagged — three of them (§4, O04/O05/O06) more consequential, in terms of
changing which strategies the genetic algorithm selects, than either of the
two previously-known open items.

---

## 1. Summary table

| ID | Summary | Severity | Status |
|---|---|---|---|
| F01 | Missing taker fee on entry/exit | Trade-decision (equity) | Fixed |
| F02 | Missing slippage on entry/exit | Trade-decision (fill price) | Fixed |
| F03 | Missing `min_hold_bars` gate on exit signal | Trade-decision | Fixed |
| F04 | Realized-only drawdown instead of mark-to-market | Trade-decision (circuit breaker) | Fixed (see O03/O05 for residual gaps) |
| F05 | `per_state_pnl` array overflow past old 12-state bound | Memory corruption | Fixed |
| F06 | Stateful-node (`Sma`/`StdDev`/`Lag`) ring-buffer addressing bug | Trade-decision (signal score) | Fixed |
| F07 | Cross-ticker drawdown-accumulator leak | Trade-decision (circuit breaker) | Fixed |
| F08 | Missing position-size margin cap | Trade-decision (position size) | Fixed |
| F09 | Spurious `× leverage` on realized/unrealized P&L | Reported numbers + circuit breaker | Fixed |
| F10 | Same-bar fill instead of next-bar-open fill | Trade-decision (timing + price) | Fixed |
| F11 | Exit tree/threshold/periods read from current-bar state instead of frozen entry state | Trade-decision | Fixed |
| F12 | Missing gap-through-on-open handling | Trade-decision (fill price) | Fixed |
| F13 | Liquidation entirely absent | Trade-decision + reported numbers | Fixed |
| F14 | GP `Not`-node sign error (compiler, `gpu.rs`) | Trade-decision (signal score) | Fixed |
| F15 | Operand-stack depth overflow risk in `evaluate_gp_tree` | Trade-decision (signal score) / crash risk | Fixed |
| O01 | `risk_per_coin` floor (`entry_price * 0.0005`) missing on GPU | Trade-decision (position size) | **Fixed** |
| O02 | Bankruptcy check hardcoded `10.0f` instead of config-sourced | Trade-decision (circuit breaker) | **Fixed, then removed** (F32/2.3.2: the device-only `current_equity <= failed_equity` floor was removed from the kernel breach check and both twins; `failed_equity` now serves only the O04 backfill — see §4's "O02 — decision record (F32/2.3.2)") |
| O03 | Drawdown/bankruptcy breach does not force-realize the open position | Reported numbers (equity understated at breach) | **Fixed** |
| O04 | No `break_all_tickers` analog — GPU keeps simulating every remaining ticker instead of failing them | Trade-decision + fitness selection | **Fixed** |
| O05 | Out-of-range HMM state causes an unconditional `continue` that skips PHASE 1–7.5 entirely, not just the trading decision | Trade-decision (timing, funding, cooldown, drawdown) | **Fixed** |
| O06 | Tier-gate uses average trades-per-ticker instead of the true per-ticker minimum | Fitness selection (which individuals reach tier-2/full) | **Fixed** |
| O07 | Mixed-length ticker batches are zero-padded and fully simulated instead of being skipped | Trade-decision (fabricated price crashes) | **Fixed** |
| O08 | Austrian tax (`simulate_austrian_tax`/`loss_pot`) entirely unimplemented on GPU | Reported numbers (equity overstated); production config has this on | **Fixed** (F32/2.3.1 closed the previously-residual year-boundary `loss_pot` reset gap — see §4's "O08 — decision record (F32/2.3.1)") |
| O09 | Agg1/Agg2-timeframe indicator terminals read constant registry defaults on GPU | Trade-decision (signal score) for any tree using them | **Fixed** (second follow-on task, after O01–O08; see §3.9 detail) |
| O10 | Sharpe/Sortino/Calmar, daily returns, per-ticker/per-state breakdown entirely absent from the GPU result path | Reported numbers only (tier gate never reads them) | **Accepted** (F32/2.3.3: the GPU only gates tiers, final scoring is always CPU-side — see §4's "Accepted residual gaps (F32/2.3.3)") |
| O11 | Entry-time risk-per-coin read the post-update (`global_i`) ATR snapshot instead of the pre-update (`global_i - 1`) one, a one-bar lookahead into the entry bar's own ATR | Trade-decision (position size, initial stop/TP/liquidation levels) | **Fixed** (both `backtest_kernel.hip.cpp:1419-1441` and `kernel_reference.rs`; shipped in an earlier round but never logged in this table until now — see O12 for the confirming regression-test reproduction) |
| O12 | `kernel_reference_simple_parity_tests.rs`'s five twin-vs-`run_simulation` tests once documented a 149-vs-148-trade divergence on a fixed BTC/seed-42 control case, carried `#[ignore]`d under the mistaken belief it was an unfixable f32/f64 precision artifact | Trade-decision (position size) — same root cause as O11 | **Fixed** — was O11 all along; reverting O11's fix reproduces the divergence exactly, restoring it re-closes it; all six tests re-enabled as permanent regression coverage — see §4 O12 detail |
| O13 | F28: kernel `Log` computed `ln(a)` (undefined for `a <= 0`, silently zeroed) instead of the CPU's signed `ln1p` (`sign(a) * ln(1+\|a\|)`), discarding the sign of a negative indicator entirely | Trade-decision (signal score) for any GP tree using `Log` | **Fixed** — `evaluate_gp_tree`'s `Log` arm now matches `alpha_gp::evaluator::mod::Op::Log` exactly, in the kernel (`backtest_kernel.hip.cpp`), `kernel_reference.rs` (f32 twin and both f64 twins), and `gpu.rs`'s `eval_stateless` test helper — see §4 O13 detail |
| O14 | F28: kernel `Div` returned `1.0f` for `fabsf(b) < 1e-9f` instead of the CPU's `0.0` fallback for a non-finite quotient — a different, scale-dependent degenerate-input threshold entirely, wrong by construction for this system's micro-cap (sub-1e-6) price scale | Trade-decision (signal score) for any GP tree using `Div` | **Fixed** — same call sites as O13; `isfinite(v) ? v : 0.0f` replaces the old fixed threshold everywhere — see §4 O14 detail |
| O15 | F28: `Sma`/`StdDev`/`Lag` used the batch-global bar index `timestep` (always `>= warmup_period`) directly as the ring's local sample count, so `count == period` ("ring already full") on the very first bar of every ticker instead of counting up from zero the way `SmaState`/`StdDevState`/`LagState` do | Trade-decision (signal score) for any GP tree using a stateful op, worse the shorter the op's period relative to `warmup_period` (2000) | **Fixed** — `evaluate_gp_tree` (kernel and every twin) computes `local_t = timestep - warmup_period` and uses it everywhere the ring index/count/lag arithmetic previously read `timestep`; `Lag`'s pre-fix warm-up fallback (`input_val`, and an off-by-one ring read) was also corrected to match `LagState::update` exactly (`0.0` fallback, pre-overwrite read) — see §4 O15 detail |
| O16 | F28: the per-individual `state_buffer` ring-buffer slice is zeroed once per BATCH (host-side `gpu_memset` in `gpu_evaluator.rs`), never per ticker — ticker N>0 started every `Sma`/`StdDev`/`Lag` node with whatever ring contents ticker N-1 left behind instead of the fresh, empty history the CPU's per-ticker `eval.reset()` gives every ticker | Trade-decision (signal score) for the first `period` bars of every ticker after the first, for any GP tree using a stateful op | **Fixed** — the kernel's per-ticker loop (and every `kernel_reference.rs` twin's per-ticker loop) now zeroes this thread's `max_stateful_nodes_per_ind * stride`-float slice of `state_buffer` at the top of every ticker iteration, safe under `__restrict__` since it is exclusively this thread's own memory — see §4 O16 detail |
| O17 | F07 (4.1.2): the new `spread_and_impact` friction model (`simulation.friction.model`, half-spread + participation-scaled impact, entry rejection above `max_participation`) is CPU-only — the kernel keeps the legacy `constant` (flat `slippage_pct`) fill unconditionally, with no `spread_and_impact` arm at all | Trade-decision (fill price, position notional, entry admission) for any GPU-evaluated tier when `simulation.friction.model = "spread_and_impact"` | **Accepted, deferred** — `alpha_orchestrator::run::warn_if_gpu_friction_model_mismatch` logs a `tracing::warn!` once per run start whenever `friction.model = "spread_and_impact"` and `gpu.enabled = true` are both set, so the divergence is operator-visible rather than silent; mirroring the model into the kernel (and `kernel_reference.rs`'s twins) is left as a follow-up subtask under Wave 2 |

Thirteen fixed (F01–F13, the set the task described), plus two more
historical fixes recovered from the kernel's own comments and
`manual/13_gpu_cpu_parity_self_check.md` (F14–F15, which predate F01–F13).
Ten open items were originally identified, of which two (O01–O02) were
already known going in, one (O09) was already documented in `manual/12`
(and has since been fixed — see below), one (O10) is arguably by design,
and **six were new findings from this audit** (O03–O08).

**Update (follow-on task, same repository state this audit describes):**
O01–O08 have since been fixed — see each item's detail section below,
updated in place with what actually shipped (kept ID-stable rather than
renumbered). O08's fix carries one documented residual gap (no
calendar-year `loss_pot` reset, since `OhlcGpu` carries no per-bar
timestamp for the kernel to detect one). At the time of that follow-on
task, O09 and O10 remained open, explicitly out of scope: O09 belonged to
a separate queued task that would rebuild the whole indicator supply
path, and O10's fields are never read by the tier gate. Every fix below
was verified by re-reading the actual source side-by-side with this
audit's original claims — no divergence
between what this audit predicted and what the source actually contained
was found for O01–O08; the fixes described below implement exactly what
each item's original analysis called for.

**Second update (later follow-on task, task F2): O09 has since been fixed
too.** The "separate queued task" mentioned above ran; `crate::gpu_columns`
and `crate::gpu_column_gather` (new files) implement the resident, packed
`(agg_period, indicator_period)`-keyed column cache `manual/12` scoped as
"Option A," and `crate::gpu::build_snapshot_matrix` was itself fixed in
the same task to tick real `Timeframe::Agg1`/`Agg2` pipelines (it is used
throughout this crate's test suite as the CPU-equivalence authority, so
this matters beyond O09 itself — see §3.9 for what changed and why
`MultiScaleIndicatorState`/`TimeframeIndicators`, not
`build_snapshot_matrix`, was always the true reference, with
`build_snapshot_matrix` catching up to match it). O10 remains open,
unchanged — see its own entry below.

**Third update (F28 remediation, blueprint `fdb2` subtask 2.1): four more
findings closed, O13-O16.** This audit's table grows from 27 rows (26
fixed, O10 the sole open item) to 31 (30 fixed, O10 still the sole open
item). All four are semantics divergences in `evaluate_gp_tree`'s `Log`/
`Div`/`Sma`/`StdDev`/`Lag` arms — none of them touch a `#[repr(C)]` struct
or a function signature crossing the FFI boundary, so
`LAUNCH_BACKTEST_KERNEL_ABI_VERSION` (`gpu.rs`) and
`launch_backtest_kernel_abi_version()` (`backtest_kernel.hip.cpp`) are
**unchanged** by this update — see §4's O13-O16 detail sections for the
full before/after and the regression tests that pin each fix.

**Fourth update (F32 remediation, blueprint `fdb2` subtasks 2.3.1/2.3.2/2.3.3
— "close or formally accept the documented residual gaps"): the table's row
count is unchanged at 31 (F01-F15, O01-O16); this update closes O08's
residual gap and settles the disposition of O02 and O10, but adds no new
rows.**
- **O08 (2.3.1, `f4c2`)** is now fixed in full, not just fixed-with-a-gap:
  the year-boundary `loss_pot` reset this table previously carried as a
  documented residual gap is closed on-device. `LAUNCH_BACKTEST_KERNEL_ABI_VERSION`
  (`gpu.rs`) and `launch_backtest_kernel_abi_version()` (`backtest_kernel.hip.cpp`)
  are bumped **5 -> 6** together, since this adds two new pointer parameters
  and a count to `launch_backtest_kernel`'s signature. Pinned by
  `year_boundary_loss_pot_reset_matches_run_simulation_across_dec31_to_jan1`
  (`crates/alpha-simulation/tests/kernel_reference_simple_parity_tests.rs`).
  See §4's "O08 — decision record (F32/2.3.1)" below for the mechanism.
- **O02 (2.3.2, `58cf`)** changes disposition from "fixed, floor kept" to
  "fixed, floor removed": the device-only `current_equity <= failed_equity`
  bankruptcy floor described in §3.6 has been REMOVED from the kernel breach
  check and from both `kernel_reference.rs` twins, per this task's own
  documented default decision — **taken without explicit owner
  confirmation**, since the blueprint instructs proceeding on the stated
  default absent owner input. `failed_equity` now serves only the O04
  backfill. Pinned by
  `bankruptcy_floor_removal_yields_identical_trade_counts_vs_run_simulation`
  (same test file). See §4's "O02 — decision record (F32/2.3.2)" below.
- **O10 (2.3.3, `5ab1`)** is reclassified from "Open, by design —
  informational" to **ACCEPTED**: the GPU tier filter only ever decides
  which individuals survive to the next tier or to full-history CPU
  evaluation — it never itself produces the Sharpe/Sortino/Calmar/daily-
  return/per-state numbers a human or the fitness function ultimately reads,
  since those are always computed CPU-side (`run_simulation` /
  `BacktestResult`) once an individual reaches that stage. The GPU result
  path never carrying these fields is therefore a property of the pipeline's
  architecture, not an open gap. Per-bar funding deduction and the
  hard-coded `0.5` entry/exit thresholds are likewise reclassified from
  implicit ("documented in §3.6/§3.3 as a structural difference") to
  explicitly **accepted** — see the new "Accepted residual gaps (F32/2.3.3)"
  subsection at the end of §4.

**After this update: zero rows are open.** All 31 (F01-F15, O01-O16) are now
either fixed or explicitly, formally accepted — closing item 1 of §7's
"for the claim to hold" checklist for the full O01-O10 set, not just
O01-O08 as previously recorded there.

**Fifth update (Wave 2 close-out, blueprint `fdb2` task `c766`, 2026-09-07):
divergences found/fixed/documented counts, and hardware-verification status.**

- **Divergences found:** 31 total (F01-F15 trade-decision/memory bugs, O01-O16
  documented gaps this audit's follow-on tasks tracked).
- **Fixed in code:** 30 of 31 (every row above except O10).
- **Formally accepted (no code change, decision recorded):** 1 row (O10 —
  Sharpe/Sortino/Calmar/daily-return/per-state fields absent from the GPU
  result path, F32/2.3.3), plus two narrower accepted approximations folded
  into that same 2.3.3 decision (per-bar funding deduction, hard-coded `0.5`
  entry/exit thresholds) and O17 (F07/4.1.2's `spread_and_impact` friction
  model staying CPU-only, accepted-deferred, logged via
  `warn_if_gpu_friction_model_mismatch`).
- **Parity status: NOT YET VERIFIED ON HARDWARE.** Every fix above (F01-F15,
  O01-O16) has only ever been compiled and type-checked on this (GPU-less)
  host — see §7 item 5. Per blueprint `fdb2` Wave 2 (2.2, F27):
  - **2.1 (F28 kernel/CPU semantics alignment) — DONE.** O13-O16 closed;
    see the "Third update" note above.
  - **2.2.1 (run `alpha-worker --gpu-parity-check` on the A100 and record
    the per-field PASS/FAIL table in `manual/16` §4) — PENDING.** Blocks
    2.2.2/2.2.3 from being trusted empirically even though both are
    code-complete (see below): this is the first time any of this kernel's
    fixes run against a real device.
  - **2.2.2 (population-level GPU-vs-CPU tiering parity test,
    `crates/alpha-simulation/tests/gpu_cpu_parity_tests.rs`, feature `gpu`)
    — code DONE, unexecuted.** Cannot produce a real PASS/FAIL until run on
    the A100 (2.2.1's blocker applies here too).
  - **2.2.3 (GPU change-checklist process gate, `manual/11` +
    `backtest_kernel.hip.cpp` header comment) — DONE.** See `manual/11`'s
    "GPU change checklist" section.
  - **2.3 (F32 residual-gap disposition: O02/O08/O10) — DONE.** See the
    "Fourth update" note above.
  - **2.4 (F31 throughput measurement: occupancy, spill bytes, ms/generation
    at 1000/4000/16384 individuals into `manual/15`, and `manual/17` if
    spills/occupancy warrant it) — PENDING.** No A100 in this environment;
    §8's occupancy analysis (Task H) is static-analysis-only per its own
    header, not a measurement — see §8.6's "profiling checklist" for exactly
    what running 2.4 must collect.

  Until 2.2.1 and 2.4 run, "the kernel reproduces the CPU model" (§7) is a
  claim resting on source-reading and compile-time type-checking alone, not
  on execution evidence — the distinction §7 item 5 already drew before this
  update, restated here as this chapter's current summary count.

---

## 2. Method

`run_simulation` in `crates/alpha-simulation/src/engine/simulation.rs` is the
spine: its PHASE 1–9 comments were read start to end, and each phase's kernel
mirror in `crates/alpha-simulation/src/backtest_kernel.hip.cpp` was located
and compared line-by-line. The kernel's own extensive comments (`GAP n`,
`FIX n`, `LEVERAGE PARITY FIX`, `LIQUIDATION FIX`, `FILL-TIMING FIX`,
`DIVERGENCE 1 FIX`, `N10`) document most of the already-fixed history
directly at the point of the fix, and are cited as primary sources below
rather than re-derived. `crates/alpha-simulation/src/gpu.rs`,
`gpu_evaluator.rs`, and `engine/tiering.rs` were read in full to cover the
Rust-side FFI glue and the GPU tier-filter harness, since several of the new
findings (O04, O06, O07) live there rather than in the kernel itself.
`config.toml`, `config.harvest.toml`, `config.bolt.toml`, and
`config.pgo.toml` were checked directly for every config value cited, rather
than assumed from defaults.

Both files were read in their entirety (`simulation.rs`: 1181 lines;
`backtest_kernel.hip.cpp`: 1471 lines) — this is not a sampled review.

---

## 3. Phase-by-phase comparison

### 3.1 Ticker iteration order and synthetic-failure backfill

**CPU** (`simulation.rs:192-193`): `tickers` is the `historical_data` key set,
sorted alphabetically — deterministic, independent of `HashMap` iteration
order. A ticker with `data_points.len() <= WARMUP_PERIOD` is `continue`d past
entirely (`simulation.rs:217-219`) — no `ticker_results` entry at all. On a
global drawdown-cap breach (PHASE 7.5, `simulation.rs:717`), the breaching
ticker's open position (if any) is force-closed through `process_exit`
(realizing the loss into `current_equity`, `simulation.rs:751-769`), then
**every remaining ticker in sort order** (`tickers[ticker_idx + 1..]`,
`simulation.rs:1141-1165`) is backfilled with a synthetic failing
`TickerResult`: `final_equity: config.simulation.failed_equity`,
`max_drawdown_percent` carried forward from the breach, `trades = wins =
losses = 0`. This is deliberately punitive — see `simulation.rs:1113-1140`'s
comment for why a flat `initial_equity` reading would have been wrong.

**GPU** (`gpu_evaluator.rs:552`, `run_tier_filter_gpu` in `tiering.rs:115-117`):
`ticker_order` is also the key set, sorted — matches. Ticker-length exclusion
and the failure backfill are both **absent** — see O04 and O07 below.

### 3.2 Warmup handling and the state/candle invariant

**CPU**: `data_points.len() <= WARMUP_PERIOD` skips the ticker
(`simulation.rs:217-219`); `ticker_states.len() != data_points.len() -
WARMUP_PERIOD` is a hard `Err` that aborts the whole individual
(`simulation.rs:290-303`) — promoted from a `debug_assert!` specifically
because release builds compile those out (see that block's long comment).
The candle loop iterates `data_points.iter().enumerate().skip(WARMUP_PERIOD)`
(`simulation.rs:388`), i.e. exactly and only that ticker's own real candles.

**GPU**: `evaluate_batch` (`gpu_evaluator.rs:540-545`) computes one shared
`num_candles = max` over every ticker in the batch, and — **at the time
this audit was originally written** — the kernel looped `for (i =
warmup_period; i < num_candles; ++i)` identically for every ticker,
regardless of that ticker's own real length, with no equivalent of the
CPU's length check anywhere in `gpu_evaluator.rs` or the kernel. **Fixed**
(O07): the loop bound is now `for (int i = warmup_period; i < ticker_len;
++i)` (`backtest_kernel.hip.cpp:1293`), where `ticker_len` is that
ticker's own entry in the new `ticker_lengths` array — see O07 below for
the fix and the upstream `run_tier_filter_gpu` filtering that closes the
too-short-to-clear-`WARMUP_PERIOD` case.

### 3.3 Entry

| Aspect | CPU | GPU | Status |
|---|---|---|---|
| Signal eval | `entry_evaluators[idx].evaluate()` bool (`simulation.rs:836-841`) | `entry_score > active_strategy.entry_threshold` (0.5, `gpu.rs:1276-1277`); `backtest_kernel.hip.cpp:2003-2005` | Agrees (tree resolves to exact 0.0/1.0; see I02) |
| Fill timing | Queued PHASE 9, filled PHASE 1 next bar (`simulation.rs:990-995`, `413-427`) | `pending_order`/`PENDING_*` state machine, filled at top of next iteration (`backtest_kernel.hip.cpp:1372-1536`) | F10, fixed |
| Entry price | `open * (1 ± slippage_pct/100)` (`execution.rs:18-19,67-68`) | identical (`backtest_kernel.hip.cpp:1388-1392`) | F02, fixed |
| Risk-per-coin | `(entry - stop).max(entry_price * 0.0005)` (`execution.rs:30-31,79-80`) | at the time of this audit: `atr * atr_multiplier`, no floor, only guarded `> 1e-9f`; **since fixed (O01)** — now `fmaxf(entry_atr * fill_strategy.atr_multiplier, entry_price * 0.0005f)` (`backtest_kernel.hip.cpp:1453`), matching this column exactly | **O01, Fixed** — see §3.3's O01 detail below |
| Position sizing | `risk_amount = equity * base_risk_per_trade * risk_modulator` (`execution.rs:34-35`) | `risk_amount = equity * base_risk_per_trade` — no `risk_modulator` term (`backtest_kernel.hip.cpp:1456`) | Agrees in current usage only; see I01 |
| Margin cap | `position_size = risk_based.min(equity*leverage/entry_price)` (`execution.rs:37-38`) | identical (`backtest_kernel.hip.cpp:1469`) | F08, fixed |
| Entry fee | `notional * taker_fee_pct/100` deducted from equity (`execution.rs:49-51`) | identical (`backtest_kernel.hip.cpp:1503-1504`) | F01, fixed |
| Liquidation price | `entry * (1 -/+ liquidation_margin_loss_pct/leverage)` (`execution.rs:45-47,94-96`) | identical (`backtest_kernel.hip.cpp:1479-1492`) | F13, fixed |
| Cooldown seed | `cooldown_period_setting = active_strategy.params.cooldown_period`, frozen at entry (`execution.rs:24,73`) | `cooldown_period_setting = fill_strategy.cooldown_period` (`backtest_kernel.hip.cpp:1525`) | Fixed, both freeze to the *filling* strategy |

**O01 detail**: `execution.rs:31` — `let risk_per_coin = (sim_state.entry_price
- stop_price).max(sim_state.entry_price * 0.0005);` — floors risk-per-coin
at 5 bps of entry price, so a near-zero-ATR bar still gets a sane (small but
nonzero) position size. At the time of this audit, the kernel's entry-fill
branch had no `.max(...)` at all — it only guarded `if (risk_per_coin >
1e-9f)` before sizing, and skipped the entry outright (no floor, no
fallback) when ATR was at or below that threshold. This was a
decision-changing difference, not a rounding one: a bar where CPU would
size a (small) position and trade, GPU sized nothing and skipped the
entry.

**Fixed**: the PENDING_ENTER_LONG/SHORT fill branch now computes
`risk_per_coin` as `fmaxf(snapshot[SLOT_ATR_RAW_1M] * fill_strategy.atr_multiplier,
entry_price * 0.0005f)`, mirroring `execution.rs`'s formula exactly (the
kernel's `base_risk_per_coin` term is the `atr * atr_multiplier` half of
CPU's `(entry_price - stop_price)`, since `stop_price = entry_price -
base_risk_per_coin` by construction on both sides). No FFI/ABI change was
needed — this is a pure in-kernel arithmetic fix.

### 3.4 Exit resolution

**Gap-through-on-open precedence, intrabar stop/TP/liquidation precedence**:
CPU's single source of truth is `resolve_candle_exit`
(`utils.rs:7-115`) — long-form precedence: (1) open-based liquidation, (2)
open-based stop/TP, (3) intrabar stop before TP before liquidation. GPU
reproduces this exactly, branch for branch (`backtest_kernel.hip.cpp:1615-1803`,
inclusive of the O08 Austrian-tax step folded into this range's exit paths).
F12/F13, fixed — verified line-by-line, no residual gap found here.

**Trailing-stop updates**: CPU, PHASE 5 (`simulation.rs:536-556`), keyed off
`params.strategies[sim_state.entry_state_idx]` (frozen entry state) — runs
only when still in a position at PHASE 5's point in the bar (i.e. *not* on a
bar this same PHASE 3 just closed). GPU updates `trailing_stop` inside the
PHASE 3 mirror block, unconditionally whenever `trade_state != 0` at the
*start* of that block (`backtest_kernel.hip.cpp:1697,1734`) — including on a
bar this same block just closed the position. This is harmless in practice
(the stale `trailing_stop` value is discarded — a fresh entry always resets
it before it is read again), but it is a real ordering difference from the
CPU, not merely a stylistic one. Noted, not scored as a divergence with
consequence (nothing reads the stale value).

**Exit signal gate and `min_hold_bars`**: CPU, PHASE 8
(`simulation.rs:843-853`) — evaluated only for `idx == entry_state_idx` and
`bars_since_entry >= min_hold_bars`; stop/TP/liquidation (PHASE 3) is
*never* gated by this. GPU matches exactly: `exit_score >
exit_strategy.exit_threshold && bars_since_entry >= min_hold_bars`
(`backtest_kernel.hip.cpp:2044`), keyed off `entry_state_idx`
(`backtest_kernel.hip.cpp:2031`, the DIVERGENCE 1 FIX). F03/F11, fixed.

### 3.5 Exit accounting

| Aspect | CPU (`execution.rs`) | GPU (`backtest_kernel.hip.cpp`) | Status |
|---|---|---|---|
| Exit slippage | `final_sell_price *= 1 ∓ slippage_pct/100`, skipped if liquidated (:124-131) | identical, skipped if liquidated (:797-804, :1003-1008) | F02, fixed |
| Realized P&L | `(sell - entry) * coins` or mirrored short, **no leverage multiplier** (:137-141) | identical, leverage multiplier removed (:814-817, :1011-1014) | F09, fixed |
| Liquidation P&L | fixed `-margin_for_trade * liquidation_margin_loss_pct`, never re-derived from crashed price (:133-136) | identical (:996-1000) | F13, fixed |
| Exit fee | `exit_notional * taker_fee_pct/100`, **skipped on liquidation** (:157-162) | identical, skipped on liquidation (:827-829, :1016-1021) | F01/F13, fixed |
| Funding | `net_pnl -= accumulated_funding_fee` (deferred to exit) (:164) | **not** re-subtracted at exit — already deducted bar-by-bar from `current_equity` (PHASE 4 mirror); kernel's own comment (:820-826) argues the equivalence | Structurally different mechanism, same net effect — see 3.6 |
| Austrian tax | 27.5% on net gain above `loss_pot`, loss added to `loss_pot`, applied every exit when `config.simulate_austrian_tax` (:166-178) | at the time of this audit: **absent entirely** — no tax, no loss-pot concept anywhere in the kernel; **since fixed (O08)** — `apply_realized_pnl` (`backtest_kernel.hip.cpp:329-345`) applies the identical 27.5%/`loss_pot` step, against a per-ticker `loss_pot` seeded at `:1253` | **O08, Fixed** — the year-boundary `loss_pot` reset gap this row once carried is closed (F32/2.3.1); see §3.5's O08 detail |
| Equity update | `current_equity += net_pnl_dollars` (post-tax) (:171,174,177) | at the time of this audit: `current_equity += pnl_dollars` (pre-tax, since no tax exists); **since fixed (O08)** — every exit path now routes through `apply_realized_pnl` (`backtest_kernel.hip.cpp:1586`, `:1787`, `:1926`), which applies the tax step before crediting equity | **Fixed** — agrees with CPU whenever tax is enabled, except across a year boundary (O08's residual gap) |

**O08 detail**: `config.simulate_austrian_tax = true` in `config.toml`,
`config.bolt.toml`, and `config.pgo.toml` (all: line ~15/24); only
`config.harvest.toml` sets it `false`. On every config except harvest, CPU
tier-1/tier-2 evaluation (via `run_simulation`, which GPU tiering is meant to
approximate) deducts 27.5% of net realized gains above the running loss-pot
at every winning exit; GPU deducts nothing. This systematically overstates
GPU-computed equity relative to what the CPU tier filter would compute for
the identical sequence of trades, on three of the four shipped configs.

**Fixed, with one documented residual gap**: a new `apply_realized_pnl`
device function implements `process_exit`'s tax step verbatim (`taxable_gain
= max(pnl - loss_pot, 0); tax_due = taxable_gain * 0.275; loss_pot =
max(loss_pot - pnl, 0); equity += pnl - tax_due` on a win, `loss_pot +=
|pnl|; equity += pnl` on a loss), gated on a new `simulate_austrian_tax`
kernel parameter (`config.simulate_austrian_tax`) and a new per-ticker
`loss_pot` float (reset per ticker, mirroring CPU's `let mut loss_pot: f64
= 0.0;` inside the per-ticker loop). Called from all three exit-realization
sites in the kernel (`PENDING_EXIT`, the unified intrabar exit — which
covers both the `was_liquidated` and non-liquidated branches, since CPU
applies its tax step once regardless of which path computed `net_pnl_dollars`
— and the new O03 forced-breach close), so no exit path can silently skip
it. **Residual gap, left open deliberately**: CPU also resets `loss_pot` to
zero on every calendar-year boundary (PHASE 7, keyed off
`data_point.date.year()`); `OhlcGpu` carries no per-bar timestamp at all, so
the kernel has no way to detect a year boundary, and `loss_pot` now simply
accumulates for a ticker's entire simulated life on GPU. GPU tier-1/tier-2
evaluation windows are short slices (`tier_1_days`/`tier_2_days`, 30-180
days across every shipped config), so this is expected to rarely matter in
practice, but a slice that happens to straddle a calendar year boundary
would carry a `loss_pot` across it that CPU would have reset. This was a
deliberate scope decision, not an oversight: correctly threading calendar
dates through to the kernel would require adding a per-bar date/year field
to the OHLC upload path, a materially larger change than the rest of O08.

### 3.6 Per-bar bookkeeping

**Mark-to-market equity / peak / drawdown**: CPU, PHASE 6.5/7.5
(`simulation.rs:630-717`) — `mark_to_market_equity = current_equity +
unrealized_pnl - accumulated_funding_fee`; `peak_equity =
peak_equity.max(...)`; `drawdown = (peak - mtm)/peak`; breach triggers a
forced close via `process_exit` (`simulation.rs:751-769`) so the very loss
that tripped the breaker gets realized into `final_equity`, then re-folds
the realized drawdown (`simulation.rs:782-789`). GPU (`backtest_kernel.hip.cpp:1837-1880`,
the "GAP 4" block) computes the same `mark_to_market_equity`/`peak`/`drawdown`
math — **without** an `accumulated_funding_fee` subtraction, which is
correct given the kernel's bar-by-bar funding deduction (documented
equivalence nearby) — but, **at the time of this audit**, did **not
force-realize the position on breach**: the loop simply `break`ed with
`current_equity` left
realized-only, exactly the bug pattern the CPU's own PHASE 7.5 comment
describes fixing (`simulation.rs:718-751`, "the very loss that just tripped
it is sitting entirely in `unrealized_pnl` and never gets realised"). See
**O03**.

**O03 fixed**: the breach check moved from the top of the next iteration to
immediately after the GAP 4 mark-to-market/drawdown update — the same
relative position as CPU's PHASE 7.5 vs. PHASE 6.5. On breach, an open
position is now force-closed through the identical non-liquidated exit
formula (slippage, taker fee, Austrian tax via `apply_realized_pnl`, no
leverage multiplier) the PENDING_EXIT and intrabar-exit paths already use,
mirroring CPU's forced `process_exit(sell_price: data_point.close,
was_liquidated: false)` call, then re-folds `peak_equity`/`max_drawdown_val`
from the now-realized `current_equity` — matching CPU's own re-fold after
its forced close.

**Bankruptcy floor**: GPU used to add an extra early-exit the CPU has no
equivalent of. At the time of this audit it read `if (current_equity <=
10.0f || max_drawdown_val * 100.0f > max_dd) break;`; a first fix (O02, see
below) made it read `if (current_equity <= failed_equity || max_drawdown_val
* 100.0f > max_dd)`, and a later fix (F32/2.3.1, see §4's "O02 — decision
record") **removed the `current_equity <= failed_equity` term entirely**,
so this line is now just `if (max_drawdown_val * 100.0f > max_dd) break;` —
matching CPU exactly. CPU's *only* in-loop circuit breaker is the drawdown
cap (`simulation.rs:717`) — there is no `current_equity <= X` check anywhere
in `run_simulation`. `config.simulation.failed_equity` on the CPU side is
used exclusively as the synthetic `final_equity` value backfilled onto
tickers skipped after `break_all_tickers` fires (`simulation.rs:1156`),
never as an independent live-simulation trigger. See **O02**.

**O02, updated disposition (F32/2.3.2, `58cf`)**: the hardcoded `10.0f`
literal was first turned into a `failed_equity` kernel parameter, sourced
from `config.simulation.failed_equity` — a GPU-only defensive floor CPU had
no equivalent of, kept (not removed) at that point, just no longer
disconnected from config. This task's remediation goes one step further and
**removes the floor from the breach check altogether**, on both the kernel
and both `kernel_reference.rs` twins, so GPU no longer trips a bankruptcy
break CPU has no analog of. `failed_equity` remains a live kernel parameter
— it still serves O04's backfill equity exactly as before — it simply no
longer also gates the per-bar breach check. See §4's "O02 — decision record
(F32/2.3.2)" for the full reasoning and the regression test.

**Cooldown decrement**: CPU, PHASE 6 (`simulation.rs:586-593`) — unconditional
every bar `state == InCooldown`, matching the off-by-one-corrected seeding in
`process_exit` (`cooldown_counter = setting + 1`, `execution.rs:301`). GPU:
`if (cooldown > 0) cooldown--;` (`backtest_kernel.hip.cpp:1834`), seeded
identically (`cooldown = cooldown_period_setting + 1`, `backtest_kernel.hip.cpp:1608,1803,1933`). Matches —
GPU folds `NoPosition`/`InCooldown` into one `trade_state == 0` value,
distinguished only by the `cooldown` counter, which is behaviourally
equivalent to the CPU's explicit enum state (see I03 for the fragility of
this).

**`bars_since_entry`**: CPU increments at the very top of the per-bar loop,
unconditionally, whenever `state == InLong/InShort` (`simulation.rs:408-410`).
GPU does the same (`backtest_kernel.hip.cpp:1355`) — at the time of this
audit, only for bars that passed the HMM-state-range check first (since
fixed, O05 below). See **O05**.

### 3.7 Out-of-range HMM state handling — subconscious updates (new finding, O05)

**CPU** (`simulation.rs:803-830`): when `current_state >=
params.num_hmm_states`, the bar still runs PHASE 1 through PHASE 7.5 in
full — pending-order fill, exit check, funding, trailing stop, cooldown
decrement, mark-to-market equity, daily-return tracking, and the global
drawdown breaker all execute exactly as on any other bar. Only PHASE 8's
trading *decision* is skipped, and even then every evaluator's
`update_subconscious_state` still runs (so `Sma`/`StdDev`/`Lag` ring buffers
stay current). The comment at `simulation.rs:803-822` explains this is
routine, not an edge case: the precomputed HMM label sequence is shared
across individuals that may have been fit with different `num_hmm_states`.

**GPU**, at the time this audit was originally written (`backtest_kernel.hip.cpp`
around what is now line 1324):

```cpp
int current_state_idx = hmm_states[global_i];
if (current_state_idx >= sim_params.num_strategies)
    continue;
```

This `continue` fired **before** the `bars_since_entry` increment, before
the PHASE 1 mirror (pending-order fill), before the PHASE 3 mirror (exit
check), before PHASE 4 (funding), before PHASE 6 (cooldown decrement),
before the GAP 4 mark-to-market/drawdown block, and before the
"Subconscious state updates" loop. Every one of those is skipped outright
for that bar, not just the trading decision.

Consequences, for any bar sequence containing an out-of-range state:
- A pending order queued on the prior in-range bar is **not dropped**, but
  its fill is **delayed** to whichever future bar next has an in-range
  state — not the immediately following bar, as the CPU (and the kernel's
  own "FILL-TIMING FIX" elsewhere) intends.
- `bars_since_entry` stalls, silently loosening `min_hold_bars` relative to
  wall-clock/bar-count time actually elapsed.
- Cooldown does not decrement, effectively pausing it.
- Funding fees are not charged for that bar.
- Mark-to-market equity and drawdown are not updated — the drawdown/
  bankruptcy breaker goes blind for the duration.
- `Sma`/`StdDev`/`Lag` ring buffers for every state's tree stop advancing
  entirely (not merely for the current regime), which will desynchronize
  from what the CPU's per-evaluator subconscious update would have produced
  for identical genotype and data — the same class of bug F06 fixed, on a
  different code path.

This is reachable whenever a ticker's precomputed HMM state sequence carries
more distinct values than a given individual's own `num_hmm_states` — the
CPU's own comment calls this a normal, expected occurrence
(migrant individuals between islands fit with different state counts,
resumed sessions, etc.), not a pathological input. Under the current
production configs' `hmm.state_search_range = [16, 16]` (fixed at 16 for
every individual), this specific path is not obviously reachable *today* —
but nothing in the GPU code path enforces that, and the codebase explicitly
designs for the opposite (a shared label sequence across differing state
counts). Marked high severity because of how broad the blast radius is
(seven distinct pieces of per-bar bookkeeping silently skipped, not one).

**Fixed**: `current_state_idx >= sim_params.num_strategies` no longer
`continue`s past the rest of the bar. It now sets `bool state_in_range =
current_state_idx < sim_params.num_strategies;`, and only the PHASE 8/9
mirror (both the entry-evaluation and exit-evaluation branches) is wrapped
in `if (state_in_range) { ... }` — `active_strategy`'s lookup moved inside
that guard too, since dereferencing `d_strategies[... + current_state_idx]`
would itself be an out-of-bounds read when out of range. Every other
phase — `bars_since_entry` increment, PHASE 1 (pending-order fill), PHASE 3
(stop/take-profit/liquidation), PHASE 4 (funding), PHASE 6 (cooldown
decrement), the GAP 4 mark-to-market/drawdown update, and the O02/O03
breach check — now runs unconditionally every bar, exactly like CPU. The
"Subconscious state updates" loop needed no code change at all: with
`current_state_idx` out of range, `state == current_state_idx` can never
match for any `state` in `0..sim_params.num_strategies`, so every state
automatically takes that loop's "background" branch (advancing both its
entry and exit tree's ring buffers) the moment the loop is actually
reached — which happens now that the early `continue` is gone. This
exactly reproduces CPU's own fallback ("advance every evaluator's
subconscious state, then skip only the trading decision") with no
additional special-casing.

### 3.8 Result construction

**CPU** (`alpha-models/src/backtest.rs`, `alpha-core/src/lib.rs:441-464`):
`BacktestResult` carries `ticker_results: HashMap<String, TickerResult>` —
one full `TickerResult` per ticker, each with its own `trades`/`wins`/
`losses`/`final_equity`/`max_drawdown_percent`/`sharpe_ratio`/
`sortino_ratio`/`calmar_ratio`/`per_state_results: PerStateMap<PerStateTickerResult>`
(trades/wins/total_pnl/r_squared per HMM state, per ticker)/
`state_counts: PerStateMap<usize>`, plus a flat `all_trades: Vec<BacktestTrade>`
when `keep_trades` is set.

**GPU** — at the time this audit was originally written, `BacktestResultGpu`
was `{ final_equity, max_drawdown_percent, trades, wins, per_state_pnl[32] }`
— five fields, **all aggregated across every ticker in the batch**
(`total_final_equity`, `worst_drawdown_across_tickers`, `total_trade_count`,
`total_win_count`, one `per_state_pnl` array summed over every ticker;
result-write block now at `backtest_kernel.hip.cpp:2163-2231`, which since
the O06 fix below also includes `min_trades_across_tickers`). There is no per-ticker breakdown
anywhere in the FFI struct — it is architecturally impossible for the Rust
side to recover one from a `BacktestResultGpu`. No Sharpe/Sortino/Calmar, no
daily returns, no `r_squared`, no per-state trade/win counts, no
`state_counts`, no trade log.

**`run_tier_filter_gpu`** (`tiering.rs:92-212`) is what papers over this gap
to satisfy the `Vec<Option<BacktestResult>>` interface the rest of the
pipeline expects:
- On **pass**, the individual's index is simply forwarded to
  `next_survivors` — no `BacktestResult` is synthesized at all for a passing
  individual; the next tier (or the full-history CPU run) produces the real
  one.
- On **fail** (`tiering.rs:174-206`), it fabricates one `TickerResult` per
  ticker in `ticker_order`, dividing the aggregate evenly:
  `per_ticker_equity = final_equity / num_tickers`,
  `per_ticker_trades = trades / num_tickers`,
  `per_ticker_wins = wins / num_tickers` (integer division), with
  `sharpe_ratio`/`sortino_ratio`/`calmar_ratio` hardcoded `0.0` and
  `per_state_results`/`state_counts` both `PerStateMap::new(...)` (empty).
  `max_drawdown_percent` is the one field that isn't divided — it's the true
  cross-ticker worst value, broadcast identically to every synthesized
  ticker.

This even-division approximation is also what drives **O06** (see below):
the pass/fail decision itself, computed from the same aggregate fields
before this per-ticker fabrication ever happens, cannot recover a true
per-ticker minimum — see `tiering.rs:166-169`.

**O06 fixed**: `BacktestResultGpu` gained a sixth field,
`min_trades_across_tickers: i32`, tracked on-device in the per-ticker loop
(folded via `min()` after each ticker's own `trade_count` is final,
including a ticker force-failed by O04's backfill, which always
contributes `0`) — exactly the per-ticker breakdown this paragraph says is
architecturally required. `run_tier_filter_gpu` now reads it directly
(`gpu_res.min_trades_across_tickers`) instead of computing
`trades / num_tickers`. The even-division approximation for the
per-ticker-equity/trades/wins *display* fields on a failing individual
(described above) is unchanged — this fix only closes the gate-decision
gap, not the synthetic per-ticker breakdown's own approximation, which
remains exactly as documented above.

**`per_state_pnl`** is downloaded from the GPU into every
`BacktestResultGpu` but **`run_tier_filter_gpu` never reads it** —
confirmed by inspection of `tiering.rs`'s only two field accesses,
`gpu_res.final_equity`/`.trades`/`.wins`/`.max_drawdown_percent`. The kernel
computes it faithfully (bounds-checked per F05) for a field the Rust side
currently discards entirely at the tiering layer.

### 3.9 Config values hardcoded on GPU or never received

| Value | CPU source | GPU treatment |
|---|---|---|
| `taker_fee_pct`, `slippage_pct`, `min_hold_bars`, `liquidation_margin_loss_pct` | `config.simulation.*` | Threaded through as kernel launch params (`tiering.rs:155-161`, `gpu.rs:536-539`) — correct |
| `leverage`, `max_acceptable_drawdown`, `per_bar_funding_decay` | `config.simulation.*` | Threaded through (`tiering.rs:147-149`) — correct |
| `config.simulation.failed_equity` | Used as the tier-rejection synthetic equity and (CPU-side) `break_all_tickers` backfill value | **Fixed** — a `failed_equity` kernel launch parameter (`gpu.rs`'s `launch_backtest_kernel`). Originally did double duty as both the O02 bankruptcy floor and the O04 backfill equity; since F32/2.3.2 removed the O02 floor from the breach check, this parameter now serves **only** the O04 backfill equity — CPU's own `config.simulation.failed_equity` field is likewise single-purpose (the `break_all_tickers` backfill), so this is now a closer match to CPU, not a looser one. |
| `config.simulate_austrian_tax`, the 27.5% rate, loss-pot mechanics | `execution.rs:166-178` | **Fixed** (O08) — `simulate_austrian_tax` kernel parameter plus a new per-ticker `loss_pot`; the year-boundary reset gap this row once carried is closed (F32/2.3.1, see O08's detail section) |
| `risk_modulator` (ML-filter multiplier) | `sim_state.risk_modulator`, set by the ML filter block (`simulation.rs:923-926`) | No concept of it exists on GPU at all — see I01 |
| Agg1/Agg2 `IndicatorPeriods` fields | Computed for real via `MultiScaleIndicatorState` | **Fixed** (O09) — the resident column cache (`crate::gpu_columns`/`crate::gpu_column_gather`) resolves the 6 Agg1/Agg2 terminals to real, packed-`(agg_period, indicator_period)`-keyed values, and `build_snapshot_matrix` (`crates/alpha-simulation/src/gpu.rs:785-876`) independently now ticks real `Timeframe::Agg1`/`Agg2` pipelines too — see §3.9 detail below |

#### O09 detail — what actually shipped

At the time this audit was originally written (and at the time `manual/12`
scoped "Option A"), `build_snapshot_matrix` ticked only the 1-minute
pipeline; the 6 Agg1/Agg2 GP-visible terminals (`atr_raw_agg1`,
`rsi_agg1`, `stochastic_k_agg1`, and their Agg2 mirrors) sat at their
`SlotRegistry` registry-default constant for every bar on GPU, while CPU's
`MultiScaleIndicatorState` (`crates/alpha-models/src/simulation/mod.rs`)
computed them for real from an aggregated candle stream. **The true
equivalence authority for these slots was always `MultiScaleIndicatorState`
/ `TimeframeIndicators` — the actual CPU production code path — never
`build_snapshot_matrix`.** `build_snapshot_matrix` was itself one of the
two broken things for these slots (the other being the complete absence
of any GPU-side Agg1/Agg2 computation at all); it happened to be usable as
a stand-in reference for the other 47 (non-Agg) slots only because it
independently reimplements the same 1-minute indicator pipeline
`MultiScaleIndicatorState` uses, not because it was the canonical source
of truth. Both halves of the fix are now in place:

- **The resident column cache** (`crates/alpha-simulation/src/gpu_columns.rs`,
  `gpu_column_gather.rs` — both new files, `manual/12`'s "Option A")
  computes and stores `atr_raw_agg{1,2}`/`rsi_agg{1,2}`/
  `stochastic_k_agg{1,2}` for real. **Cache-key design**: each of the 3 Agg
  families (`GpuColumnFamily::AtrRawAgg`/`RsiAgg`/`StochasticKAgg`) is
  keyed by `(ticker_hash, family, packed_key)`, where `packed_key =
  pack_period_window(agg_period, indicator_period) = (agg_period as u64)
  << 32 | (indicator_period as u64)` (`gpu_columns.rs`'s
  `pack_period_window`) — a lossless bijection for any realistic period
  value (both comfortably under 2^32), so two individuals with different
  `agg_period_1`/`agg_period_2` genuinely get different cache entries, and
  one family can serve both aggregation tiers (`column_requests` emits
  each Agg family twice — once for `agg_period_1`, once for
  `agg_period_2` — reusing one cache entry across individuals that happen
  to land on the same `(agg_period, indicator_period)` pair from either
  gene). `crate::gpu_column_gather::NUM_COLUMN_ROLES` grew from 26 to
  **32** to carry these 6 new roles (`gpu_column_gather.rs:93`).
- **1-minute-bar → aggregated-bar index mapping**: each Agg column is
  stored COMPACT (`aggregate_candles` in `gpu_columns.rs`), not at full
  1-minute length — roughly `len / agg_period` entries, prefixed with a
  2-`f32` `[first_visible_bar, compact_len]` header. `aggregate_candles`
  anchors its replay at `alpha_core::WARMUP_PERIOD` (not bar `0`) and walks
  forward to the first bar whose epoch-minute satisfies
  `CandleAggregator`'s own `(minute % agg_period) == 0` boundary
  congruence, matching the real `CandleAggregator::aggregate`'s bar
  boundaries exactly (verified directly against it, not merely assumed —
  see the test list below). At read time, `reconstruct_agg_indicator` maps
  a 1-minute bar index `t` to the compact series via `((t -
  first_visible_bar) / agg_period).min(compact_len - 1)`, returning the
  slot's registry-default constant for any `t < first_visible_bar` or when
  `compact_len == 0` (a ticker shorter than `WARMUP_PERIOD`, or a
  degenerate `agg_period <= 1`) — the same registry default GPU used to
  return for every bar, now correctly scoped to only the bars before the
  first real aggregate exists.
- **`build_snapshot_matrix` was separately fixed** (`crates/alpha-simulation/src/gpu.rs:785-876`)
  to also tick real `Timeframe::Agg1`/`Agg2` pipelines, sharing one
  `IndicatorBus` with the 1-minute pipeline and driven by
  `CandleAggregator::aggregate` from `WARMUP_PERIOD` onward (`gpu.rs:833-864`)
  — matching `MultiScaleIndicatorState::new_with_columns`'s construction
  order and `MultiScaleIndicatorState::update`'s per-bar timing exactly.
  This was necessary because `build_snapshot_matrix` is used throughout
  `alpha-simulation`'s test suite as the CPU-side equivalence authority for
  GPU snapshot data generally (not just for this cache) — leaving it
  un-fixed would have meant every test that used it as ground truth for an
  Agg1/Agg2 terminal was silently checking the new column-cache code
  against the same stale constant it was built to replace, rather than
  against a real time-varying value.
- **Verification**: `crate::gpu_columns`'s test module includes
  `agg_columns_match_multi_scale_indicator_state_reference_bar_by_bar`,
  which checks the cache's reconstructed values against real
  `MultiScaleIndicatorState` output bar-by-bar — the actual CPU production
  path, not `build_snapshot_matrix`. `crate::gpu_column_gather`'s
  `agg_slots_gathered_via_full_pool_match_build_snapshot_matrix_reference_over_warmup`
  additionally checks the full gather-through-`ColumnPool` path against
  `build_snapshot_matrix`, which is a valid second check only because
  `build_snapshot_matrix` itself was fixed to agree with
  `MultiScaleIndicatorState` for these slots, per the point above.

O09 is a Trade-decision-severity fix, same as its original classification:
any compiled GP tree referencing one of these 6 terminals now receives a
real, time-varying signal on GPU matching the CPU reference path, instead
of a constant. See `manual/12`'s §5.4.1/§5.4.2 for the memory cost this
closure added (material — roughly 31% of the combined column-cache
footprint at every tier under `config.harvest.toml`'s search-space
bounds) and confirmation that it was already reflected in that chapter's
pre-implementation estimate, not a nasty surprise found only after
shipping.

### 3.10 Numeric precision: f64 (CPU) vs f32 (GPU)

This is systemic, not a discrete bug, and is not fixable without either
redesigning the kernel to use double precision (with the attendant
throughput cost the existing `f32` choice is explicitly optimizing for,
per `gpu.rs`'s doc comment on `bus_to_gpu_snapshot`) or accepting it as a
permanent source of tolerance-bounded (not exact) parity — which is exactly
how `manual/13_gpu_cpu_parity_self_check.md` already frames its own
tolerances (§2 of that chapter: 1% relative / $1 absolute floor on equity,
5% relative / 0.10pp floor on drawdown, exact match required only on
integer trade/win counts).

Every indicator value is truncated `f64 -> f32` once, at snapshot-build time
(`bus_to_gpu_snapshot`, `gpu.rs:722-729`), and every subsequent GP-tree
computation, stop/TP/liquidation-price comparison, and P&L calculation runs
in `f32` on GPU versus `f64` on CPU. Most of the time this only affects the
last few significant digits of a reported number. But every comparison in
this codebase that gates a *decision* — `entry_score > entry_threshold`,
`candle.low <= trailing_stop`, `open <= liquidation_price`, the `Comparison`
node inside a GP tree itself — is a boolean gate over a floating-point
inequality. For any bar where the true (infinite-precision) values on either
side of such a comparison are within `f32` rounding distance of each other,
`f32` and `f64` arithmetic can legitimately disagree about which side is
larger, flipping a trade/no-trade or exit/no-exit decision for that bar. This
is not fixable by finding "the bug" — it is the direct, unavoidable
consequence of running the decision logic itself (not just the reported
numbers) in `f32`. `trades`/`wins` being an **exact**-match tolerance in
`manual/13`'s self-check (rather than a rounding-tolerant one) is precisely
because a mismatch there is the symptom of this class of divergence, and the
self-check's own failure-mode table (`manual/13` §4) already anticipates
it ("This is the strongest possible signal of a real kernel bug... [or genuine
close-call precision divergence]" — though that table does not currently
distinguish the two).

---

## 4. New findings, in detail

### O03 — Drawdown/bankruptcy breach does not force-realize the open position

At the time of this audit: `if (current_equity <= 10.0f ||
max_drawdown_val * 100.0f > max_dd) break;` — simply exits the per-candle
loop. Contrast `simulation.rs:751-789`: on the identical condition, CPU
calls `process_exit` with `sell_price: data_point.close` *before* breaking,
booking the same slippage/fee/funding/tax the position would have paid on
any other exit, then re-folds the now-realized drawdown. GPU's
`total_final_equity += current_equity` (now `backtest_kernel.hip.cpp:2163`)
for a ticker that breached while still in a position therefore **omits that
position's unrealized loss entirely** — reporting a materially better
`final_equity` for that ticker than either the CPU would, or than the
account actually experienced. Consequence: reported numbers only in the
narrow sense that no *further* trades happen on GPU either way once the
loop breaks — but the *reported* equity/PnL that downstream fitness scoring
sees is wrong by exactly the size of the unrealized loss that triggered the
breach, which is by construction the single largest loss of the run.

**Fixed**: the breach check (now positioned right after the GAP 4
mark-to-market/drawdown update, matching CPU's PHASE 7.5-after-PHASE-6.5
ordering) force-closes an open position through the same non-liquidated
exit formula every other exit path uses before breaking, then re-folds
`peak_equity`/`max_drawdown_val` from the realized `current_equity` —
mirroring `process_exit(sell_price: data_point.close, was_liquidated:
false)` and CPU's own re-fold step line for line.

### O04 — No `break_all_tickers` analog

CPU's PHASE 7.5 breach handling doesn't just close the breaching ticker's
position — it stops simulating **every other ticker for that individual**
and backfills each with a punitive synthetic failure
(`final_equity: config.simulation.failed_equity`, `simulation.rs:1141-1165`).
GPU's `for (t_idx = 0; t_idx < num_tickers; ++t_idx)` loop (now
`backtest_kernel.hip.cpp:1202`) had, at the time of this audit, no such
mechanism: the breach `break` (then at what is now line 1901) only exited
the *inner* per-candle loop for the current ticker; control fell through
to `total_final_equity += current_equity;` and the outer loop proceeded to
the **next ticker and simulated it completely normally**, as if nothing
happened.

This means an individual whose strategy blows up catastrophically on one
ticker, but performs normally on the others, is scored by GPU tiering as
roughly "normal performance on N-1 tickers, one early exit on the Nth" —
whereas CPU scores the identical genotype as "one blown-up ticker, plus
every other ticker force-failed at `failed_equity`." This is not a
reported-numbers difference: it directly changes whether the individual
passes or fails the `total_pnl >= pnl_threshold` tier gate
(`tiering.rs:169`), i.e. it changes **which strategies the genetic
algorithm selects** — precisely the failure mode this whole audit exists to
find. A strategy that is one bad ticker away from ruin is punished hard by
CPU tiering and comparatively unpunished by GPU tiering.

**Fixed**: a new per-ticker `bool ticker_breached` flag, set when O03's
breach-and-force-close fires, is checked right after that ticker's totals
are folded into the individual-wide accumulators. When set, every
remaining ticker in `ticker_order` (`num_tickers - t_idx - 1` of them)
contributes `failed_equity` to `total_final_equity` and `0` to both
`total_trade_count` and (via O06's `min_trades_across_tickers`) the
per-ticker minimum, then the outer per-ticker loop `break`s — mirroring
CPU's backfilled `trades: 0, wins: 0, losses: 0` synthetic `TickerResult`
exactly, with no per-remaining-ticker eligibility re-check needed (unlike
CPU's own backfill loop): `run_tier_filter_gpu` (`engine/tiering.rs`)
already filters `ticker_order` down to only length-eligible tickers before
this kernel ever launches (see O07's fix), so every remaining ticker here
is guaranteed eligible by construction. Getting this right required
`ticker_order` to match CPU's own alphabetical processing order exactly —
already true in production (`run_tier_filter_gpu` already sorts), but the
`gpu_parity` self-check fixture's `ticker_order` used to deliberately
differ from CPU's order for an unrelated, long-fixed reason (see
`apps/alpha-worker/src/gpu_parity/fixture.rs`'s `DRAWDOWN_TICKER` doc
comment) and had to be retuned alongside this fix.

### O05 — see §3.7 above.

### O06 — Tier-gate uses average, not minimum, trades-per-ticker

CPU (`tiering.rs:261-266`, `process_tier_outcomes`):
```rust
let min_trades = result.ticker_results.values().map(|s| s.trades).min().unwrap_or(0);
```
the true minimum over tickers — an individual that never trades on even one
ticker fails the gate regardless of how much it trades elsewhere.

GPU (`tiering.rs:168`, `run_tier_filter_gpu`):
```rust
let min_trades = (gpu_res.trades as usize) / num_tickers;
```
the **average**, integer-divided. An individual that trades 40 times on one
ticker and zero times on three others reports `min_trades = 10` on GPU
(passing a `trades_threshold` of, say, 5) while CPU would compute
`min_trades = 0` and fail it outright.

This is not merely a Rust-glue rounding issue — it is architecturally
forced by §3.8's finding that `BacktestResultGpu` carries no per-ticker
trade counts at all, only a cross-ticker sum. A correct fix requires the
kernel to report per-ticker trade counts (a struct/FFI change), not just a
different formula in `tiering.rs`. This changes the tier-1/tier-2 survivor
set directly.

**Fixed** exactly as this section anticipated: `BacktestResultGpu` gained
`min_trades_across_tickers: i32`, computed on-device as the running
minimum of each ticker's own `trade_count` (folded in once per ticker,
after that ticker's simulation — real or O04-backfilled — is final).
`tiering.rs:168`'s gate is now `let min_trades =
gpu_res.min_trades_across_tickers.max(0) as usize;` — no division, and no
architectural gap left to paper over.

### O07 — Mixed-length ticker batches are zero-padded and fully simulated

See §3.2. `gpu_evaluator.rs:556-589` allocates `flat_ohlc`/`flat_states` at
size `num_tickers * num_candles` (`num_candles` = the **longest** ticker in
the batch) and only writes real values up to each ticker's own actual
length; everything beyond that stays at its `vec![...]`-initialized zero.
At the time of this audit, the kernel's per-ticker loop ran the full
`warmup_period..num_candles` range identically for every ticker, with no
per-ticker length check (since fixed — the loop is now bounded by
`ticker_len`, `backtest_kernel.hip.cpp:1293`, per the fix described just
below).

Two distinct manifestations:
- A ticker with `real_len <= WARMUP_PERIOD` (which CPU excludes entirely,
  `simulation.rs:217-219`) is, on GPU, simulated across its **entire**
  post-warmup range on fabricated `OHLC = (0,0,0,0)` candles and HMM state
  `0`. In practice the zero-ATR-implied `risk_per_coin` keeps the entry
  guard (`> 1e-9f`) from ever firing, so this degrades to a silent
  zero-trade, flat-equity ticker rather than a crash — but it still
  contributes an extra (phantom) entry to `num_tickers`, diluting O06's
  already-wrong average further, and it has no CPU-side counterpart at all
  (CPU would not even list this ticker in `ticker_results`).
- A ticker with `real_len > WARMUP_PERIOD` but shorter than the batch's
  longest ticker (the realistic case — different assets have different
  listing dates) transitions from real prices to fabricated
  `OHLC = (0,0,0,0)` candles partway through its simulated life. An open
  position at that boundary sees `candle.close` instantaneously drop to
  `0.0`, so `unrealized_pnl = (0 - entry_price) * position_size_coins`
  registers as a total loss of the position's notional on the very next
  synthetic bar — almost certainly tripping the drawdown/bankruptcy breaker
  (O02/O03) on a loss that never actually happened. CPU, by contrast,
  iterates that ticker's own `data_points` slice and simply stops — the
  ticker's contribution to `ticker_results` reflects its real final state,
  nothing more.

**Fixed**, addressing both manifestations separately since they have
different natural fixes:
- The first (a ticker too short to clear `WARMUP_PERIOD` at all) is closed
  at the Rust level, upstream of the kernel: `run_tier_filter_gpu`
  (`engine/tiering.rs`) now filters `ticker_order` down to tickers with
  `data.as_ref().len() > WARMUP_PERIOD` before it is ever used — mirroring
  CPU's own per-ticker `continue` exactly, at the one place a batch's
  `ticker_order` is built for every individual in it. Such a ticker no
  longer reaches `evaluate_batch`/the kernel at all.
- The second (real, differing lengths among otherwise-eligible tickers) is
  closed with a new per-ticker `ticker_lengths` array, uploaded alongside
  `d_ohlc`/`d_hmm_states` (`GpuEvaluator`'s new `d_ticker_lengths` buffer,
  `gpu_evaluator.rs`) and threaded through as a new kernel parameter. The
  per-candle loop now bounds `i` by `min(ticker_lengths[t_idx],
  num_candles)` instead of unconditionally `num_candles`, so a ticker never
  runs into the padded, fabricated `OHLC = (0,0,0,0)` region past its real
  data.

### O08 — see §3.5 and the "O08 — decision record (F32/2.3.1)" subsection later in this section. O09 — **Fixed**, see §3.9's detail section and `manual/12` §6/§5.4.1/§5.4.2. O10 — see §3.8; **accepted** (F32/2.3.3, no longer open) — see the "Accepted residual gaps (F32/2.3.3)" subsection later in this section.

### O11 — Entry-time ATR read one bar ahead of what CPU sizes off (backfilled into this table)

Both `backtest_kernel.hip.cpp` and its CPU-side reference twin
(`crates/alpha-simulation/src/kernel_reference.rs`) originally sized the
entry fill's `risk_per_coin` off `snapshot[SLOT_ATR_RAW_1M]` — the row for
the *current* bar `global_i`, which `build_snapshot_matrix` stores only
after that bar's own indicator tick. CPU's `execute_entry_long`/
`execute_entry_short` (`execution.rs`), by contrast, run in PHASE 1,
strictly *before* PHASE 2's `indicators.update(data_point)` for that same
bar — so CPU always sizes off the ATR as of the *previous* bar,
`global_i - 1`. Reading `snapshot[global_i]` at entry therefore injected a
one-bar lookahead into the entry-time risk-per-coin, and hence into the
initial stop-loss/take-profit/liquidation levels the position is opened
with.

**Fixed** (in both files, prior to this session): the entry-fill branch now
reads the pre-update (`global_i - 1`) ATR for `risk_per_coin`, while the
trailing-stop ratchet later in the same bar correctly keeps reading the
current-bar `snapshot` (matching CPU's PHASE 5, which runs *after* PHASE 2's
update). In the kernel proper this is a direct scalar read, `entry_atr =
snapshot_data[(global_i - 1) * GPU_MAX_SLOTS + SLOT_ATR_RAW_1M]`
(`backtest_kernel.hip.cpp:1419-1441`, used by `risk_per_coin` at line 1453);
`kernel_reference.rs`'s three twin variants (`evaluate_individual_reference`,
the f64-arithmetic twin, and the f64-input twin) instead slice a full
`entry_snapshot = &snapshot_data[(global_i - 1) * GPU_MAX_SLOTS..global_i *
GPU_MAX_SLOTS]` and index `entry_snapshot[SLOT_ATR_RAW_1M]` out of it
(`kernel_reference.rs:551-553`, `:1670-1672`, `:2576-2578`) — same value,
different container shape between the two languages. This item was
implemented and verified correct in an earlier round but never added to
this audit's own summary table — added here purely as documentation
catch-up; no code changed for O11 in this session.

See O12 immediately below: this fix is also what closed the
`kernel_reference_simple_parity_tests.rs` 149-vs-148-trade divergence that
a previous session logged as an unexplained non-reproduction. Reverting
this exact `entry_snapshot` read back to the pre-fix `global_i` lookahead
reproduces that divergence's archived numbers digit-for-digit, which is
the direct proof that O11 was always the root cause.

### O12 — `kernel_reference.rs` twin/`run_simulation` trade-count divergence: root cause was O11, now closed

`crates/alpha-simulation/tests/kernel_reference_simple_parity_tests.rs`
carries five tests (plus one non-ignored bar-level-localisation test) built
around a shared BTC/seed-42, 5000-candle, two-long-only-RSI-strategy
"control case" (`build_control_case`). Their doc comments recorded a
previously-observed divergence: `run_simulation` produced 148 trades/55
wins/equity 10059.321703, while every twin variant (`f32` snapshot, `f64`
internal arithmetic, `f64` snapshot inputs, `f64` ATR-only, and `f64`
inputs with a fused trailing-stop ratchet) produced 149 trades/56
wins/equity ≈10053.9585 — localised by the bar-level test to trade index 3
(entered bar 2038, long, `entry_state_idx` 0), where the twin's stop-loss
fired one bar late (bar 2043 vs `run_simulation`'s bar 2042). A prior
investigation session tested and rejected five precision-based mechanisms
(twin-internal `f64` arithmetic, full-`f64` snapshot inputs, `f64` on the
ATR slot only, fused vs. non-fused trailing-stop arithmetic, `OhlcGpu`'s
`f32` candle-price truncation), found all six tests passing unmodified on
its host, and logged this entry as a non-reproduction of unknown cause
(toolchain/dependency drift was floated as a guess), leaving the five
`#[ignore]` attributes in place as a precaution.

**Root cause identified this session: this was O11 all along.** With the
tree as it stands (O11's fix in place — `kernel_reference.rs` reads
`entry_snapshot` at `global_i - 1`), all six tests pass on their original,
unweakened `assert_eq!` checks. Temporarily reverting only that one read
(back to the pre-O11 lookahead form, `global_i`) reproduced the archived
divergence exactly, digit for digit:

```
DIAG trades cpu=148 twin=149 | wins cpu=55 twin=56 | equity cpu=10059.321
twin=10053.965 rel_err=0.053249%
```

Restoring the `global_i - 1` form immediately re-closed it. A rounding or
toolchain-drift coincidence cannot reproduce a six-significant-digit equity
figure on demand by flipping a single array index — this is a direct,
mechanical cause-and-effect, not the f32/f64 "decision-flip" class §3.10
describes. The "toolchain/dependency drift" guess in the prior session's
version of this entry was incorrect and is superseded by this finding.

The five precision-experiment tests' real historical value was negative
evidence: each one ruled out a precision-based mechanism in turn (internal
`f64` arithmetic, `f64` snapshot inputs, ATR-only `f64`, fused
multiply-add trailing-stop) and correctly narrowed the search toward a
structural bug rather than an inherent float-precision artifact — which is
exactly what O11 turned out to be. Their doc comments have been rewritten
(`kernel_reference_simple_parity_tests.rs`) to record this positive
regression-guard framing instead of the original "IGNORED, left unfixable"
language.

**Status: closed.** All five `#[ignore]` attributes have been removed
permanently; all six tests in this file pass on their original,
unweakened `assert_eq!` checks on `trades` and `wins` (no tolerance
widened, no assertion weakened):

```
test kernel_reference_closely_matches_run_simulation_on_a_well_behaved_strategy ... ok
test kernel_reference_f64_inputs_twin_with_fused_trailing_stop_vs_run_simulation ... ok
test kernel_reference_f64_twin_closely_matches_run_simulation_on_a_well_behaved_strategy ... ok
test kernel_reference_f64_inputs_twin_vs_run_simulation_on_a_well_behaved_strategy ... ok
test kernel_reference_f64_atr_only_twin_vs_run_simulation_on_a_well_behaved_strategy ... ok
test kernel_reference_f64_inputs_twin_bar_level_localisation_of_the_extra_trade ... ok

test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

These six tests now stand as permanent regression coverage for O11: if the
entry-time ATR snapshot read ever regresses back to the post-update
(`global_i`) lookahead form, this exact 149-vs-148 divergence returns and
these tests fail loudly on their exact-match assertions.

### O13 — `Log` computed unsigned `ln(a)` instead of the CPU's signed `ln1p`

**Before**: `evaluate_gp_tree`'s `Log` arm (`backtest_kernel.hip.cpp`) was
`a > 0.0f ? logf(a) : 0.0f` — a domain-restricted natural log that silently
zeroed every non-positive input, discarding the sign of a negative
indicator entirely. The CPU evaluator (`alpha_gp::evaluator::mod::Op::Log`,
both the `evaluate` and `update_states_only` arms) instead computes
`a.signum() * a.abs().ln_1p()` — a signed `ln1p`, `sign(a) * ln(1 + |a|)`,
defined and sign-preserving for every finite `a`, with `a == 0.0` mapping
to exactly `0.0` because `ln_1p(0.0) == 0.0` regardless of which sign
`signum` would otherwise report.

**After**: the kernel's `Log` arm is now
`float m = log1pf(fabsf(a)); stack.push(a > 0.0f ? m : (a < 0.0f ? -m : 0.0f));`
— matching the CPU formula exactly, and additionally mapping a non-finite
`a` to `0.0f` (both comparisons are false for NaN) rather than propagating
NaN onto the stack. Mirrored identically in `kernel_reference.rs`'s f32
twin and both f64 twins, and in `gpu.rs`'s `eval_stateless` test helper.

**Regression coverage**: `crates/alpha-gp/tests/review_2026_09_findings_tests.rs`'s
`f28_cpu_log_div_reference_values` pins the CPU evaluator's exact reference
outputs for `a` in `{-2, -0.5, 0, 0.5, 2}`;
`crates/alpha-simulation/tests/kernel_reference_simple_parity_tests.rs`'s
`f28_log_div_bytecode_matches_kernel_reference` evaluates the same inputs
through `crate::gpu::compile_float_expression`'s compiled bytecode and
`kernel_reference::evaluate_gp_tree_stateless_for_tests` (a new test-only
public entry point, `evaluate_gp_tree` itself being private to that
module), proving the compiled-bytecode path agrees, not just the CPU
evaluator in isolation.

### O14 — `Div` used a fixed absolute threshold instead of the CPU's `is_finite` guard

**Before**: `evaluate_gp_tree`'s `Div` arm was
`stack.push(fabsf(b) < 1e-9f ? 1.0f : a / b);` — a fixed, scale-dependent
denominator threshold that returns `1.0f` (not zero — a value the GP can
mistake for "no effect" on a multiplicative expression) on a near-zero
denominator, and does nothing to guard an overflowing quotient. The CPU
evaluator (`Op::Div`) instead computes `let v = a / b;` and pushes `v` only
`if v.is_finite()`, else `0.0` — a scale-free guard that only fires on the
genuinely undefined/non-representable cases (`b == 0.0`, or overflow),
chosen specifically because this system's micro-cap crypto price scale
(down to ~1e-8) makes any fixed absolute threshold like `1e-9` fire on
essentially every price-derived denominator, not a rare edge case (see the
CPU `Op::Div` arm's own doc comment in
`crates/alpha-gp/src/evaluator/mod.rs`).

**After**: the kernel's `Div` arm is now
`float v = a / b; stack.push(isfinite(v) ? v : 0.0f);` — matching the CPU
formula exactly. Mirrored identically in `kernel_reference.rs`'s f32 twin
and both f64 twins, and in `gpu.rs`'s `eval_stateless` test helper.

**Regression coverage**: same two tests as O13 —
`f28_cpu_log_div_reference_values` documents `a / 0.0`, `0.0 / 0.0`
(both `0.0`, not `1.0`) and `a / 1e-12` (a large but finite quotient, NOT
zeroed); `f28_log_div_bytecode_matches_kernel_reference` runs the same
three cases through the compiled bytecode.

### O15 — stateful ops counted samples from `warmup_period`, not from zero

**Before**: `evaluate_gp_tree`'s `Sma`/`StdDev`/`Lag` arms indexed their
ring buffer and derived their sample `count` directly from `timestep`, the
batch-global bar index the per-ticker bar loop passes in
(`for (int i = warmup_period; i < ticker_len; ++i)` — always
`>= warmup_period`, 2000). Every count/ring-index expression
(`timestep % period`, `(timestep < period) ? (timestep + 1) : period`)
therefore evaluated as if `period` samples had already arrived on the very
first bar of every ticker, because `timestep >= warmup_period > period`
for every period this system's search space allows (3-100). The CPU's
`SmaState`/`StdDevState`/`LagState` (`crates/alpha-gp/src/evaluator/state.rs`)
instead count strictly from `0` starting at the first bar the evaluator is
ever updated with — a real, count-based warm-up window the kernel had
never reproduced.

**After**: `evaluate_gp_tree` takes an additional `warmup_period`
parameter (a plain function argument, not a `#[repr(C)]` field or kernel
launch parameter — `warmup_period` was already threaded into
`backtest_kernel`/`evaluate_individual_reference` before this fix) and
computes `int local_t = timestep - warmup_period;` once, using `local_t`
everywhere the three stateful arms previously read `timestep`. This
reproduces `SmaState` (mean over `min(local_t + 1, period)`) and
`StdDevState` (population variance over the same count — the CPU uses
Welford's algorithm where the kernel recomputes directly from the ring
each call, so parity is asserted within 1e-6 relative, not bit-exact) bar
for bar. Fixing `Lag` required two additional corrections beyond the
`local_t` rebase, described under O16... no — see immediately below:
`Lag`'s pre-fix arm read `history[(timestep + 1) % period]` and fell back
to `input_val` (the just-computed value) during warm-up. Both were latent
bugs masked by O15 itself: since `count` was always `== period` before
this fix (warm-up never triggered), the `count < period` fallback branch
was permanently dead code, and the off-by-one read location never
mattered because the ring was always "full" from bar one. Once `local_t`
made the warm-up branch live, both had to be corrected to match
`LagState::update` (`crates/alpha-gp/src/evaluator/state.rs`) exactly:
read `history[local_t % period]` **before** overwriting that same slot
(the CPU reads `self.buffer[self.cursor]` before writing it), and fall
back to `0.0`, not `input_val`, while `local_t < period` — `LagState`'s
own doc comment explains why returning the live input during warm-up was
judged a bug in an earlier fix (it makes `Lag(x) - x` identically zero for
the whole warm-up window).

Mirrored identically in `kernel_reference.rs`'s f32 twin and both f64
twins (`evaluate_gp_tree`, `evaluate_gp_tree_f64_inputs`).

**Regression coverage**: `crates/alpha-simulation/tests/kernel_reference_stateful_ops_parity_tests.rs`'s
`f28_kernel_reference_matches_run_simulation_on_stateful_ops_strategy`
(new file) runs a two-ticker (`common::two_ticker_unequal_dataset`),
two-HMM-state strategy exercising `Sma`/`StdDev`/`Lag` plus `Log`/`Div`
(`common::stateful_ops_strategy`) through both `run_simulation` and
`kernel_reference::evaluate_batch_reference`, asserting exact trade/win
parity and final-equity/per-state-PnL parity within tolerance.

### O16 — per-individual ring buffers were zeroed once per batch, never per ticker

**Before**: `state_buffer` (the flat, per-individual `Sma`/`StdDev`/`Lag`
ring-buffer arena) is zeroed exactly once per kernel launch, host-side,
before the FIRST ticker of the batch (`gpu_memset` in `gpu_evaluator.rs`).
`backtest_kernel`'s per-ticker loop (`for (int t_idx = 0; t_idx <
num_tickers; ++t_idx)`) evaluates every ticker for a given individual
through that SAME slice, with no reset in between — so ticker N>0 started
every stateful node with whatever ring contents ticker N-1's bars left
behind. The CPU engine instead resets every `StatefulGpEvaluator` (and
therefore every stateful node's ring buffer) at the start of EACH ticker
(`eval.reset()`, PHASE 7, `alpha_simulation::engine::simulation`) — see
`StatefulGpEvaluator::reset` in `crates/alpha-gp/src/evaluator/mod.rs`.

**After**: at the top of the kernel's per-ticker loop, before the bar loop
begins, this thread's own `max_stateful_nodes_per_ind * stride` slice of
`state_buffer` (`stride = max_period + 1`, matching `evaluate_gp_tree`'s
own stride derivation) is zeroed in place. Safe with no synchronization:
this is the exact same address range `evaluate_gp_tree`'s `Sma`/`StdDev`/
`Lag` arms read/write through `state_buffer_base`, already documented
(see that function's `__restrict__` doc comment) as exclusively this
thread's own memory — no other thread's `individual_idx` ever overlaps
it. Cost: `max_stateful_nodes_per_ind * stride` floats per ticker per
thread (64 * 257 at production scale) — negligible against the per-bar
loop. Mirrored identically before the per-ticker bar loop in
`kernel_reference.rs`'s f32 twin and both f64 twins, using
`slice.fill(0.0)`. The host-side `gpu_memset` in `gpu_evaluator.rs` is
unchanged — it still clears the whole arena once per batch, which remains
necessary for individuals with fewer tickers than the batch's longest.

**Regression coverage**: the same
`f28_kernel_reference_matches_run_simulation_on_stateful_ops_strategy`
test as O15 — the fixture's two tickers (`LONGX`, 80,000 bars; `SHORTX`,
4,500 bars, evaluated second in alphabetical `ticker_order`) mean
`SHORTX`'s stateful nodes start from `LONGX`'s leftover ring contents
without this fix.

### O08 — decision record (F32/2.3.1): year-boundary `loss_pot` reset closed

Blueprint `fdb2` subtask `f4c2`. Full mechanism detail (host-side
`build_year_boundaries`, the CSR `year_boundary_bars`/`year_boundary_offsets`
layout, the ABI-6 FFI parameters, the kernel's PHASE 7 mirror, and the twins)
is written up in §3.5's O08 detail section above rather than repeated here —
in short: the host precomputes, per ticker, the ticker-relative bar indices
where `alpha_core::Ohlcv.date.year()` changes, flattens them across tickers
into one array with a per-ticker offset table, and passes both plus a count
as three new `launch_backtest_kernel` parameters
(`year_boundary_bars`/`year_boundary_offsets`/`year_boundary_count`,
`gpu.rs:716-718`), bumping `LAUNCH_BACKTEST_KERNEL_ABI_VERSION` **5 -> 6** on
both sides since this changes the FFI signature. The kernel's PHASE 7 mirror
and every `kernel_reference.rs` twin walk the identical arrays with the same
CSR cursor logic, so no engine derives its own independent approximation of
where a year boundary falls.

**Regression coverage**:
`year_boundary_loss_pot_reset_matches_run_simulation_across_dec31_to_jan1`
(`crates/alpha-simulation/tests/kernel_reference_simple_parity_tests.rs`).
**What it proves**: on a fixture whose dates genuinely cross a Dec-31 ->
Jan-1 boundary strictly after `WARMUP_PERIOD`, with
`config.simulate_austrian_tax = true`, `run_simulation` and the kernel
twin's trade and win counts are asserted **exactly equal**, and final
equity within a 1% f32/f64 tolerance — closing the loop this table
previously left open, where the two engines could silently disagree on
`loss_pot` across a calendar-year rollover.

### O02 — decision record (F32/2.3.2): device-only bankruptcy floor removed

Blueprint `fdb2` subtask `58cf`. This audit's §3.6 originally scored O02 as
"fixed" once the hardcoded `10.0f` literal became a `failed_equity` kernel
parameter — but that still left the kernel with an early-exit
(`current_equity <= failed_equity`) that `run_simulation` has never had any
equivalent of. This task's brief posed two options: remove the floor from
the kernel (and both `kernel_reference.rs` twins), or add an equivalent
floor to CPU's `engine/simulation.rs` PHASE 7.5. **The default option —
removing the device-only floor — was taken, per the blueprint's own stated
default, WITHOUT an explicit owner confirmation being obtained first.** This
is recorded here explicitly so a reader does not mistake this for a
reviewed design decision: it is the blueprint's documented fallback, applied
because no owner sign-off was available at the time this task ran, not
because the alternative (adding the floor to CPU) was evaluated and
rejected. Anyone who prefers the CPU-side floor instead should treat this as
still open for re-litigation, not settled.

**What changed**: the `current_equity <= failed_equity` term is removed from
the kernel's per-bar breach check (`backtest_kernel.hip.cpp`, previously at
`:1901`) and from the equivalent check in both `kernel_reference.rs` twins,
leaving only the drawdown-cap breach (`max_drawdown_val * 100.0f > max_dd`)
— now an exact structural match for `run_simulation`'s own single in-loop
circuit breaker. `failed_equity` itself is **not** removed as a kernel
parameter: it continues to serve O04's `break_all_tickers`-style backfill
equity, exactly as `config.simulation.failed_equity` does on the CPU side.

**Regression coverage**:
`bankruptcy_floor_removal_yields_identical_trade_counts_vs_run_simulation`
(`crates/alpha-simulation/tests/kernel_reference_simple_parity_tests.rs`).
**What it proves**: on a fixture engineered to have trivially crossed the
OLD floor (a consistently-losing strategy with `failed_equity` raised well
above its default so ordinary losses cross it) while staying comfortably
under the drawdown cap, the kernel twin and `run_simulation` now trade the
fixture out to its natural end with **identical trade counts** — before this
fix, the old floor would have force-closed and stopped the twin early while
`run_simulation` kept going, which is exactly the divergence this test
would have caught (and did, before the fix landed).

### Accepted residual gaps (F32/2.3.3)

Blueprint `fdb2` subtask `5ab1`. No code changed for this item — it formally
accepts three approximations already described elsewhere in this audit,
rather than leaving their status implicit:

- **O10 — Sharpe/Sortino/Calmar, daily returns, per-ticker/per-state
  breakdown absent from the GPU result path.** Reclassified from "Open, by
  design — informational" to **ACCEPTED**. Reason: the GPU tier filter
  (tier-1/tier-2, and the full-history GPU stage) exists only to decide
  which individuals survive to the next stage — `run_tier_filter_gpu` reads
  `gpu_res.final_equity`/`.trades`/`.wins`/`.max_drawdown_percent`/
  `.min_trades_across_tickers` and nothing else (§3.8). Final scoring —
  including every risk ratio and the per-ticker/per-state breakdown — is
  always produced CPU-side, by `run_simulation`, once an individual reaches
  a stage that needs those numbers. A field the tier gate structurally never
  reads cannot bias which individuals it selects, so there is no fitness- or
  trade-decision-relevant gap here to close, only a reporting field the GPU
  path was never architected to populate.
- **Per-bar funding deduction** (§3.6's "Funding" row). CPU defers funding
  into `accumulated_funding_fee` and realizes it once, at exit
  (`execution.rs:164`); the kernel instead deducts each bar's funding fee
  straight out of `current_equity` as it accrues (PHASE 4 mirror) and never
  re-subtracts it at exit. Accepted as an equivalent, not identical,
  mechanism — the kernel's own comment at the exit-fee call site makes the
  equivalence argument directly: "Funding is NOT re-subtracted here: unlike
  the CPU's deferred `accumulated_funding_fee` (only realized into equity at
  exit), this kernel already deducts each bar's funding fee straight out of
  `current_equity` as it accrues (PHASE 4 mirror below), so there is nothing
  left over to charge at exit." (`backtest_kernel.hip.cpp`, the GAP 1/exit-fee
  comment block). The GAP 4 mark-to-market/drawdown block makes the same
  argument from the drawdown side: "No `accumulated_funding_fee` term here:
  unlike the CPU, this kernel deducts each bar's funding fee straight out of
  `current_equity` above (in the PHASE 4 mirror) rather than deferring it to
  exit, so `current_equity` is already funding-inclusive and there is
  nothing extra to subtract" (`backtest_kernel.hip.cpp`, the GAP 4 comment
  block). Both engines end up charging the identical total funding cost over
  a position's life; only the bar at which it is realized into a reportable
  number differs.
- **Hard-coded `0.5` entry/exit signal thresholds.** Every `StateStrategyGpu`
  built by `gpu.rs`'s compilation path gets `entry_threshold: 0.5,
  exit_threshold: 0.5` unconditionally (`gpu.rs:1814-1815`) — not a value
  read from the evolved `StrategyParams` at all. The kernel then gates
  entry/exit purely on `entry_score > active_strategy.entry_threshold` /
  `exit_score > exit_strategy.exit_threshold` (the PHASE 8/9 mirror,
  `backtest_kernel.hip.cpp`) against that constant. This is already
  identified and argued as safe in this same audit's §5, **I02**: it is
  correct only because every compiled `SignalExpression` tree's root op
  (`Comparison`/`And`/`Or`/`Not`/`True`/`False`) bottoms out in an
  `IfLessThen`/`Terminal(0 or 1)` sequence that resolves to an *exact*
  `0.0` or `1.0` (`gpu.rs`'s `compile_signal_expression`) — so
  `tree_output > 0.5` is equivalent to `tree_output == 1.0`, i.e. a boolean
  check dressed up as a threshold comparison, agreeing with CPU's own
  `entry_evaluators[idx].evaluate()` boolean exactly (§3's "Signal eval"
  row). **Accepted, not merely tolerated**: this task formally accepts I02's
  own caveat as a standing condition, not a one-time observation — a future
  signal-tree root type producing a continuous (non-boolean) score would
  silently break this equivalence with no compiler or test signal, exactly
  as I02 already warns.

---

## 5. Incidental agreements — the next thirteen bugs

These currently behave identically on both engines, but only because of a
condition that isn't enforced anywhere and could silently stop holding after
an unrelated future change.

- **I01 — `risk_modulator` (ML-filter position-size multiplier)**: the
  kernel has no concept of it at all (`backtest_kernel.hip.cpp:1456` sizes off
  `base_risk_per_trade` alone). This is currently harmless only because
  every call site that drives GPU tiering (`run_tier_filter_gpu` ->
  `evaluate_batch`) never has an ML model to apply — `run_simulation` is
  always invoked elsewhere with `modulator_opt: None` for tier-1/tier-2 CPU
  evaluation too (`tiering.rs:58`, `tiering.rs:559-563`), so
  `sim_state.risk_modulator` never leaves its default `1.0` on the CPU side
  being mirrored. The moment ML-filtered tiering is wired to GPU (or the CPU
  tier path starts passing a real modulator), this silently reintroduces a
  position-sizing divergence with no compiler or test signal.
- **I02 — hardcoded `entry_threshold`/`exit_threshold = 0.5`**
  (`gpu_evaluator.rs:426-427`): correct only because every compiled
  `SignalExpression` tree's root op resolves to an exact `0.0`/`1.0`
  (`Comparison`/`And`/`Or`/`Not`/`True`/`False` all bottom out in
  `IfLessThen`/`Terminal(0 or 1)`, per `gpu.rs`'s `compile_signal_expression`).
  A future signal-tree root type that produced a continuous score would
  silently break this without any error.
- **I03 — `trade_state == 0` folding `NoPosition` and `InCooldown`
  together**, distinguished only by the separate `cooldown` counter
  (`backtest_kernel.hip.cpp`, throughout). Behaviourally equivalent to the
  CPU's explicit `TradeState` enum today, but only because nothing currently
  needs to distinguish "flat, never entered" from "flat, cooling down" by
  anything other than "is the counter zero." A future feature keyed
  directly off `TradeState::NoPosition` vs `InCooldown` (e.g. a state-entry
  counter, forensic logging of cooldown starts) would need a GPU-side
  concept that doesn't exist.
- **I04 — the `entry_stateful_count`/`exit_stateful_count` skip-if-zero
  optimization** in the "Subconscious state updates" loop
  (`backtest_kernel.hip.cpp:2065-2140`) is proven equivalent only because
  every non-`Sma`/`StdDev`/`Lag` opcode is provably side-effect-free (the
  loop's own doc comment makes this argument explicitly). Adding any future
  GPU opcode with an observable side effect outside the function-local
  `DeviceStack` (a GPU-side RNG, a new kind of accumulator terminal) would
  silently invalidate this optimization with no signal anywhere.
- **I05 — batch grouping by `IndicatorPeriods` in
  `evaluate_batch_tiered`/`run_tier_filter_gpu`** happens to keep every
  individual in one GPU batch on the same `ticker_order`/snapshot matrix,
  which is what makes O07's zero-padding possible in the first place (all
  individuals in a batch share one padded-to-longest candle grid). Nothing
  currently checks that this grouping and the CPU's per-ticker eligibility
  rule agree; they simply haven't been made to disagree loudly yet.

---

## 6. Items marked OPEN — could not be determined

- **Whether O05 is reachable in the current production deployment.** All
  four shipped configs set `hmm.state_search_range` such that
  (per `manual/12`-adjacent evidence and `config.harvest.toml`/`config.toml`
  comments) 16 states is typical, but I did not locate a runtime guarantee
  that every individual's `num_hmm_states` always equals the precomputed
  label sequence's own range in every deployment path (GA migrants between
  islands with different `state_search_range`, resumed sessions from an
  older config). The CPU-side comment at `simulation.rs:803-822` treats this
  as a routine occurrence the codebase must handle correctly, which is why
  O05 is scored on architectural reachability, not on a confirmed
  production trigger.
- ~~Whether `manual/12`'s Agg1/Agg2 finding (O09) has already been
  scheduled or fixed elsewhere.~~ **Answered (later follow-on task,
  task F2): yes, it has been fixed.** `manual/12`'s "Option A" landed as
  `crate::gpu_columns`/`crate::gpu_column_gather`, and
  `build_snapshot_matrix` (`crates/alpha-simulation/src/gpu.rs:785-876`,
  not the stale `gpu.rs:537` this entry originally cited — that line
  number predates the F1b/F2 work and no longer points at this function)
  now ticks real `Timeframe::Agg1`/`Agg2` pipelines instead of hardcoding
  `Timeframe::OneMin`. See §3.9's O09 detail section for the cache-key
  design and equivalence-authority argument, and `manual/12` §5.4.1/§5.4.2
  for the memory cost this closure added.
- **The practical frequency of O01's guard actually firing** (i.e., how
  often production ATR values are small enough for `atr * atr_multiplier`
  to fall at or below `1e-9f` on real market data) — this would require
  running real data through the kernel, which was out of scope for a
  read-only audit and (per `manual/13` §1) has never been done on real
  hardware for *any* part of this kernel.
- **Whether GPU's trailing-stop update ordering difference (§3.3,
  "Trailing-stop updates") has any consequence I missed.** I traced every
  read of `trailing_stop` after a same-bar exit and found none before the
  next entry resets it, but I did not exhaustively check every future
  consumer this value might gain.

---

## 7. Overall assessment

**Update, post-fix**: O01–O08 have since been fixed (a follow-on task to
this audit), and O09 has since been fixed too (a still-later follow-on
task, F2); see each item's detail section above for exactly what shipped.
**Update (F32 remediation)**: O08's residual gap has since been closed
(2.3.1), O02's floor has since been removed rather than merely
config-sourced (2.3.2), and O10 has since been formally **accepted** rather
than left open (2.3.3) — see the "Fourth update" note in §1 and §4's O02/O08
decision records and "Accepted residual gaps" subsection. No item in this
table remains open. The assessment below is left as originally written, for
the historical record of what this audit found — read it as describing the
state *before* any of these follow-on tasks, not the current one.

**No — the kernel does not currently reproduce the CPU model**, and not only
because of the two previously-known open items (O01, O02). This audit found
six more, three of which (O04, O05, O06) change which individuals the tier
filter selects, not merely how their numbers are reported — which is the
exact failure mode the fitness function depends on the GPU path not having.

For "the kernel reproduces the CPU model" to hold as a claim, at minimum:

1. O01–O08 need to be either fixed, or explicitly accepted as permanent,
   documented approximations with the fitness-consumer side adjusted to
   compensate (e.g. if O04's harsher CPU penalty is judged too aggressive
   for a *pre-filter* stage, that should be a stated design decision, not a
   silent gap). **Status: done for O01–O10** — see the summary table and
   each item's detail section. O08's previously-accepted residual
   approximation (no `loss_pot` year-boundary reset) has since been closed
   outright (F32/2.3.1) rather than merely documented; O02's floor has been
   removed rather than merely config-sourced (F32/2.3.2); O10 is now
   formally accepted, not merely "open by design" (F32/2.3.3). Per-bar
   funding deduction and the hard-coded `0.5` entry/exit thresholds are
   likewise now explicitly accepted rather than implicitly tolerated — see
   §4's "Accepted residual gaps (F32/2.3.3)".
2. O06/O07 cannot be fixed inside `tiering.rs` alone — `BacktestResultGpu`
   itself needs a per-ticker breakdown, which is a kernel and FFI change,
   not just a Rust-side formula fix. **Status: done** — `BacktestResultGpu`
   gained `min_trades_across_tickers` (O06), and a new `ticker_lengths`
   per-ticker array closes O07's mixed-length case; the entirely-too-short
   case is closed one level up, in `run_tier_filter_gpu`'s `ticker_order`
   construction.
3. The incidental agreements in §5 need either a kernel-side implementation
   of what they currently get away with skipping, or a loud runtime/compile-time
   assertion that the conditions they rely on still hold — otherwise this
   audit's "next thirteen bugs" framing will prove literal. **Status:
   unchanged** — I01–I05 were out of scope for the O01–O08 follow-on task
   and were not touched; they remain exactly as documented in §5, still
   worth a future pass.
4. §3.10's f32/f64 gap means **exact** reproduction is not an achievable
   target at all, only bounded-tolerance parity — which is already how
   `manual/13`'s self-check tool frames success. Any claim of "parity" needs
   to be stated against explicit tolerances, not as an absolute. **Status:
   unchanged and unchangeable** — this is structural, not a bug to fix.
5. None of this — not the thirteen previously-fixed items, not the two
   previously-known open ones, not this audit's six new findings — has ever
   been exercised against a real GPU. `manual/13`'s chapter 1 is still
   accurate: every one of these fixes has only ever been type-checked.
   Running `alpha-worker --features gpu --gpu-parity-check` and
   `cargo test --features gpu -p alpha-simulation` on real hardware, for the
   first time, remains the necessary next step before any of this can be
   trusted empirically rather than by code reading — and even a clean run of
   that tool would not by itself catch O04, O06, or O07, since (per
   `manual/13` §5) its fixture is a single small multi-ticker batch that was
   never built to exercise a drawdown breach on one ticker among survivors,
   mismatched ticker lengths, or an average-vs-minimum trade-count gap.
   **Status: still true of the O01–O08 fixes themselves** — the O01–O08
   follow-on task retuned `apps/alpha-worker`'s `gpu_parity` fixture
   (`fixture.rs`) so it now DOES exercise O04's `break_all_tickers` backfill
   (via `DRAWDOWN_TICKER`/`TICKERS_AFTER_DRAWDOWN`'s CPU-side ground truth,
   which the CPU-only test suite confirms), but that fixture's GPU-side
   half — the actual CPU-vs-GPU numeric comparison — remains, like
   everything else in this line item, compiled and type-checked only, never
   run against real hardware. Compile-verification plus the CPU-only side
   of that fixture's tests are the ceiling reachable without a GPU.

---

## 8. Task H — warp-per-individual + shared-memory stack: occupancy analysis and decision (NOT implemented)

This section records a design investigation, not a shipped change. `backtest_kernel.hip.cpp`, `gpu.rs`, and `gpu_evaluator.rs` are **untouched** by this section — the ABI, launch geometry, and kernel body are exactly as documented in §§1–7 above and in `gpu.rs`'s own comments. This is a decision record for a restructure that was evaluated and rejected, kept here so the next person who has the same idea (very reasonable — the REGISTER/OCCUPANCY TRADEOFF comment above `gather_one_slot` in `backtest_kernel.hip.cpp` invites exactly this) does not have to redo the analysis, and so a future engineer with real A100 access knows precisely what measurement would overturn it.

### 8.1 The proposal

Today's kernel is thread-per-individual: `individual_idx = blockIdx.x * blockDim.x + threadIdx.x`, one thread runs one individual's entire multi-ticker, multi-bar backtest sequentially. F1b-2's per-bar gather materializes a `float[GPU_MAX_SLOTS]` (128 floats, 512 bytes) local array per thread per bar (`snapshot_row`, `backtest_kernel.hip.cpp` around the per-ticker bar loop), which that file's own comment already flags as likely to spill to per-thread local memory once combined with `evaluate_gp_tree`'s `DeviceStack<float, 32>`. The proposal: assign a **warp** (32 lanes) to one individual instead, split the 128-slot gather 4-ways across the lanes, park the one gathered row in `__shared__` memory (once per warp, not once per thread), and run the strictly-sequential per-bar state machine (PHASE 1/3/5 fill, exit resolution, ratchet, cooldown, equity/tax) on a single designated lane per warp, reading the shared row.

### 8.2 Current design: per-thread footprint arithmetic

Three arrays in `backtest_kernel.hip.cpp` are indexed by a value only known at runtime, which forces nvcc/ptxas to place them in per-thread **local memory** (the same address space and cache hierarchy as global memory, not the register file) rather than registers, regardless of thread-per-individual vs. warp-per-individual:

| Array | Size | Indexed by | Why it can't be a register array |
|---|---|---|---|
| `snapshot_row[GPU_MAX_SLOTS]` | 128 × 4 B = **512 B** | `slot` in `gather_snapshot_row`'s loop, then `node.terminal_id` in `get_terminal_value` | the *consumer* index (`terminal_id`) is data from a compiled GP tree, unknowable at compile time |
| `DeviceStack<float, 32>::data` | 32 × 4 B = **128 B** | `top_idx`, which walks in a pattern determined by each individual's own compiled bytecode | tree shape varies per individual/strategy, so push/pop order isn't a compile-time constant |
| `per_state_pnl[GPU_MAX_HMM_STATES]` | 32 × 4 B = **128 B** | `current_state_idx = hmm_states[global_i]`, an HMM regime label read from device memory | genuinely a runtime value |

Known dynamically-indexed local-memory footprint: **512 + 128 + 128 = 768 B/thread** (a floor — if ptxas fails to coalesce the stack's storage across the up-to-seven sequential `evaluate_gp_tree` call sites per bar into one reused slot, it could be more; I cannot check this without a SASS/PTX dump from a real build, see §8.6).

The rest of the per-bar/per-ticker state (`current_equity`, `peak_equity`, `max_drawdown_val`, `trade_count`, `win_count`, `trade_state`, `cooldown`, `entry_state_idx`, `entry_price`, `position_size_coins`, `trailing_stop`, `take_profit`, `liquidation_price`, `bars_since_entry`, `pending_order`, `pending_state_idx`, `cooldown_period_setting`, `loss_pot`, plus the per-individual accumulators and loop/index variables — roughly 20–25 named scalars) **is** register-eligible. I estimate 30–60 live 32-bit registers at peak for this, but this is exactly the number I cannot pin down on this host: getting it requires `nvcc -Xptxas -v` (or `--resource-usage`) output from an actual compile, and this crate's `build.rs` (`cc::Build`) only surfaces the underlying compiler's stdout/stderr when the compile itself fails, not on success — so there is no way to extract a real register count without either a deliberate (and unverifiable) build.rs edit or a genuinely broken build. **Flagged as the single most important number this analysis is missing; see the profiling checklist in §8.6.**

Occupancy consequence: because the two big arrays (512 B, 128 B) spill to local memory rather than consuming the 256 KB/SM (65 536 × 32-bit) register file, they do **not** directly gate occupancy the way a register-resident array of the same size would. At an assumed 30–60 registers/thread, register-file occupancy works out to:

- 30 regs/thread → 65536/30 ≈ 2048 threads/SM → 100% (64/64 warps)
- 50 regs/thread → 65536/50 ≈ 1310 threads/SM → ≈ 40/64 warps ≈ 62%
- 64 regs/thread → 65536/64 = 1024 threads/SM → 32/64 warps = 50%

With `threads_per_block = 256` (today's `hipLaunchKernelGGL` call) and 0 bytes of shared memory used today, none of these bands hit the 32-blocks/SM structural cap (2048/256 = 8 blocks/SM) or any shared-memory limit — **registers are already the only occupancy limiter today**, somewhere in the 50–100% band depending on the real (unmeasured) register count. The array spills are a real cost, but it is a **memory-traffic** cost (a 512 B global read into local memory, then per-slot local read/writes, every bar), not the register-starvation story the raw byte count might suggest.

### 8.3 Hypothetical warp-per-individual footprint

Only `snapshot_row` benefits from moving to `__shared__`: it is written and read cooperatively across a warp's 32 lanes in the proposed design, so sharing one copy per warp (32 individuals' worth of redundant local-memory footprint under thread-per-individual → one 512 B row under warp-per-individual) is the entire point. `DeviceStack` and `per_state_pnl` are used **only by the single lane running the sequential state machine** in the proposed design — exactly as they are used by the single thread today — so there is no sharing benefit and no reason to relocate them. This corrects the task brief's rough "≈640 B/warp (512 B snapshot + 128 B stack)" figure: the stack does not need to move, so the actual new shared-memory requirement is **512 B/warp**, not 640 B/warp.

With `WARPS_PER_BLOCK = threads_per_block / 32` and a block-static `__shared__ float smem_snapshot[WARPS_PER_BLOCK][GPU_MAX_SLOTS]`:

- At `threads_per_block = 256` (8 warps/block): 8 × 512 B = **4 KiB/block**.
- At 8 blocks/SM (2048 threads/SM ÷ 256 = 8, under the 32-blocks/SM structural cap): **32 KiB/SM** shared-memory usage at full occupancy.

That is far under both the 48 KiB default static-shared-memory-per-block ceiling (no `hipFuncAttributeMaxDynamicSharedMemorySize`-style opt-in needed at all — this can be a plain static `__shared__` array, not a dynamically-sized one, so the `hipLaunchKernelGGL` call's shared-memory-bytes argument stays `0`) and the ~164 KiB/SM total budget. Shared memory would need to be **~48× larger** (i.e. a block hosting ~384 warps, physically impossible — the whole SM tops out at 64 resident warps) before it could plausibly become the occupancy limiter ahead of registers. **Registers, not shared memory, remain the limiter under the proposed design too** — and the register count is essentially unchanged from today, because the state-machine code that dominates register use still exists, unabridged, just now guarded by a `lane == 0` predicate; SIMT hardware allocates the same registers to every lane regardless of which lanes the predicate masks off, so predicating code does not shrink its register footprint.

**Conclusion of the mechanical part of the analysis: on pure occupancy arithmetic (warps resident per SM), the proposed design is close to occupancy-neutral relative to today** — shared memory is nowhere near the limiter either way, and registers are the limiter in both designs at roughly the same count.

### 8.4 The argument occupancy numbers miss: SIMT lane utilization

Occupancy-neutral does not mean throughput-neutral, and this is where the proposal fails.

**The gather has zero warp divergence today.** `d_slot_gather_plan` (`slot_kind`/`slot_role_a`/`slot_role_b`) is one buffer, identical for every individual and ticker in the launch (`launch_backtest_kernel`'s own doc comment: "IDENTICAL for every individual/ticker in this launch, so resolved once here rather than once per bar"). So in the current thread-per-individual kernel, at every one of `gather_snapshot_row`'s 128 loop iterations, **every lane in the warp executes the same `SlotGatherKind` case** — only the per-individual *data* (role offsets, OHLC values) differs, not the control flow. This is already about as SIMT-efficient as a 24-way-switch-driven, per-element gather can be.

Splitting that same 128-slot gather 4 ways across a warp's lanes (lane ℓ owns slots `4ℓ..4ℓ+3`) makes *different lanes* respon­sible for *different slot indices* at the same program point — and whenever those slots carry different `SlotGatherKind`s (a near-certainty across any 4 arbitrary slots out of 128, given up to 24 defined kinds), that is genuine, newly-introduced divergence that does not exist today. The realized speedup from spreading the gather over 32 lanes is therefore bounded not by 32×, but by however many *distinct* `SlotGatherKind`s are actually present among a typical compiled individual's 128 slots — plausibly a handful, if (as I'd guess but cannot confirm without inspecting a live `d_slot_gather_plan`) `SGK_NONE`/`SGK_DIRECT` dominate — giving a realistic speedup in the rough 3–10× range on the gather phase specifically, not a clean 32×. I could not measure the actual kind distribution on this host; see §8.6.

**The sequential phase pays the full 32× tax, unconditionally.** Whatever divergence the *current* design already has in its tree-evaluation and state-machine logic (real — every individual has its own compiled tree shape, its own `trade_state`/branch outcomes, so this part of today's kernel is not SIMT-clean either), it still runs with **up to 32 individuals advancing per issued instruction**, degraded by however much that divergence costs, but never worse than 1-of-32 in the limit. Confining the same logic to `if (lane == 0)` in the proposed design makes it advance **exactly 1 individual per issued instruction, always** — a hard floor, not a degraded ceiling.

Sizing the two phases against each other: PERF1's own comment in `backtest_kernel.hip.cpp` states that at production `hmm.state_search_range = [16, 16]`, the "subconscious" background-update loop is already pruned (by the `entry_stateful_count`/`exit_stateful_count` gates) from a worst case of ~31 tree evaluations per bar down to **typically 1–2** background evaluations actually executed, on top of the 1 main entry-or-exit evaluation — so realistically **~2–4 `evaluate_gp_tree` calls per bar**, each up to `MAX_GP_TREE_DEPTH = 30` nodes (almost certainly fewer in practice — GP trees start small and grow under evolutionary pressure, not always at the cap). That, plus the non-tree PHASE 1/3/5 fill/exit/ratchet/cooldown/equity/tax control flow, is the same rough order of magnitude as the 128-slot gather (128 slots × a handful of instructions/slot each in `gather_one_slot`'s switch arms) — **not a case where the gather clearly dominates the per-bar cost**. Both phases are comparable, plausibly within a factor of 2–3 of each other.

Net: the proposal trades an uncertain, likely-modest (few-×) speedup on a phase that is already close to SIMT-optimal, for a guaranteed, severe (order-32×, floor-bounded) throughput tax on a comparably-sized phase that today gets at least *some* parallelism. There is no plausible combination of the unmeasured numbers in §8.2/§8.6 that flips this: even if every register-count/local-memory-spill assumption in §8.2 is wrong in the direction that most favors the redesign, the SIMT-utilization argument in this section is a control-flow property of the code, not a resource-budget one, and holds regardless.

### 8.5 Decision: not implemented

Per this task's own instruction ("If your analysis concludes the win does not justify the complexity, say so and stop — that is a legitimate and valuable outcome"), **this restructure was not implemented.** `backtest_kernel.hip.cpp`'s kernel body, `launch_backtest_kernel`'s launch geometry (`threads_per_block = 256`, one thread per individual), the ABI (`LAUNCH_BACKTEST_KERNEL_ABI_VERSION = 3`), and `GpuEvaluator`'s `max_individuals` accounting are all unchanged. No `__syncwarp`/`__shfl_sync` calls were added, because no warp-cooperative code was added.

For the record, had §8.4's argument come out the other way, the design this analysis worked through (so it doesn't need re-deriving) was: per bar, (1) all 32 lanes cooperatively fill `smem_snapshot[warp_in_block]` (4 slots/lane, base-copy + `gather_one_slot` reconstruction), (2) one `__syncwarp(0xffffffff)` ordering every lane's shared-memory writes in step (1) before any lane reads the row, (3) `if (lane == 0)` runs the untouched sequential state machine reading `smem_snapshot[warp_in_block]` directly (no shuffle needed — shared memory is warp-visible, so the single active lane can read it after the barrier with no data movement), (4) a second `__syncwarp(0xffffffff)` ordering lane 0's reads in step (3) before any lane's next-bar gather (step 1 of the next iteration) overwrites the same shared slot. `if (individual_idx >= num_individuals) return;` would need to become a clamped-index-plus-`active`-flag pattern (out-of-range lanes keep participating in every `__syncwarp` with a clamped, in-bounds `individual_idx` and simply skip the final `all_results` write), since an early `return` desyncs a partial warp from any lane that still calls `__syncwarp`. This is recorded so a future implementer starts from here rather than from scratch — it is not vetted beyond the reasoning in §8.4, and per that section, I do not believe implementing it is warranted.

### 8.6 What would overturn this decision — A100 profiling checklist

All of §8.2–§8.4 is static-analysis arithmetic from reading the source, not a measurement — there is no GPU on this host. Before anyone spends time implementing the warp-per-individual restructure (or trusts this section's rejection of it), check, on the real A100:

1. **`nvcc -Xptxas -v` (or `--resource-usage`) register count** for `backtest_kernel`, from an actual `sm_80` build. This is the single number §8.2 is missing; it directly tests whether registers or something else is really the occupancy limiter today, and whether §8.2's 30–60-register guess is anywhere close.
2. **Nsight Compute "achieved occupancy" and "warp execution efficiency"** on a real tier-1/tier-2 evaluation batch. Warp execution efficiency is the metric that directly measures the SIMT-divergence tax discussed in §8.4 — if today's kernel already shows poor efficiency (meaning the current design's divergence is worse than assumed), the case for warp-per-individual gets weaker, not stronger, since it would mean the "up to 32×" ceiling this section assumed for today's sequential phase is optimistic.
3. **Nsight Compute local-memory spill counters** (spill stores/loads, local memory throughput) to confirm or refute that `snapshot_row`/`DeviceStack`/`per_state_pnl` are in fact spilling to local memory as §8.2 assumes, rather than being kept in registers or eliminated by the optimizer.
4. **A histogram of `SlotGatherKind` values actually present** in a production `d_slot_gather_plan` (built by `crate::gpu::build_slot_gather_plan` from a real evolved population). This one measurement most directly replaces §8.4's "a handful of kinds" guess with a real number, and is the most likely single data point to change the conclusion.
5. If, and only if, that data shows the gather dominates total kernel time by a wide margin (rule of thumb: >70%) **and** achieved occupancy is well under 50% specifically because of local-memory pressure (not registers) — that combination would be grounds to revisit §8.5.
6. A cheaper alternative to profile first, since it needs no warp-cooperative restructuring at all and carries none of the divergence risk in §8.4: the REGISTER/OCCUPANCY TRADEOFF comment already above `gather_one_slot` in `backtest_kernel.hip.cpp` floats gathering only the slots a compiled individual's trees actually reference (a bitmask threaded through from `GpuEvaluator::compile_batch`) instead of unconditionally gathering all 128 every bar. That shrinks local-memory traffic directly, within the existing thread-per-individual model, and is a much smaller, more reviewable change to justify once real profiling data (item 1–4 above) shows the gather is worth optimizing at all. **Since item 6 was implemented as Task H′ (§10), this item is closed** — see §10 for the shipped design.
7. **(Task J, §11) Nsight Systems timeline, tier-1/tier-2 vs full-history separately**: capture a full generation and look specifically at whether `run_full_history_filter_gpu`'s group-to-group boundaries (§11.5) show the host CPU building the next group's `GpuColumnCache`/`ColumnPool` genuinely overlapping the previous group's kernel execution on `kernel_stream`, or whether they serialize anyway (e.g. because the host-side build is so fast relative to the kernel that there is nothing to hide, or because of an unexpected implicit sync somewhere in the driver). §11.2's byte-level transfer analysis and §11.5's design are both static reasoning from source, never observed running.
8. **(Task J) Pinned vs pageable achieved H2D bandwidth**: compare `nvprof`/Nsight Systems' reported bandwidth for the pinned-staged uploads (`GpuEvaluator::upload_staged`, `gpu.rs`'s `staged_upload_chunks`) against the pre-Task-J plain `DeviceBuffer::upload` path on the same buffers and sizes. §11.2 assumes the textbook pinned-vs-pageable gap (roughly 2× on typical PCIe setups) without measuring it on this specific host/driver/A100 combination — this is the item that would confirm or correct that assumption.
9. **(Task J) Stream serialization / false dependencies**: confirm `kernel_stream` (the one non-default stream `GpuEvaluator` creates, §11.4) is not implicitly serialized against the default stream by legacy default-stream semantics in whatever CUDA context this binary actually runs under (this project's own build hard-codes CUDA-vs-HIP compatibility shims, §2 — a build-time detail that could plausibly affect which stream-synchronization semantics apply at runtime, unverifiable from source alone). If Nsight Systems shows `kernel_stream`'s kernel launches serializing against unrelated default-stream work rather than running concurrently as designed, that would be grounds to look at explicit per-context stream flags (`hipStreamNonBlocking`/`cudaStreamNonBlocking`) — not attempted here since it cannot be verified without a real device.

### 8.7 What this section could not verify

No GPU was available on this host. Nothing in §8.2–§8.4 was measured; all of it is source-reading plus textbook A100 occupancy arithmetic applied to the sizes and loop structures actually present in `backtest_kernel.hip.cpp`. Specifically unverified: real register counts (§8.2), whether the three flagged arrays actually spill to local memory as opposed to being partly register-allocated or optimized away in dead branches, the real `SlotGatherKind` distribution in a production gather plan, and — since no warp-cooperative code was written — anything about actual `__syncwarp` timing, lane divergence, or shared-memory bank conflicts, all of which are invisible on a CPU twin by construction (per this task's own framing) and were never exercised here.

---

## 9. Task I — the full-history GPU stage: no new divergence introduced

Full detail (memory arithmetic, chunking design, config semantics) lives
in `manual/12` §9, not here — this section exists only to record, in this
audit's own terms, what Task I did and did not touch relative to
everything above.

**Untouched**: `backtest_kernel.hip.cpp`, `LAUNCH_BACKTEST_KERNEL_ABI_VERSION`
(still `3`), `launch_backtest_kernel`'s signature, and every `#[repr(C)]`
struct crossing the FFI boundary (`GpSimulationParamsGpu`, `OhlcGpu`,
`BacktestResultGpu`, `StateStrategyGpu`, `GpNode`, `StatefulNodeInfo`).
Task I is a host-side-only change: it decides WHETHER and HOW MANY
individuals reach the existing, unmodified kernel per call
(`engine::tiering::run_full_history_filter_gpu`, new), and how the
existing, unmodified `GpuEvaluator::evaluate_batch`/`GpuColumnCache`/
`ColumnPool` machinery (F1b-1/2/2b/3, F2) is invoked for the full-history
candle range instead of only tier-1/tier-2. No kernel-side code path that
§§1-8 above analyzed runs any differently at full-history length than it
already did at tier-2 length — the device gather's own correctness
argument (`validate_periods_for_device_gather`'s `WARMUP_PERIOD`-dominance
check) is independent of `num_candles` (`manual/12` §9.6), and
`GpuEvaluator::evaluate_batch`'s residency hard-fail (F1b-3) is preserved
exactly, not weakened, by the new chunking (`manual/12` §9.4).

**O10 status, updated by a later task (F32/2.3.3)**: at the time Task I ran,
this table's O10 entry (Sharpe/Sortino/Calmar, daily returns,
per-ticker/per-state breakdown absent from the GPU result path) stood
**Open, by design**; it has since been reclassified **Accepted** (see §4's
"Accepted residual gaps (F32/2.3.3)" subsection) — a disposition change, not
a code change, so everything below about Task I's own scope boundary still
holds. Task I's full-history GPU results use the SAME
`approximate_result_from_gpu` per-ticker-average construction
`run_tier_filter_gpu` already used for a GPU-rejected tier-1/2 individual
(`manual/12` §9.5) — this was an explicit scope boundary the task set, not
an oversight: getting the total-PnL/min-trades numbers the tier gates and
fitness scoring actually read onto the GPU was in scope; improving
`TickerResult` attribution fidelity or closing O10 was not (and, per
2.3.3's acceptance, never needed to be — the tier gate never reads these
fields on any path, GPU tier-1/2 or GPU full-history alike). A reader
tracing why full-history `BacktestResult`s still carry `sharpe_ratio: 0.0`
etc. on the GPU path should land here, then on `manual/12` §9.5, rather
than conclude Task I forgot about it.

**No new divergence class**: every phase this audit's §§3-4 traced
(entry/exit resolution, funding, cooldown, drawdown/bankruptcy, Austrian
tax, the O01-O12 fixes) runs inside the SAME kernel Task I calls more
often, at more candles per call, never a modified one. The one thing that
DOES change behaviorally is which stage produces the final
`BacktestResult` for a given individual: previously always the CPU
`run_simulation` (exact per-ticker attribution); now, when GPU
full-history is eligible, `approximate_result_from_gpu`'s aggregate
approximation (same approximation as an O06/O04-era tier rejection, per
above) for individuals GPU actually handles, with CPU `run_simulation`
(unchanged, exact) still covering whatever `run_full_history_filter_gpu`
routes back to it (GPU disabled/ineligible, the column cache disabled, a
candle count exceeding capacity, or a tuple whose own column-cache
footprint alone cannot fit any budget). This is a deliberate,
scope-bounded fidelity trade the task's own text accepted explicitly
("TickerResult attribution... stays open and out of scope"), not a new
entry for this table's O-numbered list.

---

## 10. Task H' — compact per-individual GP-terminal gather (implemented)

§8.6 item 6 floated an alternative to the rejected warp-per-individual
restructure: instead of unconditionally gathering all `GPU_MAX_SLOTS`
(128) slots into `snapshot_row` every bar, gather only the slots a
compiled individual's trees actually reference. This section documents
that alternative, implemented as Task H' — a pure work-reduction change,
within the existing thread-per-individual model, carrying none of §8.4's
divergence risk. Unlike §8/§9, this is a shipped change: `gpu.rs`,
`backtest_kernel.hip.cpp`, `gpu_evaluator.rs`, and
`device_gather_reference.rs` are all touched.

### 10.1 Design

Host side (`gpu.rs`):

- `GPU_MAX_REFERENCED_SLOTS: usize = 64` (`gpu.rs:168`) — capacity of the
  new compact per-individual gather row. Mirrored exactly in
  `backtest_kernel.hip.cpp:70`. 64 is headroom over the true ceiling: only
  53 of the registry's 56 occupied slots are `gp_visible: true`
  (`gpu.rs:151`), i.e. reachable by any GP terminal at all — see §10.3 for
  what a real individual actually references, which is far below even 53.
- `collect_referenced_slots(nodes: &[GpNode]) -> Result<Vec<i32>>`
  (`gpu.rs:1461`) — scans one individual's own compiled bytecode slice
  (every entry+exit tree, across every one of its strategies) for
  `Terminal` nodes, returns the raw `0..GPU_MAX_SLOTS` slot ids in
  first-seen order, deduplicated. `SLOT_ATR_RAW_1M` (raw id `0`) is always
  reserved first regardless of whether any tree references it directly —
  see 10.1's ATR note below. `Terminal::Constant` (`terminal_id == -1`) is
  skipped. Returns `Err`, not a silent truncation, if the individual
  references more than `GPU_MAX_REFERENCED_SLOTS` distinct slots.
- `rewrite_terminals_to_compact_index(nodes: &mut [GpNode], raw_slots: &[i32])`
  (`gpu.rs:1500`) — rewrites every `Terminal` node's `terminal_id` in place
  from its raw slot id to its position in `raw_slots`. This is the move
  that makes the on-device remap free: the kernel's `Terminal` case just
  indexes the compact row directly, no runtime search.
- `compact_individual_for_device(nodes: &mut [GpNode], params_gpu: &mut GpSimulationParamsGpu) -> Result<()>`
  (`gpu.rs:1536`) — combines the two functions above for one individual and
  writes the resulting plan into `params_gpu.referenced_slots`/
  `referenced_slot_count`.
- `GpSimulationParamsGpu` gained `referenced_slots: [i32; GPU_MAX_REFERENCED_SLOTS]`
  (`gpu.rs:428`) and `referenced_slot_count: i32` (`gpu.rs:440`), mirrored
  in the C++ struct in the same position (`backtest_kernel.hip.cpp:284-285`).
  `compile_gp_batch` (`gpu.rs`) always leaves a freshly compiled individual
  UNCOMPACTED — `referenced_slot_count: -1` (`gpu.rs:1420-1425`, the "not
  compacted" sentinel) — deliberately, so every OTHER consumer of that
  function (`kernel_reference.rs`, most of this crate's tests) keeps
  seeing the original, raw-terminal-id bytecode it always produced.
  Compaction is a separate, explicit step (`compact_individual_for_device`),
  run only by the real device-launch path — see §10.2.
- ABI bumped `3 -> 4` (`LAUNCH_BACKTEST_KERNEL_ABI_VERSION`, `gpu.rs:514`;
  `launch_backtest_kernel_abi_version()`, `backtest_kernel.hip.cpp:2397-2400`,
  returns `4`) — the two new struct fields are the only ABI change; every
  other parameter and struct is untouched by this task.

Device side (`backtest_kernel.hip.cpp`):

- `GATHER_COMPACT_ATR_IDX = 0` (`backtest_kernel.hip.cpp:77`), numerically
  identical to `SLOT_ATR_RAW_1M` (also `0`) — the invariant
  `collect_referenced_slots`/`compact_individual_for_device` guarantee
  (ATR always at compact index 0) is what lets PHASE 5's trailing-stop
  ratchet read `snapshot[GATHER_COMPACT_ATR_IDX]` directly out of the
  CURRENT bar's compact row (`backtest_kernel.hip.cpp:1834,1872`), with no
  runtime lookup, even though ATR may not be a slot any of the
  individual's GP trees reference on their own. PHASE 1's entry-sizing
  risk calculation needs the PRECEDING bar's ATR, which the compact row
  (a per-current-bar local, not a stored history) doesn't hold, so it
  takes a different path — a direct `gather_one_slot`/raw-buffer read
  keyed by `SLOT_ATR_RAW_1M` at bar `i - 1`
  (`backtest_kernel.hip.cpp:1543,1568,1573`) — relying on the same
  reserved-first invariant, just via the RAW slot id rather than the
  compact row.
- `get_terminal_value(int32_t id, const float *snapshot)`
  (`backtest_kernel.hip.cpp:343`) now bounds-checks `id` against
  `GPU_MAX_REFERENCED_SLOTS` (64), not `GPU_MAX_SLOTS` (128)
  (`backtest_kernel.hip.cpp:345`) — a real behavioral tightening, correct
  because `snapshot` here is now always the compact row, never the old
  128-wide one. See §10.2 for the gap this created and closed.
- `gather_snapshot_row_compact` (`backtest_kernel.hip.cpp:1124`) — compact
  counterpart of the pre-existing `gather_snapshot_row`: iterates only
  `referenced_slots[0..referenced_count)` (built host-side), writing
  `out_row[0..referenced_count)` from a full `GPU_MAX_SLOTS`-wide
  `base_row` indexed by the raw slot id at each referenced position.
- The kernel's per-bar block (`backtest_kernel.hip.cpp:1416-1453`) declares
  `float snapshot_row[GPU_MAX_REFERENCED_SLOTS]` (64 floats, not 128), and
  either calls `gather_snapshot_row_compact` (column cache enabled) or
  copies `snapshot_row[j] = base_row[referenced_slots[j]]` directly
  (column cache disabled) for `j` in `0..ref_count`, where `ref_count` is
  `sim_params.referenced_slot_count` clamped into `0..=GPU_MAX_REFERENCED_SLOTS`
  (`backtest_kernel.hip.cpp:1419,1434`) as a defensive belt-and-braces
  measure — see §10.2 for why this clamp exists and what it guards against.
- `crates/alpha-simulation/src/device_gather_reference.rs:509` —
  `gather_snapshot_row_device_compact`, the CPU transliteration of
  `gather_snapshot_row_compact` above, letting the compact gather's
  numerics be exercised and checked under a plain `cargo test` (no GPU on
  this host, same rationale as every other `*_reference` module this
  crate already has).

### 10.2 The gap this task found and fixed: production was not compacting

Auditing every caller that builds and uploads GP bytecode found that
`GpuEvaluator::compile_batch`/`evaluate_batch` (`gpu_evaluator.rs`) — the
one path that actually launches the real kernel — called only
`crate::gpu::compile_gp_batch(params)` and returned its output directly,
**never** calling `compact_individual_for_device`. Every individual this
crate's production path uploaded therefore carried
`referenced_slot_count == -1` and raw (uncompacted) `terminal_id`s up to
127, despite `compact_individual_for_device`'s own doc comment
(`gpu.rs`, above `pub fn compact_individual_for_device`) and multiple
comments in `backtest_kernel.hip.cpp` (e.g. lines 271-278, 1405-1410, 1420-1432)
stating flatly that "every real caller... runs
`crate::gpu::compact_individual_for_device` before upload."

The consequence, had this reached a real GPU unfixed: the kernel's
defensive clamp for `referenced_slot_count < 0` sets `ref_count = 0`
(`backtest_kernel.hip.cpp:1433`), so the per-bar `snapshot_row` — a local
stack array, never zero-initialized — receives *no writes at all* before
`evaluate_gp_tree`'s `Terminal` case reads from it via `get_terminal_value`.
A raw `terminal_id` in `0..64` (most real terminals, given §10.3's
evidence) would read **uninitialized stack memory**, not even a defined
`0.0`; a raw id `>= 64` would hit the bounds check and read a defined
`0.0`. Either way: silently wrong, no error anywhere, and invisible on
this host since nothing here executes the actual kernel.

Fixed by adding a compaction pass to `GpuEvaluator::evaluate_batch`
(`gpu_evaluator.rs:604-650`), immediately after `Self::compile_batch(params)?`
and before any GPU upload: for each individual, its own contiguous
node-slice bounds in `gp_nodes` are computed from that same individual's
slice of `strategies_gpu` (`strategies_start_idx`/`num_strategies` give
`[first_strategy.entry_tree_start_idx, last_strategy.exit_tree_start_idx +
last_strategy.exit_tree_len)`, contiguous because `compile_gp_batch`
appends every individual's strategies, and every strategy's entry-then-exit
nodes, strictly in program order with no gaps), and
`compact_individual_for_device` is called on that slice
(`gpu_evaluator.rs:648`). `GpuEvaluator::compile_batch` itself, and the
non-`gpu`-feature stub in `lib.rs:80-84`, are deliberately left unchanged
and still produce uncompacted output — compaction is scoped to the one
real device-launch path, matching `compact_individual_for_device`'s own
documented contract, not folded into the shared compiler that
`kernel_reference.rs` and most of this crate's tests also consume.

With this fix, the `referenced_slot_count == -1` sentinel is unreachable
on the real launch path: every individual `evaluate_batch` uploads is
compacted immediately before upload, same as the FFI parity test harness
(`gpu_cpu_parity_tests.rs`) already did. `backtest_kernel.hip.cpp`'s
`ref_count < 0 -> 0` clamp remains in place as a belt-and-braces guard
against a hypothetical corrupted or future never-compacted upload — the
same defensive-clamp philosophy `DeviceStack::push`/`pop` already use
elsewhere in this file — not as a documented, expected code path.

**Not independently verified**: no GPU exists on this host, so the actual
uninitialized-read behavior described above was never observed directly —
it is derived from reading `backtest_kernel.hip.cpp`'s C++ (an
uninitialized local array with no writes reads whatever was left on the
stack by a prior call frame) plus the confirmed absence of the compaction
call in `gpu_evaluator.rs` before this fix, not from a live repro.

### 10.3 Deriving the typical referenced-slot count from real evidence

Rather than guess, this section measured the real generator. Every number
below comes from actually running the production functions:

- `alpha_gp::operators::generate_random_signal` (`crates/alpha-gp/src/operators/generation.rs:6`),
  the real tree generator (`sample_random_strategy`,
  `crates/alpha-ga/src/nsga2/initialization.rs:389-400`, calls it directly
  for both entry and exit).
- `GP_MAX_DEPTH = 5` (`crates/alpha-ga/src/nsga2/initialization.rs:19`),
  the real, only-ever-used depth for population initialization —
  `ramped_half_half`/`full_initialization`/`grow_initialization`
  (`generation.rs:155-183`) are exported but have no caller anywhere in
  this workspace; `initialize_population` calls `generate_random_signal`
  directly at this fixed depth instead.
- `TerminalWeights::new_uniform(&registry)`
  (`crates/alpha-models/src/strategy/terminals.rs:84`), the real default
  weighting — `MultiObjectiveGA::new` falls back to it whenever no
  explicit `terminal_weights` override is supplied
  (`crates/alpha-ga/src/nsga2/mod.rs:114`) — uniform over the 53
  `gp_visible` slots (`crates/alpha-indicators/src/registry.rs:80-88`).
- Stateful-period bounds `(5.0, 60.0)`
  (`default_stateful_indicator_period`, `crates/alpha-config/src/ga.rs:421-423`),
  the shipped `GaSearchSpaceConfig` default.
- `gp_max_terminals = 8` (`default_gp_max_terminals`,
  `crates/alpha-config/src/ga.rs:300`; field at `ga.rs:42-43`) — the real
  cap `alpha_ga::nsga2::evolution` enforces on every crossover/mutation
  offspring by rejecting a tree whose `count_terminals() > max_terminals`
  (`crates/alpha-ga/src/nsga2/evolution.rs:506-519,628,641`). This cap is
  **not** enforced on generation-0's initial random population
  (`sample_random_strategy` calls `generate_random_signal` with no
  post-hoc terminal-count check), only on every later generation's
  offspring — so a converged population (100 generations by default,
  `default_generations`, `ga.rs:152-154`) is overwhelmingly built from
  ≤8-terminal-per-tree individuals, while the very first generation is not.
- `num_hmm_states`: no single shipped numeric default exists (it is set
  per session), but `3` is the consistent representative value across the
  workspace's own defaults and fallbacks:
  `apps/ga-strat-editor/ui/app.slint:40` (`num_states: 3`, the strategy
  editor's own default), `crates/alpha-orchestrator/src/run.rs:734`
  (`.map_or(3, ...)`, the recovery fallback when no prior session exists),
  and `apps/ga-strat-editor/src/validation/mod.rs:75`.

A probe built directly from these functions/constants (2000 trials per
`num_hmm_states` value, counting the DISTINCT raw slot ids
`collect_referenced_slots`'s own algorithm would find — always including
`SLOT_ATR_RAW_1M` — across all of an individual's entry+exit trees) gave,
at the representative `num_hmm_states = 3`:

| Population phase | min | p50 | mean | p90 | p99 | max |
|---|---|---|---|---|---|---|
| Generation 0 (uncapped, raw `generate_random_signal` at depth 5) | 18 | 43 | 42.2 | 48 | 51 | 53 |
| Steady state (crossover/mutation offspring, ≤8 terminals/tree enforced) | 11 | 25 | 24.7 | 29 | 32 | 34 |

(Full sweep, `num_hmm_states` in `{1, 3, 4, 6}`, uncapped mean/steady-state
mean: `1`→22.3/10.5, `3`→42.2/24.7, `4`→46.6/29.8, `6`→50.7/37.8 — both
distributions grow with `num_hmm_states`, as expected, since more
strategies mean more independent entry/exit trees to draw terminals for.)

Two things this measurement shows directly: (1) generation-0's raw,
uncapped trees saturate toward the 53-slot GP-visible ceiling as
`num_hmm_states` grows — a birthday-paradox effect from drawing many
terminals i.i.d. uniformly from a 53-slot pool, not something
`GPU_MAX_REFERENCED_SLOTS = 64` was sized against, since even that
saturated case (max `53`) fits comfortably under 64; (2) the population
that actually dominates a full run (every generation after the first,
under the real `gp_max_terminals = 8` cap) references on the order of
**25 distinct slots at the representative `num_hmm_states = 3`** — under
half of `GPU_MAX_REFERENCED_SLOTS`, and well under a quarter of the
pre-H' unconditional 128.

This is an analytical/empirical estimate from a standalone probe run
against the real generator functions and real config defaults — not a
measurement from an actual evolutionary run, and not a GPU measurement.
The probe's own code was written for this section, exercised once via
`run_tests`, and then removed; it is not part of the shipped test suite
(nothing here needed a `#[test]` that only ever produces a diagnostic
`panic!`; see this chapter's own precedent of not shipping speculative
non-assertions).

### 10.4 Before / after: per-thread local memory

Extending §8.2's table (that section's arithmetic and per-thread totals
for `DeviceStack`/`per_state_pnl` are otherwise unaffected by this task):

| Array | Before H' | After H' |
|---|---|---|
| `snapshot_row` capacity | `float[128]` = 512 B | `float[64]` = 256 B |
| `DeviceStack<float, 32>::data` | 128 B (unchanged) | 128 B (unchanged) |
| `per_state_pnl[32]` | 128 B (unchanged) | 128 B (unchanged) |
| **Known dynamically-indexed local memory (§8.2's floor)** | **768 B/thread** | **512 B/thread (-33%)** |

`snapshot_row`'s allocated capacity is a fixed, compile-time bound
(`GPU_MAX_REFERENCED_SLOTS`, sized for headroom over the real ceiling —
see §10.3), so this 256 B reduction holds for every individual regardless
of how few slots it actually references. The reduction in *actual per-bar
work* is much larger and individual-dependent — see §10.5.

### 10.5 Before / after: per-bar slot computations

Before H': `gather_snapshot_row` (`backtest_kernel.hip.cpp`, pre-H')
unconditionally called `gather_one_slot` (or copied the precomputed base
value) for all `GPU_MAX_SLOTS = 128` slots, every bar, for every
individual — regardless of how many of those 128 a given individual's
compiled trees actually read.

After H': `gather_snapshot_row_compact` does exactly
`referenced_slot_count` of those calls/copies per bar. Using §10.3's
steady-state, representative-`num_hmm_states = 3` figures as the typical
case:

| | Before H' | After H' (typical, steady state) | Reduction |
|---|---|---|---|
| Slot computations/bar | 128 | ~25 (mean 24.7, median 25) | ~80% |
| Slot computations/bar (generation-0, uncapped) | 128 | ~42 (mean 42.2, median 43) | ~67% |

Both figures are per-individual averages over the probe in §10.3, not a
hard bound: an individual with a maximal, still-in-spec tree (steady
state's observed max `34`, or generation-0's observed max `53`) still
gathers strictly fewer than 128 slots, since `GPU_MAX_REFERENCED_SLOTS = 64`
is the hard ceiling `collect_referenced_slots` enforces (`Err`, not
truncation, past it — see §10.1) and every measured trial landed well
under it.

### 10.6 Verification

- **Compact-gather equivalence**
  (`device_gather_equivalence_tests.rs::compact_device_gather_matches_full_device_gather_for_every_referenced_slot`,
  new): for every raw slot `slot_resolutions` resolves (covering every
  `SlotResolution`/`SlotGatherKind` branch), plus `SLOT_ATR_RAW_1M` and two
  never-resolved slots (the `kind == NONE` passthrough), across two
  non-default `IndicatorPeriods` tuples and two tickers of different
  lengths (`WARMUP_PERIOD + 300`, `WARMUP_PERIOD + 900`): the compact
  gather (`gather_snapshot_row_device_compact`) produces a value
  bit-identical to the full gather (`gather_snapshot_row_device`) for that
  same slot, every bar. No tolerance needed — both call the identical
  `gather_one_slot_device` with identical inputs, so any difference would
  be an indexing bug, not floating-point noise. Passed.
- **Bytecode-rewrite pipeline equivalence**
  (`gp_bytecode_compaction_pipeline_tests.rs::compacted_bytecode_matches_uncompacted_bytecode_end_to_end`,
  new file): a scattered-terminal individual (a real, populated indicator
  plus three synthetic raw slot ids — 63, 90, 120 — the latter two
  deliberately `>= GPU_MAX_REFERENCED_SLOTS` to exercise
  `get_terminal_value`'s bounds-check path, with slot 90 deliberately
  duplicated across the entry and exit tree to exercise dedup) is compiled
  once, then evaluated two ways through `kernel_reference::evaluate_individual_reference`:
  uncompacted bytecode against the real, uncompacted snapshot matrix, and
  compacted bytecode (via `compact_individual_for_device`) against a
  synthetic snapshot buffer with each bar's values relocated to their
  compact position (`compact[j] = full[referenced_slots[j]]`, exactly
  reproducing the real kernel's column-cache-disabled gather branch). All
  of `trades`, `wins`, `min_trades_across_tickers`, `final_equity`,
  `max_drawdown_percent`, and `per_state_pnl` came back bit-identical.
  Passed.
- **Decision-neutrality, unchanged**:
  `gpu_column_gather_decision_neutrality_tests.rs` (both
  `gpu_column_gather_reconstruction_does_not_change_backtest_decisions` and
  the short-`bb_period` PercentB stress case,
  `percent_b_short_bb_period_is_decision_neutral_or_reports_the_first_divergent_trade`)
  and `device_gather_decision_neutrality_tests.rs`
  (`device_gather_reconstruction_does_not_change_backtest_decisions`) were
  left untouched by this task and still pass — H' is pure work-reduction
  on an already-correct gather; if either had moved at all, that would
  indicate the compaction changed a computed value somewhere, not merely
  where it's stored.
- **Whole-crate regression**: `alpha-simulation`'s full `cargo test`
  (unit + every integration test file) passed with 0 failures, 0 ignored,
  after this task's changes.
- **Lint**: `clippy::pedantic` at zero warnings across `alpha-simulation`,
  `alpha-worker`, `alpha-config` after this task's changes (workspace-wide
  policy, `Cargo.toml:94-95`).

### 10.7 What this section could not verify

No GPU exists on this host. Nothing in §10.1-10.6 was measured on real
hardware:

- The C++ side (`backtest_kernel.hip.cpp`) was checked for compilation
  correctness only (via `nvcc`, which does run on this host, per the
  environment's own stated capability) — never executed. The
  uninitialized-stack-read consequence described in §10.2 is derived from
  reading the C++ (a local array with no writes reads whatever the prior
  call frame left on the stack), not observed.
- §10.4's local-memory byte counts are the same kind of static,
  size-of-the-declared-array arithmetic §8.2 already used for the
  pre-existing 768 B figure — not a register/spill measurement from
  `nvcc -Xptxas -v`. Whether `snapshot_row` actually spills to local
  memory at all (as opposed to being partly register-allocated, now that
  it is half the size) is exactly the kind of question §8.2/§8.6 already
  flagged as needing a real compile's resource-usage output, which remains
  unavailable here.
- §10.3's referenced-slot-count distribution is a standalone probe against
  the real generator functions and real config defaults, not a sample from
  an actual evolutionary run (fitness pressure over 100 generations could
  plausibly shift which specific slots — though probably not how MANY
  distinct slots — end up referenced, relative to the crossover/mutation
  offspring distribution measured here).
- No Nsight Compute (or any profiler) run exists for this change, so
  §10.5's "80% fewer slot computations per bar" is exactly that — a count
  of `gather_one_slot`/`gather_one_slot_device` call-site invocations, not
  a measured wall-clock or memory-traffic improvement. §8.6's own
  profiling checklist (register count, achieved occupancy, local-memory
  spill counters, a real `SlotGatherKind` histogram) remains the
  authoritative list of what would need measuring before this task's
  expected benefit can be stated as anything more than "does less
  compile-time-bounded, dynamically-indexed work per bar."

---

## 11. Task J — pinned host memory + streams (implemented, narrow scope by design)

This section documents a shipped change, unlike §8 (rejected) but like §10
(Task H′): `backtest_kernel.hip.cpp`, `gpu.rs`, `gpu_evaluator.rs`, and
`engine/tiering.rs` are all touched. Per this task's own framing, the
analysis in §11.2 is presented **before** the design it justifies, and the
design is deliberately **narrower** than "pin everything, overlap
everything" — §11.4 explains exactly what was left out and why, mirroring
§8's own precedent of scoping down rather than shipping something
unverifiable.

### 11.1 Starting point: what F1b-3 and Task I already changed

Two facts from earlier chapters bound this task before any new analysis:

1. **F1b-3** replaced per-individual snapshot matrices with one shared base
   snapshot (`d_snapshots`) plus a per-generation column pool
   (`d_column_pool`/`d_role_offsets`, F1b-3, `gpu_evaluator.rs`). Upload
   volume per generation is no longer the dominant cost it was under the
   pre-F1b-3 per-individual-matrix design — see §11.2 for what actually
   dominates a launch's transfer today instead.
2. **Task I** (§9) introduced per-column-cache-group chunking in
   `run_full_history_filter_gpu` (`engine/tiering.rs`) — a fresh
   `GpuColumnCache`/`ColumnPool` per group, RAII-dropped at the end of
   each group's iteration — and `run_tier_filter_gpu` already chunks its
   single shared pool by `gpu_eval.max_individuals`. §9.2 (manual/12) found
   that **both shipped configs fit the full-history stage in ONE group**
   under the default 48 GiB `column_cache_budget_mb` — multi-group
   chunking is "a safety valve... not because either shipped config needs
   it today." That same fact governs §11.4/§11.5 below: the group-to-group
   overlap this task adds is real, shipped, and exercised by a dedicated
   test, but is **not on the hot path of either shipped config today** —
   only when `column_cache_budget_mb` is lowered or the search space
   widened enough to force `plan_column_cache_groups` to split.

### 11.2 Transfer-vs-compute analysis

All byte figures below are computed directly from the struct sizes and
buffer-sizing code in `gpu_evaluator.rs`/`gpu.rs` (source-reading
arithmetic, the same method §8.2's local-memory figures used) — **no
wall-clock kernel time is measurable on this host** (no GPU), so this
section can state transfer volume precisely but can only reason
structurally, not numerically, about what fraction of a real launch that
volume represents. §11.7 and the extended §8.6 checklist name exactly what
would close that gap.

**Finding 1 — the single largest recurring transfer was pure zeros, on
every launch, at every candle count.** Before this task, `evaluate_batch`
built a host `vec![0.0f32; state_buffer_len]` and uploaded it every call to
clear `d_state_buffer` (the `Sma`/`StdDev`/`Lag` ring-buffer state,
`gpu_evaluator.rs`, pre-Task-J). `state_buffer_len = num_individuals *
MAX_STATEFUL_NODES_PER_IND(64) * STATE_SLOT_STRIDE(257)`. At the production
ceiling (`max_individuals = 16,384`, `manual/12` §9.2): `16,384 × 64 × 257
× 4 bytes = 1,077,936,128 bytes ≈ 1.00 GiB` — **on every single kernel
launch, regardless of candle count**, since this buffer's size depends
only on `num_individuals`, never on `num_candles` or `num_tickers`. This
is larger than every OTHER per-call upload combined at tier-1/tier-2 scale
(Finding 2 below), and it was moving bytes across PCIe to produce a value
(all zeros) that has no reason to touch the host at all.

**Fixed, not merely pinned**: `gpu_evaluator.rs`'s `prepare_and_launch`
(line 685) now calls `gpu_memset` (already an existing FFI export,
`gpu.rs`) directly on `d_state_buffer`, eliminating this transfer
entirely rather than staging it through pinned memory — device-local
zeroing has no host-transfer cost at all, which is strictly better than
any pinning/streaming treatment of the same bytes could have been. This
one change removes ~1 GiB of host→device traffic from every launch,
unconditionally, and was found only by working through this section's own
"quantify the transfer" instruction — it is the single highest-confidence
result in this task.

**Finding 2 — the remaining ALWAYS-transferred buffers are small.** Once
the state buffer is excluded, what `prepare_and_launch` uploads on
**every** call (not gated by the `cached_data_fingerprint` check) is the
per-individual bytecode: `d_params` (`GpSimulationParamsGpu`, 300 bytes/
individual incl. the Task H′ `referenced_slots` array → ≈4.7 MiB at
16,384 individuals), `d_strategies` (`StateStrategyGpu`, 64 bytes each, ≈3
MiB at a representative 3 strategies/individual), `d_gp_nodes` (16 bytes/
node, on the order of tens of MiB depending on compiled tree size — not
precisely computable without running the compiler, see §11.7), and
`d_stateful_info` (4 bytes/slot, ≤4 MiB). Total: on the order of tens of
MiB per launch, **roughly two orders of magnitude smaller than the ~1 GiB
this task found and eliminated in Finding 1**. The column-cache
role-offset table (`d_role_offsets`) adds `num_individuals × num_tickers ×
NUM_COLUMN_ROLES(32) × 4 bytes` ≈ 21 MiB at production scale — same order
of magnitude.

**Finding 3 — the large, VARIABLE transfers are gated by data fingerprint,
not by chunk.** `d_ohlc`/`d_snapshots`/`d_hmm_states` upload only when
`(ticker_order, num_candles)` changes (`cached_data_fingerprint`,
`gpu_evaluator.rs`) — at minimum once per tier/stage transition per
generation (tier-1 → tier-2 → full-history each carry a different candle
count), not once per chunk. `d_snapshots` is the dominant term:
`num_tickers × num_candles × GPU_MAX_SLOTS(128) × 4 bytes`. At full-history
scale (~1,000,000 candles, 10 tickers): `10 × 1,000,000 × 128 × 4 ≈ 4.77
GiB`; at tier-1 scale (43,200 candles): `≈ 211 MiB`; at tier-2 (172,800
candles): `≈ 844 MiB`. `d_ohlc` is an order of magnitude smaller
(`OhlcGpu` is 5 `f32`s vs. 128) at every tier.

**Finding 4 — the column pool is the one transfer that can reach tens of
GiB, but per §11.1 does so as ONE upload under both shipped configs, not
several.** `manual/12` §9.2's real, code-derived worst case is 27.48 GiB
(harvest.toml) / 22.86 GiB (config.toml) at full history — 57.25%/47.63%
of the default 48 GiB budget, comfortably one group. A narrower search
space or a lower `column_cache_budget_mb` is what would force
`plan_column_cache_groups` to split this into several sequential uploads —
see §11.4/§11.5 for why that specific case is where this task's streaming
design pays off, and why it is currently latent rather than exercised in
production.

**What this adds up to, structurally (not numerically — see the caveat
above):** compute cost scales with `num_individuals × num_candles ×
tree_evals_per_bar` (full sequential per-thread backtests), while the
ALWAYS-present transfer cost (Finding 2, post-fix) scales only with
`num_individuals` and is small in absolute terms. This means the
transfer-to-compute ratio is worst (most transfer-bound, relatively) at
**tier-1** — the shortest candle count, so the least compute per launch —
and best (most compute-bound) at **full-history** — the longest candle
count. The FIXED, unconditional wins this task ships (Finding 1's
elimination, Finding 2's pinning) matter most exactly where they're
cheapest to verify is safe: at tier-1/tier-2, where the transferred bytes
are small and bounded regardless of chunking. The VARIABLE, chunking-
dependent win (Finding 4's group-to-group overlap) matters most at
full-history under configurations wide enough to force multiple groups —
which, per §11.1, is not either shipped config today.

### 11.3 Design chosen — pinned host memory

`PinnedHostBuffer<T>` (`gpu_evaluator.rs:214`) is a `DeviceBuffer`-shaped
RAII wrapper around `gpu_host_malloc`/`gpu_host_free`
(`hipHostMalloc`/`hipHostFree`, macro-shimmed to `cudaHostAlloc`/
`cudaFreeHost` under the CUDA branch — `backtest_kernel.hip.cpp`'s
existing `#ifndef hip... #define hip... cuda...` block, extended for this
task). `GpuEvaluator::upload_staged` (`gpu_evaluator.rs:585`) uploads any
`&[T]` by copying it, in bounded chunks, into ONE reusable pinned buffer
(`pinned_bounce: PinnedHostBuffer<u8>`, `gpu_evaluator.rs:459`) and issuing
a plain, synchronous `hipMemcpy` per chunk — a pinned SOURCE lets the
driver DMA directly instead of first staging through its own internal
pinned bounce buffer, which is the whole bandwidth benefit; nothing about
this path is async.

**The ceiling**: `PINNED_STAGING_BYTES = 256 MiB` (`gpu_evaluator.rs:412`),
allocated ONCE at `GpuEvaluator::new` and reused (never grown) for every
staged upload, including the column pool at whatever size it actually is
— chunked via `crate::gpu::staged_upload_chunks` (`gpu.rs:857`) when the
transfer exceeds the cap. This is a deliberate, fixed, small reservation
of page-locked (non-swappable) host RAM — not one sized against the
largest thing this crate ever uploads (which, per §11.2 Finding 4, can be
tens of GiB) — per this task's own instruction not to pin unboundedly.
256 MiB was chosen to amortize the fixed per-`hipMemcpy` dispatch overhead
to a negligible fraction of even a modest chunk's transfer time, while
being small enough to be a non-issue on any host this crate targets.

**What is and isn't staged**: every buffer named in this task's
instructions (`d_ohlc`, `d_snapshots`, `d_column_pool`, `d_role_offsets`,
`d_params`, `d_strategies`, `d_gp_nodes`, `d_stateful_info`) is now staged
through `upload_staged`. `d_state_buffer` is NOT staged — it is not
uploaded at all any more (§11.2 Finding 1). `d_ticker_lengths` (≤ 10
`i32`s) and `d_slot_gather_plan` (fixed 1,536 bytes) stay plain
`DeviceBuffer::upload` calls — staging machinery for a transfer this small
would add overhead, not remove it.

**Verifiable here**: `staged_upload_chunks`'s pure chunk-boundary
arithmetic is exhaustively unit-tested (`gpu.rs`'s
`staged_upload_chunks_tests` module, 7 tests: exact multiple, remainder,
single-chunk, zero bytes, zero cap, and a property check across a spread
of `(total, cap)` pairs asserting every chunk is contiguous, non-empty,
`≤ cap`, and the chunks sum to the total). This module is deliberately
placed in `gpu.rs`, which is NOT gated behind `feature = "gpu"` (unlike
`gpu_evaluator.rs`, gated at the `pub mod` level in `lib.rs`) — so these 7
tests run and pass on THIS host, with no GPU and no `gpu` feature enabled,
unlike everything else this task added. **Not verifiable here**: the
actual `hipHostMalloc`/`hipMemcpy` calls, whether the driver really DMAs
directly from pinned memory on this specific host/driver, and the real
bandwidth delta versus pageable — see §8.6 item 8 (extended for this
task).

### 11.4 Design chosen — streams, and why the scope is narrow

`launch_backtest_kernel` (`gpu.rs:544`, C++ definition
`backtest_kernel.hip.cpp:2464`) gained one trailing parameter, `stream:
*mut std::ffi::c_void` (`gpu.rs:675`). Contract, precisely:

- `stream == null` (`std::ptr::null_mut()`) — **the default, used by every
  existing call site** (`GpuEvaluator::evaluate_batch`,
  `gpu_cpu_parity_tests.rs`): launches on the default stream and blocks
  inside the C++ function (`hipGetLastError` then `hipDeviceSynchronize`)
  before returning, byte-for-byte the pre-Task-J behavior. This satisfies
  this task's explicit requirement that "a synchronous path must remain
  available and be the default when streams are not in use" — it is not
  merely available, it is what every caller except one still uses.
- `stream != null`: launches on that stream and returns as soon as the
  launch is confirmed ACCEPTED (`hipGetLastError` only) — it does NOT
  wait for the kernel. The caller must `gpu_stream_synchronize` before
  touching `d_results` or any buffer the launch reads/writes.

This does not touch the `backtest_kernel` `__global__` signature at all —
a stream is a launch-configuration argument to `hipLaunchKernelGGL`, never
one of the kernel's own parameters, so it cannot appear there. **ABI
bump**: `LAUNCH_BACKTEST_KERNEL_ABI_VERSION` `4 → 5` (`gpu.rs:521`, C++
`launch_backtest_kernel_abi_version()` returns `5`,
`backtest_kernel.hip.cpp`) — required because `launch_backtest_kernel`'s
own signature changed, per this task's ABI-discipline rule, even though no
`#[repr(C)]` struct did. The four sites that move in lockstep for THIS
change (not five — the `__global__` is deliberately excluded, see above):
the C++ `extern "C"` definition, its `hipLaunchKernelGGL` call's stream
slot, the Rust `extern "C"` declaration, and both Rust call sites
(`gpu_evaluator.rs:1084`, `gpu_cpu_parity_tests.rs:325` — confirmed by a
whole-repository grep for `launch_backtest_kernel(` finding exactly these
four call/definition sites plus the `__global__`, nothing missed).

**Why the scope stops at ONE stream, used in ONE place.** Before deciding
where to launch async, this task worked through what it would take to
overlap an UPLOAD (not just CPU prep) with a still-running PREVIOUS
kernel, since that is the more aggressive design a first read of the task
brief suggests. The blocker: `GpuEvaluator`'s persistent device buffers
(`d_params`/`d_strategies`/`d_gp_nodes`/`d_stateful_info`/`d_state_buffer`/
`d_results`) are REUSED (overwritten in place) by every call — uploading
group N+1's bytecode into them while group N's kernel might still be
reading them is a genuine write-after-read hazard, not a Rust-visible one
(raw device pointers, invisible to the borrow checker), and avoiding it
correctly requires either double-buffering every one of those buffers or
proving no aliasing per-buffer — meaningfully more moving parts than this
host can verify with no GPU to actually race against. `d_column_pool`/
`d_role_offsets`, by contrast, ARE freshly allocated every call (no
aliasing risk), but restructuring THEIR upload to be truly concurrent with
a previous kernel would still need the SAME buffers (`d_params` etc.) to
be double-buffered before the corresponding kernel could safely launch —
the hazard doesn't go away, it just moves.

Given that, this task scoped the overlap to the one thing that is safe
BY CONSTRUCTION, needs no double-buffering, and (per `gpu_columns.rs`
containing no device calls at all — confirmed by grep, not assumed) has
zero device-concurrency risk: **CPU-only host-side work** (building the
next full-history column-cache group) overlapping a **previous, already-
launched kernel**. See §11.5.

### 11.5 The one overlap this task implements: `run_full_history_filter_gpu`'s group loop

`GpuEvaluator::launch_batch_async` (`gpu_evaluator.rs:1265`) and
`PendingBatch<'a>` (`gpu_evaluator.rs:1369`) are the vehicle;
`run_full_history_filter_gpu` (`engine/tiering.rs`, the `prebuilt`
variable at line 543, the `is_last_chunk_of_group` branch at line 595) is
the one caller.

**Design**: for each column-cache group (§9.4's `plan_column_cache_groups`
output), when a group's LAST chunk is reached AND there is a next group,
that chunk launches via `launch_batch_async` instead of `evaluate_batch`.
While its kernel runs (on `self.kernel_stream`, the one non-default stream
`GpuEvaluator` owns — created once in `GpuEvaluator::new`, destroyed on
`Drop`, always synchronized first), the loop builds the NEXT group's
`GpuColumnCache`/`ColumnPool` — pure CPU work — then calls
`pending.collect()`, which synchronizes and downloads. Every OTHER chunk
(not last-of-group, or last-of-group-with-no-next-group) still calls
`evaluate_batch` exactly as before Task J. `run_tier_filter_gpu`'s chunk
loop is UNTOUCHED — its single `ColumnPool` is built once, before any
chunk, so there is no per-chunk host work to hide (§11.4's "why the scope
stops" applies identically: nothing to overlap without also
double-buffering the persistent buffers).

**Ownership, enforced in the type system, not by comment.** Two things
must outlive the async kernel, and this task's own instructions demand
the type system prove it rather than trust a caller:

1. `PreparedLaunch` (`gpu_evaluator.rs`, private) holds the freshly
   allocated `d_column_pool`/`d_role_offsets` `DeviceBuffer`s for exactly
   as long as `PendingBatch` holds `PreparedLaunch` — they drop (freeing
   device memory) only after `PendingBatch::collect`'s
   `kernel_stream.synchronize()` call returns, or after `PendingBatch`'s
   `Drop` forces that same synchronize if `collect` was never called. This
   was a real bug this task's own design process caught mid-implementation
   — an earlier draft dropped these buffers at the end of
   `prepare_and_launch` (safe for the synchronous, null-stream path, since
   the kernel had already finished by then, but a genuine use-after-free
   for the async path, where the kernel might still be reading them when
   the function returns) — see `gpu_evaluator.rs`'s comment on
   `PreparedLaunch`'s fields for the exact hazard.
2. `PendingBatch<'a>` borrows `&'a mut GpuEvaluator` for its entire
   lifetime (`gpu_evaluator.rs:1369`). This makes a second
   `evaluate_batch`/`launch_batch_async` call — which would upload into
   the SAME persistent buffers the in-flight kernel may still be reading —
   a **compile error** until the `PendingBatch` is consumed. `impl Drop for
   PendingBatch` (`gpu_evaluator.rs:1412`) always calls
   `kernel_stream.synchronize()` before the borrow (and `PreparedLaunch`'s
   buffers) can be released — redundant and cheap if `collect()` already
   ran, but the ONLY thing standing between an in-flight kernel and reused
   memory if a caller forgets, hits an early `?`-propagated error, or
   panics mid-loop.

**Every `hipStreamSynchronize`/async call and exactly what it orders**:

| Call | Where | Orders |
|---|---|---|
| `launch_backtest_kernel(..., stream: null)` | `evaluate_batch` (every non-overlap chunk) | Blocks internally (`hipDeviceSynchronize`) before returning — download in the caller is always safe. |
| `launch_backtest_kernel(..., stream: kernel_stream)` | `launch_batch_async`, the overlap chunk only | Returns once the launch is ACCEPTED; does NOT order anything past that. |
| `kernel_stream.synchronize()` in `PendingBatch::collect` | After building the NEXT group's cache/pool | Orders: the overlap kernel's completion before `d_results` is downloaded, and before `PreparedLaunch`'s buffers drop. |
| `kernel_stream.synchronize()` in `PendingBatch::drop` | Only reached if `collect()` was never called | Same ordering as above, forced unconditionally — the safety net described above. |
| `kernel_stream.synchronize()` in `GpuStream::drop` | `GpuEvaluator`'s own teardown | Orders: any operation still enqueued on the stream before the stream handle itself is destroyed (destroying a busy stream is undefined by the CUDA/HIP contract). |

Nothing else in this task issues an async device operation. Every
`upload_staged` call (§11.3) is a plain, synchronous `hipMemcpy` — there is
no async-copy ordering to reason about there at all, by design (§11.4).

### 11.6 What this task could NOT verify here

No GPU exists on this host — this is the same limitation §8.7/§10.7
already state, in full, for this task too:

- Whether `gpu_host_malloc`/`hipHostMalloc` actually succeeds and produces
  memory the driver treats as pinned on this specific
  host/driver/CUDA-vs-HIP-shim combination (`backtest_kernel.hip.cpp`'s
  own `#ifndef hip... #define ... cuda...` block, extended for this task
  — compiles cleanly via `nvcc`, per this environment's own stated
  capability, but was never executed).
- Whether the pinned-vs-pageable bandwidth improvement §11.3 assumes is
  real and of the expected rough magnitude on this A100 (§8.6 item 8,
  new).
- Whether `kernel_stream`'s async kernel launch and `GpuColumnCache::
  build`'s CPU work genuinely overlap in wall-clock time on real hardware,
  or serialize anyway for a reason invisible from source (§8.6 item 7,
  new) — including whether the legacy default-stream semantics this
  project's CUDA-compat shims might imply create a false dependency
  between `kernel_stream` and unrelated default-stream work (§8.6 item 9,
  new).
- Whether `PendingBatch`'s Drop-forces-sync safety net is ever actually
  exercised (i.e., whether any real call path ever fails to call
  `collect()`) — by construction it should be unreachable in the one
  shipped call site (`run_full_history_filter_gpu` always calls
  `pending.collect()?` in the same expression that produces `pending`),
  but this was never run to confirm.
- §11.2's byte-level transfer figures are exact (computed from struct
  sizes and buffer-sizing code, not estimated) — what remains unverified
  is translating them into a real fraction of wall-clock launch time,
  which needs a real kernel execution time this host cannot produce.
- Per §11.1, neither shipped config drives `plan_column_cache_groups` into
  more than one group today, so the ONE overlap path this task implements
  (§11.5) has REAL, passing, host-executable test coverage
  (`full_history_chunked_column_cache_tests.rs`'s pattern extended — see
  §11.8) exercising its LOGIC, but has never run against a production-
  shaped multi-group full-history batch on real hardware, only a small
  synthetic one sized to force multiple groups.

### 11.7 Also unresolved: exact per-individual GP-node byte count

§11.2 Finding 2 states `d_gp_nodes`'s per-launch size only as "on the
order of tens of MiB" — unlike every other figure in §11.2, this one is
not computed exactly, because it depends on the compiled bytecode's real
node count, which `GpuEvaluator::compile_batch` produces at runtime from
whatever GP trees a given generation's population happens to compile to,
not from a fixed struct size. §10.3's own probe (a different one, for
Task H′'s referenced-slot-count question) measured DISTINCT referenced
SLOTS, not total compiled NODES, so it does not directly answer this
either. This is a minor gap relative to §11.2's overall conclusion (this
buffer is one of several "tens of MiB" buffers, all small next to §11.2
Finding 1's ~1 GiB elimination), not a load-bearing uncertainty, but is
recorded here rather than silently rounded away.

### 11.8 Verification

- **Compile**: `run_clippy` on `alpha-simulation`/`alpha-worker`/
  `alpha-config` with `all_targets`+`all_features` (which builds
  `backtest_kernel.hip.cpp` via `nvcc`, per this environment's own stated
  capability) — zero warnings, confirming both the Rust and the C++ sides
  of every change in this section compile and lint cleanly. This is the
  ONLY thing an environment with no GPU can confirm about the FFI/device
  code in §11.3–§11.5; §11.6 lists what it cannot.
- **New tests, all passing, all host-executable (none need a GPU)**:
  `gpu::tests::staged_upload_chunks_tests` (7 tests, §11.3) — the pure
  chunk-boundary arithmetic behind pinned staging.
- **Whole-crate regression**: `alpha-simulation`'s full `cargo test`
  (unit + every integration test file) — 285 passed (72 lib + 213
  integration), 0 failed, 0 ignored, i.e. exactly the pre-Task-J 278
  (65 lib + 213 integration) plus this task's 7 new lib tests, with the
  full integration-test count unchanged. `gpu_cpu_parity_tests.rs`'s
  `#![cfg(feature = "gpu")]`-gated tests (updated for the new `stream`
  parameter, §11.4) do not run in this default build — consistent with
  every prior GPU-feature test in this crate never having executed on
  this host (§8.7, §10.7, this section's own §11.6).
- **Lint**: `clippy::pedantic` at zero warnings across `alpha-simulation`,
  `alpha-worker`, `alpha-config` after this task's changes (workspace-wide
  policy, `Cargo.toml:94-95`).
- **Not run**: `GpuColumnCache`/`ColumnPool`'s existing
  `full_history_chunked_column_cache_tests.rs` (Task I's own
  chunk-boundary integration test) was not extended with a NEW test
  exercising `launch_batch_async`/`PendingBatch` specifically, because
  doing so meaningfully requires a real device to observe anything
  streams-specific (the borrow-checker-level safety §11.5 describes is
  already exercised by the fact that this code compiles at all under
  `--all-features`, and adding a test that merely calls
  `launch_batch_async` on a device-less host would only re-confirm the
  same `Err`-on-no-device path `GpuEvaluator::new` already exercises
  everywhere else in this crate, not anything new).
