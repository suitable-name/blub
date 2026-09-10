# Chapter 12: GPU Indicator Column Cache — Investigation for Option A

This chapter is an investigation document, not a description of shipped behavior. It is the input to a planned change ("Option A"): replacing `build_snapshot_matrix`'s per-`IndicatorPeriods`-group snapshot rebuild in the GPU backtest path with a resident, per-`(timeframe, aggregation period, indicator period value)` column cache that the kernel gathers from. No code changes were made while producing this document — every claim below is either backed by a file:line citation or explicitly marked **OPEN**.

---

## 1. Summary of findings

- The `IndicatorBus` has 128 slots (`alpha_indicators::bus::MAX_SLOTS`, `crates/alpha-indicators/src/bus.rs:8`). **56 are occupied** by `build_default_registry()` (`crates/alpha-indicators/src/registration.rs:144-487`): **48 on the 1-minute timeframe, 3 on Agg1, 3 on Agg2**, plus **2 written externally by the HMM engine** (never touched by any indicator pipeline). 53 of the 56 are GP-visible (`gp_visible: true`); 3 are internal-only (`obv_1m`, `lag_close_1m`, `vwap_1m`).
- `IndicatorPeriods` has **31 fields** (`crates/alpha-core/src/lib.rs:207-244`). Exactly **26 are evolved** — randomized in `sample_indicator_periods` (`crates/alpha-ga/src/nsga2/initialization.rs:286-381`), mutated in `mutation_indicator_periods` (`crates/alpha-gp/src/operators/mutation.rs:105-179`), crossed in `crossover.rs:183-266` — and **5 are permanently fixed at `Default`** (`mcv_adx_period`, `mcv_roc_period`, `mcv_sma_period`, `mcv_atr_period`, `mcv_atr_window`; never appear in any of the three call sites above, only ever populated via `..IndicatorPeriods::default()`). This is the crux the task called out: a slot keyed only by a fixed field is **one column for the whole run**, not a family.
- One evolved field, **`atr_volatility_window`, is a dead gene as far as indicator output is concerned** — it is drawn, mutated, crossed, and hashed into `structural_hash`/`max_lookback`, but no indicator runner in `registration.rs` ever reads it. It inflates warmup sizing and cache-key cardinality for zero column-cache benefit.
- Of the six joint (2-period) dependencies the task flagged, **five collapse to two independent single-key columns each** once you read past the constructor signature into the actual math — verified directly in the runner source (§4). Only the raw *arrival* of a joint key (stochastic K/D, vol-osc fast/slow, physics slow/fast, phase-space slow/fast, Bollinger period/σ) is real; the *storage* is not jointly keyed. The sixth (`mcv_atr_period`/`mcv_atr_window`) is jointly keyed but **both** periods are fixed, so it is one column, full stop.
- Applying every collapse found, a full-range precompute of every period-dependent 1m + Agg1 + Agg2 column, at 5 tickers, is **~36.1 GiB (f32) / ~72.2 GiB (f64) at full history** (~1,000,000 candles). Without the collapses, the same computation is **~1.34 TiB (f32)** — dominated almost entirely by Bollinger's period×σ product. The collapses are not an optimization footnote; they are what makes a resident cache possible at all.
- `build_snapshot_matrix`, **at the time this investigation was originally written**, built **only** `Timeframe::OneMin`. Agg1/Agg2 slots were never written by it, so they sat at their `SlotRegistry` default for every bar on GPU (not uniformly `0.0` — `rsi_agg1`/`rsi_agg2`/`stochastic_k_agg1`/`stochastic_k_agg2` default to `50.0`, `atr_raw_agg1`/`atr_raw_agg2` default to `0.0`; `registration.rs:232-267`), while the CPU path (`MultiScaleIndicatorState`, `crates/alpha-models/src/simulation/mod.rs:61-83`) computed them for real. **6 of the 53 GP-visible terminals (11.3%) were affected.** **This has since been fixed** (`manual/14` O09): `build_snapshot_matrix` (`crates/alpha-simulation/src/gpu.rs:785-876`) now ticks real `Timeframe::Agg1`/`Agg2` pipelines too — see §6 for the fix detail and §5.4.1/§5.4.2 for the memory consequence, which was already assumed in §5 below even before the fix shipped.

---

## 2. Per-slot inventory

All slots below come from `build_default_registry()` (`crates/alpha-indicators/src/registration.rs:144-487`) and the constructor calls in `build_default_pipeline_with_bus` (`registration.rs:502-700`), which is the authoritative source for which `IndicatorPeriods` field(s) feed which slot.

### 2.1 1-minute timeframe (48 slots)

| Slot | Period field(s) | Citation | Notes |
|---|---|---|---|
| `atr_raw_1m` | `atr_period` | `registration.rs:516-520` | root; feeds `normalized_atr`, `liquidation_cascade_prob` |
| `rsi_1m` | `rsi_period` | `registration.rs:523-526` | |
| `stochastic_k_1m` | `stochastic_k_period` | `registration.rs:527-532` | |
| `stochastic_d_1m` | `stochastic_k_period` **and** `stochastic_d_period` (jointly, in the constructor) | `registration.rs:527-532`; `stochastic.rs:19-29,73-85` | **derived** — see §4, actually `SMA(k, d_period)` |
| `roc_1m` | `roc_period` | `registration.rs:536-539` | |
| `obv_1m` | none (period-independent) | `registration.rs:541` | not GP-visible |
| `bollinger_middle_1m` | `bb_period` | `registration.rs:604-612`; `bollinger.rs:227-264` | **not** dependent on `bb_std_dev_x100` — see §4 |
| `bollinger_upper_1m` | `bb_period` **and** `bb_std_dev_x100` | same | joint, see §3/§4 |
| `bollinger_lower_1m` | `bb_period` **and** `bb_std_dev_x100` | same | joint, see §3/§4 |
| `bollinger_band_width` | `bb_period` **and** `bb_std_dev_x100` | same | joint, see §3/§4 |
| `percent_b_1m` | `bb_period` **and** `bb_std_dev_x100` | same | joint, see §3/§4 |
| `log_return_1m` | none (period-independent) | `registration.rs:621-623`; `pipeline.rs:524-535` | pure function of `CandleInput.log_return` |
| `lag_close_1m` | `lag_period` | `registration.rs:632-635`; `pipeline.rs:653-693` | trivial index-shift of `close`, see §4; not GP-visible |
| `normalized_atr` | `atr_period` (via `atr_raw_1m`) | `registration.rs:628-631`; `pipeline.rs:600-620` | **derived**, see §4 |
| `mcv_trend_intensity_1m` (= `adx_1m`) | `mcv_adx_period` (**fixed**) | `registration.rs:638-642`; `adx.rs:258-280` | exact alias of `adx_1m`, see §4 |
| `mcv_volatility_state_1m` | `mcv_atr_period` **and** `mcv_atr_window` (**both fixed**) | `registration.rs:654-659`; `atr.rs:131-147` | joint but both fixed — 1 column |
| `mcv_trend_bias_1m` | `mcv_sma_period` (**fixed**) | `registration.rs:649-652` | |
| `mcv_velocity_1m` | `mcv_roc_period` (**fixed**) | `registration.rs:644-647` | |
| `mcv_directional_efficiency_1m` | `ter_period` | `registration.rs:660-663` | |
| `adx_1m` | `mcv_adx_period` (**fixed**) | `registration.rs:638-642` | same runner/value as `mcv_trend_intensity_1m` |
| `return_skewness` | `return_skewness_period` | `registration.rs:551-554` | |
| `price_correlation` | `price_correlation_period` | `registration.rs:542-545` | |
| `hurst` | `hurst_period` | `registration.rs:570-573` | |
| `candle_body_ratio` | none (period-independent) | `registration.rs:618-620`; `pipeline.rs:480-521` | |
| `kinetic_energy` | `physics_fast` (`roc_period` arg) | `registration.rs:574-580`; `physics.rs:104-141` | **not** dependent on `physics_slow` — see §4 |
| `potential_energy` | `physics_slow` (`vwap_period` arg) | same | **not** dependent on `physics_fast` — see §4 |
| `lagrangian` | derived from the two above | same | **derived**, see §4 |
| `momentum_acceleration` | `roc_period` | `registration.rs:581-584` | separate column family from `roc_1m` despite the shared key — internals not inspected, see §7 OPEN |
| `microstructural_entropy` | `atr_period` | `registration.rs:585-588` | separate family from `atr_raw_1m` — internals not inspected, OPEN |
| `phase_space_angular_velocity` | `phase_space_fast` (`roc_period` arg) | `registration.rs:589-594`; `phase_space.rs:79-179` | **not** dependent on `phase_space_slow` alone — see §4 |
| `phase_space_radius` | `phase_space_fast` **and** `phase_space_slow` jointly at the final step | same | see §4 |
| `volume_imbalance_ratio` | `roc_period` | `registration.rs:595-598` | separate family, OPEN |
| `thermo_entropy` | `atr_period` | `registration.rs:599-603` | shares one runner/family with `thermo_temp`, OPEN internals |
| `thermo_temp` | `atr_period` | same | |
| `downside_volatility` | `downside_vol_period` | `registration.rs:566-569` | |
| `volume_oscillator` | `vol_osc_fast` **and** `vol_osc_slow` | `registration.rs:546-550`; `volume_osc.rs:60-92` | joint at the ratio, single-key at the primitive — see §4 |
| `dist_to_vwap` | `vwap_period` (via `vwap_1m`) | `registration.rs:561-565`; `vwap.rs:62-90` | **derived**, see §4 |
| `dist_to_sma` | `price_to_sma_period` | `registration.rs:556-559` | |
| `high_low_disparity` | none (period-independent) | `registration.rs:624-627`; `pipeline.rs:559-598` | |
| `close_location_value` | none (period-independent) | `registration.rs:615-617`; `pipeline.rs:434-455` | |
| `liquidation_cascade_prob` | `lcp_period` **own**, plus reads `atr_raw_1m` (⇒ `atr_period`) and `momentum_acceleration` (⇒ `roc_period`) | `registration.rs:670-675`; `alpha_physics.rs:104-215` | effectively 3-way coupled, see §3 |
| `kinematic_jerk` | none of its own; reads `momentum_acceleration` | `registration.rs:666-669`; `alpha_physics.rs:14-72` | **derived**, see §4 |
| `hmm_persistence` | n/a | `registration.rs:436-441` | written externally by the HMM engine, not by any `IndicatorPipeline` — out of this cache's scope |
| `hmm_regime_stability` | n/a | `registration.rs:442-447` | same |
| `constant_zero` | none | `registration.rs:448-453,678-681` | write-once, hoisted out of the runtime pipeline by `IndicatorPipeline::build_with_bus` (`pipeline.rs:331-350`) |
| `constant_oversold` | none | `registration.rs:454-459,682-685` | |
| `constant_overbought` | none | `registration.rs:460-465,686-689` | |
| `constant_mid` | none | `registration.rs:466-471,690-693` | |
| `constant_small_pos` | none | `registration.rs:472-477,694-697` | |
| `vwap_1m` | `vwap_period` | `registration.rs:561-565`; `vwap.rs:49-90` | not GP-visible (internal) |

That is 14 (1m-standard) + 5 (MCV) + 5 (advanced stats) + 18 (physics/thermo/misc) + 5 (constants) + 1 (`vwap_1m` internal) = **48 rows**, minus the double-listing of `adx_1m`/`mcv_trend_intensity_1m` (one runner, two slots) which the table lists both for completeness.

### 2.2 Agg1 (3 slots) and Agg2 (3 slots)

Built only when `build_default_pipeline_with_bus` runs with `Timeframe::Agg1` / `Agg2` (`registration.rs:493-532`); `stochastic_d_opt` returns `None` for both, so there is no `stochastic_d_agg{1,2}` (`registration.rs:134-139`).

| Slot | Period field(s) | Citation |
|---|---|---|
| `atr_raw_agg1` | `atr_period` **and implicitly** `agg_period_1` (the candle series it consumes is downsampled by `CandleAggregator::new(periods.agg_period_1)`, `crates/alpha-models/src/simulation/mod.rs:80`) | `registration.rs:516-520,232-237` |
| `rsi_agg1` | `rsi_period` and implicitly `agg_period_1` | `registration.rs:523-526,238-243` |
| `stochastic_k_agg1` | `stochastic_k_period` and implicitly `agg_period_1` | `registration.rs:527-532,244-249` |
| `atr_raw_agg2` | `atr_period` and implicitly `agg_period_2` | `registration.rs:516-520,250-255` |
| `rsi_agg2` | `rsi_period` and implicitly `agg_period_2` | `registration.rs:523-526,256-261` |
| `stochastic_k_agg2` | `stochastic_k_period` and implicitly `agg_period_2` | `registration.rs:527-532,262-267` |

Unlike the 1m-vs-atr_period pairing, this "implicit" coupling to `agg_period_{1,2}` is **real, not illusory**: a different `agg_period_1` produces a genuinely different bar series (`CandleAggregator::aggregate`, `crates/alpha-models/src/simulation/aggregator.rs`, not read in full but its role is unambiguous from the call site), so `atr_raw_agg1` for `(atr_period=14, agg_period_1=5)` and `(atr_period=14, agg_period_1=8)` are different columns, not the same column at two names.

### 2.3 Totals

- **Occupied slots: 56 / 128.** Split: **48 OneMin, 3 Agg1, 3 Agg2, 2 external (HMM)**.
- **Period-independent** (no `IndicatorPeriods` field drives them at all): `obv_1m`, `log_return_1m`, `high_low_disparity`, `close_location_value`, `candle_body_ratio`, plus the 5 constants and the 2 HMM slots — **12 slots**.

---

## 3. Evolved vs. fixed periods

`IndicatorPeriodsSearchSpaceConfig` (`crates/alpha-config/src/ga.rs:428-483`) lists exactly 26 fields. Cross-checked against three independent call sites that must agree for the GA to actually explore them:

1. **Sampling** — `sample_indicator_periods` (`crates/alpha-ga/src/nsga2/initialization.rs:290-380`) draws all 26 from `param_bounds[...]`, then falls back to `..IndicatorPeriods::default()` (line 379) for everything else.
2. **Mutation** — `mutation_indicator_periods` (`crates/alpha-gp/src/operators/mutation.rs:120-149`) mutates the same 26, by name, against the same `param_bounds` map.
3. **Crossover** — `crossover.rs:183-266` swaps the same 26 fields between parents (`std::mem::swap`).

None of the three ever touches `mcv_adx_period`, `mcv_roc_period`, `mcv_sma_period`, `mcv_atr_period`, `mcv_atr_window` — confirmed by grepping all three files for those names (no hits). `alpha-core/src/lib.rs:238` labels them "New non-tunable (calibration)" in a comment, which matches the code behavior exactly.

**`Config::get_param_bounds`** (`crates/alpha-config/src/lib.rs:539-767`) inserts a bound for every one of the 26 evolved fields (lines 653-764) and none of the 5 fixed ones — a fourth independent confirmation.

### 3.1 Effective ranges under `config.harvest.toml` (the production config)

`config.harvest.toml`'s `[ga.search_space.indicator_periods]` section (lines 172-181) **only overrides 9 of the 26** evolved fields:

```
atr_period = [5, 50]        rsi_period = [5, 50]         stochastic_k_period = [5, 50]
stochastic_d_period = [2,15] roc_period = [3, 40]         lag_period = [2, 30]
agg_period_1 = [2, 20]      agg_period_2 = [21, 120]      hurst_period = [25, 300]
```

The other 17 evolved fields are **not** keys in that TOML table. Because the enclosing `[ga.search_space.indicator_periods]` section *is* present, serde's per-field `#[serde(default = "...")]` fires for each missing key (`ga.rs:429-482`) rather than the whole-section fallback the big comment at `ga.rs:674-727` warns about — so these 17 silently take their compiled-in default range, not `(0.0, 0.0)`:

| Field | Effective range | Source |
|---|---|---|
| `adx_period` | `[7, 30]` | `default_atr_p_bounds`, `ga.rs:485-487`, reused per `ga.rs:811` comment |
| `ter_period` | `[20, 100]` | `default_ter_p_bounds`, `ga.rs:512-514` |
| `price_correlation_period` | `[10, 40]` | `default_corr_p_bounds`, `ga.rs:515-517` |
| `vol_osc_fast` | `[5, 20]` | `default_vol_osc_fast_bounds`, `ga.rs:518-520` |
| `vol_osc_slow` | `[30, 100]` | `default_vol_osc_slow_bounds`, `ga.rs:521-523` |
| `return_skewness_period` | `[20, 60]` | `default_skew_p_bounds`, `ga.rs:524-526` |
| `price_to_sma_period` | `[20, 200]` | `default_sma_p_bounds`, `ga.rs:527-529` |
| `vwap_period` | `[10, 50]` | `default_vwap_p_bounds`, `ga.rs:530-532` |
| `downside_vol_period` | `[10, 60]` | `default_dv_p_bounds`, `ga.rs:533-535` |
| `physics_fast` | `[2, 10]` | `default_phys_fast_bounds`, `ga.rs:536-538` |
| `physics_slow` | `[15, 40]` | `default_phys_slow_bounds`, `ga.rs:539-541` |
| `lcp_period` | `[10, 40]` | `default_lcp_p_bounds`, `ga.rs:542-544` |
| `bb_period` | `[5, 50]` | `default_bb_p_bounds`, `ga.rs:545-547` |
| `bb_std_dev_x100` | `[100, 400]` | `default_bb_mult_p_bounds`, `ga.rs:548-550` |
| `atr_volatility_window` | `[100, 500]` | `default_vol_window_bounds`, `ga.rs:551-553` — **unused by any indicator, see below** |
| `phase_space_slow` | `[15, 40]` | reuses `default_phys_slow_bounds` per `ga.rs:828` comment |
| `phase_space_fast` | `[2, 10]` | reuses `default_phys_fast_bounds` per `ga.rs:829` comment |

### 3.2 `atr_volatility_window` is a dead gene for this cache

Grepping the whole `crates/` tree for `atr_volatility_window` (all matches inspected) turns up: the `IndicatorPeriods` field itself, its `Default` (`lib.rs:272`), its contribution to `structural_hash` (`lib.rs:320`) and `max_lookback` (`lib.rs:376`), its use in `engine/cache.rs:187` (a warmup-sizing sum), its search-space bound, and mutation/crossover/tests. **No indicator constructor in `registration.rs` ever reads it.** It evolves, mutates, crosses, and fragments the `structural_hash`-keyed grouping in `run_tier_filter_gpu` (`crates/alpha-simulation/src/engine/tiering.rs:123-129`) for zero column-cache payoff — an individual can differ from another *only* in this field and still be forced into a separate GPU snapshot-matrix group today, and would still be treated as a distinct cache key by any naive `IndicatorPeriods`-keyed design tomorrow, even though no column depends on it.

### 3.3 The six joint dependencies

| Slot family | Fields | Both evolved? | Citation |
|---|---|---|---|
| `stochastic_k_1m` / `stochastic_d_1m` | `stochastic_k_period`, `stochastic_d_period` | Both evolved | `registration.rs:527-532` |
| `volume_oscillator` | `vol_osc_fast`, `vol_osc_slow` | Both evolved | `registration.rs:546-550` |
| `kinetic_energy`/`potential_energy`/`lagrangian` | `physics_slow` (passed as `vwap_period`), `physics_fast` (passed as `roc_period`) | Both evolved | `registration.rs:574-580` |
| `phase_space_angular_velocity`/`phase_space_radius` | `phase_space_slow`, `phase_space_fast` | Both evolved | `registration.rs:589-594` |
| `bollinger_*` (5 outputs) | `bb_period`, `bb_std_dev_x100` | Both evolved | `registration.rs:604-612` |
| `mcv_volatility_state_1m` | `mcv_atr_period`, `mcv_atr_window` | **Neither** evolved (both `Default`-only) | `registration.rs:654-659` |

The first five are genuine combinatorial risks *at the naive-cache-key level* — but see §4, where reading the actual math shows four of them are not jointly keyed in storage at all, only at the point of combining two independently-cacheable single-key primitives. The sixth is trivially one column because neither key ever changes.

**An additional joint dependency not on the task's pre-spotted list**: `liquidation_cascade_prob` (`alpha_physics.rs:104-215`) owns `lcp_period` but also *reads* `atr_raw_1m` (`atr_in`, keyed by `atr_period`) and `momentum_acceleration` (`accel_in`, keyed by `roc_period`) as live bus values inside its own O(`lcp_period`) rolling window (`sum_vol`/`sum_atr` over `history`, `alpha_physics.rs:130-169`). A column-cache implementation that stores LCP as its own array must key it by the full `(lcp_period, atr_period, roc_period)` triple, or restructure it to read the already-cached ATR/momentum columns at gather time and do the O(`lcp_period`) reduction itself. This is flagged **OPEN** — I did not determine which approach the kernel side should take; both are viable, but they have very different memory/complexity trade-offs and the choice is squarely an implementation decision for Option A.

---

## 4. Derived-relationship collapses

### 4.1 `stochastic_d_1m = SMA(stochastic_k, stochastic_d_period)` — confirmed

`StochasticRunner::step_concrete` (`stochastic.rs:53-86`) computes `%K` from a `SlidingWindowMinMax` over `k_period`, pushes it into a `VecDeque<f64>` capped at `d_period`, and sets `d_slot` to `k_values.iter().sum::<f64>() / d_period as f64` (line 82) once the window fills. This is exactly a plain SMA over the K series — no other state feeds it. **Collapses a `k_period × d_period` product into one `stochastic_k`-keyed column plus a cheap in-kernel SMA windowed by `d_period`.**

### 4.2 Bollinger: only `bb_period` needs to be a stored dimension

`calculate_bollinger_bands_scalar` (`bollinger.rs:227-264`, and the AVX2/AVX-512 variants that compute the identical values) derives:

```
middle_band = mean(close, bb_period)
std_dev     = sqrt(variance(close, bb_period))
upper_band  = middle_band + std_dev * std_dev_mult
lower_band  = middle_band - std_dev * std_dev_mult
band_width  = (upper_band - lower_band) / middle_band
percent_b   = (close - lower_band) / (upper_band - lower_band)
```

`std_dev_mult` (`= bb_std_dev_x100 / 100.0`, `alpha-core/src/lib.rs:286-288`) is a **pure scalar multiplier** applied at the end. Caching `(mean, std_dev)` keyed by `bb_period` alone is sufficient to reconstruct all 5 outputs for *any* `bb_std_dev_x100` at gather time. **This is the single largest collapse in the codebase**: naive storage is `bb_period × bb_std_dev_x100 × 5 outputs`; actual storage need is `bb_period × 2` (mean, std).

### 4.3 Volume oscillator is a ratio of two independent volume-SMAs

`VolumeOscRunner::step_concrete` (`volume_osc.rs:59-92`) maintains two running sums (`short_sum`, `long_sum`) over volume and sets the output to `short_sma / long_sma` (lines 82-89) — a pure ratio of `SMA(volume, vol_osc_fast)` and `SMA(volume, vol_osc_slow)`. Caching one "rolling SMA of volume" family keyed by a single period (covering the union of the `vol_osc_fast` and `vol_osc_slow` ranges) and dividing two lookups at gather time collapses the `vol_osc_fast × vol_osc_slow` product into `|fast range| + |slow range|` columns.

### 4.4 Physics: `kinetic_energy`/`potential_energy` are each single-key, not jointly keyed

`MarketPhysicsRunner::step_concrete` (`physics.rs:66-142`) computes `kinetic_energy` purely from `roc` (which only depends on `roc_period`, i.e. `physics_fast`) and `volume` (line 104-108); it computes `potential_energy` purely from `vwap` (which only depends on `vwap_period`, i.e. `physics_slow`) and `volume` (lines 110-119). Both then pass through **independent** percentile-rank windows (`ke_window`/`pe_window`, each a `SortedWindow::new(200, 6)` — a **hardcoded 200**, not evolved, line 33-34). `lagrangian = ke_rank - pe_rank` (line 137) is a trivial subtraction. So: `kinetic_energy` needs a column keyed only by `physics_fast`; `potential_energy` needs a column keyed only by `physics_slow`; `lagrangian` needs no storage of its own. The apparent 2D joint (one runner, two period arguments, three outputs) is an artifact of how the runner is packaged, not of the math.

### 4.5 Phase space: same pattern as physics, one level further

`PhaseSpaceRunner::step_concrete` (`phase_space.rs:79-179`) computes `raw_x` from `vwap` (⇒ `phase_space_slow`, the `vwap_period` argument) and z-scores it using `x_history`/`x_sum`/`x_sq_sum` over a **hardcoded** `NORMALIZATION_WINDOW = 500` (line 10) to get `x_z`. It separately computes `raw_v` from `roc` (⇒ `phase_space_fast`, the `roc_period` argument) and z-scores it the same way to get `v_z` — `x_z` and `v_z` accumulate into entirely disjoint state (`x_history`/`x_sum`/`x_sq_sum` vs. `v_history`/`v_sum`/`v_sq_sum`). Only at the very end are `radius_sq = v_z² + x_z²` and `angular_velocity = (x_z·dv - v_z·dx)/radius_sq` (lines 163-172) combined — both are one-step diffs plus a scalar combine. **`x_z` needs a column keyed only by `phase_space_slow`; `v_z` needs a column keyed only by `phase_space_fast`; `radius`/`angular_velocity` need no storage of their own**, just a gather-time combine of a matched `(x_z, v_z)` pair.

### 4.6 Other collapses

- **`adx_1m` ≡ `mcv_trend_intensity_1m`**: `McvAdxRunner::step_concrete` (`adx.rs:268-279`) runs one `AdxState` and does `bus.set(self.mcv_slot, bus.get(self.adx_slot))` (line 277) — bit-identical value, not merely correlated. One column serves both slots.
- **`normalized_atr = (atr_raw_1m / close) * 100`** — `NormalizedAtrRunner::step_concrete` (`pipeline.rs:606-615`) is a pure per-tick function of the already-cached `atr_raw_1m` column and the (always-resident) close price. No storage needed beyond `atr_raw_1m`.
- **`dist_to_vwap = ((close - vwap_1m) / close) * 100`** — `VwapRunner::step_concrete` (`vwap.rs:79-89`) computes it inline from `vwap` and `close`; it is not a separately evolving computation. No storage needed beyond `vwap_1m`.
- **`kinematic_jerk = accel[t] - accel[t-1]`** — `JerkRunner::step_concrete` (`alpha_physics.rs:31-37`) is a one-step diff of `momentum_acceleration`. No storage needed beyond `momentum_acceleration`.
- **`lag_close_1m` is an index-shift of `close`** — `LagRunner::step_concrete` (`pipeline.rs:670-680`) pushes `close` into a deque and returns the value `period+1` ticks back; it is `close[t - lag_period]`, needing no computed column at all, only an index offset into the raw close-price series already uploaded for the OHLC buffer. (It is also `gp_visible: false`, `registration.rs:220-225`, so it is not even GP-reachable today.)

None of these five require reading further indicator files to confirm; each is a direct, small, non-stateful (or single-accumulator) function visible in full in the cited `step_concrete`.

---

## 5. Column-count and memory table

### 5.1 Method

For each 1m column family, the stored-column count is the **collapsed** count from §3/§4 (i.e., the actual minimum a correctly-designed cache needs), using the effective ranges from §3.1. For Agg1/Agg2, the real joint coupling with `agg_period_{1,2}` (§2.2) is honored: a column at aggregation period `p` has `num_candles / p` entries, so the element cost for one Agg-family is `(own-period range size) × num_candles × Σ_p (1/p)` over the aggregation period's range — using harmonic-number approximations `Σ_{p=2}^{20}(1/p) ≈ 2.5977` and `Σ_{p=21}^{120}(1/p) ≈ 1.7711`.

### 5.2 1-minute families (collapsed column counts, per ticker)

| Family | Key(s) | Range size(s) | Columns |
|---|---|---|---|
| `atr_raw_1m` | `atr_period` | 46 | 46 |
| `rsi_1m` | `rsi_period` | 46 | 46 |
| `stochastic_k_1m` (`stochastic_d` derived) | `stochastic_k_period` | 46 | 46 |
| `roc_1m` | `roc_period` | 38 | 38 |
| `price_correlation` | `price_correlation_period` | 31 | 31 |
| `return_skewness` | `return_skewness_period` | 41 | 41 |
| `dist_to_sma` | `price_to_sma_period` | 181 | 181 |
| `vwap_1m` (`dist_to_vwap` derived) | `vwap_period` | 41 | 41 |
| `downside_volatility` | `downside_vol_period` | 51 | 51 |
| `hurst` | `hurst_period` | 276 | 276 |
| `momentum_acceleration` | `roc_period` | 38 | 38 |
| `microstructural_entropy` | `atr_period` | 46 | 46 |
| `volume_imbalance_ratio` | `roc_period` | 38 | 38 |
| `thermo_entropy`+`thermo_temp` (1 family, 2 outputs) | `atr_period` | 46 | 46 |
| `mcv_directional_efficiency_1m` (TER) | `ter_period` | 81 | 81 |
| `liquidation_cascade_prob` | `lcp_period` (own key; also reads atr/momentum columns, §3.3 OPEN) | 31 | 31 |
| **Subtotal, single-key** | | | **1077** |
| `volume_oscillator` (SMA-of-volume, §4.3) | union of `vol_osc_fast`(16) ∪ `vol_osc_slow`(71) | 16+71 | 87 |
| `kinetic_energy`+`potential_energy` (§4.4; `lagrangian` derived) | `physics_fast`(9) + `physics_slow`(26) | 9+26 | 35 |
| `phase_space` `x_z`+`v_z` (§4.5; radius/velocity derived) | `phase_space_slow`(26) + `phase_space_fast`(9) | 26+9 | 35 |
| `bollinger` mean+std (§4.2) | `bb_period` × 2 | 46×2 | 92 |
| `mcv_volatility_state_1m` | `mcv_atr_period`+`mcv_atr_window`, both fixed | 1 | 1 |
| **Subtotal, collapsed-joint** | | | **250** |
| `mcv_trend_intensity_1m`/`adx_1m`, `mcv_velocity_1m`, `mcv_trend_bias_1m` | each a fixed field | 1×3 | 3 |
| period-independent (`obv_1m`, `log_return_1m`, `high_low_disparity`, `close_location_value`, `candle_body_ratio`) | none | 1×5 | 5 |
| **1m total per ticker** | | | **1335** |

(`lag_close_1m` and `normalized_atr`/`dist_to_vwap`/`kinematic_jerk`/`mcv_trend_intensity_1m`-as-alias are excluded from the count — zero marginal storage per §4.)

### 5.3 Agg1/Agg2 (per ticker, elements expressed as an equivalent full-length-column count)

| Family | Own-period range | Agg reciprocal-sum | Equivalent full-length columns |
|---|---|---|---|
| `atr_raw_agg1`, `rsi_agg1`, `stochastic_k_agg1` (×3) | 46 each | `agg_period_1` ∈ [2,20]: Σ(1/p) ≈ 2.5977 | 46 × 2.5977 × 3 ≈ **358.5** |
| `atr_raw_agg2`, `rsi_agg2`, `stochastic_k_agg2` (×3) | 46 each | `agg_period_2` ∈ [21,120]: Σ(1/p) ≈ 1.7711 | 46 × 1.7711 × 3 ≈ **244.4** |
| **Agg total (equivalent)** | | | **≈ 602.9** |

### 5.4 Totals, 5 tickers

Per-ticker equivalent full-length columns: `1335 + 602.9 = 1937.9`. Across 5 tickers: `9689.5`. Total stored elements = `9689.5 × num_candles`.

| Tier | `num_candles` | Elements | f32 (4B) | f64 (8B) |
|---|---|---|---|---|
| Tier-1 | 43,200 | 9689.5 × 43,200 ≈ 418,586,400 | ≈ 1.56 GiB | ≈ 3.12 GiB |
| Tier-2 | 172,800 | 9689.5 × 172,800 ≈ 1,674,345,600 | ≈ 6.24 GiB | ≈ 12.47 GiB |
| Full history | 1,000,000 | 9689.5 × 1,000,000 = 9,689,500,000 | ≈ 36.09 GiB | ≈ 72.19 GiB |

(1 GiB = 1,073,741,824 bytes; e.g. full-history f32: `9,689,500,000 × 4 / 1,073,741,824 = 36.09`.)

#### 5.4.1 Verified against the shipped implementation (O09 closure, `manual/14`)

The Agg1/Agg2 fix (`crates/alpha-simulation/src/gpu_columns.rs`'s `GpuColumnFamily::AtrRawAgg`/`RsiAgg`/`StochasticKAgg`, `aggregate_candles`, `compute_agg_column`) shipped with EXACTLY the joint `(agg_period, indicator_period)` keying §5.1/§5.3 assumed — there is no further collapse available for these three families (unlike Bollinger/physics/phase-space/volume-oscillator, ATR/RSI/StochasticK's aggregated-timeframe computation has no period-independent shared primitive to factor out: the whole accumulator depends on both the resampled candle stream, keyed by `agg_period`, and the indicator's own window, keyed by `atr_period`/`rsi_period`/`stochastic_k_period`, jointly). So §5.3's harmonic-sum figures are not merely an estimate of what a correct implementation *would* cost — they are what the shipped one actually costs, modulo one detail the estimate could not have accounted for in advance:

**Per-column header overhead.** Each stored Agg column is `[first_visible_bar, compact_len, val_0, val_1, ...]`, not a bare value array (see `GpuColumnFamily::AtrRawAgg`'s doc comment) — 2 extra `f32`s per DISTINCT `(agg_period, indicator_period)` key, not per element. Worst-case distinct-key count per family: `46 (own-period range) × (19 (agg_period_1 range) + 100 (agg_period_2 range)) = 46 × 119 = 5474` keys. Across all 3 families: `3 × 5474 = 16,422` keys × 2 `f32`s = `32,844` `f32`s = 131,376 bytes ≈ **128.3 KiB total, worst case, at ANY tier size** (the header cost is per-key, not per-candle, so it does not scale with `num_candles` at all). Against the full-history total of 36.09 GiB, this is **≈ 0.00035%** — genuinely negligible, not merely asserted to be. The §5.4 table above therefore stands as the verified real figure, not just the pre-implementation estimate; no correction is needed at any of the three tiers.

**Sanity-checked against the task's own rough guess** ("a few hundred MiB at tier-1"): Agg's share of the tier-1 total is `602.9 / 1937.9 ≈ 31.1%` of `1.56 GiB ≈ 0.485 GiB ≈ 497 MiB` — consistent with "a few hundred MiB," confirmed rather than merely repeated.

**Re-verified (this update)**: the 46/19/100 range sizes were re-read directly from the live `config.harvest.toml` (`atr_period`/`rsi_period`/`stochastic_k_period = [5, 50]` → 46; `agg_period_1 = [2, 20]` → 19; `agg_period_2 = [21, 120]` → 100 — lines 173-180) and from `crate::gpu_column_gather::NUM_COLUMN_ROLES` (`gpu_column_gather.rs:93`, now **32**, up from 26: the extra 6 are exactly the O09 closure's 2 aggregation tiers × 3 families, per that constant's own doc comment and `column_requests`'s emission order, `gpu_columns.rs:372-430`). Both match §5.1/§5.3's assumptions exactly — no correction needed. See §5.4.2 for a second shipped config (`config.toml`) that overrides the same 9 fields with different bounds, and for the actual GPU-side byte budget this design is sized against.

### 5.4.2 Agg1/Agg2's isolated cost, a second shipped config, and the real GPU-side budget

**Agg-only bytes, isolated from the combined total** (harvest.toml bounds — the same 46/19/100 range sizes as §5.3/§5.4.1, `602.9` equivalent full-length columns per ticker, `3014.5` across 5 tickers):

| Tier | `num_candles` | Agg-only elements | Agg-only f32 | Agg-only f64 | Share of that tier's combined total (§5.4) |
|---|---|---|---|---|---|
| Tier-1 | 43,200 | 3014.5 × 43,200 ≈ 130,227,226 | ≈ 0.485 GiB (≈497 MiB) | ≈ 0.970 GiB | 31.1% |
| Tier-2 | 172,800 | 3014.5 × 172,800 ≈ 520,908,904 | ≈ 1.940 GiB | ≈ 3.881 GiB | 31.1% |
| Full history | 1,000,000 | 3014.5 × 1,000,000 ≈ 3,014,519,118 | ≈ 11.23 GiB | ≈ 22.46 GiB | 31.1% |

**This is material, not marginal.** Agg1/Agg2 is not a rounding error on top of the 1m design — it is very close to a third of the entire cache's footprint at every tier, exactly as §8 flagged before implementation ("Agg1/Agg2 support ... contributes ~31% of the collapsed-design total"). Implementation confirmed the estimate; it did not shrink it.

**A second shipped config disagrees on the bounds.** `config.toml` (the master/default config, also `tickers = ["BTC", "ETH", "SOL", "TURBO", "BONK"]` — 5 tickers) overrides the identical 9 `[ga.search_space.indicator_periods]` fields as `config.harvest.toml` (§3.1) but with different numbers: `atr_period`/`rsi_period`/`stochastic_k_period = [7, 40]` (34, not 46), `agg_period_1 = [3, 30]` (28, not 19), `agg_period_2 = [31, 120]` (90, not 100) — `config.toml` lines 148-155. Recomputing just the Agg-only figures against these narrower bounds (own-period range 34; `Σ_{p=3}^{30}(1/p) ≈ 2.4950`; `Σ_{p=31}^{120}(1/p) ≈ 1.3739`):

| Tier | Agg-only elements (`config.toml` bounds) | Agg-only f32 | Agg-only f64 |
|---|---|---|---|
| Tier-1 | 394.62 × 5 × 43,200 ≈ 85,238,906 | ≈ 0.317 GiB | ≈ 0.635 GiB |
| Tier-2 | 394.62 × 5 × 172,800 ≈ 340,955,624 | ≈ 1.270 GiB | ≈ 2.540 GiB |
| Full history | 394.62 × 5 × 1,000,000 ≈ 1,973,122,827 | ≈ 7.35 GiB | ≈ 14.70 GiB |

Smaller than harvest.toml's figures (narrower `agg_period_2` and a narrower own-period range dominate), but still multiple GiB at full history — not negligible under either shipped config. §5.2-§5.4's main table uses `config.harvest.toml`'s (wider) bounds throughout, including the 1m-family counts, matching §3.1's original framing of it as "the production config"; this row is a cross-check that the Agg conclusion (material, GiB-scale) holds under the other shipped config too, not a claim that `config.toml`'s full combined total is 36 GiB (its narrower `atr_period`/`rsi_period`/`stochastic_k_period`/`roc_period`-adjacent bounds would also shrink several 1m families in §5.2's table; that full recomputation is out of this update's scope). The compiled-in `ga.rs` defaults (`default_agg1_p_bounds` = `[3, 15]`, `default_agg2_p_bounds` = `[16, 60]`, `default_atr_p_bounds`/`default_rsi_p_bounds`/`default_stoch_k_p_bounds` = `[7, 30]`, `alpha-config/src/ga.rs:485-508`) are narrower still, but neither shipped config leaves any of these 9 fields unset, so those compiled-in values are never actually reached in production — listed here only because the task that produced this update asked for them explicitly.

**The budget this cache is actually checked against is 48 GiB, not 32 GiB.** §5.5 below cites `ga.column_cache_budget_mb = 32768` (32 GiB) — that is the CPU-side `IndicatorWarmupCache` budget (5 indicators only) and was never the right number for the GPU-resident cache this chapter is about. The GPU-side cache `gpu_columns.rs`/`gpu_column_gather.rs` implement is checked against `alpha_config::gpu::GpuConfig::column_cache_budget_mb` (`crates/alpha-config/src/gpu.rs:26-28`, default `49_152` MiB = **48 GiB**), wired via `GpuColumnCache::new(config.gpu.column_cache_budget_mb)` at `crates/alpha-simulation/src/engine/tiering.rs:205`; neither shipped config overrides it, so both run against the 48 GiB default. Against that real ceiling, the full-history, 5-ticker, all-slot combined total from §5.4 (36.09 GiB f32, `config.harvest.toml` bounds, Agg included) is **≈75.2%** of the budget — real headroom (≈11.9 GiB) remains, and it is a full-range worst case rather than a single generation's actual high-water mark (a population only ever has as many *distinct* periods in flight as its individuals happen to sample, not the full range product), but 75% of a dedicated budget is a tighter margin than "comfortably under" suggests. `crates/alpha-config/src/gpu.rs:17-22`'s own doc comment, which cites this chapter's §5.4/§5.5 and calls the ~36 GiB figure "an all-1m-slot precompute," is stale on that description now that Agg1/Agg2 is real and already folded into that same 36.09 GiB number (§5.3/§5.4) — the code comment itself was not changed as part of this documentation-only update, but a reader should not take "all-1m-slot" at face value there.

### 5.5 Why the collapses in §4 are not optional

Repeating the full-history calculation **without** applying §4.2-4.5 (i.e., a naive cache keyed by the raw joint tuples) — holding the single-key families and the Agg total fixed, and replacing only the five collapsed-joint rows:

- `stochastic`: `46 × 14 = 644` instead of `46` (Δ +598)
- `volume_oscillator`: `16 × 71 = 1136` instead of `87` (Δ +1049)
- `physics` (3 outputs): `9 × 26 × 3 = 702` instead of `35` (Δ +667)
- `phase_space` (2 outputs): `9 × 26 × 2 = 468` instead of `35` (Δ +433)
- `bollinger` (5 outputs): `46 × 301 × 5 = 69,230` instead of `92` (Δ **+69,138**)

Naive per-ticker equivalent columns: `1335 + 71,885 = 73,220`; plus the (unchanged) Agg contribution `602.9` ⇒ `73,822.9`. Across 5 tickers: `369,114.5` — **38x** the collapsed figure. At full history: `369,114.5 × 1,000,000 = 369,114,500,000` elements ⇒ **f32 ≈ 1.34 TiB, f64 ≈ 2.68 TiB**. Bollinger's period×σ product alone accounts for essentially all of the blow-up.

For context: `config.harvest.toml` (line 65) already sets `ga.column_cache_budget_mb = 32768` (32 GiB) for the *existing* CPU-side cache, which covers only 5 indicators (`ColumnKind::{Hurst, PriceCorrelation, ReturnSkewness, DownsideVol, McvAtrWindow}`, `crates/alpha-indicators/src/column.rs:40-55`). Even the fully-collapsed, all-indicator, all-timeframe design in §5.4 (36 GiB f32 at full history) **exceeds that existing budget**, and the naive version is 40-80x over it. A byte-budgeted, LRU-evicted cache — the pattern `IndicatorWarmupCache`/`column_cache_budget_mb` already establishes — is required regardless of which collapses are implemented; it is not an optional refinement. (This CPU-side 32 GiB figure is not, in fact, the budget the shipped GPU-side cache is checked against — see §5.4.2 for the actual `alpha_config::gpu::GpuConfig::column_cache_budget_mb` default of 48 GiB and how the §5.4 total compares against it.)

---

## 6. GP terminal reachability

`TerminalWeights::new_uniform` (`crates/alpha-models/src/strategy/terminals.rs:82-97`) builds its terminal set from `TerminalNode::generate_all(registry)` (line 85), which is `registry.gp_visible_slots()` (`terminals.rs:68-71`) — **every** GP-visible slot in the `SlotRegistry`, regardless of timeframe. `SlotRegistry::gp_visible_slots` (`crates/alpha-indicators/src/registry.rs:86-89`) returns the pre-cached `gp_visible_ids` populated at `register()` time (`registry.rs:52-71`), driven purely by each `SlotDescriptor.gp_visible` flag set in `registration.rs`.

Of the 56 occupied slots, 53 are GP-visible (`obv_1m`, `lag_close_1m`, `vwap_1m` are the 3 exceptions, all `gp_visible: false`). Of those 53: **6 are Agg1/Agg2** (`atr_raw_agg1`, `rsi_agg1`, `stochastic_k_agg1`, `atr_raw_agg2`, `rsi_agg2`, `stochastic_k_agg2` — all `gp_visible: true`, `registration.rs:232-267`), and 47 are 1m or fixed-timeframe (MCV/HMM/constants).

**6 / 53 ≈ 11.3%** of GP-reachable terminals pointed at slots that `build_snapshot_matrix` never wrote on GPU, at the time this chapter was originally written. `IndicatorBus::new(&registry.defaults())` seeded every slot at its registered default before ticking, and only the `Timeframe::OneMin` pipeline was ticked, so any GP tree that referenced an Agg1/Agg2 terminal read a **constant** for every bar on GPU — `50.0` for `rsi_agg{1,2}`/`stochastic_k_agg{1,2}` and `0.0` for `atr_raw_agg{1,2}` (`registration.rs:238-267` for the defaults) — while the same tree read a real, time-varying value on the CPU reference path (`MultiScaleIndicatorState`, `crates/alpha-models/src/simulation/mod.rs`). That was a pre-existing GPU/CPU scoring divergence (`manual/14`'s O09), independent of the column-cache work, but the column-cache redesign was the natural point to fix it, since it required building the Agg1/Agg2 pipelines in the first place (§2.2, §5.3).

**Fixed.** `crate::gpu::build_snapshot_matrix` (`crates/alpha-simulation/src/gpu.rs:785-876`) now ticks real `Timeframe::Agg1`/`Timeframe::Agg2` pipelines too: it constructs `TimeframeIndicators` for both (`gpu.rs:833-834`) alongside the 1m `IndicatorPipeline` (`gpu.rs:813-815`), all three sharing one `IndicatorBus`, and drives them through `CandleAggregator::aggregate` from `WARMUP_PERIOD` onward (`gpu.rs:855-864`) — matching `MultiScaleIndicatorState::new_with_columns`'s construction order and `MultiScaleIndicatorState::update`'s per-bar timing exactly (that function's own doc comment cites the specific anchor-point argument for why `WARMUP_PERIOD` matters here, not bar `0`). All 53 GP-visible terminals, not 47, now read real, time-varying values out of `build_snapshot_matrix` — verified bar-by-bar against `MultiScaleIndicatorState` by `crate::gpu_columns`'s `agg_columns_match_multi_scale_indicator_state_reference_bar_by_bar` test, and end-to-end through the column-cache gather by `crate::gpu_column_gather`'s `agg_slots_gathered_via_full_pool_match_build_snapshot_matrix_reference_over_warmup` test. This makes `build_snapshot_matrix` a complete, not partial, equivalence reference for the column cache in `crate::gpu_columns`/`crate::gpu_column_gather` — see §5.4.1/§5.4.2 for the memory consequence and `manual/14`'s O09 entry for the fix in the divergence-audit's own framing.

---

## 7. Items marked OPEN

- **`momentum_acceleration`, `microstructural_entropy`, `volume_imbalance_ratio`, `thermo_entropy`/`thermo_temp`** are all single-period-keyed per `registration.rs`, so they carry no combinatorial risk regardless of their internals — but I did not read `kinematics.rs`, `microstructure.rs`, or `thermodynamics.rs` to check whether any of them are *also* cheap derived functions of an already-cached column (the way `normalized_atr`/`dist_to_vwap`/`kinematic_jerk` turned out to be). If any are, that only shrinks the totals in §5 further, never grows them — the numbers there are a safe upper bound for these four families.
- **`liquidation_cascade_prob`'s effective key**: §3.3 identifies the `(lcp_period, atr_period, roc_period)` coupling but does not resolve whether the implementation should (a) store LCP as its own array keyed by the full triple, or (b) restructure it to gather from the already-cached ATR/momentum-acceleration columns and run the O(`lcp_period`) rolling reduction at kernel time. This is a real design decision with materially different memory cost and is left to the implementer.
- ~~Whether Option A is intended to also fix the Agg1/Agg2-reads-defaults bug (§6)~~ — **resolved**: it was in scope and has since shipped (`manual/14` O09, now Fixed). `crate::gpu_columns`/`crate::gpu_column_gather` implement exactly the packed `(agg_period, indicator_period)` keying §5.1/§5.3 assumed, and `crate::gpu::build_snapshot_matrix` was separately fixed to tick real `Timeframe::Agg1`/`Agg2` pipelines, so it remains a valid equivalence reference for these slots too. The Agg1/Agg2 rows in §5.3 (≈602.9 of the ≈1937.9 per-ticker total under `config.harvest.toml`'s bounds, ≈31%) were NOT dropped, and §5.4.2 re-verifies them against the live shipped code and a second config.
- **`CandleAggregator::aggregate`'s exact bar-boundary semantics** (`crates/alpha-models/src/simulation/aggregator.rs`) were not read; the "column length ≈ `num_candles / agg_period`" approximation in §5.1/§5.3 is standard for this kind of downsampling but the exact off-by-one behavior at the start/end of a series was not verified against source.
- **Whether any GPU-side double-precision kernel already exists for parts of this** (`crate::gpu::compute_column_gpu`, referenced from `column.rs:157-160` under `#[cfg(feature = "gpu")]`) was not investigated — it may already establish a precedent for how a GPU-resident column should be computed and laid out, which would directly inform Option A's implementation rather than just its sizing.

---

## 7.1 What the column cache actually saved, per generation

The figure below is the headline justification for the whole Option A
effort and was, until now, recorded only in the F1b-3 implementation
report and not in this document — `manual/16`'s bring-up sheet correctly
refused to cite it for exactly that reason. It is restated here with its
arithmetic so it is checkable rather than folklore.

Setting: `config.harvest.toml` — population 16,000, 5 tickers, tier-1 =
30 days = 43,200 candles. `GpuEvaluator::new(16384, ...)` (see
`apps/alpha-worker/src/controller.rs`) means the whole population fits in
one chunk.

One snapshot matrix is `num_candles × GPU_MAX_SLOTS × 4 bytes × num_tickers`:

`43,200 × 128 × 4 × 5 = 110,592,000 bytes ≈ 105.5 MiB`

**Before F1b-3.** The kernel takes ONE `d_snapshots` per launch, so
individuals could only share a launch if they shared an
`IndicatorPeriods` tuple. With 26 evolved period genes, tuple collisions
are statistically negligible, so this was ~one launch per individual,
each re-uploading its own full matrix:

`16,000 × 110,592,000 B ≈ 1.77 × 10^12 B ≈ 1,648 GiB ≈ 1.61 TiB` per generation.

(This also explains the ~0.1% occupancy observed before batch coalescing:
those launches were effectively one individual wide.)

**After F1b-3.** One periods-invariant base snapshot, one column pool, one
role-offset table:

| Component | Bytes |
|---|---|
| Base snapshot (built once, at `IndicatorPeriods::default()`) | 110,592,000 B ≈ 105.5 MiB |
| Column pool (tier-1, full-range residency — §5.4) | ≈ 1.56 GiB |
| Role-offset table (`16,000 × 5 × 26 × 4 B`) | ≈ 7.9 MiB |
| **Total** | **≈ 1.67 GiB** |

**≈ 1,648 GiB → ≈ 1.67 GiB, a ~987× reduction.** Tier-2 (120 days,
172,800 candles) scales the same way: ≈ 6.44 TiB → ≈ 6.66 GiB, ≈ 990×.

Both figures are **analytical, derived from config and the arithmetic
above — never measured.** No GPU was available while this was
implemented. Task K's harness (`manual/15`) is what turns this into a
measured number; treat a disagreement on the A100 as information about
where the model is wrong, not as a failure.

Note the role-offset row predates the Agg1/Agg2 closure (O09), which grew
`NUM_COLUMN_ROLES` from 26 to 32; at 32 roles that row is ≈ 9.8 MiB,
which does not move the total.

## 8. Implications for the column-cache implementation

**Cheap, essentially free:**
- `stochastic_d`, `normalized_atr`, `dist_to_vwap`, `kinematic_jerk`, `lag_close_1m`, `mcv_trend_intensity_1m` (§4) need zero dedicated storage — they are one-step diffs, index shifts, or scalar combines of columns (or raw price data) that must be resident anyway.
- The five MCV/calibration slots that key off fixed fields (`mcv_trend_intensity_1m`/`adx_1m`, `mcv_velocity_1m`, `mcv_trend_bias_1m`, `mcv_volatility_state_1m`) are each exactly one column for the entire run, regardless of population size or generation count.
- Most single-key 1m families (ATR, RSI, ROC, correlation, skewness, VWAP, downside-vol, TER, Hurst) are already the cheap, well-behaved case the existing `IndicatorWarmupCache`/`column.rs` machinery was built for — 5 of them (Hurst, PriceCorrelation, ReturnSkewness, DownsideVol, McvAtrWindow) are *already* cached there today.

**Expensive, and worth the design effort to get right:**
- Bollinger, volume-oscillator, physics, and phase-space are the ones that *look* like they need a jointly-keyed column per (period-a, period-b) pair. §4.2-4.5 show that in every one of these four cases the actual computation only needs two independently-keyed single-period primitives (mean/std, or two independent SMAs, or two independently-ranked/z-scored series) combined at gather time. **Implementing the collapse, not just the cache, is what separates a ~36 GiB design from a >1 TiB one at full history** — this is not a minor efficiency tweak, it is the difference between shippable and not.
- `liquidation_cascade_prob`'s multi-column read (§3.3, §7) needs an explicit decision before implementation, since it is the one case that doesn't cleanly decompose into "cache one thing, combine at gather time" — it has genuine O(window) state of its own layered on top of two other columns.
- Agg1/Agg2 support (§2.2, §5.3, §6) was simultaneously the fix for a real GPU/CPU scoring bug (§6) and the single biggest scope question for memory — it contributes ~31% of the collapsed-design total, and its cost only exists because the aggregation period itself is evolved, genuinely changing the resampled series being cached, not because of any indicator combinatorics. **This has since shipped** (`manual/14` O09, Fixed) with exactly this keying; §5.4.2 has the re-verified byte figures, including under a second shipped config and against the real 48 GiB GPU-side budget.

**The one design decision that most affects memory**: whether the four "de-jointed" families (§4.2-4.5) are actually implemented as two single-key primitives combined at gather time, or naively as their raw joint key. That single choice is a ~38x memory multiplier at every dataset size in §5.4/§5.5, dwarfing every other design parameter examined in this document (including the byte budget itself, and including whether Agg1/Agg2 support is bundled in).

---

## 9. Task I: putting the full-history stage on GPU

Everything above (§1-§8) was written for tier-1/tier-2, where the F1b column
cache already shipped (`crate::gpu_columns`/`crate::gpu_column_gather`,
`crates/alpha-simulation/src/engine/tiering.rs`'s `run_tier_filter_gpu`).
Full history — the terminal stage, ~1,000,000 candles — stayed hardcoded to
CPU regardless of GPU availability, gated by
`alpha_config::gpu::GpuConfig::max_snapshot_candles` (pre-Task-I default
`300_000`, well under any realistic full-history length). This section
documents Task I: putting that stage on GPU, with real memory arithmetic
computed from the SHIPPED code (not a hand estimate) and a chunking
mechanism as the safety valve for whatever headroom that arithmetic
doesn't already cover.

### 9.1 Survivor count: bounded by the search space, not by population size

The task that requested this work asked "how many individuals actually
reach full history" as the first question, expecting the answer to drive a
chunk-size choice. It turns out not to matter, for a reason worth stating
plainly: **the maximum number of DISTINCT `(family, period)` columns the
full-history stage can ever need is bounded by the GA's own search-space
ranges (§3.1), not by how many individuals survive tier-2.** A column is
keyed by `(ticker_hash, GpuColumnFamily, period value)`
(`crate::gpu_columns::GpuColumnKey`) — an integer period field can only take
as many distinct values as its own `[min, max]` range allows, no matter how
many thousands of individuals sample it. `config.toml`'s
`ga.population_size = 10000` and `config.harvest.toml`'s `population_size =
16000` (both lines ~62-63 of their respective files) are therefore
irrelevant to this arithmetic — even if literally every one of 16,000
individuals reached full history with a distinct `IndicatorPeriods`, the
distinct-column count could not exceed one column per value in each
field's own range, which is exactly the §9.2 "full-range worst case." Real
generations sample far less than the full range (especially early on, and
increasingly less as the population converges), so §9.2's figure is a
ceiling, not a typical case.

`config.ga.evaluation_tiers` (`crates/alpha-config/src/ga.rs:270-284`) has
no field for "how many individuals reach the terminal stage" — tier-1/
tier-2 pass/fail is threshold-emergent (`tier_1_min_total_pnl`/
`tier_1_min_trades_per_asset`, scaled for tier-2 by
`tier_2_days / tier_1_days`, `engine/tiering.rs`'s
`evaluate_batch_tiered`), not a fixed count, so there is no config value to
cite here. The range-bound argument above is what makes that not matter.

### 9.2 Real memory arithmetic, computed from the shipped code

`manual/12`'s §5.4 figure (~36.09 GiB, harvest.toml bounds) was a
pre-implementation hand estimate — written before `crate::gpu_columns`
shipped, using a harmonic-sum approximation for the three Agg families and
assuming `liquidation_cascade_prob` would get its own ~31-column
allocation (§3.3's OPEN item). The shipped implementation instead
reconstructs LCP from already-cached ATR/momentum columns at gather time
(no dedicated `GpuColumnFamily` variant for it at all — confirmed by
reading the full `GpuColumnFamily` enum, `gpu_columns.rs:119-323`, which
has no LCP variant), so that assumption never became real cost.

Task I re-derives the figure directly from the shipped
`GpuColumnFamily`/`column_requests` (`gpu_columns.rs:372-430`) via two new,
tested functions in `crates/alpha-simulation/src/gpu_columns.rs`:

- `estimate_one_column_bytes` (line 473) — the exact stored byte cost of
  one `(family, period)` column at a given ticker length: `ticker_len × 4`
  bytes for a plain family, `ticker_len × 2 × 4` for the two
  interleaved-pair families (`BollingerMeanStd`, `ThermoEntropyTemp`), and
  `(ticker_len.div_ceil(agg_period) + 2) × 4` for the three Agg families
  (an over-estimate of the real compacted length, by construction — see
  that function's own doc comment).
- `estimate_column_cache_bytes` (line 503) — sums the above over every
  DISTINCT `(family, period)` key a `periods_list` requests, deduplicated
  exactly like `GpuColumnCache::build` itself deduplicates.

`gpu_columns::tests::task_i_full_range_worst_case_bytes_for_shipped_configs`
reconstructs §5.2's "full range, every family independently maxed"
worst case using these functions and the LIVE `[ga.search_space.
indicator_periods]` bounds from both shipped configs, at 1,000,000
candles across 5 tickers. Running it gives the real, ground-truth figures
(not hand arithmetic):

| Config | Bytes | GiB | % of 48 GiB default budget |
|---|---|---|---|
| `config.harvest.toml` (lines 172-181) | 29,508,851,480 | 27.482 | 57.25% |
| `config.toml` (lines 147-156) | 24,549,820,880 | 22.864 | 47.63% |

Both fit the default `alpha_config::gpu::GpuConfig::column_cache_budget_mb`
(49,152 MiB = 48 GiB, `crates/alpha-config/src/gpu.rs:69-90`, unchanged by
Task I) with real headroom — harvest.toml's worst case uses 57.25% of the
budget, leaving ~20.5 GiB free even in the theoretical maximum-diversity
case. This is a MORE comfortable margin than manual/12 §5.4.2's
pre-implementation ~75% estimate, and per §9.1, an actual generation's
real footprint is bounded well below even this full-range figure.

**Other buffers, added to the same picture:**

- **Base snapshot** (`GpuEvaluator`'s `d_snapshots`,
  `crates/alpha-simulation/src/gpu_evaluator.rs:300`): sized
  `max_tickers × max_candles × GPU_MAX_SLOTS(128) × 4` bytes at
  construction (`GpuEvaluator::new`, `gpu_evaluator.rs:266-284`). At the
  production `max_tickers = 10` (`apps/alpha-worker/src/main.rs`,
  `controller.rs`) and the new `GPU_EVALUATOR_CANDLE_CAPACITY = 1_500_000`
  (`gpu_evaluator.rs:221`, see §9.3), this is `10 × 1,500,000 × 128 × 4 =
  7,680,000,000` bytes ≈ **7.15 GiB** — up from ≈1.43 GiB at the
  pre-Task-I `max_candles = 300_000`. `d_ohlc`
  (`OhlcGpu` = 5 `f32`s) adds `10 × 1,500,000 × 20 = 300,000,000` bytes ≈
  286 MiB; `d_hmm_states` adds `10 × 1,500,000 × 4 = 60,000,000` bytes ≈
  57 MiB. These three are PERSISTENT for the evaluator's whole lifetime
  (held even during tier-1/tier-2 work, not just full-history), unlike the
  column pool below.
- **GP bytecode / per-individual state** (`d_params`, `d_strategies`,
  `d_gp_nodes`, `d_stateful_info`, `d_state_buffer`,
  `gpu_evaluator.rs:299-309`): scale with `max_individuals`
  (16,384 in production) and small per-individual constants
  (`MAX_STATEFUL_NODES_PER_IND`, `STATE_SLOT_STRIDE`), independent of
  candle count — unchanged by Task I, on the order of tens of MiB total at
  production scale.
- **Column pool** (`ColumnPool::data`, allocated fresh per
  `evaluate_batch` call, `gpu_evaluator.rs`'s `d_column_pool_buf`): this is
  the ≈22-27 GiB worst case computed above, freed (RAII) the instant each
  `evaluate_batch` call returns.

Sum at the worst case: ≈27.48 GiB (column pool) + ≈7.15 GiB (base
snapshot) + ≈0.34 GiB (OHLC + HMM states) + tens of MiB (bytecode/state) ≈
**35 GiB**, comfortably inside an 80 GiB card with the 48 GiB column-cache
budget and the ~32 GiB `alpha_config::gpu::GpuConfig::
column_cache_budget_mb`'s own doc comment (`gpu.rs:30-36`) already reserves
for "the OHLC buffer, per-individual GP bytecode/state buffers, and
CUDA/HIP runtime overhead."

**Bottom line: for both shipped configs, the full-history stage fits the
existing 48 GiB budget without raising it, even at the theoretical
maximum-diversity worst case.** Chunking (§9.4) exists as a safety valve
for configs with wider ranges than either shipped one, or for a future
raise to `column_cache_budget_mb` interacting badly with a wider search
space — not because either shipped config needs it today.

### 9.3 `max_snapshot_candles` and the evaluator's own candle capacity

Two separate numbers gate whether a stage can run on GPU at all, and Task I
had to reconcile them:

1. **`alpha_config::gpu::GpuConfig::max_snapshot_candles`**
   (`crates/alpha-config/src/gpu.rs:86`, default raised from `300_000` to
   **`1_500_000`** by `default_max_snapshot_candles`, `gpu.rs:26-40`) — an
   operator-configurable ceiling, checked by `engine::tiering::
   evaluate_batch_tiered` for ALL THREE stages now (tier-1, tier-2, AND
   full-history), not just tier-1/tier-2 as before Task I.
2. **`GpuEvaluator::max_candles`** (`crates/alpha-simulation/src/
   gpu_evaluator.rs`'s `GpuEvaluator` struct field, set once at
   construction from the `max_candles` argument to `GpuEvaluator::new`,
   `gpu_evaluator.rs:266`) — the ACTUAL capacity `d_ohlc`/`d_snapshots`/
   `d_hmm_states` were allocated for. Exceeding this is not a graceful
   fallback: `evaluate_batch`'s `assert!(total_candles <=
   self.d_ohlc.len())` panics.

Before Task I these two numbers happened to agree by coincidence — both
`apps/alpha-worker/src/main.rs` and `controller.rs` hardcoded the literal
`300_000` for `GpuEvaluator::new`'s `max_candles` argument, matching the
config default, but with no code tying them together: an operator who
raised `max_snapshot_candles` in a synced config without also changing
those two literals would have passed the gate check only to panic inside
`evaluate_batch`. Task I closes that gap two ways:

- `engine::tiering::evaluate_batch_tiered`'s gate for every stage is now
  `max_c <= config.gpu.max_snapshot_candles.min(gpu_eval.max_candles)`
  (tier-1: `tiering.rs` around the `use_gpu_tier1` binding; tier-2: the
  identical `use_gpu_tier2` binding; full-history: `use_gpu_full`,
  `tiering.rs:898-911`) — the smaller of the two always wins, so a config
  value that exceeds the evaluator's real allocation can never reach the
  panic; it silently (and correctly) falls back to CPU for that stage
  instead.
- `apps/alpha-worker/src/main.rs` and `controller.rs` now both construct
  their `GpuEvaluator` with `alpha_simulation::
  GPU_EVALUATOR_CANDLE_CAPACITY` (`gpu_evaluator.rs:221`, `1_500_000`) in
  place of the old hardcoded `300_000` literal — matching the new
  `max_snapshot_candles` default exactly, and documented as the one place
  that needs to grow if the historical window ever exceeds it. `Config`
  (from which `max_snapshot_candles` would otherwise come) is not
  available yet at either construction site — both run before the first
  Hive Master config sync — which is why this is a separate Rust constant
  rather than a config read.

An operator who explicitly sets `max_snapshot_candles` LOWER than the
default (e.g. `100_000`, as `apps/alpha-worker/tests/worker_gpu_tests.rs`
already does) keeps the original whole-stage-veto behavior for whichever
stages exceed it, full-history included — this is a deliberate,
backward-compatible reading of the field, not a behavior change for
anyone who already pins it.

### 9.4 Chunking design: `plan_column_cache_groups`

`crate::gpu_columns::plan_column_cache_groups`
(`crates/alpha-simulation/src/gpu_columns.rs:549`) is the safety valve
§9.2 found unnecessary for either shipped config but that the task
explicitly required regardless. Given the full-history survivor set's
DISTINCT `IndicatorPeriods` tuples and the real per-ticker candle
lengths, it:

1. Computes each tuple's OWN request-set byte cost via
   `estimate_column_cache_bytes`. A tuple whose own cost alone exceeds 85%
   of the configured budget (a margin absorbing the Agg-family
   over-estimate and any bookkeeping overhead the real cache pays that
   this pure arithmetic doesn't model) is routed to a `cpu_only` list —
   this tuple can never fit under ANY chunking, and is never handed to
   `GpuColumnCache::build` at all.
2. Greedily first-fits every other tuple into the most recently opened
   group, tracking the group's running de-duplicated `(family, period)`
   set so a tuple that shares columns with what's already in the group
   only pays the MARGINAL cost of its new columns. A tuple that doesn't
   fit the current group opens a new one.

This is chunk-size-DERIVED, not hardcoded: the group boundaries fall out
of the actual configured budget (`config.gpu.column_cache_budget_mb`) and
the actual candle counts (`ticker_lengths`) passed in, every call — never
a fixed constant. `engine::tiering::run_full_history_filter_gpu`
(`tiering.rs:411`) is the caller: for each group, it builds a FRESH
`GpuColumnCache`/`ColumnPool` (RAII-dropped at the end of that group's
iteration, so one group's resident columns never count against the next
group's budget — `tiering.rs:499-506`'s inline comment), runs
`GpuEvaluator::evaluate_batch` in `gpu_eval.max_individuals`-sized chunks
against that group's pool (mirroring `run_tier_filter_gpu`'s existing
launch-sizing pattern), and returns the indices it could NOT place on GPU
(the `cpu_only` tuples' survivors, plus anything if no eligible tickers or
the candle count exceeds `gpu_eval.max_candles`) for the caller
(`evaluate_batch_tiered`) to run through the EXISTING CPU `run_simulation`
path unchanged.

**What happens when a config genuinely cannot fit**: `run_full_history_
filter_gpu` never weakens `GpuEvaluator::evaluate_batch`'s residency
hard-fail (F1b-3) — a group that WOULD evict under budget is never
constructed in the first place (step 1 above), and if `evaluate_batch`
still hits a `NO_COLUMN` eviction despite that pre-planning (an
under-estimate this arithmetic didn't anticipate), the error propagates
as a whole-batch `Err` — this function does not catch it and silently
retry on CPU. A tuple whose OWN footprint alone cannot fit gets a
DELIBERATE, silent-nothing CPU fallback (correct, not a workaround) via
the `cpu_only` routing; an unexpected mid-flight eviction gets a loud
`Err` (also correct, per this crate's existing "fail loudly rather than
produce a silently wrong number" rule). Neither path ever raises
`column_cache_budget_mb` or weakens the hard-fail check to make a bad fit
"work."

### 9.5 Result fidelity: reusing the existing GPU approximation, not inventing a new one

`BacktestResultGpu` (`crates/alpha-simulation/src/gpu.rs:363-385`) has
always been a cross-ticker aggregate (`final_equity`, `trades`, `wins`, the
O06 `min_trades_across_tickers`) with no true per-ticker breakdown.
`run_tier_filter_gpu` already synthesizes an approximate `BacktestResult`
from it for a GPU-rejected tier-1/2 individual, spreading `trades`/`wins`
evenly across tickers by integer division and reporting the same
whole-batch `max_drawdown_percent` for every ticker. Task I extracts that
construction into a shared `approximate_result_from_gpu`
(`tiering.rs:124`) and reuses it, unchanged, for EVERY individual
`run_full_history_filter_gpu` handles (not just rejects, since full
history has no pass/fail threshold of its own) — this is the SAME
approximation already accepted at tier-1/2, not a new or looser one.
`TickerResult`'s `sharpe_ratio`/`sortino_ratio`/`calmar_ratio` stay `0.0`
(O10, `manual/14`, explicitly out of scope) exactly as they already do at
tier-1/2. Per this task's own scope discipline, TickerResult attribution
accuracy was NOT a goal here — getting the total-PnL/min-trades numbers
the tier gates (and, upstream, fitness) actually read onto the GPU was.

### 9.6 Correctness: warmup, gather semantics, and test coverage

Full-history length (up to `GPU_EVALUATOR_CANDLE_CAPACITY`, ~1.5M candles)
does not change any of the device gather's own correctness arguments:
`crate::device_gather_reference::validate_periods_for_device_gather` (still
invoked, unmodified, at the top of `GpuEvaluator::evaluate_batch`) bounds
`stochastic_d_period`/`vol_osc_slow`/`vwap_period`/`lcp_period` combine-step
scalars against `WARMUP_PERIOD` (2000) — a check that is independent of
`num_candles` entirely, since it only constrains how far a combine step can
look back relative to the FIRST post-warmup bar, not how many bars follow
it. No new divergence class is introduced by running the SAME gather over
more bars.

Test coverage added:

- `crate::gpu_columns::tests`: 8 new unit tests for
  `estimate_one_column_bytes`/`estimate_column_cache_bytes`/
  `plan_column_cache_groups` (exactness for plain/interleaved/Agg
  families, deduplication, single-group-when-it-fits,
  splits-when-it-doesn't, unfittable-tuple-routes-to-cpu-only), plus
  `task_i_full_range_worst_case_bytes_for_shipped_configs` (§9.2's
  ground-truth regression).
- `crates/alpha-simulation/tests/full_history_chunked_column_cache_tests.rs`
  (new file): the required host-side integration test, modelled on
  `f1b3_column_cache_pipeline_tests.rs`. A 4-individual, 2-ticker batch
  with a budget deliberately sized (via this batch's OWN measured
  per-individual cost, not a guessed constant) to force
  `plan_column_cache_groups` into more than one group, asserting the
  chunked pipeline — a SEPARATE `GpuColumnCache`/`ColumnPool` per group —
  reproduces exactly what a per-individual `build_snapshot_matrix` call
  produces, for every individual regardless of which group it landed in.
  This is the chunk-BOUNDARY test the task called out as the main risk:
  an individual gathered against the wrong group's pool (or silently
  falling back to the base snapshot for a slot that should have come from
  its own group's column) would be a wrong value that would not
  necessarily look implausible.

Not verified on this host (no GPU available): the actual CUDA/HIP kernel
launch (`launch_backtest_kernel`) at full-history length, and the real
`GpuColumnCache::build`'s wall-clock cost of computing ≈20-27 GiB of
columns for a production-sized survivor set. `backtest_kernel.hip.cpp`
compiles cleanly via `nvcc` under this change (no kernel-side code, ABI
struct, or `extern "C"` signature was touched by Task I — the entire
change is host-side sizing, gating, and chunking logic), but nothing
GPU-gated was executed.
