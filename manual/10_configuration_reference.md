# Chapter 10: Exhaustive Master Configuration Reference

This chapter serves as the definitive reference manual for all configuration parameters in Alpha Suite (`config.toml` and `config.harvest.toml`). Every section, key, data type, valid range, default value, and practical tuning guideline is documented below.

---

## 1. Top-Level Core & Infrastructure Settings

```toml
max_threads = 16
websocket_url = "wss://crypto.financialmodelingprep.com"
api_key = "your_api_key_here"
questdb_pg_uri = "postgresql://user:password@127.0.0.1:8812/qdb"
questdb_ilp_address = "tcp::addr=127.0.0.1:9009;"
refresh_interval = 5
tickers = ["BTC", "ETH", "SOL", "TURBO", "BONK"]
allow_short_selling = false
simulate_austrian_tax = true
```

| Parameter | Type | Default | Description & Tuning Advice |
|---|---|---|---|
| `max_threads` | Integer | `0` (auto) | Cap on Rayon CPU worker thread pool. Set to match physical cores for maximum throughput. |
| `websocket_url` | String | FMP URL | WebSocket endpoint for live price ingestion. |
| `api_key` | String | Secret | API key for historical data and live WebSocket feeds. |
| `questdb_pg_uri` | String | Postgres URI | QuestDB PostgreSQL wire protocol endpoint for SQL queries and session restores. |
| `questdb_ilp_address` | String | TCP Address | QuestDB InfluxDB Line Protocol (ILP) endpoint for ultra-high-speed tick and trade streaming. |
| `refresh_interval` | Integer | `5` | Seconds between database health checks and poll cycles. |
| `tickers` | Array of Strings | `["BTC", ...]` | Basket of crypto/asset symbols evaluated in simulations. Wide baskets promote universal market physics. |
| `allow_short_selling` | Boolean | `false` | When `false`, forces strategy direction genes strictly to `Long` ($[0.0, 0.0]$ bounds). |
| `simulate_austrian_tax` | Boolean | `true` | When `true`, models 27.5% capital gains tax with loss-pot deduction. Disable in harvest mode to observe pure alpha. |

---

## 2. `[simulation]` & `[simulation.friction]`

```toml
[simulation]
initial_equity = 100.0
leverage = 10.0
failed_equity = 20.0
max_acceptable_drawdown = 60.0
liquidation_margin_loss_pct = 0.5

min_sharpe_ratio = 1.0
max_sharpe_ratio = 15.0
min_sortino_ratio = 1.0
max_sortino_ratio = 25.0
min_calmar_ratio = 0.5
max_calmar_ratio = 40.0

[simulation.friction]
taker_fee_pct = 0.075
slippage_pct = 0.05
model = "constant"
impact_k = 0.1
adv_window_bars = 1440
max_participation = 0.02
spread_window_bars = 1440
# M3 (blueprint 81f1), off by default, not shown in the abbreviated block:
# intraday_profile = false
# intraday_bucket_minutes = 60
# intraday_profile_days = 20

[simulation.funding]
mode = "legacy_decay"
allow_legacy_with_rates = false

# Present in [simulation] itself, not shown in the abbreviated block above:
# min_hold_bars = 5
# annual_funding_rate_pct = 10.0
# bar_interval_minutes = 1.0
# use_real_funding = false
```

| Parameter | Type | Code Default | Description |
|---|---|---|---|
| `initial_equity` | Float | `100.0` | Starting account balance per asset. `Config::validate` rejects `<= 0.0`. |
| `leverage` | Float | `10.0` | Margin multiplier. `Config::validate` rejects `<= 0.0`. |
| `failed_equity` | Float | `10.0` | Account balance triggering bankruptcy/ruin failure. |
| `max_acceptable_drawdown` | Float | `15.0` | **Hard, global** drawdown cap (%) — see `manual/06` §1.6's circuit breaker. Distinct from the per-island *soft* target `ga.objective_annealing`/`ga.harvest.drawdown_target` shape. |
| `liquidation_margin_loss_pct`| Float | `0.5` | Fraction of margin lost on liquidation (0.5 = 50%). |
| `min_sharpe_ratio` / `max_sharpe_ratio` | Float | `0.0` / `f64::MAX` | `RatioBoundsPenalty` bounds (`manual/06` §5). The `0.0` lower bound is NOT a no-op — any negative Sharpe already breaches it and accrues penalty out of the box. |
| `min_sortino_ratio` / `max_sortino_ratio` | Float | `0.0` / `f64::MAX` | Same shape as Sharpe bounds. |
| `min_calmar_ratio` / `max_calmar_ratio` | Float | `0.0` / `f64::MAX` | Same shape as Sharpe bounds. |
| `min_hold_bars` | Integer | `5` | Bars a position must be held before the exit signal tree is even evaluated — `manual/06` §1.4. Does not gate stop/take-profit/liquidation. |
| `annual_funding_rate_pct` | Float | `10.0` | Annualized perpetual-funding rate spread evenly across bars — `manual/06` §1.3. |
| `bar_interval_minutes` | Float | `1.0` | Wall-clock minutes each simulated bar spans, used to derive the per-bar funding decay. **Nothing verifies this against the real candle spacing** — pointing the simulation at non-1-minute data without updating this field silently mis-scales every funding charge. |
| `taker_fee_pct` | Float | code `0.07`; shipped `config.toml` sets `0.075` | Taker fee % applied to **Total Notional Value** per side (entry and exit). Check which value actually applies to your run — the code-level default differs from the shipped config file. |
| `slippage_pct` | Float | `0.05` | Expected bid-ask slippage % on market entry/exit. Used verbatim under `friction.model = "constant"`; superseded by the `spread_and_impact` cost formula (`manual/06` §2.2b) under the other model. |
| `model` (F07, 4.1.2) | String | `"constant"` | Selects the per-side price-impact cost model in `execution.rs` — `manual/06` §2.2b. `"constant"` reproduces the `slippage_pct` formula above bit-for-bit. `"spread_and_impact"` prices each fill from the Corwin-Schultz half-spread proxy plus a volatility-scaled, participation-dependent impact term, and can reject a fill outright (see `max_participation` below). `Config::validate` rejects any other value. |
| `impact_k` (F07, 4.1.2) | Float | `0.1` | `spread_and_impact` impact coefficient: `cost_pct = est_half_spread_pct + impact_k * sigma_bar_pct * sqrt(participation)`. Ignored under `model = "constant"`. `Config::validate` rejects `< 0.0` or NaN. Scaled by `stressed_config` (`ga.friction_stress_multiplier`) the same way `slippage_pct`/`taker_fee_pct` are. |
| `adv_window_bars` (F07, 4.1.2) | Integer | `1440` | Rolling window (bars) the new `adv_notional` indicator slot (`AdvNotionalRunner`, `alpha-indicators`) sums `close * volume` over, for the participation denominator. `Config::validate` rejects `0`. Reaches the engine via `alpha_simulation::engine::friction_windows_from_config` → `alpha_indicators::FrictionWindows` on every live CPU path (`run_simulation`, every `IndicatorWarmupCache`, ensemble, ML-filter training); config-less paths (HMM, phenotype hashing, GPU/kernel-reference snapshots) keep the 1440-bar default — see `manual/06` §2.2b's note. |
| `max_participation` (F07, 4.1.2) | Float | `0.02` | `spread_and_impact` entry-only rejection threshold: a fill whose `notional / adv_notional` exceeds this is rejected outright (no entry, no fee). Never applied to exits. `Config::validate` requires `(0.0, 1.0]`. |
| `spread_window_bars` (F07, 4.1.2) | Integer | `1440` | Rolling window (bars) `SpreadProxyRunner` (4.1.1) averages its per-pair Corwin-Schultz estimate over. Mirrors `DEFAULT_SPREAD_PROXY_WINDOW_BARS`. `Config::validate` rejects `0`. Threaded into the engine exactly like `adv_window_bars` above (CPU engine paths only; the GPU kernel runs the `constant` model, `manual/14` row O17). |
| `spread_stress_multiplier` (F07, 4.1.2) | Float | `1.0` | Not intended to be set directly in `config.toml` — `stressed_config` writes `ga.friction_stress_multiplier` here so `execution.rs::friction_cost_pct` can scale the runtime-indicator-sourced half-spread the same way `impact_k`/`slippage_pct` are scaled (a `Config` scalar can't multiply a bus-read value directly). Stays `1.0` (no-op) outside GA evaluation. |
| `intraday_profile` (M3, blueprint 81f1) | Boolean | `false` | Off-by-default switch selecting the causal time-of-day volume profile (`alpha_simulation::engine::IntradayVolumeProfile`) over the flat `adv_notional` denominator for `spread_and_impact`'s participation formula — `manual/06` §2.2b-ii. CPU-only: `alpha_orchestrator::run::check_intraday_profile_gpu_conflict` hard-errors a run that sets this `true` together with `gpu.enabled = true`. Persisted per-run to `ga_run_metadata.friction_intraday_profile` (see the note at the end of this section). |
| `intraday_bucket_minutes` (M3, blueprint 81f1) | Integer | `60` | Time-of-day bucket width, in minutes, for `intraday_profile`. `Config::validate` requires `1..=1440` AND that it evenly divides `1440` — validated even when `intraday_profile = false`. |
| `intraday_profile_days` (M3, blueprint 81f1) | Integer | `20` | Trailing COMPLETED calendar days `IntradayVolumeProfile` averages each bucket over. `Config::validate` requires `>= 1` — validated even when `intraday_profile = false`. |
| `funding.mode` (F07, 4.1.3) | String | `"legacy_decay"` | Declarative funding-model selector — `manual/06` §1.3.1. `"legacy_decay"` is the flat, always-a-cost `per_bar_funding_decay` model above. `"real"` uses actual signed per-interval funding rates when loaded. ORs with the older `use_real_funding` boolean (`SimulationConfig::real_funding_enabled`) — either one being "on" enables the real path. `Config::validate` rejects any other value. Persisted per-run to `ga_run_metadata.funding_mode` (see the note at the end of this section). |
| `funding.allow_legacy_with_rates` (F07, 4.1.3) | Boolean | `false` | Escape hatch for the guard in `alpha_orchestrator::run` (runs right after `db::load_funding_rates`, since `Config::validate` cannot see the database): normally a run with `funding.mode = "legacy_decay"` fails to start when real funding rates are actually loaded for its tickers, since that silently discards sign-correct rate data. Set `true` to explicitly allow it anyway (e.g. deliberately reproducing a pre-F07 run). |

`simulation.friction.model` and `simulation.funding.mode` are persisted per-run to `ga_run_metadata.friction_model` / `.funding_mode` (`db::log_run_friction_and_funding_mode`, called once at run start alongside `log_run_start`). Unlike `objective_mode`/`partition_mode`/`persist_population_mode` (§17), these two columns (plus M3's `friction_intraday_profile`, added to the same writer/row) are read back via a "latest row where the column is NOT NULL" query (`db::get_run_friction_model` / `get_run_funding_mode` / `get_run_friction_intraday_profile`) rather than `LATEST ON timestamp` — later `ga_run_metadata` writes (periodic checkpoints, completion) do not repeat these columns, so the plain `LATEST ON timestamp` pattern the other three follow would read back `NULL` for any run that logged so much as one later row.

---

## 3. `[ga]` Evolutionary Algorithm Master Settings

```toml
[ga]
population_size = 10000
generations = 300
tournament_size = 4
column_cache_budget_mb = 32768
state_cache_budget_mb = 16384

param_crossover_eta_start = 5.0
param_crossover_eta_end = 25.0
param_mutation_eta_start = 10.0
param_mutation_eta_end = 60.0
param_mutation_prob_start = 0.4
param_mutation_prob_end = 0.05
gp_crossover_prob_start = 0.85
gp_crossover_prob_end = 0.35
gp_mutation_prob_start = 0.4
gp_mutation_prob_end = 0.05

hof_injection_prob = 0.20
gp_max_terminals = 10
pnl_saturation_factor = 0.65
profit_pressure_factor = 0.60
min_utility_weight = 0.5
harvest_mode = false
```

| Parameter | Type | Code Default | Explanation |
|---|---|---|---|
| `population_size` | Integer | `3200` | Total individuals across all islands. |
| `generations` | Integer | `100` | Maximum evolutionary generations before stopping. |
| `tournament_size` | Integer | `3` | Competitors in tournament selection. Effective size is silently capped at `min(size, 8)` regardless of a larger configured value (`multi_winner_tournament_indexed`). |
| `column_cache_budget_mb` | Integer | `0` (**disabled**) | RAM budget (MiB) for the shared CPU indicator-column cache (`IndicatorWarmupCache`, 5 indicator kinds only — Hurst, PriceCorrelation, ReturnSkewness, DownsideVol, McvAtrWindow). Unlike `state_cache_budget_mb`, this is off by default — the example above (`32768`) is a config-file choice, not the code default. See §4 below for the shared-budget semantics. |
| `state_cache_budget_mb` | Integer | `4096` | RAM budget (MiB) for the indicator-warmup-*state* cache (`MultiScaleIndicatorState` snapshots). Load-bearing, not optional — left unbudgeted this grows without bound across a run. `0` explicitly opts back into unbounded growth. |
| `param_crossover_eta_start` / `_end` | Float | `5.0` / `20.0` | SBX crossover distribution index, annealed across the run. Lower = wider exploration, higher = fine exploitation. |
| `param_mutation_eta_start` / `_end` | Float | `10.0` / `50.0` | Polynomial mutation distribution index, annealed. |
| `param_mutation_prob_start` / `_end` | Float | `0.3` / `0.05` | Per-gene mutation probability, annealed. |
| `gp_crossover_prob_start` / `_end` | Float | `0.8` / `0.4` | GP AST subtree crossover probability, annealed. |
| `gp_mutation_prob_start` / `_end` | Float | `0.3` / `0.05` | GP AST subtree mutation probability, annealed. |
| `hof_injection_prob` | Float | `0.1` | Probability of splicing a Hall of Fame champion strategy into an underperforming state. |
| `gp_max_terminals` | Integer | `8` | Soft limit on terminal leaf count per AST tree; also bounds the stateful-comparison-operand count `manual/04` §5.1 discusses. |
| `pnl_saturation_factor` | Float | `0.75` | Log-utility cap preventing massive outlier gains from dominating fitness. |
| `min_utility_weight` | Float | `0.5` | Weight given to the worst-performing asset's utility. |
| `profit_pressure_factor` | Float | `0.5` | Scaling weight for the weakest-link portfolio-protection bonus. |
| `state_locking_enabled` | Boolean | `false` | Enables per-state gene locking once a state's performance exceeds `state_locking_unlock_threshold`. |
| `state_locking_unlock_threshold` | Float | `0.75` | Threshold (fraction) for `state_locking_enabled`. |
| `dynamic_locking_enabled` | Boolean | `false` | Enables dynamic locking gated on solved-state count. |
| `dynamic_locking_min_solved_states` | Integer | `3` | Minimum solved states before `dynamic_locking_enabled` engages. |
| `dynamic_locking_strength_ratio` | Float | `3.0` | Locking strength ratio for dynamic locking. |
| `harvest_mode` | Boolean | `false` | Enables behavioral niche archiving and wide parameter exploration (see `[ga.harvest]`, §4). |
| `wait_for_first_result` | Boolean | `true` | Holds `generation_id` at 0 until at least one individual scores positive — this is exactly the plateau `manual/07` §4.5 explains `dispatch_epoch` exists to work around. |
| `seed` | `Option<Integer>` | unset | Top-level RNG seed. `None` means the orchestrator generates and logs one at startup, writing it back so a resumed run reuses it. |

> [!WARNING]
> Seven `GaConfig` sub-structs (`IslandModelConfig`, `GaSearchSpaceConfig`, `IndicatorPeriodsSearchSpaceConfig`, `ObjectiveAnnealingConfig`, `EvaluationTiersConfig`, plus the two above) are hand-written `Default` impls, **not** `#[derive(Default)]`, specifically because a derived impl would ignore every `#[serde(default = "...")]` function and silently zero every field. This matters operationally: if a TOML file includes `[ga]` but omits, say, `[ga.island_model]` entirely, serde falls back to that section's `Default::default()` — which correctly resolves to `num_islands = 16`, not `0`, only because of this fix. An operator on a build that predates it would get `num_islands: 0`, a divide-by-zero panic in island migration.

---

## 4. `[ga]` Sub-Tables: Regularization, Tiers, & Island Models

### `[ga.parsimony_penalty]` (Anti-Bloat)
```toml
[ga.parsimony_penalty]
enabled = true
base_penalty_factor = 0.01
complexity_exponent = 1.5
node_soft_cap = 50
```
- **`node_soft_cap`**: Cap compared against **average** AST nodes per HMM state, not the raw total across all states — see `manual/04` §9.4 / `manual/06` §5 for the corrected formula.
- **`complexity_exponent`**: Non-linear penalty multiplier ($\text{Excess}^{1.5}$). The overall shrink is a plain `(1 - Penalty)` clamped at zero — no `tanh` saturation.

### `[ga.evaluation_tiers]` (Tier-1 / Tier-2 of the Three-Stage Cascade)
Configures only the first two stages of the three-stage cascade (`manual/06` §4) — the terminal full-history stage has no threshold keys of its own; it runs unconditionally on every tier-2 (or tier-1, if tier-2 is inactive) survivor.
```toml
[ga.evaluation_tiers]
enabled = true
tier_1_days = 30
tier_2_days = 180
tier_1_min_total_pnl = -5.0
tier_1_min_trades_per_asset = 1
selection = "quantile"
quantile_keep_fraction = 0.3
window_anchor = "latest"
```
- **`tier_1_days`**: Length of fast pre-screening window.
- **`tier_1_min_total_pnl`**: Minimum portfolio PnL required in Tier-1 to qualify for full Tier-2 backtest.
- **`selection`** (String, code default `"quantile"`, F14): `"quantile"` keeps the top `quantile_keep_fraction` of each evaluation batch by rank (total PnL, then trade count, then index) regardless of the absolute floors above — batch-local successive halving, so a harsh regime or an early generation never culls a batch to zero survivors (INVARIANT: the top-1 candidate is always unconditionally promoted even if it misses the floor). `"absolute"` is the legacy all-or-nothing mode — both floors above must be met independently — still selectable, but can stall evolutionary progress. `Config::validate` rejects any other value.
- **`quantile_keep_fraction`** (Float, code default `0.3`): the fraction of each evaluation batch `selection = "quantile"` keeps. `Config::validate` requires `(0.0, 1.0]`. Has no effect under `selection = "absolute"`.
- **`window_anchor`** (String, code default `"latest"`, F14): `"latest"` always culls each ticker against its own most-recent `tier_1_days`/`tier_2_days` window — reproduces every run from before this key existed. `"random"` draws ONE offset per GENERATION (shared by every individual dispatched that generation, from `alpha_ga::rng_seed::seeded_rng` keyed on `tags::TIER_WINDOW` and the generation number) and shifts the window backward in time by that many candles for every ticker, so the tier-1/tier-2 culling gate isn't systematically biased toward whatever fits the CURRENT market regime. The chosen offset is carried on the wire as `SimulationBatch::tier_window_offset` so a distributed run's workers and the master's own local-fallback path slice the identical window for a generation. `Config::validate` rejects any other value.

### `[ga.island_model]` (Distributed Speciation)
```toml
[ga.island_model]
num_islands = 32
migration_interval = 10
migration_rate = 5
health_check_interval = 5
```
- **`num_islands`**: Number of isolated sub-populations (e.g. 32 islands with 312 individuals each). Code default `16`; `Config::validate` rejects `0` (would divide-by-zero sizing each island's population) and rejects `population_size < num_islands`.
- **`migration_interval` / `rate`**: Every N generations (code default `15`), M elite individuals (code default `3`) migrate to neighboring islands. `Config::validate` rejects `migration_interval = 0` (divide-by-zero in the migration-destination modulus).
- **`health_check_interval`**: Generations between periodic out-of-sample health checks (`ga_health_checks`/`ga_early_stopping` telemetry). Code default `15`. **`0` disables health checks entirely** -- no OOS backtests are run and both tables get no rows for the run; this is the one interval field `Config::validate` deliberately does NOT reject at `0`, since it is the documented off-switch.

### `[ga.objective_annealing]` (Dynamic Difficulty Ratchet)
```toml
[ga.objective_annealing]
min_trades_start = 20
min_trades_end = 4000
max_drawdown_start = 60.0
max_drawdown_end = 15.0
```
- Ratchets required trade count and maximum allowable drawdown targets from loose start values to strict production targets over the run horizon. Code defaults: `min_trades_start = 10`, `min_trades_end = 1000`, `max_drawdown_start = 25.0`, `max_drawdown_end = 5.0`.
- Only consulted when `ga.objective.mode = "legacy"` (below) — stationary mode uses `ga.objective.stationary`'s FIXED thresholds instead and never ratchets.

### `[ga.objective]` (F04: Fitness Objective Mode Switch)
```toml
[ga.objective]
mode = "legacy"

[ga.objective.stationary]
sharpe_lcb_z = 1.0
min_trades = 30
max_drawdown = 20.0
```
Selects which fitness objective `calculate_log_utility_fitness` (`evaluator/utility.rs`) scores individuals with — `manual/06` §5 (legacy) / §5b (stationary) has the full formulas and rationale for each.

| Parameter | Type | Code Default | Description |
|---|---|---|---|
| `mode` | String | `"legacy"` | `"legacy"`: the eleven-component pipeline in `manual/06` §5 — annealed targets, ratcheted per-asset floors, periodic ratchet decay. `"stationary"`: the fixed-threshold, per-era-Sharpe-LCB objective in `manual/06` §5b — no annealing, no ratchet, no decay. `Config::validate` rejects any other value. **Requires `ga.era_fitness.num_eras >= 3`** when set to `"stationary"` (the general `[2, 8]` bound from `[ga.era_fitness]` above still applies otherwise) — the per-era Sharpe LCB needs at least 3 eras to estimate dispersion at all. |
| `stationary.sharpe_lcb_z` | Float | `1.0` | $z$ in the Lo (2002) Sharpe-ratio standard-error formula (`manual/06` §5b). `Config::validate` rejects `<= 0.0` or NaN. Distinct from `ga.lcb_expectancy.z` (default `1.64`) — a different, trade-level LCB concept not used by stationary mode. |
| `stationary.min_trades` | Integer | `30` | FIXED per-ticker trade-count feasibility floor — never annealed, never ratcheted. A ticker below this routes the whole individual to the graduated failure band (`manual/06` §4's "Graduated Failure Band", §5b's exact formula). `Config::validate` rejects `0` (a zero floor is a silent no-op). |
| `stationary.max_drawdown` | Float | `20.0` | FIXED per-ticker max-drawdown feasibility ceiling (percent), same treatment as `min_trades`. `Config::validate` requires `(0.0, 100.0]`. |

Persisted as `ga_run_metadata.objective_mode` by every `db::log_run_final_targets` / `db::log_run_completion_status` call, alongside `final_min_trades` / `final_max_drawdown` (which carry the FIXED stationary thresholds, not annealed island targets, when this mode is active) — see `manual/06` §5b's "Generation independence and persistence". `db::resolve_objective_mode` mirrors `db::resolve_final_targets`'s legacy-run fallback: a pre-F04 run (or one whose write never landed) has no persisted value, and callers fall back to today's config with a warning.

### `[ga.search_space]` (Evolvable Gene Bounds)
Every gene the GA can evolve is bounded by a `(min, max)` pair here. These are not cosmetic — `manual/06` §2.3 and `manual/09` §4 both depend on strategies never exploring outside these bounds (the liquidation/stop ordering argument, and the sensitivity-analysis clamp, respectively).

```toml
[ga.search_space]
atr_multiplier = [1.0, 12.0]
reward_ratio = [1.0, 15.0]
cooldown_period = [5.0, 240.0]
base_risk_per_trade = [0.01, 0.15]
ml_filter_threshold = [0.5, 0.90]
stateful_indicator_period = [5.0, 60.0]
```

| Gene | Default Bounds |
|---|---|
| `atr_multiplier` | `[1.0, 12.0]` |
| `reward_ratio` | `[1.0, 15.0]` |
| `cooldown_period` | `[5.0, 240.0]` bars — see `manual/06` §1.7 for exact cooldown-bar semantics. |
| `base_risk_per_trade` | `[0.01, 0.15]` |
| `ml_filter_threshold` | `[0.5, 0.90]` |
| `stateful_indicator_period` | `[5.0, 60.0]` — GP-tree `Sma`/`StdDev`/`Lag` node periods. |

`[ga.search_space.indicator_periods]` bounds every indicator lookback period the GA can evolve (all in bars, all `(min, max)`):

| Period | Default Bounds | Period | Default Bounds |
|---|---|---|---|
| `atr_period` | `[7, 30]` | `bb_period` | `[5, 50]` |
| `rsi_period` | `[7, 30]` | `bb_std_dev_x100` | `[100, 400]` |
| `stochastic_k_period` | `[7, 30]` | `atr_volatility_window` | `[100, 500]` |
| `stochastic_d_period` | `[3, 10]` | `phase_space_slow` | `[15, 40]` (shares `physics_slow`'s bounds) |
| `roc_period` | `[5, 30]` | `phase_space_fast` | `[2, 10]` (shares `physics_fast`'s bounds) |
| `lag_period` | `[3, 20]` | `adx_period` | `[7, 30]` (shares `atr_period`'s bounds) |
| `agg_period_1` | `[3, 15]` | `ter_period` | `[20, 100]` |
| `agg_period_2` | `[16, 60]` | `price_correlation_period` | `[10, 40]` |
| `hurst_period` | `[50, 200]` | `vol_osc_fast` | `[5, 20]` |
| `return_skewness_period` | `[20, 60]` | `vol_osc_slow` | `[30, 100]` |
| `price_to_sma_period` | `[20, 200]` | `physics_fast` | `[2, 10]` |
| `vwap_period` | `[10, 50]` | `physics_slow` | `[15, 40]` |
| `downside_vol_period` | `[10, 60]` | `lcp_period` | `[10, 40]` |

`IndicatorPeriods::enforce_invariants()` additionally guarantees `agg_period_1 < agg_period_2`, `vol_osc_fast < vol_osc_slow`, `physics_fast < physics_slow`, `phase_space_fast < phase_space_slow`, and every period `>= 2`, on top of these bounds.

### `[ga.periods]` (M2, blueprint 81f1: Shared-Period-Family Genome Shrink)
```toml
[ga.periods]
sharing = "none"
```

| Parameter | Type | Code Default | Description |
|---|---|---|---|
| `sharing` | String | `"none"` | `"none"`: historical behavior, UNCHANGED — every one of the 25 shared-eligible `IndicatorPeriods` fields (see the family table below; excludes `bb_std_dev_x100` and the five `mcv_*` fields) is drawn, mutated and crossed over independently. `"family"`: the GA evolves ONE representative period per `alpha_core::PeriodFamily` and broadcasts it to every member of that family (`IndicatorPeriods::expand_from_families`), then clamps each expanded field back to ITS OWN bound from the table above. `Config::validate` rejects any other value with an error naming `ga.periods.sharing`. |

**Family table** (`alpha_core::PeriodFamily`, `manual/04` §"Shared indicator-period families" has the full rationale):

| Family | Members |
|---|---|
| `Momentum` | `rsi_period`, `roc_period`, `stochastic_k_period`, `adx_period`, `ter_period` |
| `Volatility` | `atr_period`, `downside_vol_period`, `atr_volatility_window`, `bb_period` |
| `Structure` | `hurst_period`, `price_correlation_period`, `return_skewness_period`, `lcp_period`, `price_to_sma_period`, `vwap_period` |
| `Fast` | `vol_osc_fast`, `physics_fast`, `phase_space_fast` |
| `Slow` | `vol_osc_slow`, `physics_slow`, `phase_space_slow` |
| `Smooth` | `stochastic_d_period`, `lag_period` |
| `Agg` | `agg_period_1`, `agg_period_2` |

`Fast`/`Slow` are kept as two SEPARATE families (rather than one six-member family) specifically so `fast < slow` — each pair's bounds are individually disjoint, e.g. `physics_fast` tops out at `10` while `physics_slow` starts at `15` — survives family sharing for every pair simultaneously: sharing one FAST representative and one SLOW representative, each clamped to its own field's bound, can never invert any of the three pairs, because clamping only narrows a value within its own already-non-overlapping range. `Agg` is the one family where this DOESN'T hold: `agg_period_1`'s bound (`[3, 15]`) and `agg_period_2`'s bound (`[16, 60]`) are also disjoint, but they share the SAME representative (drawn/mutated against `agg_period_1`'s bound), so a shared value below 16 clamps `agg_period_2` down to its own floor (`16`) regardless of what `agg_period_1` holds — an accepted crudeness of family sharing for that one family, not a bug.

`bb_std_dev_x100` (a standard-deviation multiplier, not a lookback) and the five `mcv_*` fields (F08: fixed per run, define the HMM regimes) are excluded from every family and continue to evolve (or, for `mcv_*`, stay fixed) exactly as under `"none"` mode regardless of `sharing`.

The expanded `IndicatorPeriods` — never the compact per-family representation — is what `structural_hash()` hashes and the F02 JSON codec persists, so old runs, the phenotype hash (F11), and every downstream reader load unchanged regardless of which mode produced a given row. Persisted as `ga_run_metadata.periods_sharing_mode` by `db::log_run_periods_sharing_mode`; `db::get_run_periods_sharing_mode` / `db::resolve_periods_sharing_mode` mirror `get_run_cpcv_enabled` / `resolve_friction_model`'s legacy-run fallback exactly — a pre-M2 run (or one whose write never landed) has no persisted value, and `resume_from_db` / `run_post_ga_pipeline_from_db` fall back to today's config with a `warn!`, also warning on a mismatch between the persisted mode and today's config.

### `[ga.harvest]` (Quality-Diversity Harvest Tuning)
Only consulted when `ga.harvest_mode = true`. Controls how finely strategies are binned into behavioral niches (harvest yield vs. distinctness trade-off) and how aggressively stagnation is perturbed:

```toml
[ga.harvest]
trade_count_bands = [20, 50, 100, 250, 500, 1000, 2500]
drawdown_band_pct = 5.0
drawdown_target = 50.0
injection_interval = 15
injection_fraction = 0.2
```
- **`trade_count_bands`**: Upper edges of the trade-count niche descriptor — a strategy falls into the first band whose edge it does not exceed. Finer/closer edges → more, less-distinct niches.
- **`drawdown_band_pct`**: Width (percent) of each drawdown niche band.
- **`drawdown_target`**: Per-island soft-knee drawdown target during harvest. Must sit clearly below `simulation.max_acceptable_drawdown`, or the soft-knee penalty has no gradient and the population becomes nearly all catastrophic blow-ups.
- **`injection_interval`** / **`injection_fraction`**: Every N stagnant generations, replace fraction F of each island with fresh random individuals.

### `[ga.deflation]` (F05/3.1.2: Multiple-Testing Overfitting Gate)

```toml
[ga.deflation]
enabled = false
base_threshold = 0.0
k = 1.0
mode = "heuristic"
min_dsr_probability = 0.95
```
- **`enabled`** (Boolean, code default **`false`**): gates the Hall-of-Fame admission and running-champion checks in `manual/05` §6.3. When `false`, neither hurdle below is enforced — candidates are admitted purely on raw score, as before this feature existed.
- **`mode`** (String, code default **`"heuristic"`**, F05/3.1.2): selects which hurdle `enabled` enforces. `Config::validate` rejects any value other than the two below.
  - **`"heuristic"` (default)**: `base_threshold + k * sqrt(2 * ln(max(cumulative_evaluations, 2))) * pop_score_std_dev`. Kept selectable (convention 9) so a run persisted under it stays reproducible.
  - **`"dsr"`**: the Deflated Sharpe Ratio probability (`manual/05` §6.3) must clear `min_dsr_probability`, using the true distinct-phenotype trial count (`n_trials`, F05/3.1.1) rather than `cumulative_evaluations`.
- **`base_threshold`** (Float, code default **`0.0`**) / **`k`** (Float, code default **`1.0`**): the heuristic hurdle's additive floor and multiple-testing scale factor. `Config::validate` rejects `k < 0.0` or either value being `NaN`.
- **`min_dsr_probability`** (Float, code default **`0.95`**, F05/3.1.2): minimum Deflated Sharpe Ratio probability a candidate must clear when `mode = "dsr"`. `Config::validate` requires the open interval `(0.0, 1.0)` — a value at or beyond either bound can never gate anything meaningfully (`<= 0.0` admits everything, `>= 1.0` admits nothing).

### `[ga.sweep]` (F05/3.1.4: Seed Recurrence Admission Gate)

```toml
[ga.sweep]
min_recurrence = 2
```
- **`min_recurrence`** (Integer, code default **`2`**, F05/3.1.4): minimum number of distinct seeds that must independently rediscover a structure (`sweep::group_by_recurrence`'s `recurrence_count`) before `seed-sweep` admits its representative candidate to validation/Execute — `manual/09` §5a. `Config::validate` rejects `0` (a zero floor would admit every candidate regardless of recurrence, reproducing the pre-fix "recurrence count is purely informational" defect). See `manual/09` §5a for why this is a SEPARATE knob from the `seed-sweep` CLI's own `--min-recurrence` reporting-threshold flag.

### Shared Cache Budgets Are Process-Wide, Not Per-Instance
`column_cache_budget_mb` and `state_cache_budget_mb` are **process-wide ceilings**, not limits applied independently to each cache instance. Both the worker (`DataCache`'s `tier1_warmup_cache`/`tier2_warmup_cache`/`full_warmup_cache`) and the master with evaluation tiers enabled (`warmup_cache`/`cache_tier1`/`cache_tier2`) construct up to three sibling `IndicatorWarmupCache` instances via `new_with_shared_budgets`, which makes those siblings share the same underlying byte-usage and budget atomics — so the configured value bounds their **combined** footprint, not `3×` it. An earlier version of this cache gave every sibling an independent budget, silently tripling the real ceiling; `state_cache_budget_mb`'s default of `4096` (≈4 GiB) is sized against the shared-budget arithmetic, not the old tripled one.

---

## 5. `[hmm]` Regime Discovery Configuration

```toml
[hmm]
training_iterations = 300
training_tolerance = 0.0000001
classification_window = 60
state_search_range = [16, 16]
student_t_nu = 3.0
state_selection = "bic"
expose_regime_confidence = false

[hmm.features]
mcv_adx_period = 14
mcv_roc_period = 14
mcv_sma_period = 50
mcv_atr_period = 14
mcv_atr_window = 200
```
- **`state_search_range`**: `[16, 16]` trains a fixed 16-state model (skips BIC search entirely when `start == end`, `manual/03` §1); code default is `(3, 12)`, a genuine BIC search range.
- **`student_t_nu`**: Degrees of freedom $\nu = 3.0$ for the per-feature Student-t emission distributions (`manual/03` §2.2 — diagonal, not a full covariance matrix). Code default `3.0`.
- **`training_tolerance`**: EM log-likelihood convergence threshold. Code default `1e-6`.
- **`training_iterations`**: Maximum EM iterations per training restart. Code default `100`. Not to be confused with `NUM_RESTARTS = 20`, a hardcoded Rust constant (`alpha-hmm/src/setup.rs`) governing how many independent restarts are attempted — there is no config key for restart count.
- **`classification_window`**: Code default `50`.
- **`state_selection`** (String, code default **`"bic"`**, F08): selects the state-count policy `find_optimal_hmm_states` uses — `manual/03` §8.1. `Config::validate` rejects any value other than the two below.
  - **`"bic"` (default)**: runs the full BIC sweep over `state_search_range` and logs the score table — unchanged from every pre-F08 config (the default reproduces today's behaviour exactly, including the historical shortcut where `state_search_range.0 == .1` skips the sweep regardless of this key).
  - **`"fixed"`**: trains a single model at `state_search_range.0`, never consulting `.1`. `Config::validate` additionally rejects `"fixed"` when `state_search_range.0 != .1`, since a non-degenerate range in this mode would otherwise silently discard `.1`.
- **`[hmm.features]`** (F08): the HMM classifier's own fixed MCV feature window (`manual/03` §8.2), independent of the GA's evolved `IndicatorPeriods`. These are the only five `IndicatorPeriods` fields that reach the HMM's observation vector at all (see `manual/03` §1.2 / §8.2) — an earlier version of this sub-table instead exposed `roc_period: 10, lag_period: 5, agg_period_1: 5`, which compiled and validated but had zero effect on the trained model, since none of those three fields feed any of the four MCV slots the HMM reads.
  - **`mcv_adx_period`** (Integer, code default **`14`**): drives `mcv_trend_intensity_1m` via `McvAdxRunner`. `Config::validate` rejects `0`.
  - **`mcv_roc_period`** (Integer, code default **`14`**): drives `mcv_velocity_1m` via `McvRocRunner`. `Config::validate` rejects `0`.
  - **`mcv_sma_period`** (Integer, code default **`50`**): drives `mcv_trend_bias_1m` via `McvSmaRatioRunner`. `Config::validate` rejects `0`.
  - **`mcv_atr_period`** (Integer, code default **`14`**): drives `mcv_volatility_state_1m` via `McvAtrWindowRunner`, paired with `mcv_atr_window`. `Config::validate` rejects `0`.
  - **`mcv_atr_window`** (Integer, code default **`200`**): the rolling window `McvAtrWindowRunner` ranks `mcv_atr_period`'s raw ATR reading against. `Config::validate` rejects `0`.
- **`expose_regime_confidence`** (Boolean, code default **`false`**, M1/blueprint 81f1): registers the GP-visible `regime_confidence` terminal (the causal filtered-posterior confidence, `manual/03` §8.4) when `true`. Off means the slot is not registered in the indicator registry at all (not merely non-GP-visible), so slot count, every existing slot id, and `SlotRegistry::version_hash()` stay bit-identical to a pre-M1 registry for the default `false`. `Config::validate` rejects `true` combined with `gpu.enabled = true`: the GPU snapshot path does not gather this externally-written column yet, so a GPU-evaluated bar would otherwise silently see the slot's registry default instead of a live value — see `manual/03` §8.4 for the full reasoning and how M1.4 wires the terminal through the GA loop's evaluator, holdout/health-check, post-GA pipeline and CPCV scoring. `Config::validate` also rejects `true` combined with a `[distributed]` section present at all (M1.4): the distributed dispatch protocol has no wire field for the confidence column, so a distributed run would silently evaluate GPU workers without it rather than erroring — M1.4 closes that gap by rejecting the combination outright instead of extending the protocol.

This is the complete `[hmm]` key set — seven top-level fields plus the `[hmm.features]` sub-table, all shown above.

---

## 6. `[ensemble]` & `[validation]`

```toml
[ensemble]
ensemble_size = 12
quorum_threshold = 4
ml_filter_threshold = 0.65
cooldown_period = 30
trailing_stop_atr_multiplier = 2.5
override_params = false

[validation]
num_champions_to_validate = 5
monitor_fraction = 0.05

[validation.walk_forward]
num_folds = 5
mode = "fixed_champion"
generations = 30

[validation.sensitivity]
mode = "lhs_all_genes"
samples = 64
epsilon = 0.10
num_steps = 3
perturbation = 0.08

[validation.monte_carlo]
mode = "block_bootstrap"
block_len_bars = 240
num_simulations = 100

[validation.capacity_curve]
enabled = false

[validation.cpcv]
groups = 6
embargo_bars = 1440
max_pbo = 0.5
```
- **`ensemble_size`** (code default `10`) **/ `quorum_threshold`** (code default `3`): council size and minimum agreeing votes — but see `manual/09` §5: an entry additionally requires **zero** opposing-direction votes, not just meeting the threshold. `ml_filter_threshold` (code default `0.55`) and `cooldown_period` (code default `30`) and `trailing_stop_atr_multiplier` (code default `2.0`) round out five of the six keys in the full `EnsembleConfig` key set.
- **`override_params`** (Boolean, code default **`false`**, F06): as of F06, `cooldown_period` and `trailing_stop_atr_multiplier` above are no longer applied unconditionally to every executed trade. By default (`false`), the ensemble engine instead applies each trade's OWN evolved genes: the trailing-stop `atr_multiplier` of whichever member's vote decided the entry, and the MEDIAN `cooldown_period` gene across the winning-direction consensus voters — see `manual/09` §5 for the full mechanics. Set `override_params = true` to restore the pre-F06 behaviour of applying the two `[ensemble]` values above verbatim to every trade regardless of which member(s) decided it (convention 9: old path retained for reproducing a pre-F06 run bit-for-bit). This is the complete `EnsembleConfig` key set — six keys total, all shown above.
- **`num_folds`**: contiguous out-of-sample folds evaluated per `mode` below — `manual/09` §2. Code default `5`.
- **`walk_forward.mode`** (String, code default **`"fixed_champion"`**, F09/3.4.3): selects between segmented-OOS scoring and true walk-forward re-optimization — `manual/09` §2. `Config::validate` rejects any value other than the two below.
  - **`"fixed_champion"` (default)**: every fold is scored against the SAME champion genome (no per-fold retraining) — this is the "Segmented OOS" panel on the Grafana dashboard (`grafana/strategy.json`), renamed from "Walk-Forward Out-of-Sample Equity Curve" in F09/3.4.3 because that's what it actually measures: one fixed genome's robustness across separate out-of-sample windows.
  - **`"reoptimize"`**: for fold `i > 0`, a short GA (`walk_forward.generations` below, population from `ga.population_size`) re-optimizes the champion against fold `i - 1`'s own data — mutations only, seeded from a `WALK_FORWARD`-tagged RNG stream (`alpha_ga::rng_seed::tags::WALK_FORWARD`) — and the resulting best individual scores fold `i`. Fold `0` has no preceding fold, so it always scores the original champion. Persisted to the same `ga_walk_forward_results` table (a new `mode` column distinguishes the two), and charted on its own "Walk-forward (re-optimized)" Grafana panel.
- **`walk_forward.generations`** (Integer, code default **`30`**): generations run per fold's mini-GA when `mode = "reoptimize"`; ignored under `"fixed_champion"`. `Config::validate` rejects `0`.
- **`sensitivity.mode`** (String, code default **`"lhs_all_genes"`**, F09/3.4.2): selects the sensitivity sweep algorithm — `manual/09` §4. `Config::validate` rejects any value other than the two below.
  - **`"lhs_all_genes"` (default)**: Latin-hypercube design over every numeric `StrategyParams`/`IndicatorPeriods` gene at once, driven by `samples`/`epsilon` below. Logs to `ga_sensitivity`.
  - **`"two_gene"` (legacy)**: the original `atr_multiplier`/`reward_ratio`-only coordinate sweep, driven by `num_steps`/`perturbation` below. Kept selectable (convention 9) so a champion validated under it stays reproducible against its already-persisted `ga_sensitivity_results` rows.
- **`sensitivity.samples`** (Integer, code default **`64`**): number of Latin-hypercube design rows sampled per champion when `mode = "lhs_all_genes"`. Each row scores one full simulation, so this is directly proportional to validation cost. `Config::validate` rejects `0`.
- **`sensitivity.epsilon`** (Float, code default **`0.10`**): per-gene perturbation half-width for `mode = "lhs_all_genes"` — each design row multiplies every gene by a factor drawn uniformly from `[1 - epsilon, 1 + epsilon]`. `Config::validate` requires `(0.0, 1.0)`: `<= 0.0` would draw every factor as exactly `1.0` (no perturbation, zero variance for the elasticity regression to measure), and `>= 1.0` can drive a gene's factor to zero or negative, which is nonsensical for a multiplicative perturbation.
- **`num_steps`** / **`perturbation`**: sweep steps and per-step perturbation fraction for `atr_multiplier`/`reward_ratio` only, used when `sensitivity.mode = "two_gene"` — `manual/09` §4.1. Code defaults `2` / `0.10`.
- **`monte_carlo.mode`** (String, code default **`"block_bootstrap"`**, F09/3.4.1): selects the Monte Carlo resampler — `manual/09` §3. `Config::validate` rejects any value other than the two below.
  - **`"block_bootstrap"` (default)**: stationary block bootstrap over `block_len_bars`-length runs of consecutive returns, preserving the synthetic series' own short-range autocorrelation.
  - **`"permutation"` (legacy)**: the original single-bar shuffle, which destroys essentially all autocorrelation. Kept selectable (convention 9) so a run persisted under it stays reproducible.
- **`monte_carlo.block_len_bars`** (Integer, code default **`240`**): block length (in bars) for the stationary block bootstrap, used when `monte_carlo.mode = "block_bootstrap"`. `Config::validate` rejects `< 2` — a length of `1` degenerates the block bootstrap into a per-bar shuffle indistinguishable from `"permutation"`.
- **`num_simulations`**: correlated-return-resampling Monte Carlo runs, **not** trade reshuffles — `manual/09` §3. Code default `100`.
- **`capacity_curve.enabled`** (Boolean, code default **`false`**, F07/4.1.4): gates the capacity-curve stage in `run_post_ga_pipeline` — `manual/06` §2.2c. When `true`, re-runs the selected champion at `{1, 2, 5, 10, 20}x` `simulation.initial_equity` under forced `friction.model = "spread_and_impact"` and persists edge-decay rows to `ga_capacity_curve`. Off by default (convention 9): this adds five extra full-history simulations per run, so existing pipelines stay bit-identical unless explicitly opted in.
- **`monitor_fraction`** (Float, code default **`0.05`**, F05/3.1.5): fraction of the GA's own in-sample span carved off its TAIL as the sealed health-check monitor band — `manual/09` §0.1. `Config::validate` requires the open interval `(0.0, 1.0)`; whether the REMAINING GA span still clears `data.min_common_days` for a given dataset is checked at partition-build time in `run.rs`, not here (that check is data-dependent and degrades to a `warn!` + disabled health check rather than a config-load failure).
- **`cpcv.groups`** (Integer, code default **`6`**, F05/3.1.3): number of contiguous groups the GA's in-sample span is split into for Combinatorial Purged Cross-Validation — `manual/09` §4a. `C(groups, 2)` splits are evaluated (`15` at the default). `Config::validate` rejects `< 3` (fewer than 3 groups leaves no train groups once 2 are held out for test).
- **`cpcv.embargo_bars`** (Integer, code default **`1440`**, F05/3.1.3): bars of embargo applied on top of the purge window at every train/test boundary — `manual/09` §4a.
- **`cpcv.max_pbo`** (Float, code default **`0.5`**, F05/3.1.3): a front candidate whose per-candidate Probability of Backtest Overfitting exceeds this is excluded from the ensemble/genome selection (logged, not silently dropped) — `manual/09` §4a. `Config::validate` requires `(0.0, 1.0]`.
- **`cpcv.enabled`** (Boolean, code default **`false`**, F05/3.1.3): gates whether `run_post_ga_pipeline` actually runs CPCV/PBO over the top-N front and excludes `pbo > max_pbo` candidates before the Execute stage — `manual/09` §4a. Default `false` (convention 9): an existing run's Execute/ensemble selection stays bit-identical unless an operator opts in.

This is the complete `[validation]` key set: `num_champions_to_validate` and `monitor_fraction` (code defaults `5` / `0.05`) plus the five sub-tables above.

---

## 7. `[distributed]` & Hive-Mind Settings

> [!WARNING]
> **Never commit a real `auth_token` to `config.toml`.** An earlier revision of this manual (and of `DownloaderConfig`'s own default) showed the literal `"alpha-secret-key-2026"` as an example value — that string is in this repository's git history and must be treated as already compromised. Leave `auth_token` unset in the file and set the `HIVE_AUTH_TOKEN` environment variable instead; `WorkerManager::new` refuses to start the HiveMind listener at all if neither resolves to a non-empty token (fail-closed).

```toml
[distributed]
enabled = true
server_port = 50051
batch_size = 300
steal_eligible_after_secs = 180
sync_cache_ttl_mins = 30
# auth_token intentionally omitted -- set HIVE_AUTH_TOKEN instead
```

| Key | Type | Default (code) | Description |
|---|---|---|---|
| `enabled` | Boolean | *(required, no default)* | The section itself gates HiveMind mode — `Config.distributed: Option<DistributedConfig>` is `None` when `[distributed]` is absent, which is how the master decides to run standalone. When the section IS present, `enabled` still exists as an explicit field (it is **not** merely "section presence = enabled" — verified against `DistributedConfig` in `alpha-config/src/distributed.rs`). |
| `server_port` | u16 | *(required)* | Master TLS listening port for worker connections. |
| `batch_size` | usize | `50` | Individuals bundled into one `SimulationBatch`. `evaluator.rs`, `handlers/request_work.rs`, and `local_fallback.rs`'s `baseline_batch_size` all read this same default via `DEFAULT_DISTRIBUTED_BATCH_SIZE` — kept as one constant precisely so the three sites can't silently drift (they used to disagree, 10 vs. 50). |
| `steal_eligible_after_secs` | u64 | `180` | How long a batch must have been *actually running* (per `WorkerHeartbeat::started_batches`, not merely dispatched or queued) before the master's local-fallback task will speculatively race it (`manual/07` §4.4, Mechanism 1). Tune above the p50 batch runtime for this deployment's `batch_size` — too low turns every healthy in-flight batch into a race, wasting CPU on both sides for nothing. |
| `sync_cache_ttl_mins` | u64 | `30` | TTL for the master's per-ticker `SyncData` payload cache. `0` means "never expire" (not "don't cache") — the cache is still bounded by `SYNC_DATA_CACHE_MAX_BYTES` (2 GiB) regardless. |
| `auth_token` | `Option<String>` | unset — falls back to `HIVE_AUTH_TOKEN` env var | Pre-shared secret every worker must present at handshake (`manual/07` §3.1). Config value takes precedence over the env var when both are set. |

Two further mechanisms live in this section's territory but have no config key of their own — the **five-connection design** (control/submit/heartbeat/fail/telemetry) and the **watchdog's `IN_FLIGHT_TIMEOUT`** (hardcoded to 5 minutes, `constants.rs`, not exposed as a config key) — see `manual/07` §4.2–§4.4 for both.

---

## 8. `[ml_filter]` — LightGBM Trade Filter

```toml
[ml_filter]
min_risk_modulator = 0.25
min_threshold = 0.5
feature_set = "scale_free"
include_taken_flag = false

[ml_filter.lightgbm_params]
objective = "binary"
metric = "auc"
boosting_type = "gbdt"
num_leaves = 15
learning_rate = 0.05
feature_fraction = 0.7
bagging_fraction = 0.7
bagging_freq = 1
min_child_samples = 20
n_estimators = 1000
early_stopping_round = 50
verbose = -1
n_jobs = -1
seed = 42
```
- **`min_risk_modulator`** (code default `0.25`): floor of the linear risk-scaling range described in `manual/08` §1.B.
- **`min_threshold`** (code default `0.5`, validated to `[0.0, 1.0]`): floor the *swept, winning* `ml_filter_threshold` must meet or exceed before the post-GA pipeline bothers activating a modulator at all — see `manual/08` §1.B.
- **`feature_set`** (code default `"scale_free"`, F18; `"legacy"` also accepted): which `ml_filter::features` builder a NEW `train-filter` run uses — `"scale_free"` (ratios/z-scores + a categorical asset id, orders-of-magnitude more comparable across assets of very different price levels) or the original, frozen `"legacy"` 48-feature layout. Only steers what training writes NEXT; an already-trained model on disk always predicts under whatever schema its `<ml_model>.schema.json` companion file says it was trained with, regardless of this live setting — see `manual/08` §1.A.
- **`include_taken_flag`** (code default `false`, F18, `"scale_free"` only): appends one more feature recording whether a given training row's signal was actually executed versus a hypothetical-outcome-only signal the GP evaluator fired on but that never became a real trade. Ignored under `feature_set = "legacy"`.
- **`lightgbm_params`**: an arbitrary `serde_json::Value` (a TOML inline table) passed straight through to `lightgbm3::Booster::train` — any parameter `lightgbm3` accepts can be set here; the code default seeds the eleven keys shown above. This is the complete `[ml_filter]` key set — five keys plus `lightgbm_params`.

---

## 9. `[nn]` — Alpha Architect Transformer & Neural Seeding

```toml
[nn]
use_neural_seeding = false
model_path = "models/strategy_brain.safetensors"
seeding_ratio = 0.2
inference_device = "cpu"

[nn.training]
learning_rate = 0.0003
batch_size = 32
validation_split = 0.1

[nn.model]
d_model = 128
n_head = 4
n_layer = 4

[nn.context_model]
auto_train = false
model_path = "models/context_brain.safetensors"
epochs = 30
input_dim = 5
d_model = 128
n_head = 4

[nn.dataset]
mix_fraction = 0.3
strata = 5
```
- **`use_neural_seeding`** (code default `false`): gates whether `generate_neural_suggestions` runs at all — `manual/08` §3.
- **`model_path`**: path to the trained `StrategyTransformer` weights.
- **`seeding_ratio`** (code default `0.2`; shipped configs set `0.8`): **parsed but not read anywhere in the seeding code path** — see `manual/08` §3 for the full correction. Do not rely on this key changing seeding behavior.
- **`inference_device`** (code default `"cpu"`, validated to `"cpu"` or `"auto"`, Wave 6/F33): which device `Generator::generate`/`decode` run inference on. `"cpu"` forces CPU determinism (candle's CUDA/Metal kernel selection and reduction order are not bit-deterministic across runs); `"auto"` opts into `regime_context::get_device`'s CUDA/Metal-preferring selection, trading that reproducibility guarantee for throughput. See `manual/08` §4.1.
- **`[nn.training]`**: `learning_rate` (`0.0003`), `batch_size` (`32`), `validation_split` (`0.1`) — hyperparameters for `nn-trainer`'s training subcommand.
- **`[nn.model]`**: `d_model` (`128`), `n_head` (`4`), `n_layer` (`4`) — Transformer architecture dimensions, matching `manual/08` §2.
- **`[nn.context_model]`**: `auto_train` (`false`), `model_path` (`"models/context_brain.safetensors"`), `epochs` (`30`) — the separate market-context encoder model. As of Wave 6/F34.3, also carries that model's own architecture dims: `input_dim` (`5`, must match `market_character_vectors`' feature count), `d_model` (`128`), `n_head` (`4`, validated to evenly divide `d_model`) — replacing the `MarketContextTransformer::new(5, 128, 4, ..)` literals `ContextTrainer::train` used to hardcode. See `manual/08` §4.4.
- **`[nn.dataset]`** (new table, Wave 6/F34.2): `mix_fraction` (`0.3`, validated to `[0.0, 1.0]`) — fraction of `load_elite_strategies`'s harvested corpus drawn as a stratified random sample of the ranked remainder rather than the unconditional top rows; `strata` (`5`, validated `>= 1`) — number of rank-ordered buckets that remainder sample is stratified across. See `manual/08` §4.3.

---

## 10. `[paths]` — Model & Data File Locations

```toml
[paths]
hmm_model = "hmm_model.json"
ml_model = "lgbm_modulator.model"
context_model = "market_context.safetensors"
ml_training_data = "lgbm_training_data.csv"
```
Four keys, all `String` filesystem paths, all shown above with their code defaults. `hmm_model` is the `SerializableHmm` JSON the HMM training/classification path reads and writes (`manual/03`); `ml_model` is what `AdaptiveTradeModulator::new` loads and `train_and_save_model` (`manual/08` §1.C) writes; `ml_training_data` is the CSV `generate_training_data_multi` produces and `train_and_save_model` consumes.

---

## 11. `[api]` — Forensic API

```toml
[api]
listen_address = "127.0.0.1:12326"
strategy_dashboard_url = "http://127.0.0.1:3000/d/alpha-strategy/alpha-suite-unified-strategy-dossier?var-ga_run_id={ga_run_id}&var-params_hash={params_hash}"
max_concurrent_forensic_jobs = 1
# auth_key intentionally omitted -- set ALPHA_FORENSIC_API_AUTH_KEY instead
```
- **`listen_address`**: bind address for `forensic-api`. The shipped `config.toml` binds `0.0.0.0` (not loopback) so a browser on another machine can reach the Command Center's data link.
- **`strategy_dashboard_url`**: redirect template for the `/forensic` route's 303 redirect, containing the literal placeholders `{ga_run_id}`/`{params_hash}` — substituted only after both are validated (`validate_identifier`).
- **`max_concurrent_forensic_jobs`** (code default `1`): semaphore bound on concurrent `run_forensic_analysis` calls. Kept low because that function runs CPU-bound work (HMM precompute, `run_simulation`) synchronously on a shared `smol` executor thread, not via `smol::unblock` — too many concurrent jobs starves the accept loop and `/forensic/status`.
- **`auth_key`**: shared secret for the `/forensic` route (`X-Forensic-Auth` header or `key=` query param), constant-time compared. Resolves from an explicit TOML value first, then `ALPHA_FORENSIC_API_AUTH_KEY`; **fails closed** — an unset key rejects every request rather than opening the route. `/health` and `/forensic/status` are deliberately exempt.
- The whole `[api]` section is optional (`Config.api: Option<ApiConfig>`); when absent, `ApiConfig::default()` applies (`listen_address = "127.0.0.1:12326"`, the same dashboard URL template, `max_concurrent_forensic_jobs = 1`, and `auth_key` still resolved from the environment variable).

### 11.1 `[api]` — Control plane keys

**All eight keys below are implemented (C3: six, C4: `auth_token`/`require_token_for_reads`).** See `manual/17_control_plane.md` for the full design.

- **`api.supervisor.enabled`** (default `false`, nested under `[api.supervisor]`): the off switch. While `false`, `forensic-api` behaves exactly as it does today — `main` never constructs a `Supervisor` at all (no `ga_run_supervisor` table check, no startup adoption query, no background dispatch/liveness tasks), and `AppState::supervisor` is `None`. Every `/v1/*` route (C4) responds `503 supervisor_disabled`, and `/forensic`, `/forensic/status`, `/health` are the only working routes, byte-identical to the current behavior pinned by the C0 characterization tests. `SupervisorConfig` is a nested struct (rather than a bare `supervisor_enabled: bool`) purely to leave room for this section to grow without another top-level `[api]` key.
- **`api.max_concurrent_runs`** (default `4`): CPU-slot count for the control-plane scheduler (§4 of manual/17, `apps/forensic-api/src/scheduler.rs`). Independent of `api.max_concurrent_forensic_jobs`, which continues to bound only the existing `/forensic` worker. Validated `>= 1` by `Config::validate` — a supervisor with zero slots could never dispatch a single `Queued` job.
- **`api.runs_dir`** (default `"runs"`, relative to the process's working directory): filesystem root under which each job gets `<runs_dir>/<job_id>/` (control socket, `overrides.toml`, `stderr.log`, and wherever `alpha_ga::session::save_session` lands a checkpoint for that run). Validated non-empty. A `systemd`-managed deployment should set this to an absolute path.
- **`api.ga_runner_path`** (default `"ga-runner"`, resolved via `$PATH`): path to the `ga-runner` binary the supervisor spawns as a child process (`smol::process::Command::new`). Validated non-empty. A production deployment should set an absolute path rather than depend on the service's own `$PATH`.
- **`api.gpu_exclusive`** (default `true`): whether the single GPU token (§4 of manual/17) is enforced at all; exists as a switch in case a future host has more than one GPU to allocate independently, which is out of scope for this blueprint. Concretely, a job's GPU-token requirement is `config.gpu.enabled && api.gpu_exclusive` (`scheduler::resource_requirement_for_config`) — note `GpuConfig::enabled` itself already defaults to `true`, so in practice almost every profile needs the GPU token unless it explicitly disables GPU mode.
- **`api.distributed_exclusive`** (default `true`): whether the single distributed-run token (§4 of manual/17) is enforced. Decision D6 requires this to stay `true` in every deployment this blueprint targets — the key exists for explicitness in config, not to invite disabling it. A job's requirement is `config.distributed.is_some() && api.distributed_exclusive`.

Implemented by C4:

- **`api.auth_token`** (no literal default): sourced the same way `ApiConfig::auth_key` is today — an explicit `api.auth_token` value in `config.toml` wins when present; only when it is absent from the TOML does the code fall back to the `ALPHA_API_AUTH_TOKEN` environment variable (`crates/alpha-config/src/api.rs::default_auth_token`). An empty value (neither source set) means fail closed: `alpha_core::security::verify_shared_secret` already rejects every credential against an empty `expected`, so an operator who never sets this locks the entire `/v1/*` surface rather than opening it. Verified against `Authorization: Bearer <token>` — `alpha_core::security::parse_bearer_token` (new, C4) extracts the token case-insensitively on the `Bearer` scheme name and never panics on malformed input.
- **`api.require_token_for_reads`** (default `true`): when `true`, `GET /v1/jobs*` and `GET /v1/runs/*` require the same bearer token as the mutating routes; set to `false` only when network access to `forensic-api` is already restricted by other means. Every `POST`/`DELETE` route and the `GET /v1/jobs/{id}/events` SSE stream require the bearer UNCONDITIONALLY regardless of this switch — only the plain `GET` reads are gated by it.

---

## 12. `[gpu]` — GPU Acceleration

```toml
[gpu]
enabled = true
device_id = 0
max_snapshot_candles = 1500000
column_cache_budget_mb = 65536
```
- **`enabled`** (code default `true`): master switch for GPU-accelerated tier-1/tier-2/full-history evaluation (`manual/06` §4).
- **`device_id`** (code default `0`): 0-indexed CUDA/HIP device.
- **`max_snapshot_candles`** (code default `1_500_000`): per-stage candle-count ceiling before that stage falls back to CPU — gates **all three** tiering stages uniformly (tier-1, tier-2, full-history), each additionally clamped by the evaluator's actual persistent buffer capacity (the smaller of the two always wins).
- **`column_cache_budget_mb`** (code default `65536` = 64 GiB): byte budget for the GPU-resident indicator column cache (`GpuColumnCache`) — see `manual/12` for the full per-slot memory arithmetic this is sized against. **`0` disables the GPU column cache entirely**: tier-1/tier-2 fall back to per-group replay, and the full-history stage falls back to CPU entirely (a full-history survivor set's period diversity makes per-group replay infeasible at ~1,000,000 candles). Deliberately a **separate** budget from `ga.column_cache_budget_mb` — the two cache completely different things (GPU-resident device-adjacent columns vs. CPU-side `IndicatorWarmupCache`) and must be tuned independently.

---

## 13. `[downloader]` — Worker Bootstrap Distribution Server

```toml
[downloader]
enabled = false
listen_address = "0.0.0.0"
port = 12399
payload_path_win = "download.win.bin"
payload_path_linux = "download.bin"
client_bootstrap_path_win = "/path/to/alpha-launcher.exe"
client_bootstrap_path_linux = "/path/to/alpha-launcher"
# client_auth_key and bin_auth_token intentionally omitted --
# set ALPHA_DOWNLOADER_CLIENT_AUTH_KEY / ALPHA_DOWNLOADER_AUTH_TOKEN instead
```
- **`enabled`** (code default `false`): starts the HTTP bootstrap server (`ga-runner` only) that serves `alpha-launcher` binaries and payloads to new worker machines.
- **`client_auth_key`** / **`bin_auth_token`**: shared secrets for the `/client` route and the `/ping`/`/hash`/`/download` routes respectively. Both **used to be literal secrets committed to source** (`"alpha-secret-key-2026"` for the client key, a hardcoded 512-bit token for the bin token) — both literals are in git history and must be treated as compromised regardless of this fix. Resolve only from config or their respective environment variables (`ALPHA_DOWNLOADER_CLIENT_AUTH_KEY`, `ALPHA_DOWNLOADER_AUTH_TOKEN`); an unset value fails closed.
- **`payload_path_win`** / **`payload_path_linux`**: on-disk payload cache paths. Code defaults `"download.win.bin"` / `"download.bin"`.
- **`client_bootstrap_path_win`** / **`client_bootstrap_path_linux`**: where a pre-built `alpha-launcher` binary is looked for before falling back to `cargo build`. Code defaults point at a specific machine's ramdisk layout (`/mnt/ramdisk/fmp-websocket/target/...`) — override on any other machine.

---

## 14. `[price_capture]` — Live Ingestion Moving Averages

```toml
[price_capture]
ma_windows = [20, 50, 200]
```
One key: `ma_windows`, a fixed 3-element array of moving-average window lengths (bars) computed by `apps/price-capture`. Code default `[20, 50, 200]`.

---

## 15. `[data]` — Historical Data Loading Integrity (F26) & Calendar Partitioning (F03)

```toml
[data]
dedupe_timestamps = false
min_common_days = 180
allow_partial_history = false
partition_mode = "date"
```
`dedupe_timestamps` (Boolean, code default **`false`**) governs what happens when a per-ticker candle series fails the `windows(2).all(|w| w[0].date < w[1].date)` strictly-increasing-timestamp check applied at the two points a series first becomes available: `alpha_orchestrator::db::data::load_historical_data` (a fresh `QuestDB` query) and `alpha_worker::cache::DataCache::load_from_disk` (a worker's on-disk cache). Both call the same pure helper, `alpha_core::enforce_strictly_increasing_timestamps`.

- **`false` (default, fail closed)**: the first duplicate or out-of-order timestamp found for any ticker aborts the load with an error naming the ticker and the offending timestamp. A duplicate timestamp should never occur in a clean dataset, so failing loudly is the safer default.
- **`true`**: for each run of candles sharing a timestamp, only the LAST one is kept (the earlier ones are dropped) and the number dropped is logged via `tracing::warn!`. A genuine chronological inversion (candles truly out of order, not merely repeated) is still rejected even with this set — deduplication only repairs *repeats*, not disorder.

**Why this exists**: `alpha_gp::evaluator::StatefulGpEvaluator::evaluate`'s per-bar memo (`manual/04`) is keyed on `snapshot.timestamp` alone — a call whose timestamp matches the previous call's short-circuits to the previous cached boolean instead of recomputing. That is correct only when timestamps are unique: if the underlying series ever contained two *different* candles stamped with the same timestamp, the memo would silently reuse the first candle's result for the second one's inputs instead of recomputing — a plausible-looking signal computed from stale data, not a crash. This config key is what makes that memo's uniqueness assumption an enforced invariant rather than an unchecked one.

### F03: Calendar (Date-Based) Partition Planning — see `manual/09` §0

- **`partition_mode`** (String, code default **`"date"`**): selects how `alpha_orchestrator::run` and its GA/ML/holdout callers carve a multi-ticker dataset into GA/ML-train/ML-sweep/Validation-Test/Holdout bands. `Config::validate` rejects any value other than the two below.
  - **`"date"` (default)**: `alpha_core::plan_partitions` computes one shared calendar boundary set (`PartitionPlan`) from every ticker's own WARM-UP-COMPLETE date (its `WARMUP_PERIOD`-th candle date, NOT its first candle date) and last date, so every ticker's slice of a given band covers the SAME wall-clock window AND every ticker used to build the plan already has a full warmup window behind `common_start` by construction — the F03 fix (`manual/09` §0). `alpha_orchestrator::run::build_partition_plan` is what derives these per-ticker dates from `historical_data` and calls `plan_partitions`; a ticker with `<= WARMUP_PERIOD` candles cannot produce one at all and is rejected by name (or excluded under `allow_partial_history`, below) before a plan is even built.
  - **`"index"`**: the pre-F03 behaviour, retained per convention 9 (never remove an old algorithm outright — keep it selectable so an already-completed run stays reproducible under the settings it actually ran with). `calculate_data_splits` derives one absolute candle-index cutoff from the SHORTEST configured ticker and applies that same raw index to every ticker regardless of its own calendar.
  - Persisted verbatim alongside the resolved plan in `ga_run_metadata` (`partition_mode`, plus six `partition_*` boundary timestamp columns) so a resumed run (`resume-db`) reloads the same mode — and, when it was `"date"`, the exact same boundaries — it started with, and so dashboards/sweeps can filter runs by which algorithm they used.
- **`min_common_days`** (Integer, code default **`180`**): minimum number of common calendar days every configured ticker's history must jointly cover (`common_start..common_end`, i.e. `max(warmup-complete dates)..min(last dates)`) before a `"date"`-mode partition plan is trusted. `Config::validate` rejects `0` (`plan_partitions` requires a positive minimum). Guards against a plan whose bands would collapse to a sliver too short for a meaningful backtest — or, in the degenerate case, whose tickers' histories don't overlap in calendar time at all — failing loudly at plan-building time instead of silently proceeding on garbage data. Only consulted in `"date"` mode.
- **`allow_partial_history`** (Boolean, code default **`false`**): when `false`, either (a) a ticker with `<= WARMUP_PERIOD` candles, which can never complete warmup at all, or (b) a ticker whose history does not fully cover `[common_start - WARMUP_PERIOD candles, common_end]` (checked defensively by `validate_partition_coverage` — see `manual/09` §0 for why that check still exists once `common_start` is warmup-complete-date-derived) makes the whole run fail fast with an error naming every such ticker. Set to `true` to instead exclude those tickers from cross-asset scoring only, logged at `warn` rather than silently narrowing the run with no operator visibility. Only consulted in `"date"` mode.

---

## 16. `[pipeline]` — Execute-Stage Engine Selection (F06)

```toml
[pipeline]
execute_mode = "ensemble"
```

One key. `execute_mode` (String, code default **`"ensemble"`**) selects which engine `run_post_ga_pipeline`'s STAGE 3b ("EXECUTE") uses to run real trades against real data, after GA discovery and ML-threshold refinement have already picked a champion. `Config::validate` rejects any value other than the two below.

- **`"ensemble"` (default)**: unchanged pre-F06 behaviour — the Execute stage runs the multi-strategy voting engine (`alpha_simulation::ensemble::run_ensemble_simulation`, `manual/09` §5) over the top `ensemble.ensemble_size` Pareto-front individuals. Kept as the default so every existing run/config reproduces bit-for-bit (convention 9: no behaviour change without a config switch, old path retained).
- **`"genome"`**: the Execute stage instead calls `run_simulation` — the exact engine the GA used to score every individual — directly on the single best-ranked Pareto-front individual, passing it the SAME `Option<&AdaptiveTradeModulator>` the ensemble path would otherwise have received, and logs that genome mode is active. This is the zero-approximation path: see `manual/09` §5 for the list of approximations the ensemble engine still makes (averaged vs. single-member genes, no per-member position tracking, no `risk_modulator` feedback, and more) that genome mode has none of.

This is the complete `[pipeline]` key set — one key total.

---

## 17. `[logging]` — GA Population Persistence Policy (F12)

```toml
[logging]
persist_population = "elites"
```

One key. `persist_population` (String, code default **`"elites"`**) selects which individuals `db::logging::build_population_log_batches` writes to `ga_runs`/`ga_strategies` on a logging generation, outside harvest mode. `Config::validate` rejects any value other than the three below. `ga.harvest_mode = true` continues to behave like `"all"` unconditionally, regardless of this key (`db::logging::effective_persist_population` unifies the two).

- **`"winners"`**: reproduces the pre-F12 hardcoded gate exactly — only individuals passing `quality_label` (profitable AND `max_drawdown <= simulation.max_acceptable_drawdown` AND `avg_sharpe >= simulation.min_sharpe_ratio`) are persisted. Before F12 this was the *only* behaviour outside harvest mode, and it made `ga_runs` a survivors-only view: `run::resume_from_db` (which reseeds a resumed run's islands from these tables) could only ever rebuild a population out of winners — no population diversity, every resumed island starting from the same narrow, already-converged slice of the search space.
- **`"elites"` (default, F12 fix)**: every member of NSGA-II front 0 (`MultiObjectiveIndividual::rank == 0`) on every island, plus any individual matching one of the run's current Hall-of-Fame entries for one of its HMM states, is persisted regardless of `quality_label`. `is_valid` keeps meaning `quality_label` either way, so the Grafana leaderboard filter (`grafana/overview.json`, which filters on `is_valid`) is unchanged by this default flipping. This is a **behaviour change from the pre-F12 default** (convention 9's usual "old default preserved" is satisfied one config value away: set `"winners"` to reproduce exactly what every pre-F12 config did).
- **`"all"`**: what `ga.harvest_mode = true` already does unconditionally — persist every non-catastrophic individual (gated only by the drawdown ceiling), admitted through the behavioural archive's niche gate rather than a quality label. Setting this explicitly, outside harvest mode, reproduces that permissive persistence policy without also turning on harvest mode's other effects (annealing pinned to the loose end, stagnation-triggered diversity injection, every-generation logging cadence).

Every `ga_runs`/`ga_strategies` row also carries an `island_id` column (F12) so `run::resume_from_db` can group persisted elites by the island that produced them (`db::data::fetch_top_hashes_for_generation`'s `island_id` filter parameter) and seed each island from its OWN elites, padding the remainder through `initialize_population`'s existing heuristic/random path — instead of chunking one combined, provenance-blind pool evenly across islands. Rows persisted before this column existed read back `NULL`/`None` there; `run::group_resume_rows_by_island` falls back to the pre-F12 even-chunk behaviour when every row for a resumed generation predates it, logging a warning naming the affected generation.

The effective mode is also persisted into `ga_run_metadata.persist_population_mode` by every writer in `db::mod` (`log_run_start`, `log_run_partition_plan`, `log_run_final_targets`, `log_run_completion_status`) so a dashboard or sweep can filter runs by which policy they used. `log_run_final_targets` and `log_run_completion_status` additionally carry `ga.objective.mode` into `ga_run_metadata.objective_mode` (F04, §3 above) on the same cadence.

This is the complete `[logging]` key set — one key total.

---

## 18. Blueprint `fdb2` Close-out Audit (Wave 8, `c766`)

Every `Config`/sub-config field introduced across blueprint `fdb2`'s Waves 1-7 (F01-F34) was cross-checked against this chapter's sections 1-17 as part of Wave 8 close-out (2026-09-07), reading each field's current `Default` impl and `Config::validate` rule directly from `crates/alpha-config/src/*.rs` rather than the blueprint's own subtask notes (a few, e.g. `[hmm.features]` in §5, deliberately ended up with different field names than their subtask originally specified — see that section's own note on why). Result: **no missing keys found** — every config key the remediation introduced (`data.*` §15, `ensemble.override_params`/`pipeline.execute_mode` §6/§16, `logging.persist_population` above, `ml_filter.min_threshold`/`feature_set`/`include_taken_flag` §8, `hmm.state_selection`/`[hmm.features]` §5, `ga.evaluation_tiers.selection`/`.window_anchor` §4, `ga.objective.*` §4, `ga.deflation.*` §4, `ga.sweep.min_recurrence` §4, `validation.monitor_fraction`/`.cpcv.*`/`.walk_forward.*`/`.sensitivity.*`/`.monte_carlo.*`/`.capacity_curve.enabled` §6, `simulation.friction.*`/`simulation.funding.*` §2, `nn.inference_device`/`.context_model.*`/`.dataset.*` §9) was already documented by the wave that introduced it, per each wave's own Definition-of-Done. This section is a verification stamp, not new content.

---

## 19. Shipped Config Profiles (blueprint c2d1 task P4, 2026-09-07)

The four shipped config files predated blueprints `fdb2`/`81f1` and none of them opted into most of the validation/cost-model/persistence/genome switches documented above. All four were retuned against this chapter and `Config::validate`'s cross-key rules; each got a header comment naming its use case and this date. `config.pgo.toml`/`config.bolt.toml`'s headers additionally note that `ga-runner pgo-train` (blueprint `c2d1` task P1) is now the *default* PGO/BOLT training workload — these two files are the optional real-data alternative, for when a realistic profile of the DB/ILP and real-data validation paths is specifically wanted.

A pre-existing bug surfaced while adding the shipped-config validation test below: `config.toml`'s `[hmm]` table was missing `student_t_nu`, which has no `#[serde(default)]` in `HmmConfig` and is therefore REQUIRED whenever `[hmm]` is present at all — `config.toml` could never actually load via `Config::from_file` before this fix. It now sets `student_t_nu = 3.0` (the code default), matching the other three files.

| Key | `config.toml` (research) | `config.harvest.toml` (breadth) | `config.pgo.toml` / `config.bolt.toml` (profiling) |
|---|---|---|---|
| `ga.objective.mode` | `"stationary"` — fixed Sharpe-LCB/min-trades/max-drawdown floors, the more rigorous choice | `"legacy"` (explicit) — its annealed, ratcheted floors match harvest's own loose-to-tighter schedule; `"stationary"`'s fixed thresholds would fight that | left at code default (`"legacy"`) |
| `ga.deflation.{enabled,mode}` | `true`, `"dsr"` — calibrated Deflated Sharpe Ratio gate | off (default) | off (default) |
| `ga.sweep.min_recurrence` | `2` (pinned to the code default) | `1` — the loosest legal value; `Config::validate` rejects `0`, so this is the practical "recurrence gate off" | left at code default (`2`) |
| `ga.evaluation_tiers.{selection,window_anchor}` | `"quantile"`, `"random"` | `"quantile"` (explicit); `window_anchor` left at default `"latest"` | left at code defaults |
| `ga.periods.sharing` | `"none"` (default) | `"none"` (default) | `"family"` — exercises the family-sharing code path during profiling |
| `simulation.friction.model` | `"spread_and_impact"` | `"spread_and_impact"` — harvested winners should not be constant-slippage artefacts | `"spread_and_impact"` |
| `simulation.friction.intraday_profile` | left `false` — see Deviations | n/a (not set) | `true` (with `gpu.enabled = false`, required by the hard `check_intraday_profile_gpu_conflict` guard) |
| `simulation.funding.mode` | `"real"` | n/a (not set) | n/a (not set) |
| `validation.cpcv.enabled` | `true` (`groups=6`, `embargo_bars=1440`, `max_pbo=0.5`) | off (default) — cost | `true` (`groups=3`, the minimum `Config::validate` allows) |
| `validation.walk_forward.mode` | `"reoptimize"` | `"fixed_champion"` (default) | `"reoptimize"` (`num_folds=2`, `generations=3`) |
| `validation.{sensitivity,monte_carlo,capacity_curve}` | left at existing/default settings | left at existing (already-small) settings | all turned on with small counts (`samples=5`, `num_simulations=5`, `capacity_curve.enabled=true`) so this profiling run's binary exercises every validation stage once instead of `num_champions_to_validate=0` skipping validation entirely |
| `pipeline.execute_mode` | `"genome"` — zero-approximation Execute stage, fitting the "honest" research theme | left at code default (`"ensemble"`) | left at code default (`"ensemble"`) |
| `logging.persist_population` | left at code default (`"elites"`) | `"all"` (explicit) — `harvest_mode=true` already forces this, set anyway so it survives a temporary `harvest_mode=false` debug run | left at code default (`"elites"`) |
| `data.dedupe_timestamps` | `true` | not set (default `false`) | not set (default `false`) |
| `gpu.enabled` | not set (default `true`) | not set (default `true`) | `false` — this run profiles the CPU evaluation path (the one the PGO/BOLT binaries ship instrumented for) |
| `hmm.expose_regime_confidence` | left `false` — see Deviations | left `false` (default) | left `false` — see Deviations |
| `nn.seeding_ratio` | commented out, `# dead since manual/08 S3` | commented out, `# dead since manual/08 S3` | commented out, `# dead since manual/08 S3` |

**Dead key**: `nn.seeding_ratio` is parsed into `NnConfig` but never read anywhere in the neural-seeding code path (§9 above) — all four shipped files set it to `0.8`; all four now have it commented out with a `# dead since ...` note rather than silently deleted, so the historical value stays visible.

**Deviations from the task's ideal profile** (two switches each config wanted are mutually exclusive under `Config::validate`):
- `config.toml`: `simulation.friction.intraday_profile = true` together with `gpu.enabled = true` is a **hard error** (`alpha_orchestrator::run::check_intraday_profile_gpu_conflict`, §2 above), not merely a `Config::validate` rejection. `config.toml` has no `[gpu]` section, so `gpu.enabled` defaults to `true` and this host may evaluate tiers on the GPU. Keeping GPU acceleration available for the primary research config was judged more valuable than the flat-vs-time-of-day ADV refinement, so `intraday_profile` stays at its `false` default (commented, with the reasoning inline) while `spread_and_impact` and `funding.mode = "real"` are still applied.
- `config.toml` / `config.pgo.toml` / `config.bolt.toml`: `hmm.expose_regime_confidence = true` requires BOTH `gpu.enabled = false` AND no `[distributed]` section (M1/M1.4, §5 above). `config.toml` keeps `gpu.enabled` at its default `true` (see above), so `expose_regime_confidence` stays `false` there. `config.pgo.toml`/`config.bolt.toml` do set `gpu.enabled = false`, satisfying the GPU half — but both files' `[distributed]` sections were left untouched (out of scope for this task; `auth_token` in that section is a live credential), so the distributed half of the guard still blocks `expose_regime_confidence = true`. It stays `false` in both, commented with the reasoning inline.
- `config.pgo.toml` / `config.bolt.toml`: `nn.use_neural_seeding` stays `false` — both files' `nn.model_path` (`models/pgo_dummy_model.safetensors`) is not a file that actually exists in `models/`, so turning seeding on would fail to load a model at runtime rather than exercise the seeding path cleanly.
