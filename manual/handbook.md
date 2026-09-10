# The Alpha Suite Handbook: The Definitive Guide to Autonomous Alpha Evolution

```
   █████╗ ██╗     ██████╗ ██╗  ██╗ █████╗     ███████╗██╗   ██╗██╗████████╗███████╗
  ██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗    ██╔════╝██║   ██║██║╚══██╔══╝██╔════╝
  ███████║██║     ██████╔╝███████║███████║    ███████╗██║   ██║██║   ██║   █████╗  
  ██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║    ╚════██║██║   ██║██║   ██║   ██╔══╝  
  ██║  ██║███████╗██║     ██║  ██║██║  ██║    ███████║╚██████╔╝██║   ██║   ███████╗
  ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝    ╚══════╝ ╚═════╝ ╚═╝   ╚═╝   ╚══════╝
                     The Autonomous Quantitative Research Engine
```

Welcome to the **Alpha Suite Handbook** — the comprehensive, authoritative guide to the architecture, algorithms, configuration, mathematics, and operational procedures of the Alpha Suite research platform.

Whether you are a **bloody beginner** exploring quantitative finance and genetic algorithms for the first time, a **quantitative researcher** tuning multi-objective evolutionary parameters, or a **systems developer** optimizing SIMD kernels and distributed network transports, this handbook is your complete operating manual.

---

## 1. Quickstart Guide: From Zero to Running Strategies in 5 Minutes

For new users and beginners, here is the fastest way to get up and running:

```mermaid
flowchart LR
    Step1["1. Configure<br/>config.toml"] --> Step2["2. Launch Master<br/>ga-runner run"]
    Step2 --> Step3["3. Scale Compute<br/>alpha-worker"]
    Step3 --> Step4["4. Monitor Run<br/>TUI / Grafana"]
    Step4 --> Step5["5. Harvest Alpha<br/>Pareto Front"]
```

### Step 1: Verify Configuration
Ensure your `config.toml` has valid database credentials and market ticker selections:
```toml
tickers = ["BTC", "ETH", "SOL", "TURBO", "BONK"]
max_threads = 16
questdb_pg_uri = "postgresql://user:password@127.0.0.1:8812/qdb"
```

### Step 2: Start the GA Runner Master
```bash
# Launch the full research pipeline
cargo run --release --bin ga-runner -- --config config.toml run
```

### Step 3: Connect Remote Workers (Optional Scale-Out)
If you have additional machines or cloud instances, spin up remote compute workers:
```bash
export HIVE_AUTH_TOKEN="alpha-secret-key-2026"
cargo run --release --bin alpha-worker -- --master 127.0.0.1:50051 --threads 32
```

### Step 4: Monitor Evolution
Watch the interactive terminal interface as:
1. Historical 1-minute OHLCV candles are ingested.
2. The HD-HMM regime discovery engine trains and classifies market states.
3. NSGA-II islands evolve trading logic across generations.
4. Completed runs automatically execute Walk-Forward, Monte Carlo, and Sensitivity validation.

---

## 2. Master Table of Contents

This handbook is organized into 11 deep, focused chapters. Click any chapter link below to explore the detailed technical documentation:

### [Chapter 1: End-to-End System Architecture](01_system_architecture.md)
- High-level overview of the Alpha Suite ecosystem.
- The 5-stage research pipeline (Regime Discovery $\to$ Evolutionary GA/GP $\to$ ML Filter $\to$ Ensemble Voting $\to$ Validation).
- Crate map, responsibilities, and application roles.
- Non-negotiable system invariants (Warmup period, RNG seed determinism, memory models).

### [Chapter 2: Market Physics & Streaming Indicator Engine](02_market_physics_indicators.md)
- Zero-allocation `IndicatorBus` and `SlotRegistry` architecture.
- Multi-scale timeframe aggregation (1m base, Agg1, Agg2).
- Continuous streaming data structures (`SortedWindow`, `SlidingWindowMinMax`).
- Comprehensive indicator catalog: Volatility, Momentum, Market Character Vectors (MCV), and Physical Kinematics/Thermodynamics.

### [Chapter 3: High-Definition Regime Discovery (HD-HMM)](03_regime_discovery_hmm.md)
- Unsupervised market state classification.
- Continuous multivariate Student-t fat-tailed emissions ($\nu = 3.0$).
- Log-space Baum-Welch Expectation-Maximization with Log-Sum-Exp.
- SIMD hardware acceleration (AVX-512 / AVX-2) and cross-hardware reproducibility.
- Ghost-state elimination, entropy gates, cross-ticker canonical state alignment, and Viterbi decoding.

### [Chapter 4: Genetic Programming Engine & Bytecode VM](04_genetic_programming.md)
- Symbolic logic synthesis: `SignalExpression` (Boolean rules) and `FloatExpression` (Mathematical transforms).
- The `StatefulGpEvaluator` stack-based Bytecode Virtual Machine interpreter.
- Zero-allocation `FixedStack<T, 30>` with provable depth bounds.
- Protected scale-free arithmetic and symmetric comparison clamping.
- Subconscious state tracking for cold-start regime readiness.
- Genetic operators (Ramped Half-and-Half, subtree crossover, mutation, parsimony anti-bloat regularization).

### [Chapter 5: Multi-Objective Genetic Algorithm & Island Model](05_genetic_algorithm_nsga2.md)
- NSGA-II Multi-Objective Optimization: Robustness vs $R^2$ Equity Linearity Consistency.
- Parallel fast non-dominated sorting and scale-free crowding distance calculation.
- Real-valued operators: Simulated Binary Crossover (SBX) and Polynomial Mutation.
- Distributed Island model, speciation, and ring migration.
- Adaptive annealing difficulty ratchets and stagnation freezing.
- State-Locking structural memory, local Hall of Fame (HoF) splicing, and Behavioral Niche Harvesting.

### [Chapter 6: Economic Simulation, Friction & Multi-Objective Fitness Evaluation](06_economic_simulation_fitness.md)
- Independent symbol account simulation loop.
- Volatility-based position sizing and margin management.
- Realistic economic friction: Taker fees on Total Notional Value and slippage.
- Exchange liquidation mechanics and margin loss modeling.
- Austrian capital gains tax simulation (KESt 27.5% with dynamic loss-pot).
- High-speed Two-Tier cascading evaluation (30-day fast slice $\to$ 180-day full horizon).
- The 7-stage mathematical fitness pipeline and institutional performance ratios.

### [Chapter 7: Distributed Hive Mind & Transport Architecture](07_distributed_hive_mind.md)
- Centralized master and remote worker cluster topology.
- High-throughput zero-copy `rkyv` wire framing over async TLS (`SmolTransport`).
- Fail-closed shared secret authentication and dataset hash pinning.
- Work batching, worker heartbeats, and dynamic work stealing (`steal_eligible_after_secs`).
- Fault tolerance, local evaluation fallback, and session checkpoint persistence (`.bin.lz4`).

### [Chapter 8: Machine Learning Filter & Neural Architect](08_ml_filter_neural_architect.md)
- LightGBM Adaptive Trade Modulator (ATM): Context feature extraction and trade win probability prediction.
- Dynamic risk modulation and false-breakout vetoing.
- The Alpha Architect Transformer: Tokenized strategy grammar and autoregressive AST synthesis.
- Neural Seeding: Accelerating GA convergence from learned strategy distributions.

### [Chapter 9: Robustness Validation, Ensemble Voting & Live Execution](09_validation_ensemble_execution.md)
- Automated 3-tier post-GA validation suite.
- K-Fold Walk-Forward Analysis (WFA) and Walk-Forward Efficiency (WFE) ratios.
- Monte Carlo synthetic trade reshuffling, drawdown percentiles, and ruin probability estimation.
- Parameter Sensitivity analysis and parameter cliff detection.
- Ensemble Quorum Voting Engine and correlation de-duplication.
- Live trading daemon (`apps/live-trader`) for real-time exchange execution.

### [Chapter 10: Exhaustive Master Configuration Reference](10_configuration_reference.md)
- Complete line-by-line documentation of every key and section in `config.toml` and `config.harvest.toml`.
- Data types, valid parameter ranges, default values, and practical tuning advice.
- Comprehensive coverage of Simulation, GA, Islands, Annealing, HMM, ML Filter, Ensemble, Validation, NN, and Distributed settings.

### [Chapter 11: Developer & Operations Runbook](11_developer_operations_runbook.md)
- Development workflow, zero-warning pedantic clippy policy, and test suite execution.
- Complete CLI reference for `ga-runner` (run, resume, resume-db, post-process, forensic).
- Operating remote workers with `alpha-worker`.
- QuestDB database setup, ILP streaming, and automatic schema migrations.
- Performance optimization with PGO and BOLT.
- Common engineering traps and debugging guide (Lifetimes, Clone reallocations, rkyv layouts, logging deadlocks).

---

## 3. Core System Glossary

For quick reference, here are the key concepts and terms used throughout the suite:

| Term | Definition |
|---|---|
| **Alpha** | An edge or predictive trading signal that generates excess return above market benchmarks. |
| **AST** | Abstract Syntax Tree: The hierarchical tree structure representing an evolved mathematical or Boolean trading rule. |
| **Baum-Welch** | An Expectation-Maximization (EM) algorithm used to train Hidden Markov Model parameters from historical market observations. |
| **Bytecode VM** | A high-speed stack-based interpreter that evaluates compiled linear instruction arrays (`Vec<Op>`) without recursive AST traversal. |
| **Crowding Distance** | A metric in NSGA-II measuring the density of solutions around an individual on the Pareto front to promote genetic diversity. |
| **Ensemble Quorum** | A consensus mechanism where multiple champion strategies vote independently, requiring a minimum agreeing threshold to enter trades. |
| **HD-HMM** | High-Definition Hidden Markov Model: A regime discovery model using continuous Student-t emissions to identify discrete market states. |
| **Hive Mind** | The distributed compute architecture connecting remote compute workers (`alpha-worker`) to the master (`ga-runner`) over TLS. |
| **ILP** | InfluxDB Line Protocol: High-throughput TCP protocol used to stream ticks and trades into QuestDB. |
| **MCV** | Market Character Vector: A multi-dimensional feature vector describing trend intensity, volatility state, directional efficiency, and velocity. |
| **NSGA-II** | Non-Dominated Sorting Genetic Algorithm II: A gold-standard multi-objective evolutionary algorithm. |
| **Pareto Front** | The set of optimal non-dominated solutions where no objective can be improved without degrading another objective. |
| **Parsimony Pressure** | An evolutionary penalty applied to AST node counts to prevent code bloat and encourage compact, interpretable rules. |
| **Rkyv** | A zero-copy serialization framework in Rust that allows reading network buffers without heap deserialization overhead. |
| **SBX** | Simulated Binary Crossover: A real-valued crossover operator simulating binary recombination on continuous parameters. |
| **Smol** | A minimalist, high-performance async runtime in Rust. |
| **Viterbi** | A dynamic programming algorithm that decodes the single most probable sequence of hidden HMM market regimes. |
| **WFA** | Walk-Forward Analysis: A multi-fold out-of-sample validation technique simulating real-world algorithmic trading lifecycles. |

---

## 4. Key Architectural Constants Quick-Reference

| Constant | Value | Location | Architectural Purpose |
|---|---|---|---|
| `WARMUP_PERIOD` | `2000` candles | `alpha-core` | Warmup buffer excluded from evaluation to ensure slow indicators stabilize. |
| `MAX_GP_TREE_DEPTH` | `30` | `alpha-models` | Hard depth cap for GP signal trees, bounding `FixedStack` array sizing. |
| `MAX_FRAME_LEN` | `1 GiB` | `alpha-proto` | Maximum allowed TLS network frame size. |
| `MAX_DECOMPRESSED_LZ4_LEN` | `8 GiB` | `alpha-core` | Safety ceiling for decompressed LZ4 network payloads. |
| `SESSION_SAVE_INTERVAL` | `5` generations | `alpha-orchestrator`| Generational frequency for saving compressed session checkpoints. |
| `IN_FLIGHT_TIMEOUT` | `300` seconds | `alpha-orchestrator`| Maximum timeout before an unresponsive worker batch is declared lost. |
| `CALENDAR_DAYS_PER_YEAR` | `365.0` | `alpha-core` | Day count used to annualize crypto returns (markets trade 24/7/365). |
| `SQRT_DAYS_PER_YEAR` | `19.10497...` | `alpha-core` | Precomputed $\sqrt{365}$ for annualizing Sharpe and Sortino ratios. |

---
*Alpha Suite Documentation — Maintained by Synapse & The Alpha Architect Team.*
