# Chapter 13: GPU/CPU Backtest Kernel Parity Self-Check

This chapter is an operator runbook, not a design document. It covers the one command to run on the GPU box before trusting any GPU-evaluated result, what healthy output looks like, and what each failure mode most likely means.

---

## 1. Why this exists

`backtest_kernel.hip.cpp` recently had five correctness bugs fixed (a `per_state_pnl` array overflow past the old 12-state bound, a stateful-node ring-buffer addressing bug, a cross-ticker drawdown-accumulator leak, a GP `Not`-node sign error, and an operand-stack depth bound), plus compile-flag and error-surfacing changes. **None of it has ever run on a real GPU.** There is no CUDA device on the machines this code was developed on, and `cargo test` cannot enable non-default cargo features, so even the existing gpu-gated test files have only ever been type-checked, never executed.

Run the command in §2 **first**, on the actual GPU box, before running anything else GPU-related.

---

## 2. The command

```bash
alpha-worker --features gpu --gpu-parity-check
```

If you're invoking the built binary directly rather than through `cargo run`:

```bash
cargo build --release --features gpu -p alpha-worker
./target/release/alpha-worker --gpu-parity-check
```

No master, no `QuestDB`, no network connection, no config file, and no on-disk data cache are required — the check builds its own deterministic synthetic fixture in-process and exits.

**Also run this**, to execute the CPU/GPU parity test files that (like everything else GPU-related) have never actually run:

```bash
cargo test --features gpu -p alpha-simulation
```

This runs `gpu_cpu_parity_tests.rs`, `open_findings_gpu_tests.rs`, and `terminal_id_reconciliation_tests.rs` for the first time. Run it alongside `--gpu-parity-check`, not instead of it — the two exercise overlapping but not identical scenarios (that test suite uses smaller HMM state counts and single individuals; `--gpu-parity-check`'s fixture is specifically built to hit all five recent fixes, including 16 HMM states and a multi-individual batch).

A separate, always-available check (no GPU or `--features gpu` needed) is:

```bash
cargo test -p alpha-worker
```

This runs the fixture's own CPU-only sanity tests (`gpu_parity::fixture`'s `#[cfg(test)]` module) — confirming the fixture itself is well-formed (the drawdown ticker actually trips the cap, the other tickers trade enough to be meaningful, states 12-15 actually accumulate trades) independent of whether a GPU is available at all. If this fails, the fixture is broken; if it passes but `--gpu-parity-check` fails, the problem is in the kernel or the GPU glue code.

---

## 3. Healthy output

```
================================================================================
 GPU / CPU Backtest Kernel Parity Self-Check -- alpha-worker
================================================================================

[Device]
  Name:                NVIDIA A100-SXM4-80GB
  Compute capability:  8.0
  Total memory:        79.35 GiB
  Free memory:         78.10 GiB
  NOTE: build.rs targets -gencode arch=compute_80,code=sm_80 by default
        (override via CUDA_ARCH). Confirm this matches the capability above --
        a mismatch silently falls back to JIT-compiled embedded PTX instead
        of the prebuilt SASS.

[Tolerances]
  final_equity / per_state_pnl:  1.00% relative, 1.00 absolute floor (dollars)
  max_drawdown_percent:          5.00% relative, 0.10 absolute floor
                                  (percentage points) -- looser; see the
                                  caveat below
  trades / wins:                 EXACT match required (integer counts)

[Not currently compared -- neutralized in this fixture]
  - taker fees                     (fixture: 0.0%)
  - slippage                       (fixture: 0.0%)
  - min_hold_bars                  (fixture: 0)
  - ML trade filter                (fixture: no model; this binary excludes
                                     the ml_filter module entirely)
  - Agg1/Agg2 timeframe indicators (fixture: GP trees reference only *_1m
                                     terminals)

[Fixture] 3 individual(s), 16 HMM states, 4 ticker(s) (5000 candles each,
3000 post-warmup bars), ticker order: ["DRAWDOWN", "AVAX", "ETH", "SOL"]

--- Individual 0 ---
  field                          cpu              gpu       abs diff     rel diff  tolerance                    status
  final_equity              9123.456789      9123.500000       0.043211     0.000005  1.00% rel / 1.00 abs floor  PASS
  trades                          14               14              -            -    exact match required        PASS
  wins                             6                6              -            -    exact match required        PASS
  max_drawdown_percent        18.230000        18.225000       0.005000     0.000274  5.00% rel / 0.10 abs floor  PASS
  per_state_pnl[0]           -876.500000      -876.480000       0.020000     0.000023  1.00% rel / 1.00 abs floor PASS
  ...
  per_state_pnl[15]           120.330000       120.310000       0.020000     0.000166  1.00% rel / 1.00 abs floor PASS

--- Individual 1 --- ...
--- Individual 2 --- ...

================================================================================
 SUMMARY: 60 comparison(s) across 3 individual(s) -- 60 PASS, 0 FAIL
 RESULT: PASS
================================================================================
```

Exit code `0`. The exact numbers above are illustrative, not literal — they will differ on a real run because they depend on the actual GPU floating-point results.

---

## 4. Failure modes and what they most likely mean

| Symptom | Likely cause |
|---|---|
| `Failed to query the GPU device...` | No CUDA/HIP device visible, driver not loaded, or the binary wasn't built against a CUDA install with a working `nvcc`. Check `nvidia-smi` on the box first. |
| Compute capability printed does not match the card (e.g. shows `8.0` but you're on a different GPU generation) | `build.rs`'s default `-gencode arch=compute_80,code=sm_80` doesn't match your hardware. Rebuild with `CUDA_ARCH=sm_XX` set for your card, or accept the JIT-from-PTX fallback (slower first-launch, but still correct). |
| `GpuEvaluator::new failed to allocate device memory` | Out of GPU memory — check `nvidia-smi` for other processes holding VRAM, or reduce `--gpu-capacity` if invoked via the normal worker path (not applicable to `--gpu-parity-check` itself, which sizes its own small fixture). |
| `GpuEvaluator::evaluate_batch failed -- see the GPU status message above` | A kernel launch or memcpy failed. The printed message is the **device's own** `hipGetErrorString`/`cudaGetErrorString` text (not a generic message) — read it directly; a common one is an out-of-bounds access, which would indicate one of the five fixes did not actually take on this build. |
| `trades` or `wins` FAILs (exact-match field) | This is the strongest possible signal of a real kernel bug — these are integer counts, not floats, so there is no precision explanation available. Cross-reference which `per_state_pnl` indices also failed: a failure concentrated in one HMM state (especially states 12-15) points at FIX 1 (the old `per_state_pnl` overflow); a failure spread across many states in individuals 1/2 but not individual 0 points at FIX 2 (stateful-node addressing, since only individuals 1/2 have a nonzero `stateful_start_idx`). |
| `final_equity` / `per_state_pnl` FAILs beyond the stated tolerance | Check whether `trades`/`wins` also failed for the same individual. If they match exactly but the dollar amount doesn't, suspect a genuine numeric bug in the kernel's PnL math rather than an f32/f64 rounding difference (rounding alone should not produce a >1% gap over this fixture's scale). |
| Every failing field is on **individual 0** specifically (the one carrying the `DRAWDOWN` ticker), and individuals 1/2 pass cleanly | Most likely the documented mark-to-market-vs-realized drawdown-tracking difference between the CPU engine and the GPU kernel (see `apps/alpha-worker/src/gpu_parity/mod.rs`'s module doc comment, "A caveat this tool cannot fully neutralize") — not a kernel regression. The tool prints this exact note when it detects this pattern. |
| `max_drawdown_percent` alone fails (trades/wins/final_equity all pass) | Same caveat as above — the two engines can legitimately disagree slightly on the exact drawdown percentage without disagreeing on how many trades happened or how much money was made, because of when each engine updates its drawdown accumulator. |
| The whole report prints PASS, but you changed the kernel afterward | Re-run — the report is generated fresh every invocation from the same deterministic fixture (fixed RNG seeds), so a prior PASS says nothing about a build made after it. |

---

## 5. What a PASS does *not* prove

A clean PASS means: for this fixture's specific configuration (zero taker fees, zero slippage, `min_hold_bars = 0`, no ML filter, GP trees restricted to 1-minute terminals), the GPU kernel's numbers agree with the CPU engine's within the stated tolerances. It does **not** prove:

- Parity when taker fees, slippage, `min_hold_bars`, or the ML filter are non-zero — the GPU kernel does not model any of these at all yet (separate, already-queued work).
- Parity for strategies whose GP trees reference Agg1/Agg2 (aggregated-timeframe) terminals — `build_snapshot_matrix` never computes those on GPU; they silently read a registry default instead of a real value.
- Correctness under concurrent multi-stream GPU usage, or on a card with a different compute capability than what `build.rs` targeted (see §4).

See `apps/alpha-worker/src/gpu_parity/mod.rs`'s module doc comment for the authoritative, most up-to-date version of this list — it is duplicated in the tool's own printed output specifically so a narrow PASS is never mistaken for full parity.
