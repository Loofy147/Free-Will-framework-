# LATENCY BENCHMARK REPORT

## Hardware Configuration

- **CPU:** x86_64
- **Cores:** 4 physical, 4 logical
- **Frequency:** 2100 MHz
- **RAM:** 9.0 GB
- **Python:** 3.12.3
- **Date:** 2026-02-02T19:54:11.022374

## Benchmark Results

| Component | Operation | Claimed (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Verdict |
|-----------|-----------|--------------|----------|----------|----------|---------|
| VolitionModule | MCTS_depth_50 | 50.00 | 0.02 | 0.03 | 0.05 | ✓ Within claimed latency |
| VolitionModule | cached_policy | 5.00 | 0.05 | 0.07 | 0.08 | ✓ Within claimed latency |
| MetaCognitiveCritic | veto_check | 10.00 | 0.00 | 0.00 | 0.00 | ✓ Within claimed latency |
| CounterfactualSimulator | background_planning | 100.00 | 0.39 | 0.51 | 0.76 | ✓ Within claimed latency |
| FWICalculator | full_computation | 3000.00 | 0.29 | 0.33 | 0.37 | ✓ Within claimed latency |

## Summary

- **Total Tests:** 5
- **Passed:** 5 (100.0%)
- **Failed:** 0

## Bottleneck Analysis

**Slowest Component:** CounterfactualSimulator.background_planning
- p95 Latency: 0.51 ms
- Claimed: 100.00 ms

## Recommendations


## Raw Data

Full JSON results available in `latency_benchmark_results.json`

## Methodology

- **Warmup:** 1,000 iterations
- **Measurement:** 10,000 iterations
- **Outlier Removal:** Top/bottom 1% excluded
- **Confidence:** 95% confidence intervals calculated
- **Metric:** p95 latency used for pass/fail (more realistic than mean)
