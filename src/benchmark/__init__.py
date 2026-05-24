"""基准测试与对比实验框架 — 对标中物联科技进步奖评审标准."""
from src.benchmark.baselines import (
    BenchmarkResult,
    baseline_amap_eta_direct,
    baseline_historical_mean,
    baseline_ortools_euclidean,
    baseline_random_forest,
    baseline_lstm_temporal,
    BASELINE_REGISTRY,
)
from src.benchmark.runner import BenchmarkRunner, run_all_benchmarks
