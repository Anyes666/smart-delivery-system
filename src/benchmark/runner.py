"""
对比实验运行器 — 自动运行所有 Baseline + 本系统, 输出对比报告.

用法:
    python -m src.benchmark.runner --data my_points.csv --scale 20
    python -m src.benchmark.runner --data shanghai_20_houses.csv --runs 5 --output-csv results.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.benchmark.baselines import (
    BASELINE_REGISTRY, BenchmarkResult,
)


class BenchmarkRunner:
    """对比实验运行器 — 运行所有基线并在不同规模上对比."""

    def __init__(self, n_runs: int = 5, output_csv: str = None):
        self.n_runs = n_runs
        self.output_csv = output_csv
        self.results: Dict[str, List[BenchmarkResult]] = {}

    def run(
        self, data_file: str, num_vehicles: int = 5,
        enabled_baselines: List[str] = None,
    ) -> Dict[str, List[BenchmarkResult]]:
        """运行所有已启用的 Baseline."""
        from src.utils.data_processing import load_all_data
        from src.ml.travel_time_predictor import detect_city_from_points

        print("=" * 70)
        print(f"  对比实验运行器 — {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"  数据: {data_file} | 车辆: {num_vehicles} | 每组运行: {self.n_runs} 次")
        print("=" * 70)

        # 兼容两种路径: "shanghai_20_houses.csv" 或 "data/shanghai_20_houses.csv"
        if not Path(data_file).exists() and (Path("data") / data_file).exists():
            data_file = str(Path("data") / data_file)
        data = load_all_data(data_file)
        coords = data["coords"]
        demands = data["demands"]
        time_windows = data["time_windows"]
        sim_time = getattr(settings, 'DEFAULT_SIMULATION_TIME', 36000)
        city = detect_city_from_points(coords)

        print(f"\n  规模: {len(coords)} 点 | 总需求: {sum(demands)} | 城市: {city}")
        print(f"  模拟时间: {sim_time // 3600:02d}:{(sim_time % 3600) // 60:02d}")

        # 初始化高德网络 (供 Baseline 1/2 使用)
        amap = None
        try:
            from src.map.amap_network import AmapRoadNetwork
            ak = settings.AMAP_CONFIG.get('api_key', '')
            if ak and ak != 'YOUR_AMAP_KEY':
                amap = AmapRoadNetwork(
                    api_key=ak,
                    cache_dir='.cache/amap',
                    strategy=settings.AMAP_CONFIG.get('strategy', 0),
                )
                print(f"  Amap: 已连接 (cache hits={amap.cache.stats().get('memory_hits', 0)})")
        except Exception as e:
            print(f"  Amap: 未连接 ({e})")

        # 获取距离矩阵 (通过高德或欧氏)
        from src.algorithms.shortest_path import ShortestPathCalculator
        sp_calc = ShortestPathCalculator(coords, road_network=amap)
        dist_mtx = sp_calc.distance_matrix
        print(f"  距离矩阵: {dist_mtx.shape}, 已就绪")

        # 确定要运行的基线
        if enabled_baselines is None:
            enabled_baselines = list(BASELINE_REGISTRY.keys())

        all_results: Dict[str, List[BenchmarkResult]] = {}

        # ═════════════════════════════════════
        # 运行本系统 (作为对比基准)
        # ═════════════════════════════════════
        print(f"\n{'─' * 50}")
        print(f"  [0/6] 本系统 (高德 + LightGBM + 自适应预测)")
        print(f"{'─' * 50}")
        our_results = []
        for run_i in range(self.n_runs):
            print(f"    Run {run_i + 1}/{self.n_runs}...", end=" ", flush=True)
            t0 = time.time()
            try:
                from main import DeliverySystem
                system = DeliverySystem(use_road_network=(amap is not None))
                system.points = coords
                system.demands = demands
                system.time_windows = time_windows
                system.n_points = len(coords)
                system.distance_matrix = dist_mtx
                success = system.calculate_optimal_routes(
                    num_vehicles=num_vehicles, algorithm='ortools',
                )
                elapsed = time.time() - t0
                if success and hasattr(system, '_vrp_result') and system._vrp_result:
                    r = system._vrp_result
                    br = BenchmarkResult(
                        name="0.本系统(高德+LGB+自适应)",
                        total_distance_km=r.total_distance_km,
                        total_time_seconds=0,
                        vehicles_used=r.num_active,
                        vehicle_utilization_pct=r.num_active / max(1, num_vehicles) * 100,
                        overflow_count=len(r.overflow_nodes),
                        solve_time_seconds=elapsed,
                    )
                    our_results.append(br)
                    print(f"OK dist={r.total_distance_km:.1f}km vehicles={r.num_active}")
                else:
                    print("FAILED")
            except Exception as e:
                print(f"ERROR: {e}")

        if our_results:
            # 计算统计量
            dists = [r.total_distance_km for r in our_results]
            times = [r.total_time_seconds for r in our_results]
            ref = our_results[0]
            ref.distance_mean = np.mean(dists)
            ref.distance_std = np.std(dists) if len(dists) > 1 else 0
            ref.time_mean = np.mean(times)
            ref.time_std = np.std(times) if len(times) > 1 else 0
            all_results["our_system"] = our_results

        # ═════════════════════════════════════
        # 运行每个 Baseline
        # ═════════════════════════════════════
        for idx, (name, fn) in enumerate(BASELINE_REGISTRY.items()):
            if name not in enabled_baselines:
                continue
            print(f"\n{'─' * 50}")
            print(f"  [{idx + 1}/6] {name}")
            print(f"{'─' * 50}")

            bl_results = []
            for run_i in range(self.n_runs):
                print(f"    Run {run_i + 1}/{self.n_runs}...", end=" ", flush=True)
                t0 = time.time()
                try:
                    kwargs = dict(
                        points=coords, demands=demands, time_windows=time_windows,
                        distance_matrix=dist_mtx, num_vehicles=num_vehicles,
                    )
                    if name in ("amap_eta", "historical_mean"):
                        kwargs["amap_network"] = amap
                    if name in ("historical_mean", "random_forest", "lstm_temporal"):
                        kwargs["city"] = city
                        kwargs["sim_time_seconds"] = sim_time

                    br = fn(**kwargs)
                    bl_results.append(br)
                    print(f"OK dist={br.total_distance_km:.1f}km" if br.total_distance_km > 0 else "SKIP")
                except Exception as e:
                    import traceback
                    print(f"ERROR: {e}")
                    traceback.print_exc()

            if bl_results and bl_results[0].total_distance_km > 0:
                dists = [r.total_distance_km for r in bl_results]
                times = [r.total_time_seconds for r in bl_results]
                ref = bl_results[0]
                ref.distance_mean = np.mean(dists)
                ref.distance_std = np.std(dists) if len(dists) > 1 else 0
                ref.time_mean = np.mean(times)
                ref.time_std = np.std(times) if len(times) > 1 else 0
                all_results[name] = bl_results

        self.results = all_results
        self._print_summary()
        if self.output_csv:
            self._save_csv()

        return all_results

    def _print_summary(self):
        """打印汇总对比表."""
        our_ref = self.results.get("our_system", [])
        our_dist = our_ref[0].distance_mean if our_ref else 0

        print(f"\n{'=' * 70}")
        print(f"  对比实验汇总报告")
        print(f"{'=' * 70}")
        print(f"  {'方法':30s} | {'距离均值±std':16s} | {'车辆':4s} | {'求解用时':8s} | {'vs本系统':10s}")
        print(f"  {'─' * 30}─┼─{'─' * 16}─┼─{'─' * 4}─┼─{'─' * 8}─┼─{'─' * 10}")

        all_entries = []
        for key, results in self.results.items():
            if results:
                all_entries.append(results[0])

        for br in sorted(all_entries, key=lambda x: x.distance_mean if x.distance_mean > 0 else float('inf')):
            dist_str = f"{br.distance_mean:7.1f} ±{br.distance_std:4.1f}km" if br.distance_mean > 0 else "N/A"
            vs_str = ""
            if br.distance_mean > 0 and our_dist > 0 and "本系统" not in br.name:
                delta = (br.distance_mean - our_dist) / our_dist * 100
                vs_str = f"{'+'if delta>0 else ''}{delta:.1f}%"
            print(f"  {br.name:30s} | {dist_str:16s} | {br.vehicles_used:4d} | "
                  f"{br.solve_time_seconds:7.1f}s | {vs_str:10s}")

        print(f"\n  * 'vs本系统' = Baseline 相比本系统的距离差异百分比")
        print(f"  * 负值 = Baseline 距离更短 (但可能精度更低/用欧氏距离)")

    def _save_csv(self):
        """保存结果到 CSV."""
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["方法", "距离均值km", "距离标准差km", "车辆数",
                             "利用率%", "求解用时s", "溢出点", "额外信息"])
            for key, results in self.results.items():
                if results:
                    br = results[0]
                    writer.writerow([
                        br.name, f"{br.distance_mean:.1f}", f"{br.distance_std:.1f}",
                        br.vehicles_used, f"{br.vehicle_utilization_pct:.0f}",
                        f"{br.solve_time_seconds:.1f}", br.overflow_count,
                        json.dumps(br.extra, ensure_ascii=False) if br.extra else "",
                    ])
        print(f"\n  结果已保存: {self.output_csv}")


def run_all_benchmarks(data_file: str = "my_points.csv",
                       num_vehicles: int = 5, n_runs: int = 5,
                       output_csv: str = None):
    """便捷函数: 一键运行所有对比实验."""
    runner = BenchmarkRunner(n_runs=n_runs, output_csv=output_csv)
    return runner.run(data_file=data_file, num_vehicles=num_vehicles)


def main():
    parser = argparse.ArgumentParser(description="对比实验运行器")
    parser.add_argument("-d", "--data", type=str, default="my_points.csv",
                        help="数据文件")
    parser.add_argument("-v", "--vehicles", type=int, default=5,
                        help="最大车辆数")
    parser.add_argument("-r", "--runs", type=int, default=5,
                        help="每组运行次数")
    parser.add_argument("--scale", type=int, default=0,
                        help="目标规模 (20/50/100), 覆盖 --data")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="输出 CSV 文件路径")
    parser.add_argument("--baselines", type=str, default=None,
                        help="指定基线 (逗号分隔, 默认全部)")
    args = parser.parse_args()

    data_file = args.data
    if args.scale:
        scale_files = {
            20: "shanghai_20_houses.csv",
            50: "北京大范围测试.csv",
            100: "provincial_capitals.csv",
        }
        if args.scale in scale_files:
            data_file = scale_files[args.scale]

    baselines = None
    if args.baselines:
        baselines = [b.strip() for b in args.baselines.split(",")]

    runner = BenchmarkRunner(n_runs=args.runs, output_csv=args.output_csv)
    runner.run(data_file=data_file, num_vehicles=args.vehicles,
               enabled_baselines=baselines)


if __name__ == "__main__":
    main()
