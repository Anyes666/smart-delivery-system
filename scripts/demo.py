"""
一键演示脚本 — 完整流程: 数据摘要 → 模型状态 → 优化 → 可视化 → 报告

用法:
    python scripts/demo.py                          # 完整演示 (推荐)
    python scripts/demo.py -d my_points.csv         # 指定数据文件
    python scripts/demo.py --no-viz                 # 跳过可视化 (快速)
    python scripts/demo.py --compare                # 跑完整对比实验 (需 3-5 分钟)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_bar(label: str, value, unit: str = "", width: int = 50):
    if isinstance(value, (int, float)):
        bar_len = min(width, int(value * 3)) if value > 0 else 0
        bar = chr(9608) * bar_len
        print(f"  {label:20s} {str(value):>8s} {unit:5s} {bar}")
    else:
        print(f"  {label:20s} {str(value):>8s}")


def load_data_stats(data_file: str) -> dict:
    from src.utils.data_processing import load_all_data
    data = load_all_data(data_file)
    coords = data["coords"]
    demands = data["demands"]
    time_windows = data["time_windows"]
    priorities = data["priorities"]
    ids = data["ids"]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return {
        "file": data_file,
        "n_points": len(coords),
        "n_customers": len(coords) - 1,
        "total_demand": sum(demands),
        "lon_range": (min(lons), max(lons)),
        "lat_range": (min(lats), max(lats)),
        "n_time_windows": sum(1 for tw in time_windows if tw != (0, 86400)),
        "n_priorities": sum(1 for p in priorities if p > 0),
        "coords": coords,
        "demands": demands,
        "time_windows": time_windows,
        "priorities": priorities,
        "ids": ids,
    }


def run_benchmark_comparison(data_file: str, num_vehicles: int = 5, n_runs: int = 3):
    """运行真实对比实验 — 5个 Baseline + 本系统, 输出统计检验结果."""
    print_header("方法对比 (真实实验)")
    print(f"  运行 {n_runs} 次每组, 比对 5 个 Baseline + 本系统...")

    from src.benchmark.runner import BenchmarkRunner
    runner = BenchmarkRunner(n_runs=n_runs, output_csv="output/benchmark_results.csv")
    return runner.run(data_file=data_file, num_vehicles=num_vehicles)


def run_full_demo(data_file: str, num_vehicles: int = 5,
                  skip_viz: bool = False, do_compare: bool = False):
    from config import settings

    # ═══════════════════════════════════
    # 封面
    # ═══════════════════════════════════
    print_header("智能配送系统 — 完整演示")
    print(f"  时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  数据: {data_file}")
    print(f"  算法: OR-Tools | 最大车辆: {num_vehicles}")

    # ═══════════════════════════════════
    # 1. 数据摘要
    # ═══════════════════════════════════
    stats = load_data_stats(data_file)
    print_header("1. 数据摘要")
    print_bar("配送点总数", stats["n_points"], "个")
    print_bar("客户数", stats["n_customers"], "个")
    print_bar("总需求量", stats["total_demand"], "单位")
    print_bar("有时间窗", stats["n_time_windows"], "单")
    print_bar("高优先级", stats["n_priorities"], "单")
    lon_r = stats["lon_range"]
    lat_r = stats["lat_range"]
    area = (lon_r[1] - lon_r[0]) * 111.32 * (lat_r[1] - lat_r[0]) * 111.32
    print(f"  经度: {lon_r[0]:.4f} ~ {lon_r[1]:.4f}")
    print(f"  纬度: {lat_r[0]:.4f} ~ {lat_r[1]:.4f}")
    print(f"  覆盖: ~{area:.1f} km2")

    from src.ml.travel_time_predictor import detect_city_from_points
    city = detect_city_from_points(stats["coords"])
    print(f"  城市: {city}")
    print()

    # ═══════════════════════════════════
    # 2. 模型状态
    # ═══════════════════════════════════
    print_header("2. AI 模型状态")

    # ML
    try:
        from src.ml.travel_time_predictor import TravelTimePredictor
        ml = TravelTimePredictor.load_or_fallback(city=city, points=stats["coords"])
        if ml.is_trained:
            m = ml.metrics
            print(f"  [OK] LightGBM ({city})")
            print(f"       RMSE={m.get('rmse_seconds','?')}s  "
                  f"MAE={m.get('mae_seconds','?')}s  "
                  f"样本={m.get('n_samples','?')} 条")
            imp = m.get('improvement_pct', 0)
            print(f"       vs 定速提升: {imp}%")
        else:
            print(f"  [..] LightGBM ({city}) 未训练, 回退定速 40km/h")
    except Exception as e:
        print(f"  [WARN] LightGBM: {e}")

    # 在线自适应预测
    try:
        from src.adaptive.congestion_predictor import AdaptiveCongestionPredictor
        adaptive = AdaptiveCongestionPredictor.load_or_create(city=city)
        if adaptive.is_trained:
            m = adaptive.metrics
            loss_str = f"{m['running_loss']:.4f}" if m['running_loss'] else "N/A"
            print(f"  [OK] 在线自适应预测器 ({city})")
            print(f"       在线更新: {m['updates']} 次  loss={loss_str}")
        else:
            print(f"  [..] 在线自适应预测器 ({city}) 未训练")
    except Exception as e:
        print(f"  [WARN] 自适应预测: {e}")

    # 数据
    try:
        from src.ml.data_collector import TravelTimeDataCollector
        col_stats = TravelTimeDataCollector().stats()
        total = col_stats.get("total", 0)
        city_samples = col_stats.get("cities", {}).get(city, {}).get("total", 0)
        print(f"  [OK] 训练数据: 总计 {total} 条 ({city}: {city_samples} 条)")
    except Exception as e:
        print(f"  [..] 数据: {e}")

    print()

    # ═══════════════════════════════════
    # 3. 对比实验 (可选)
    # ═══════════════════════════════════
    if do_compare:
        run_benchmark_comparison(data_file, num_vehicles, n_runs=3)
        # 对比实验内部已经运行了本系统, 跳过步骤4
        print_header("总结")
        print(f"""
  项目: 基于高德真实路况的智能配送优化系统
  城市: {city} | 规模: {stats['n_customers']} 客户 {num_vehicles} 车

  对比实验已完成, 详细结果见 output/benchmark_results.csv
  重新运行: python demo.py --compare
""")
        return

    # ═══════════════════════════════════
    # 4. 优化运行
    # ═══════════════════════════════════
    print_header("3. 路线优化")
    print(f"  运行中 (高德 API + OR-Tools + ML + 自适应预测)...")

    from main import DeliverySystem
    system = DeliverySystem(use_road_network=True)

    t0 = time.time()
    success = system.run_full_pipeline(
        filepath=data_file,
        num_vehicles=num_vehicles,
        algorithm='ortools',
        output_dir='output',
    )
    elapsed = time.time() - t0

    if success and hasattr(system, '_vrp_result'):
        r = system._vrp_result
        print(f"\n  [OK] 优化完成! 耗时 {elapsed:.1f}s")
        print_header("优化结果")
        print(f"  总行驶距离:   {r.total_distance_km:.1f} km")
        print(f"  总成本:       {r.total_cost_km_eq:.1f} km-当量")
        n_active = sum(1 for u in r.vehicles_used if isinstance(u, dict) and u.get('is_active'))
        print(f"  使用车辆:     {n_active} 辆")
        print(f"  是否溢出:     {'是 (' + str(r.overflow_nodes) + '单未送)' if r.is_overflow else '否'}")
    else:
        print(f"\n  [WARN] 优化未完成 (耗时 {elapsed:.1f}s)")

    # ═══════════════════════════════════
    # 5. 可视化报告
    # ═══════════════════════════════════
    if not skip_viz:
        print_header("输出文件")
        output_files = sorted(Path("output").glob("*202*.html")) + sorted(Path("output").glob("*202*.json"))
        shown = set()
        for f in sorted(output_files, key=lambda x: x.stat().st_mtime, reverse=True):
            base = f.name.rsplit("_", 1)[0]
            if base not in shown:
                shown.add(base)
                size_kb = f.stat().st_size / 1024
                print(f"  output/{f.name}  ({size_kb:.0f} KB)")
        if not shown:
            print(f"  (无输出文件 — 使用 --no-viz 跳过了可视化)")

    # ═══════════════════════════════════
    # 6. 总结
    # ═══════════════════════════════════
    print_header("总结")
    print(f"""
  项目: 基于高德真实路况的智能配送优化系统
  城市: {city} | 规模: {stats['n_customers']} 客户 {num_vehicles} 车

  三层优化架构:
    L1 — 距离: 高德 API 真实路网 (替代欧氏直线)
    L2 — 时间: LightGBM 行程时间预测 (15维跨城市特征)
    L3 — 拥堵: 在线自适应行程时间预测 (神经网络回归 + 在线 SGD, 越跑越准)

  后续步骤:
    # 补全时段数据 (建议 3-5 天)
    python scripts/collect_all_day.py -d {data_file} --hours 8,10,12,14,16,18,20,22

    # 数据够 500+ 多条多时段样本后重训
    python scripts/train_travel_time.py --train --city {city}

    # 自适应模型自动在线学习, 无需手动干预

  竞赛适用: 物流设计大赛 | 数学建模大赛 | 交通科技大赛
""")
    print("=" * 60)
    print("  演示完成!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="智能配送系统 — 一键演示")
    parser.add_argument("-d", "--data", type=str, default=None,
                        help="数据文件 (默认自动选择)")
    parser.add_argument("-v", "--vehicles", type=int, default=5,
                        help="最大车辆数")
    parser.add_argument("--no-viz", action="store_true",
                        help="跳过可视化生成")
    parser.add_argument("--compare", action="store_true",
                        help="运行对比实验 (3-5分钟)")
    args = parser.parse_args()

    if args.data:
        data_file = args.data
    else:
        data_dir = PROJECT_ROOT / "data"
        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            print("[ERROR] data/ 目录下无 CSV 文件")
            return
        preferred = [f for f in csv_files if "my_points1" in f.name.lower()]
        if not preferred:
            preferred = [f for f in csv_files if "my_points" in f.name.lower()]
        data_file = preferred[0].name if preferred else csv_files[0].name

    run_full_demo(data_file, args.vehicles,
                  skip_viz=args.no_viz, do_compare=args.compare)


if __name__ == "__main__":
    main()
