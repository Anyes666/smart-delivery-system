"""
Phase 1.1 — 行程时间预测模型训练脚本 (跨城市)

用法:
    python scripts/train_travel_time.py --stats              # 查看各城市数据量
    python scripts/train_travel_time.py --train              # 训练当前数据所在城市
    python scripts/train_travel_time.py --train --city Shanghai  # 训练指定城市
    python scripts/train_travel_time.py --evaluate           # 评估当前城市模型
    python scripts/train_travel_time.py --features           # 特征重要度
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.travel_time_predictor import (
    TravelTimePredictor, detect_city_from_points, _compute_center
)
from src.ml.data_collector import TravelTimeDataCollector


def cmd_stats():
    collector = TravelTimeDataCollector()
    stats = collector.stats()
    print("=" * 60)
    print("  行程时间训练数据统计 (按城市)")
    print("=" * 60)
    cities = stats.get("cities", {})
    if not cities:
        print("  无数据。请先运行几次系统以积累样本。")
        return
    for city, s in sorted(cities.items()):
        bar = "█" * min(40, s["total"] // 10)
        print(f"  {city:20s} {s['total']:5d} 条 {bar}")
    print(f"  {'总计':20s} {stats['total']:5d} 条")
    print()
    for city, s in sorted(cities.items()):
        if s["total"] >= 500:
            print(f"  [OK] {city}: 可训练 (python train_travel_time.py --train --city {city})")
        else:
            print(f"  [..] {city}: 还需 {500 - s['total']} 条")


def cmd_train(city: str = None):
    collector = TravelTimeDataCollector()

    # 确定要训练的城市
    if city:
        city_stats = collector.stats(city).get("city_stats", {})
        total = city_stats.get("total", 0)
    else:
        stats = collector.stats()
        cities = stats.get("cities", {})
        if not cities:
            print("[ERROR] 无训练数据")
            return
        # 选样本最多的城市
        city = max(cities, key=lambda c: cities[c]["total"])
        total = cities[city]["total"]
        print(f"自动选择城市: {city} ({total} 条样本)")

    samples = TravelTimeDataCollector.load_all_samples(city)
    print("=" * 60)
    print(f"  训练 {city} 行程时间预测模型")
    print("=" * 60)

    # 数据概况 + 质量标记
    from collections import Counter
    real_samples = [s for s in samples if s.get("real_time", True)]
    fake_samples = [s for s in samples if not s.get("real_time", True)]
    hours = Counter(s["hour"] for s in samples)
    dists = [s["dist_km"] for s in samples]
    durs = [s["duration_seconds"] for s in samples]
    print(f"  总样本: {len(samples)}")
    print(f"    真实时段: {len(real_samples)} 条 (早/中/晚各跑一次的)")
    print(f"    快速采集: {len(fake_samples)} 条 (时段标签不可靠)")
    print(f"  时段覆盖: {len(hours)}/24 小时")
    print(f"  距离: {min(dists):.1f}-{max(dists):.1f} km")
    print(f"  耗时: {min(durs):.0f}-{max(durs):.0f} 秒")
    if len(fake_samples) > len(real_samples) * 5:
        print(f"  ⚠ 快速采集占比过高 ({len(fake_samples)/max(len(samples),1)*100:.0f}%)")
        print(f"    时段预测可能不准, 建议补 3-5 次真实时间采集")

    # 只保留真实时间样本训练: 模拟时间数据的 duration 是调用 API 时刻的
    # 真实路况, 与模拟的 hour 不匹配, 会导致特征与标签矛盾
    samples = real_samples
    print(f"\n  训练样本 (仅真实时间): {len(samples)} 条")

    if len(samples) < 500:
        print(f"[ERROR] 真实时间样本不足: {len(samples)} < 500")
        return

    # 用样本中心作为城市中心
    all_lons = [s["origin_lon"] for s in samples] + [s["dest_lon"] for s in samples]
    all_lats = [s["origin_lat"] for s in samples] + [s["dest_lat"] for s in samples]
    center_lon = sum(all_lons) / len(all_lons)
    center_lat = sum(all_lats) / len(all_lats)

    print(f"  城市中心: ({center_lon:.2f}, {center_lat:.2f})")
    print("\n  训练中...")

    try:
        predictor = TravelTimePredictor(city_center=(center_lon, center_lat))
        metrics = predictor.train(samples)
    except Exception as e:
        print(f"[ERROR] 训练失败: {e}")
        import traceback; traceback.print_exc()
        return

    predictor.save(city=city)
    print(f"\n  [OK] {city} 模型训练完成!")
    print(f"  RMSE:  {metrics['rmse_seconds']} 秒")
    print(f"  MAE:   {metrics['mae_seconds']} 秒")
    print(f"  MAPE:  {metrics['mape_pct']}%")
    print(f"  定速:  {metrics['fallback_rmse']} 秒")
    print(f"  提升:  {metrics['improvement_pct']}%")


def cmd_evaluate(city: str = None):
    if not city:
        stats = TravelTimeDataCollector().stats()
        cities = list(stats.get("cities", {}).keys())
        if not cities:
            print("[ERROR] 无数据, 无法确定城市")
            return
        city = cities[0]
        print(f"自动选择: {city}")

    try:
        pred = TravelTimePredictor.load(city)
    except FileNotFoundError:
        print(f"[ERROR] {city} 模型不存在")
        return

    m = pred.metrics
    print("=" * 60)
    print(f"  模型评估 — {city}")
    print("=" * 60)
    print(f"  训练样本: {m.get('n_samples', '?')}")
    print(f"  RMSE:     {m.get('rmse_seconds', '?')} 秒")
    print(f"  MAE:      {m.get('mae_seconds', '?')} 秒")
    print(f"  MAPE:     {m.get('mape_pct', '?')}%")
    print(f"  定速RMSE: {m.get('fallback_rmse', '?')} 秒")
    print(f"  提升:     {m.get('improvement_pct', '?')}%")

    print("\n  预测对比 (5km路段):")
    for h in [7, 8, 12, 17, 18, 22]:
        t_ml = pred.predict((121.47, 31.23), (121.50, 31.25), 5.0, h, 2)
        t_fb = pred.fallback_predict(5.0)
        diff = t_ml - t_fb
        tag = "高峰期" if h in (7, 8, 17, 18) else "夜间" if h == 22 else "午间"
        print(f"  {h:02d}:00 ({tag:4s})  ML={t_ml/60:.1f}分  定速={t_fb/60:.1f}分  差异={diff/60:+.1f}分")


def cmd_features(city: str = None):
    if not city:
        stats = TravelTimeDataCollector().stats()
        cities = list(stats.get("cities", {}).keys())
        if not cities:
            print("[ERROR] 无数据")
            return
        city = cities[0]

    try:
        pred = TravelTimePredictor.load(city)
    except FileNotFoundError:
        print(f"[ERROR] {city} 模型不存在")
        return

    fi = pred.feature_importance
    if not fi:
        print("特征重要度不可用")
        return

    print("=" * 60)
    print(f"  特征重要度 (gain) — {city}")
    print("=" * 60)
    total = sum(fi.values()) or 1
    for name, gain in sorted(fi.items(), key=lambda x: -x[1]):
        pct = gain / total * 100
        bar = "█" * int(pct * 2)
        print(f"  {name:22s} {pct:5.1f}% {bar}")


def main():
    parser = argparse.ArgumentParser(description="行程时间预测模型训练工具 (跨城市)")
    parser.add_argument("--stats", action="store_true", help="各城市数据统计")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--evaluate", action="store_true", help="评估模型")
    parser.add_argument("--features", action="store_true", help="特征重要度")
    parser.add_argument("--city", type=str, default=None, help="指定城市 (默认自动检测)")
    args = parser.parse_args()

    if args.stats:
        cmd_stats()
    elif args.train:
        cmd_train(args.city)
    elif args.evaluate:
        cmd_evaluate(args.city)
    elif args.features:
        cmd_features(args.city)
    else:
        cmd_stats()


if __name__ == "__main__":
    main()
