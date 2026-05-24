"""
在线自适应行程时间预测器 — 离线批量训练脚本

用历史 jsonl 数据初始化/更新自适应预测模型。

用法:
    python src/adaptive/train_congestion.py --city Beijing     # 批量训练
    python src/adaptive/train_congestion.py --stats             # 查看模型状态
    python src/adaptive/train_congestion.py --evaluate          # 评估预测准确度
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive.congestion_predictor import AdaptiveCongestionPredictor, FALLBACK_SPEED_KMH
from src.ml.data_collector import TravelTimeDataCollector


def cmd_train(city: str, epochs: int = 15):
    samples = TravelTimeDataCollector.load_all_samples(city)
    if not samples:
        print(f"[ERROR] {city} 无数据")
        return

    # 过滤: 只用真实时间样本
    real_samples = [s for s in samples if s.get("real_time", True)]
    if len(real_samples) < 50:
        print(f"[ERROR] {city} 真实样本不足: {len(real_samples)} < 50")
        return

    print("=" * 60)
    print(f"  在线自适应预测器批量训练 — {city}")
    print("=" * 60)
    print(f"  样本: {len(real_samples)} 条 (已过滤非真实时段)")

    # 用样本中心作为城市中心
    all_lons = [s["origin_lon"] for s in real_samples] + [s["dest_lon"] for s in real_samples]
    all_lats = [s["origin_lat"] for s in real_samples] + [s["dest_lat"] for s in real_samples]
    center_lon = sum(all_lons) / len(all_lons)
    center_lat = sum(all_lats) / len(all_lats)

    # 显示样本质量
    from collections import Counter
    hours = Counter(s["hour"] for s in real_samples)
    print(f"  时段覆盖: {len(hours)}/24 小时: {sorted(hours.keys())}")
    dists = [s["dist_km"] for s in real_samples]
    print(f"  距离: {min(dists):.1f}-{max(dists):.1f} km")

    # 计算真实拥堵乘数分布
    multipliers = []
    for s in real_samples:
        baseline = s["dist_km"] / FALLBACK_SPEED_KMH * 3600
        if baseline > 0:
            multipliers.append(s["duration_seconds"] / baseline)
    print(f"  真实乘数: {min(multipliers):.2f}x - {max(multipliers):.2f}x "
          f"(中位数 {sorted(multipliers)[len(multipliers)//2]:.2f}x)")
    print()

    predictor = AdaptiveCongestionPredictor(city_center=(center_lon, center_lat))
    print("  训练中...")
    result = predictor.train_batch(real_samples, epochs=epochs)
    predictor.save(city)

    print(f"\n  [OK] 训练完成!")
    print(f"  更新次数: {result['updates']}")
    print(f"  最终 loss: {result['final_loss']:.4f}")
    print(f"  探索噪声: {result['exploration_noise']:.3f}")
    print(f"\n  下次运行 run_system.py 将自动加载此模型")


def cmd_stats(city: str = None):
    from src.ml.travel_time_predictor import _city_slug
    cache_dir = Path(__file__).parent.parent.parent / ".cache" / "adaptive"
    if not cache_dir.exists():
        print("无自适应预测模型文件")
        return

    models = list(cache_dir.glob("congestion_*.pkl"))
    if not models:
        print("无自适应预测模型文件")
        return

    print("=" * 60)
    print("  在线自适应预测器状态")
    print("=" * 60)
    for path in sorted(models):
        city = path.stem.replace("congestion_", "")
        try:
            pred = AdaptiveCongestionPredictor.load(city)
            m = pred.metrics
            loss_str = f"{m['running_loss']:.4f}" if m['running_loss'] else "N/A"
            print(f"  {city:15s}  updates={m['updates']:6d}  loss={loss_str}  "
                  f"noise={m['exploration_noise']:.3f}")
        except Exception as e:
            print(f"  {city:15s}  [ERROR] {e}")


def cmd_evaluate(city: str = None):
    if not city:
        stats_cache = list((Path(__file__).parent.parent.parent / ".cache" / "adaptive").glob("congestion_*.pkl"))
        if not stats_cache:
            print("[ERROR] 无模型")
            return
        city = stats_cache[0].stem.replace("congestion_", "")
        print(f"自动选择: {city}")

    try:
        pred = AdaptiveCongestionPredictor.load(city)
    except FileNotFoundError:
        print(f"[ERROR] {city} 模型不存在")
        return

    samples = TravelTimeDataCollector.load_all_samples(city)
    real_samples = [s for s in samples if s.get("real_time", True)]
    if len(real_samples) < 20:
        print(f"[ERROR] {city} 样本不足")
        return

    # 测试集 (后 20%)
    random.shuffle(real_samples)
    split = int(len(real_samples) * 0.8)
    test = real_samples[split:]

    errors = []
    fallback_errors = []
    for s in test:
        pred_mult = pred.predict(
            (s["origin_lon"], s["origin_lat"]),
            (s["dest_lon"], s["dest_lat"]),
            s["dist_km"], s["hour"], s["day_of_week"],
        )
        baseline = s["dist_km"] / FALLBACK_SPEED_KMH * 3600
        true_mult = s["duration_seconds"] / baseline if baseline > 0 else 1.0
        errors.append(abs(pred_mult - true_mult))
        fallback_errors.append(abs(1.0 - true_mult))  # 定速 = multiplier 1.0

    mae = sum(errors) / len(errors)
    fb_mae = sum(fallback_errors) / len(fallback_errors)

    print("=" * 60)
    print(f"  在线自适应预测器评估 — {city}")
    print("=" * 60)
    print(f"  测试样本: {len(test)}")
    print(f"  自适应 MAE: {mae:.4f}  (乘数误差)")
    print(f"  定速 MAE:   {fb_mae:.4f}  (乘数误差)")
    print(f"  提升:       {(1 - mae / max(fb_mae, 0.001)) * 100:.1f}%")
    print()
    print("  预测示例 (前 5 条):")
    print(f"  {'时段':>6s}  {'距离':>6s}  {'真实':>6s}  {'预测':>7s}  {'定速':>6s}")
    for s in test[:5]:
        pred_mult = pred.predict(
            (s["origin_lon"], s["origin_lat"]),
            (s["dest_lon"], s["dest_lat"]),
            s["dist_km"], s["hour"], s["day_of_week"],
        )
        baseline = s["dist_km"] / FALLBACK_SPEED_KMH * 3600
        true_mult = s["duration_seconds"] / baseline if baseline > 0 else 1.0
        print(f"  {s['hour']:02d}:00  {s['dist_km']:5.1f}km  "
              f"{true_mult:5.2f}x  {pred_mult:6.3f}x   {1.0:5.2f}x")


def main():
    parser = argparse.ArgumentParser(description="在线自适应行程时间预测器训练工具")
    parser.add_argument("--train", action="store_true", help="批量训练")
    parser.add_argument("--stats", action="store_true", help="模型状态")
    parser.add_argument("--evaluate", action="store_true", help="评估准确度")
    parser.add_argument("--city", type=str, default=None, help="城市 (默认自动检测)")
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数 (默认 15)")
    args = parser.parse_args()

    if args.train:
        city = args.city or "Beijing"
        cmd_train(city, args.epochs)
    elif args.evaluate:
        cmd_evaluate(args.city)
    elif args.stats:
        cmd_stats(args.city)
    else:
        cmd_stats()


if __name__ == "__main__":
    main()
