"""
Train adaptive congestion predictor from collected samples.
Usage: python scripts/train_adaptive.py --city Beijing --epochs 30
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adaptive.congestion_predictor import (
    AdaptiveCongestionPredictor, _compute_center,
)
from src.ml.data_collector import TravelTimeDataCollector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Beijing")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    print(f"Loading samples for {args.city}...")
    samples = TravelTimeDataCollector.load_all_samples(args.city)
    print(f"  Total: {len(samples)}")

    # 只使用真实时间样本: 模拟时间的数据 duration 是调用 API 时刻的真实路况,
    # 与模拟的 hour 不匹配, 会导致标签和特征矛盾
    real = [s for s in samples if s.get("real_time", True)]
    print(f"  Real:  {len(real)}  (simulated={len(samples)-len(real)})")

    if len(real) < 100:
        print("Not enough real-time samples to train.")
        return

    # 从样本推断城市中心 (而非硬编码北京)
    sample_coords = [(s["origin_lon"], s["origin_lat"]) for s in real]
    city_center = _compute_center(sample_coords)
    print(f"  Inferred city center: ({city_center[0]:.4f}, {city_center[1]:.4f})")

    print(f"\nTraining adaptive predictor ({args.epochs} epochs)...")
    pred = AdaptiveCongestionPredictor(city_center=city_center)
    result = pred.train_batch(real, epochs=args.epochs)

    print(f"\nTraining complete:")
    print(f"  Final loss:     {result['final_loss']:.4f}")
    print(f"  Updates:        {result['updates']}")
    print(f"  Exploration:    {result['exploration_noise']:.4f}")

    pred.save(args.city)
    print(f"\nModel saved for {args.city}.")


if __name__ == "__main__":
    main()
