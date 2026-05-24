"""
Train adaptive congestion predictor from collected samples.
Usage: python scripts/train_adaptive.py --city Beijing --epochs 30
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adaptive.congestion_predictor import AdaptiveCongestionPredictor
from src.ml.data_collector import TravelTimeDataCollector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Beijing")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    print(f"Loading samples for {args.city}...")
    samples = TravelTimeDataCollector.load_all_samples(args.city)
    print(f"  Total: {len(samples)}")

    # Only use real-time samples
    real = [s for s in samples if s.get("real_time", True)]
    print(f"  Real:  {len(real)}")

    if len(real) < 100:
        print("Not enough samples to train.")
        return

    print(f"\nTraining adaptive predictor ({args.epochs} epochs)...")
    pred = AdaptiveCongestionPredictor()
    result = pred.train_batch(real, epochs=args.epochs)

    print(f"\nTraining complete:")
    print(f"  Final loss:     {result['final_loss']:.4f}")
    print(f"  Updates:        {result['updates']}")
    print(f"  Exploration:    {result['exploration_noise']:.4f}")

    pred.save(args.city)
    print(f"\nModel saved for {args.city}.")


if __name__ == "__main__":
    main()
