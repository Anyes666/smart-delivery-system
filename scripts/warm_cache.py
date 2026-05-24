"""
课前预热缓存 — 提前跑一遍所有数据文件, 把高德 API 结果写入 .cache/amap/

上课前 10 分钟运行一次, 演示时所有距离矩阵秒出 (全部命中磁盘缓存)。

用法:
    python scripts/warm_cache.py                  # 预热所有数据文件
    python scripts/warm_cache.py -d shanghai_20_houses.csv  # 只预热指定文件
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.map.amap_network import AmapRoadNetwork
from src.utils.data_processing import load_points_csv, get_coordinates


def warm_one_file(amap, csv_path: str) -> bool:
    """预热单个 CSV 文件的距离矩阵."""
    print(f"\n  {csv_path} ", end="", flush=True)
    try:
        filepath = str(PROJECT_ROOT / "data" / csv_path) if "/" not in csv_path and "\\" not in csv_path else csv_path
        points = load_points_csv(csv_path)
        coords = get_coordinates(points)
        n = len(coords)
        print(f"({n} 点, {n * (n - 1) // 2} 对)... ", end="", flush=True)

        t0 = time.time()
        # compute_distance_matrix 会自动读写缓存
        amap.compute_distance_matrix(coords)
        elapsed = time.time() - t0

        stats = amap.cache.stats()
        print(f"OK ({elapsed:.1f}s)  内存命中:{stats['memory_hits']} "
              f"磁盘命中:{stats['disk_hits']} API调用:{stats['misses']}")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="预热 Amap API 缓存")
    parser.add_argument("-d", "--data", type=str, default=None, help="指定数据文件")
    args = parser.parse_args()

    amap_cfg = settings.AMAP_CONFIG
    api_key = amap_cfg.get("api_key", "")
    if not api_key or api_key == "YOUR_AMAP_KEY":
        print("[ERROR] 请先在 config/settings.py 中配置 AMAP_CONFIG['api_key']")
        return

    amap = AmapRoadNetwork(
        api_key=api_key,
        cache_dir=amap_cfg.get("cache_dir", ".cache/amap"),
        strategy=amap_cfg.get("strategy", 0),
        rate_limit_interval=amap_cfg.get("rate_limit_interval", 0.35),
    )
    print(f"高德 API 就绪 (rate_limit={amap.rate_limit_interval}s, "
          f"strategy={amap.strategy})")

    if args.data:
        csv_files = [args.data]
    else:
        data_dir = PROJECT_ROOT / "data"
        csv_files = sorted(f.name for f in data_dir.glob("*.csv"))
        if not csv_files:
            print("[ERROR] data/ 目录下无 CSV 文件")
            return

    print(f"\n预热 {len(csv_files)} 个数据文件...")
    print("=" * 55)

    t_start = time.time()
    ok = 0
    for f in csv_files:
        if warm_one_file(amap, f):
            ok += 1

    elapsed = time.time() - t_start
    print(f"\n{'=' * 55}")
    print(f"完成! {ok}/{len(csv_files)} 成功, 总耗时 {elapsed:.1f}s")
    print(f"缓存路径: {PROJECT_ROOT / '.cache' / 'amap'}")

    cache_files = list((PROJECT_ROOT / ".cache" / "amap").glob("*.pkl"))
    print(f"缓存文件数: {len(cache_files)}")


if __name__ == "__main__":
    main()
