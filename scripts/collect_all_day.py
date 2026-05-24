"""
全天自动采集脚本 — 按真实时间运行, 积累不同时段的真实路况数据。

用法:
    # 启动调度器, 从现在等到下一个整点开始, 每2小时跑一次
    python scripts/collect_all_day.py -d shanghai_20_houses.csv

    # 立即跑一次当前时间, 不等
    python scripts/collect_all_day.py -d shanghai_20_houses.csv --once

    # 每小时跑一次 (更密集)
    python scripts/collect_all_day.py -d shanghai_20_houses.csv --interval 1

    # 只在白天跑 (8:00-20:00)
    python scripts/collect_all_day.py -d shanghai_20_houses.csv --hours 8,10,12,14,16,18,20

运行后挂机即可, 到点自动用当前真实时间调用高德 API,
API 返回的是此刻真实路况下的行程时间。
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT_DIR = PROJECT_ROOT / "scripts"


def run_once(data_file, vehicles, algorithm):
    """运行一次系统, 使用当前真实时间."""
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_system.py"),
        "-d", data_file,
        "-v", str(vehicles),
        "-a", algorithm,
        "--real-time",
    ]
    now = datetime.now()
    print(f"\n{'=' * 55}")
    print(f"  [{now:%H:%M:%S}] 开始采集 (真实时间)...")
    print(f"{'=' * 55}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(PROJECT_ROOT)
        )
        for line in result.stdout.split("\n"):
            if any(kw in line for kw in
                   ["路线优化完成", "车辆启用", "总行驶距离",
                    "ML数据采集器", "ML行程时间模型",
                    "OK", "ERROR", "Error", "Traceback"]):
                print(f"  {line.strip()}")
        if result.returncode != 0:
            print(f"  [FAIL] 返回码 {result.returncode}")
            for line in (result.stderr + result.stdout).split("\n")[-8:]:
                if line.strip():
                    print(f"  [ERR] {line.strip()}")
        else:
            print(f"  [OK] 完成")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  [WARN] 超时")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="全天自动采集 ML 训练数据 (真实时间)")
    parser.add_argument("-d", "--data", default="my_points1.csv",
                        help="数据文件")
    parser.add_argument("-v", "--vehicles", type=int, default=5,
                        help="最大可用车辆数")
    parser.add_argument("-a", "--algorithm", default="ortools",
                        help="算法")
    parser.add_argument("--interval", type=int, default=2,
                        help="采集间隔/小时 (默认 2)")
    parser.add_argument("--hours", type=str, default=None,
                        help="指定运行的小时, 逗号分隔 (如 8,10,12,14,16,18)")
    parser.add_argument("--once", action="store_true",
                        help="只跑一次当前时间, 不进入调度循环")
    parser.add_argument("--fast", action="store_true",
                        help="快速模式: 连续跑不同模拟时间, 采集空间特征 (时间特征需真实等待)")
    args = parser.parse_args()

    if args.fast:
        run_fast(args.data, args.vehicles, args.algorithm)
        return

    # 构建目标小时列表
    if args.hours:
        target_hours = sorted(set(int(h.strip()) for h in args.hours.split(",")))
    else:
        target_hours = list(range(0, 24, args.interval))

    print("=" * 55)
    print("  全天自动采集 (真实时间模式)")
    print(f"  数据: {args.data}  算法: {args.algorithm}")
    print(f"  目标时段: {target_hours}")
    print(f"  间隔: {args.interval}h  (共 {len(target_hours)} 轮)")
    print("=" * 55)
    print("  高德 API 将返回当前真实路况的行程时间")
    print("  建议: 早上启动, 挂机一整天")
    print("=" * 55)

    if args.once:
        run_once(args.data, args.vehicles, args.algorithm)
        print("\n查看: python scripts/train_travel_time.py --stats")
        return

    success_count = 0
    total_targets = len(target_hours)

    while True:
        now = datetime.now()
        current_hour = now.hour

        # 找下一个未过的目标小时
        next_hour = None
        for h in target_hours:
            if h > current_hour or (h == current_hour and now.minute < 55):
                next_hour = h
                break

        if next_hour is None:
            # 今天所有时段都过了, 等明天
            next_hour = target_hours[0]
            next_run = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
            next_run += timedelta(days=1)
            print(f"\n今日时段已全部完成 ({success_count}/{total_targets})")
            print(f"下一轮: 明天 {next_run:%H:%M}")
            wait = (next_run - now).total_seconds()
            hours_wait = wait / 3600
            print(f"等待 {hours_wait:.1f} 小时...")
            success_count = 0
        else:
            next_run = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
            wait = (next_run - now).total_seconds()

        if wait > 60:
            wait_min = wait / 60
            print(f"\n下次运行: {next_run:%H:%M} (等待 {wait_min:.0f} 分钟)...")
            print("Ctrl+C 可随时中断, 已采集的数据不会丢失")
            try:
                time.sleep(wait)
            except KeyboardInterrupt:
                print("\n\n用户中断. 已采集的数据已保存.")
                print(f"查看: python scripts/train_travel_time.py --stats")
                return

        # 运行
        ok = run_once(args.data, args.vehicles, args.algorithm)
        if ok:
            success_count += 1


def run_fast(data_file, vehicles, algorithm):
    """快速模式: 连续跑不同模拟时间, 采集空间特征数据。

    虽然高德返回的是当前真实路况(非历史), 但不同路段的速度差异
    (高速vs小巷, 市区vs郊区) 会被完整记录。这部分数据量越大越好。
    时间特征需要配合真实时间采集。
    """
    import random
    hours = list(range(0, 24))
    random.shuffle(hours)  # 随机顺序, 避免API限流触发

    total = len(hours)
    print("=" * 55)
    print(f"  快速采集模式 — {total} 轮 (空间特征)")
    print(f"  数据: {data_file}  算法: {algorithm}")
    print("=" * 55)
    print("  注意: 高德返回的是当前真实路况, 时段标签仅作标记")
    print("  时间特征需配合真实采集 (早/中/晚各跑一次 --once)")
    print("=" * 55)

    ok_count = 0
    for i, h in enumerate(hours):
        time_str = f"{h:02d}:00"
        # 禁用缓存以获得新鲜API数据
        # --fast 模式: 使用模拟时间, 并标记采集器为"非真实时段"
        cmd = [
            sys.executable, str(SCRIPT_DIR / "run_system.py"),
            "-d", data_file,
            "-v", str(vehicles),
            "-a", algorithm,
            "--simulation-time", time_str,
            "--simulated-collection",
        ]
        print(f"\n[{i+1}/{total}] {time_str} ...")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                cwd=str(PROJECT_ROOT)
            )
            for line in result.stdout.split("\n"):
                if any(kw in line for kw in
                       ["路线优化完成", "ML数据采集器", "ML行程时间模型",
                        "OK", "ERROR"]):
                    print(f"  {line.strip()}")
            if result.returncode == 0:
                ok_count += 1
            else:
                print(f"  [FAIL]")
        except subprocess.TimeoutExpired:
            print(f"  [WARN] 超时")
        except Exception as e:
            print(f"  [ERROR] {e}")

        # 暂停一下避免API限流
        time.sleep(3)

    print(f"\n{'=' * 55}")
    print(f"  快速采集完成! 成功 {ok_count}/{total} 轮")
    print(f"  下一步: 在早/中/晚各跑一次真实时间采集:")
    print(f"    python scripts/collect_all_day.py --once")
    print(f"  然后: python scripts/train_travel_time.py --train")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()

