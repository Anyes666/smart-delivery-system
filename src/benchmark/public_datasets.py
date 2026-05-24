"""
公开数据集加载器 — Solomon VRPTW / Cordeau VRP / 滴滴 GAIA.

用法:
    loader = PublicDatasetLoader()
    points, demands, time_windows = loader.load_solomon("R101")
    csv_path = loader.convert_to_csv("R101", output_dir="data/benchmarks/")
"""

import csv
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "benchmarks"

# Solomon VRPTW 标准格式说明:
# 每行: CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME
# 第一行为配送中心 (CUST NO. = 0)


class PublicDatasetLoader:
    """公开 VRP 基准数据集加载器."""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir or DATA_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════
    # Solomon VRPTW
    # ═══════════════════════════════════════════════

    def load_solomon(self, instance: str) -> Tuple[List, List, List]:
        """加载 Solomon VRPTW 实例并转换为系统格式.

        Args:
            instance: 实例名 (如 "R101", "C101", "RC101")

        Returns:
            (points, demands, time_windows) — 与本系统兼容的格式
        """
        path = self._find_or_download_solomon(instance)
        if path is None:
            raise FileNotFoundError(
                f"Solomon 实例 '{instance}' 未找到. "
                f"请将 .txt 文件放入 {self.cache_dir}"
            )

        coords, demands, tws, ids = [], [], [], []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                try:
                    cust_no = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    demand = float(parts[3])
                    ready_time = float(parts[4])
                    due_date = float(parts[5])
                    service_time = float(parts[6])
                except (ValueError, IndexError):
                    continue

                # Solomon 网格坐标 → 真实经纬度映射
                # grid [0,100] → ~0.2 deg → ~22 km (合理的城区配送范围)
                # 时间窗口按 1 单位 = 1 分钟换算, 与缩小后的距离匹配
                lon = 116.2 + x / 500.0
                lat = 39.8 + y / 500.0

                coords.append((lon, lat))
                demands.append(demand)
                tws.append((int(ready_time * 60), int(due_date * 60)))
                ids.append(cust_no)

        return coords, demands, tws

    def _find_or_download_solomon(self, instance: str) -> Optional[Path]:
        """查找本地 Solomon 文件, 若没有则尝试自动生成内置实例."""
        # 先查找本地文件
        for pattern in [f"{instance}.txt", f"{instance}.csv",
                        f"solomon_{instance}.txt", f"SOLOMON_{instance}.txt"]:
            p = self.cache_dir / pattern
            if p.exists():
                return p

        # 内置 Solomon 经典实例 (R101: 25点, 窄时间窗)
        BUILTIN_INSTANCES = self._builtin_solomon()
        if instance in BUILTIN_INSTANCES:
            return self._write_builtin(instance, BUILTIN_INSTANCES[instance])

        return None

    def _write_builtin(self, instance: str, data: str) -> Path:
        path = self.cache_dir / f"{instance}.txt"
        with open(path, "w") as f:
            f.write(data)
        logger.info(f"内置 Solomon 实例已写入: {path}")
        return path

    @staticmethod
    def _builtin_solomon() -> Dict[str, str]:
        """内置 Solomon 实例 (典型测试集).

        R101: 25顾客, 窄时间窗, 随机分布 (经典困难实例)
        C101: 25顾客, 宽时间窗, 聚类分布
        RC101: 25顾客, 混合分布
        """
        instances = {}

        # R101 — 25 点随机分布, 窄时间窗
        r101_lines = [
            "0  35  35   0  0 230   0",
            "1  41  49  10  161 171  10",
            "2  35  17   7  50  60   10",
            "3  55  45  13  116 126  10",
            "4  55  20  19  149 159  10",
            "5  15  30  26  34  44   10",
            "6  25  30   3  99  109  10",
            "7  20  50   5  81  91   10",
            "8  10  43   9  95  105  10",
            "9  55  60  16  97  107  10",
            "10 30  60  16  124 134  10",
            "11 20  65  12  67  77   10",
            "12 50  35  19  63  73   10",
            "13 30  25  23  159 169  10",
            "14 15  10  20  32  42   10",
            "15 30   5   8  61  71   10",
            "16 10  20  19  115 125  10",
            "17  5  30  15  82  92   10",
            "18 20  40  16  38  48   10",
            "19 15  60   7  120 130  10",
            "20 45  65  14  113 123  10",
            "21 45  20  13  71  81   10",
            "22 45  10  15  98  108  10",
            "23 55   5  10  133 143  10",
            "24 65  35  16  156 166  10",
            "25 65  20  13  153 163  10",
        ]
        instances["R101"] = "\n".join(r101_lines)

        # C101 — 25 点聚类分布, 宽时间窗
        c101_lines = [
            "0  40  50   0   0 1236   0",
            "1  45  68  10  912 967  10",
            "2  45  70  30  825 870  10",
            "3  42  66  10   65 146  10",
            "4  42  68  10  727 782  10",
            "5  42  65  10   15 67   10",
            "6  40  69  20  621 702  10",
            "7  40  66  20  170 225  10",
            "8  38  68  20  255 324  10",
            "9  38  70  10  534 605  10",
            "10 35  66  10  357 410  10",
            "11 35  69  10  448 505  10",
            "12 25  85  20  652 721  10",
            "13 22  75  30   30 92   10",
            "14 22  85  10  567 620  10",
            "15 20  80  40  384 429  10",
            "16 20  85  40  475 528  10",
            "17 18  75  20   99 148  10",
            "18 15  75  20  179 254  10",
            "19 15  80  10  278 345  10",
            "20 30  50  10   10 73   10",
            "21 30  52  20  914 969  10",
            "22 28  52  20  812 883  10",
            "23 28  55  10  732 777  10",
            "24 25  50  10   65 144  10",
            "25 25  52  40  169 224  10",
        ]
        instances["C101"] = "\n".join(c101_lines)

        # RC101 — 20 点混合分布
        rc101_lines = [
            "0  40  50   0   0 240   0",
            "1  25  85  20  145 155  10",
            "2  22  75  30  53  63   10",
            "3  22  85  10  136 146  10",
            "4  20  80  40  82  92   10",
            "5  20  85  20  48  58   10",
            "6  18  75  20  46  56   10",
            "7  15  75  20  22  32   10",
            "8  15  80  10  113 123  10",
            "9  30  50  10  84  94   10",
            "10 30  52  20  0   10   10",
            "11 28  52  20  46  56   10",
            "12 28  55  10  94  104  10",
            "13 25  50  10  159 169  10",
            "14 25  52  40  34  44   10",
            "15 45  65  10  46  56   10",
            "16 45  70  30  154 164  10",
            "17 45  20  20  133 143  10",
            "18 42  10  20  102 112  10",
            "19 40   5  30  23  33   10",
            "20 40  50  10  64  74   10",
        ]
        instances["RC101"] = "\n".join(rc101_lines)

        return instances

    # ═══════════════════════════════════════════════
    # 格式转换
    # ═══════════════════════════════════════════════

    def convert_to_csv(
        self, instance: str, output_dir: str = None,
    ) -> str:
        """将 Solomon 实例转换为本系统 CSV 格式.

        Returns:
            输出 CSV 文件路径
        """
        coords, demands, tws = self.load_solomon(instance)
        output_dir = Path(output_dir or self.cache_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"solomon_{instance}.csv"

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 使用系统标准列名 (data_processing.py 要求英文字段名)
            writer.writerow(["id", "x", "y", "demand", "time_window_start", "time_window_end", "priority"])
            for i, ((lon, lat), demand, tw) in enumerate(zip(coords, demands, tws)):
                writer.writerow([
                    i,
                    f"{lon:.6f}", f"{lat:.6f}",
                    int(demand),
                    int(tw[0]), int(tw[1]),
                    0 if i == 0 else 2,  # 配送中心优先级=0, 其他=2
                ])

        logger.info(f"已转换: {output_path} ({len(coords)} 点)")
        return str(output_path)

    @staticmethod
    def list_available() -> List[str]:
        """列出所有可用的内置实例."""
        return list(PublicDatasetLoader._builtin_solomon().keys())


# ═══════════════════════════════════════════════════
# 真实数据适配接口
# ═══════════════════════════════════════════════════

class RealDataAdapter:
    """真实物流数据适配器 — 将企业格式映射为本系统格式."""

    # 常见字段映射 (映射到系统标准英文字段名)
    FIELD_MAPS = {
        "generic_gps": {
            "longitude": "x",
            "latitude": "y",
            "weight_kg": "demand",
            "delivery_start": "time_window_start",
            "delivery_end": "time_window_end",
            "priority_level": "priority",
        },
        "didi_gaia": {
            "start_lng": "x",
            "start_lat": "y",
            "weight": "demand",
            "eta_start": "time_window_start",
            "eta_end": "time_window_end",
        },
    }

    def __init__(self, field_map: str = "generic_gps"):
        self.mapping = self.FIELD_MAPS.get(field_map, self.FIELD_MAPS["generic_gps"])

    def map_row(self, row: Dict) -> Dict:
        """将一行原始数据映射为本系统格式."""
        # 内部使用标准英文字段名
        mapped = {
            "id": 0, "x": 0.0, "y": 0.0, "demand": 0,
            "time_window_start": 0, "time_window_end": 86400,
            "priority": 1,
        }
        for src_field, dst_field in self.mapping.items():
            if src_field in row:
                try:
                    mapped[dst_field] = float(row[src_field])
                except (ValueError, TypeError):
                    pass
        return mapped

    def import_csv(self, input_path: str, output_path: str = None) -> str:
        """从企业 CSV 导入并转换.

        Args:
            input_path: 原始数据 CSV
            output_path: 输出路径 (默认自动生成)

        Returns:
            输出 CSV 路径
        """
        import csv
        rows = []
        with open(input_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                mapped = self.map_row(row)
                mapped["id"] = i
                rows.append(mapped)

        if output_path is None:
            base = os.path.splitext(os.path.basename(input_path))[0]
            output_path = str(DATA_DIR / f"{base}_converted.csv")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "x", "y", "demand", "time_window_start", "time_window_end", "priority",
            ])
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"已导入: {input_path} → {output_path} ({len(rows)} 行)")
        return output_path
