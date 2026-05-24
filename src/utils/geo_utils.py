"""
共享地理坐标工具函数。

消除项目中 6 处重复的坐标格式检测逻辑。
统一判断 (lon, lat) vs (lat, lon)，并以更稳健的策略处理赤道附近城市。
"""

from typing import List, Tuple


def is_lonlat_format(points: List[Tuple]) -> bool:
    """
    判断坐标是否为 (lon, lat) 格式。

    策略:
    1. 若任何点第一维 > 90  (必为经度, 纬度范围 [-90, 90])  → (lon, lat)
    2. 若任何点第二维 > 90 → (lat, lon)
    3. 否则检查第一维范围是否更像经度 (覆盖 > 90° 跨度通常为跨国数据)
    4. 兜底: 假定 (lon, lat) — 中国场景最常见

    相比旧版 `abs(p[0]) > 90` 的单向检测, 此版本对赤道城市
    (新加坡 (103.8, 1.3)、雅加达 (106.8, -6.2)) 也能正确判断。
    """
    if not points:
        return True  # 默认 (lon, lat)

    xs = []
    ys = []
    for p in points[:min(30, len(points))]:
        try:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
        except (ValueError, TypeError, IndexError):
            continue

    if not xs:
        return True

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # 规则1: 纬度不可能超过90
    if max_x > 90 or min_x < -90:
        return True   # 第一维是经度 → (lon, lat)
    if max_y > 90 or min_y < -90:
        return False  # 第二维是经度 → (lat, lon)

    # 规则2: 经度跨度通常比纬度跨度大 (对于区域性数据集)
    x_range = max_x - min_x
    y_range = max_y - min_y
    if x_range > 180 or y_range > 180:
        # 跨国际日期线的数据, 看哪维跨度大
        return x_range > y_range

    # 规则3: 对于中国范围 (经度 73-135, 纬度 18-54)
    # 若第一维在 [70, 140] 而第二维在 [15, 55], 则是 (lon, lat)
    if 70 <= min_x <= 140 and 15 <= min_y <= 55:
        return True

    # 规则4: 兜底 — 按更常见的 (lon, lat) 处理
    return True
