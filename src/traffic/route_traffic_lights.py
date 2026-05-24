"""
Route-based traffic light detection and simulation.

在没有 OSM 路网图的情况下，从高德 polyline 中检测路口并生成红绿灯数据。
检测策略:
  1. 沿 polyline 等距采样 (模拟城市主干道路口间距 800-1500m)
  2. 检测 polyline 中显著转弯 (>50°) 的位置 (十字路口/T字路口转向点)
  3. 去重 250m 内重复点
  4. 每条路线最多保留 N 个路口 (防止超长路线过度密集)

每个红绿灯有独立的随机相位偏移，到达时刻模拟信号灯状态并计算等待时间。
"""

import math
import random
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── 常数 ──────────────────────────────────────────
AVG_SPEED_KMH = 40.0          # 城市配送平均速度
CYCLE_LENGTH = 60             # 信号灯周期 (秒)
GREEN_RATIO = 0.50            # 绿灯占比
YELLOW_SECONDS = 3            # 黄灯时长
METRES_PER_DEG_LAT = 111320.0  # 纬度每度 ≈ 米
MIN_LIGHT_SPACING_M = 250     # 最近红绿灯间距 (避免同一个大路口多个灯)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine 距离 (米)."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """点1 → 点2 的方位角 (0-360°)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2) -
         math.sin(phi1) * math.cos(phi2) * math.cos(dlambda))
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _find_index_at_dist(cum_dist: List[float], target: float) -> int:
    """二分查找 cumulative distance 中距离 target 最近的 polyline 下标."""
    if target <= 0:
        return 0
    if target >= cum_dist[-1]:
        return len(cum_dist) - 1
    lo, hi = 0, len(cum_dist) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if cum_dist[mid] < target:
            lo = mid
        else:
            hi = mid
    return lo if (target - cum_dist[lo]) < (cum_dist[hi] - target) else hi


def _deduplicate_lights(lights: List[dict], min_dist_m: float = MIN_LIGHT_SPACING_M) -> List[dict]:
    """去掉 50m 内重复的红绿灯 (不同车辆可能共用路段)."""
    kept: List[dict] = []
    for lt in lights:
        dup = False
        for k in kept:
            if _haversine_m(lt['lat'], lt['lon'], k['lat'], k['lon']) < min_dist_m:
                dup = True
                break
        if not dup:
            kept.append(lt)
    return kept


# ── 主入口 ──────────────────────────────────────

def generate_route_traffic_lights(
    routes: List[List[int]],
    points: List[Tuple],
    road_network,
    sim_time_seconds: int,
    min_spacing_m: float = 800,
    max_spacing_m: float = 1500,
    turn_threshold_deg: float = 50,
    max_lights_per_route: int = 15,
    seed: Optional[int] = None,
) -> Dict:
    """
    从高德路线 polyline 检测路口并生成红绿灯数据。

    :param routes: VRP 路线列表 [[0,3,7,0], ...]
    :param points: 所有点坐标 [(x,y), ...]
    :param road_network: AmapRoadNetwork 实例
    :param sim_time_seconds: 模拟时间 (距零点的秒数)
    :param min_spacing_m: 等距采样最小间距 (米)
    :param max_spacing_m: 等距采样最大间距 (米)
    :param turn_threshold_deg: 转弯检测阈值 (度), 只有显著转弯才视为路口
    :param max_lights_per_route: 每条路线最多红绿灯数
    :param seed: 随机种子 (None=每次不同)
    :returns: {
        'lights': [{id, lat, lon, vehicle_id, state, wait_seconds, ...}, ...],
        'summary': {total_lights, red_lights, total_wait_seconds, per_vehicle}
    }
    """
    rng = random.Random(seed)
    green_dur = int(CYCLE_LENGTH * GREEN_RATIO)
    yellow_dur = YELLOW_SECONDS

    all_lights: List[dict] = []

    for vid, route in enumerate(routes):
        if not route or len(route) < 2:
            continue

        # 1. 获取贴路 polyline
        try:
            poly = road_network.get_route_geometry(route, points)
        except Exception:
            logger.debug(f"V{vid+1}: 无法获取 polyline, 跳过红绿灯检测")
            continue

        if not poly or len(poly) < 2:
            continue

        # poly 格式: [(lat, lon), ...]

        # 2. 计算累计距离
        cum_dist = [0.0]
        for i in range(1, len(poly)):
            d = _haversine_m(poly[i - 1][0], poly[i - 1][1],
                             poly[i][0], poly[i][1])
            cum_dist.append(cum_dist[-1] + d)

        total_dist_m = cum_dist[-1]
        if total_dist_m < 50:  # 太短不考虑
            continue

        # 3. 等距采样路口点
        spacing = rng.uniform(min_spacing_m, max_spacing_m)
        offset_start = rng.uniform(0, spacing * 0.5)
        pos = offset_start
        sampled_indices: List[int] = []

        while pos < total_dist_m:
            idx = _find_index_at_dist(cum_dist, pos)
            if 0 < idx < len(poly) - 1:
                sampled_indices.append(idx)
            pos += spacing

        # 4. 检测转弯路口 (方向变化 > turn_threshold_deg)
        turn_indices: List[int] = []
        for i in range(2, len(poly) - 2):
            b1 = _bearing_deg(poly[i - 1][0], poly[i - 1][1],
                              poly[i][0], poly[i][1])
            b2 = _bearing_deg(poly[i][0], poly[i][1],
                              poly[i + 1][0], poly[i + 1][1])
            diff = abs(b2 - b1)
            if diff > 180:
                diff = 360 - diff
            if diff > turn_threshold_deg:
                turn_indices.append(i)

        # 5. 合并采样点 + 转弯点, 按距离排序去重
        all_indices = list(set(sampled_indices + turn_indices))
        all_indices.sort(key=lambda i: cum_dist[i])

        # 去重太近的点
        filtered: List[int] = []
        for idx in all_indices:
            too_close = False
            for prev in filtered:
                if abs(cum_dist[idx] - cum_dist[prev]) < MIN_LIGHT_SPACING_M:
                    too_close = True
                    break
            if not too_close:
                filtered.append(idx)

        # 5b. 如果路口数超过上限，等距下采样 (保留首尾附近的路口)
        if len(filtered) > max_lights_per_route:
            step = len(filtered) / max_lights_per_route
            filtered = [filtered[int(i * step)] for i in range(max_lights_per_route)]

        # 6. 生成红绿灯条目
        for poly_idx in filtered:
            lat, lon = poly[poly_idx]
            pos_m = cum_dist[poly_idx]

            # 预计到达时间 (距路线出发的秒数)
            arrival_offset_s = pos_m / (AVG_SPEED_KMH * 1000 / 3600)

            # 每个路口有随机的信号灯相位偏移
            phase_offset = rng.uniform(0, CYCLE_LENGTH)
            t_cycle = (sim_time_seconds + arrival_offset_s + phase_offset) % CYCLE_LENGTH

            if t_cycle < green_dur:
                state = 'green'
                wait_s = 0.0
            elif t_cycle < green_dur + yellow_dur:
                state = 'yellow'
                wait_s = 0.0
            else:
                state = 'red'
                wait_s = CYCLE_LENGTH - t_cycle  # 等到下个绿灯

            all_lights.append({
                'id': len(all_lights),
                'lat': lat,
                'lon': lon,
                'vehicle_id': vid,
                'polyline_index': poly_idx,
                'position_m': round(pos_m, 1),
                'arrival_offset_s': round(arrival_offset_s, 1),
                'state': state,
                'wait_seconds': round(wait_s, 1),
                'phase_offset': round(phase_offset, 1),
            })

    # 7. 全局去重
    deduped = _deduplicate_lights(all_lights)

    # 重新编号
    for i, lt in enumerate(deduped):
        lt['id'] = i

    # 8. 汇总统计
    red_count = sum(1 for lt in deduped if lt['state'] == 'red')
    total_wait = sum(lt['wait_seconds'] for lt in deduped)
    per_vehicle: Dict[int, int] = {}
    per_vehicle_red: Dict[int, int] = {}
    for lt in deduped:
        vid = lt['vehicle_id']
        per_vehicle[vid] = per_vehicle.get(vid, 0) + 1
        if lt['state'] == 'red':
            per_vehicle_red[vid] = per_vehicle_red.get(vid, 0) + 1

    summary = {
        'total_lights': len(deduped),
        'red_lights': red_count,
        'green_lights': len(deduped) - red_count,
        'total_wait_seconds': round(total_wait, 1),
        'per_vehicle': per_vehicle,
        'per_vehicle_red': per_vehicle_red,
    }

    logger.info(
        f"红绿灯: {summary['total_lights']} 个路口, "
        f"{summary['red_lights']} 个红灯, "
        f"总等待 {summary['total_wait_seconds']:.0f} 秒"
    )

    return {'lights': deduped, 'summary': summary}
