"""
统一异构车型配置 & 统一返回格式。

用法::

    from src.algorithms.vehicle_types import VehicleType, Fleet, UnifiedSolution

    fleet = Fleet([
        VehicleType("小型车", capacity=30,  fixed_cost=5000,  cost_per_km=800),
        VehicleType("中型车", capacity=60,  fixed_cost=10000, cost_per_km=1000),
        VehicleType("大型车", capacity=120, fixed_cost=18000, cost_per_km=1200),
    ])

    # 所有三种算法返回统一格式:
    result = UnifiedSolution(
        routes=[[0,3,7,0], [0,5,2,0]],
        vehicle_assignments=[1, 0],   # route 0用 中型车, route 1用 小型车
        fleet=fleet,
        total_distance_km=85.3,
        is_overflow=False,
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class VehicleType:
    """单种车型定义 — 对标 JT/T 1325-2020."""
    name: str                    # "微型封闭货车" / "轻型封闭货车" / "中型厢式货车" / "重型厢式货车"
    capacity: int                # 最大载重 (典型值)
    fixed_cost: float = 10000.0  # 启用成本 (单位: 米等效)
    cost_per_km: float = 1000.0  # 每公里成本 (单位: 米等效, 1000=基准)
    code: str = ""               # JT/T 1325 代码: V-MINI / V-LIGHT / V-MEDIUM / V-HEAVY

    @property
    def fixed_cost_km(self) -> float:
        return self.fixed_cost / 1000.0

    def __repr__(self):
        cpk = self.cost_per_km / 1000.0
        return (f"VehicleType({self.name}, cap={self.capacity}, "
                f"fixed={self.fixed_cost_km:.0f}km, cost/km={cpk:.2f}x)")


@dataclass
class Fleet:
    """异构车队: 包含多种车型及其数量."""
    types: List[VehicleType]
    max_total: int = 20          # 总车辆数上限

    @property
    def total_capacity(self) -> int:
        """最坏情况总容量 (全部用最小车)."""
        min_cap = min(t.capacity for t in self.types)
        return min_cap * self.max_total

    def best_for_demand(self, demand: float) -> VehicleType:
        """为给定需求量推荐综合成本最低的车型."""
        best = None
        best_cost = float('inf')
        for vt in self.types:
            if vt.capacity >= demand:
                cost = vt.fixed_cost + vt.cost_per_km * 5.0  # 预估5km路线
                if cost < best_cost:
                    best_cost = cost
                    best = vt
        return best or self.types[-1]  # 兜底用最大车

    def cheapest_per_unit(self, total_demand: float) -> VehicleType:
        """为给定总需求选择每单位成本最低的车型."""
        best, best_ratio = self.types[0], float('inf')
        for vt in self.types:
            ratio = (vt.fixed_cost + vt.cost_per_km * 10) / vt.capacity
            if ratio < best_ratio:
                best_ratio = ratio
                best = vt
        return best

    def min_vehicles_needed(self, total_demand: float) -> int:
        """最少需要多少辆车 (全部用最大车)."""
        max_cap = max(t.capacity for t in self.types)
        return max(1, int((total_demand + max_cap - 1) // max_cap))


@dataclass
class UnifiedSolution:
    """
    三种算法统一返回格式。
    """
    routes: List[List[int]]
    vehicle_assignments: List[int]
    fleet: Fleet
    total_distance_km: float = 0.0
    is_overflow: bool = False
    overflow_nodes: List[int] = field(default_factory=list)
    distance_matrix: object = None        # NxN ndarray (km) — 用于距离计算

    total_cost_km_eq: float = 0.0
    vehicles_used: List[Dict] = field(default_factory=list)

    def compute(self, demands=None):
        """根据 routes + assignments + distance_matrix 计算成本和详情."""
        dm = self.distance_matrix
        cost = 0.0
        used = []
        for rid, (route, vt_idx) in enumerate(
            zip(self.routes, self.vehicle_assignments)
        ):
            if vt_idx < 0 or len(route) <= 1:
                used.append({'route_id': rid, 'vehicle_type': None,
                             'stops': 0, 'distance_km': 0, 'load': 0,
                             'capacity': 0, 'fixed_cost_km': 0,
                             'running_cost_km': 0, 'is_active': False})
                continue

            vt = self.fleet.types[vt_idx]
            # 从 distance_matrix 计算实际距离
            dist = 0.0
            if dm is not None:
                for i in range(len(route) - 1):
                    dist += float(dm[route[i]][route[i + 1]])
            stops = len([n for n in route if n != 0])
            load = sum(demands[n] for n in route if n != 0) if demands else 0

            fixed = vt.fixed_cost_km
            running = dist * vt.cost_per_km / 1000.0
            cost += fixed + running

            used.append({'route_id': rid, 'vehicle_type': vt.name,
                         'type_index': vt_idx, 'stops': stops,
                         'distance_km': dist, 'load': load,
                         'capacity': vt.capacity,
                         'fixed_cost_km': fixed,
                         'running_cost_km': running,
                         'is_active': True})

        self.vehicles_used = used
        self.total_cost_km_eq = cost
        # 更新总距离
        if dm is not None:
            self.total_distance_km = sum(
                u['distance_km'] for u in used if u['is_active'])

    def set_loads(self, demands):
        for u in self.vehicles_used:
            if not u['is_active']:
                continue
            route = self.routes[u['route_id']]
            u['load'] = sum(demands[n] for n in route if n != 0)

    @property
    def num_active(self) -> int:
        return sum(1 for u in self.vehicles_used if u['is_active'])

    @property
    def num_available(self) -> int:
        return len(self.routes)

    def summary(self) -> str:
        lines = [
            f"车辆启用: {self.num_active}/{self.num_available} 辆",
            f"总行驶距离: {self.total_distance_km:.2f} km",
            f"综合总代价: {self.total_cost_km_eq:.2f} km(等效)",
        ]
        if self.is_overflow:
            lines.append(f"⚠ 溢出! 未分配节点: {self.overflow_nodes}")
        for u in self.vehicles_used:
            if u['is_active']:
                vt_name = u.get('vehicle_type', '?')
                lines.append(
                    f"  车辆{u['route_id']+1} ({vt_name}): "
                    f"{u['stops']}站, {u['distance_km']:.1f}km, "
                    f"载重{u['load']}/{u['capacity']}, "
                    f"成本({u['fixed_cost_km']:.0f}+{u['running_cost_km']:.0f})km"
                )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════
# 默认车队 (与 settings 同步)
# ═══════════════════════════════════════════════════

def default_fleet() -> Fleet:
    """从 settings 读取或使用默认值构建车队."""
    try:
        from config import settings
        types_cfg = getattr(settings, 'VEHICLE_TYPES', None)
        if types_cfg:
            types = [VehicleType(**t) for t in types_cfg]
            max_total = getattr(settings, 'MAX_VEHICLES', 20)
            return Fleet(types=types, max_total=max_total)
    except Exception:
        pass

    # 默认车队 (JT/T 1325-2020 国标车型)
    return Fleet(
        types=[
            VehicleType("微型封闭货车", capacity=15,  fixed_cost=3000,  cost_per_km=500,  code="V-MINI"),
            VehicleType("轻型封闭货车", capacity=30,  fixed_cost=5000,  cost_per_km=800,  code="V-LIGHT"),
            VehicleType("中型厢式货车", capacity=60,  fixed_cost=10000, cost_per_km=1000, code="V-MEDIUM"),
            VehicleType("重型厢式货车", capacity=120, fixed_cost=18000, cost_per_km=1200, code="V-HEAVY"),
        ],
        max_total=20,
    )
