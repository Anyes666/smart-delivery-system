"""
VRP 求解器 — 三种算法统一支持: 异构车队 + 自动车队规模 + 溢出检测。

统一返回格式: UnifiedSolution (见 vehicle_types.py)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from ortools.constraint_solver import routing_enums_pb2, pywrapcp

from src.algorithms.vehicle_types import (
    VehicleType, Fleet, UnifiedSolution, default_fleet,
)

logger = logging.getLogger(__name__)


class VRPSolver:
    """
    VRP 求解器 — OR-Tools / 贪心 / 聚类优先。

    用法::

        solver = VRPSolver(fleet=Fleet(...))
        result = solver.solve_with_ortools(distance_matrix, demands, ...)
        # result 是 UnifiedSolution, 包含 routes / vehicles_used / is_overflow
    """

    def __init__(self, fleet: Fleet = None, depot_index: int = 0):
        self.fleet = fleet or default_fleet()
        self.depot_index = depot_index
        self.time_windows = None
        self.priorities = None

    # ═══════════════════════════════════════════════
    # OR-Tools 精确求解 (异构车队)
    # ═══════════════════════════════════════════════

    def solve_with_ortools(
        self, distance_matrix, demands,
        time_limit=30, time_windows=None,
        travel_time_predictor=None, points=None, sim_time_seconds=None,
        travel_time_matrix=None,
    ) -> Optional[UnifiedSolution]:
        """
        OR-Tools 求解 — 支持异构车队、自动车队规模、时间窗口、ML预测。

        每辆车独立绑定车型 → 不同 Capacity / FixedCost / CostPerKm。
        OR-Tools 内部自动权衡"多派一辆小车 vs 少派一辆大车"。

        Parameters:
            distance_matrix: N×N 物理距离矩阵 (km), 用于距离维度统计。
            travel_time_matrix: N×N 行程时间矩阵 (秒), 可选。提供时用于
                arc cost (时间价值货币化) 和时间窗约束, 实现广义成本语义。
        """
        num_nodes = len(distance_matrix)
        AVG_SPEED_KMH = 40.0
        depot = self.depot_index

        # ── 构建车型数组 ──
        # 求解阶段全部使用最大容量车型, 避免小 max_total 时轮询分配
        # (如 3 辆车 → [微型,轻型,中型]) 导致路线在过小的容量约束下构建。
        # 求解后由 _optimize_vehicle_assignments 降级为成本最优车型。
        max_type = max(self.fleet.types, key=lambda t: t.capacity)
        min_fixed = min(t.fixed_cost for t in self.fleet.types)

        types_flat = [max_type] * self.fleet.max_total
        num_vehicles = len(types_flat)
        capacities = [t.capacity for t in types_flat]
        # 使用最小固定成本, 避免高固定成本抑制车辆使用
        fixed_costs = [int(min_fixed)] * num_vehicles
        cost_per_km = [t.cost_per_km for t in types_flat]

        logger.info(
            f"OR-Tools: {num_nodes}点, {num_vehicles}候选车("
            + ",".join(t.name for t in self.fleet.types) + ")"
        )

        # ── Manager & Model ──
        manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot)
        routing = pywrapcp.RoutingModel(manager)

        # ── 距离回调 (基准, 用于距离维度统计) ──
        def base_distance_cb(from_idx, to_idx):
            fn = manager.IndexToNode(from_idx)
            tn = manager.IndexToNode(to_idx)
            return int(distance_matrix[fn][tn] * 1000)

        base_idx = routing.RegisterTransitCallback(base_distance_cb)

        # ── 异构成本回调 ──
        # 有 travel_time_matrix 时: arc_cost = travel_time_sec × (cpm × AVG_SPEED / 3600)
        #   即广义成本 generalized_cost = distance·cpm × (actual_time / free_flow_time)
        # 无 travel_time_matrix 时: arc_cost = distance_km × cpm (回退到纯距离成本)
        has_tt = travel_time_matrix is not None

        def cost_cb(vehicle_id):
            cpm = cost_per_km[vehicle_id]
            if has_tt:
                cost_per_sec = cpm * AVG_SPEED_KMH / 3600.0
                def cb(from_idx, to_idx):
                    fn = manager.IndexToNode(from_idx)
                    tn = manager.IndexToNode(to_idx)
                    return int(travel_time_matrix[fn][tn] * cost_per_sec)
            else:
                def cb(from_idx, to_idx):
                    fn = manager.IndexToNode(from_idx)
                    tn = manager.IndexToNode(to_idx)
                    return int(distance_matrix[fn][tn] * cpm)
            return cb

        # 注册每辆车独立的成本回调
        for vid in range(num_vehicles):
            idx = routing.RegisterTransitCallback(cost_cb(vid))
            routing.SetArcCostEvaluatorOfVehicle(idx, vid)

        # ── 车辆启用成本 ──
        for vid in range(num_vehicles):
            routing.SetFixedCostOfVehicle(fixed_costs[vid], vid)

        # ── 异构容量约束 ──
        def demand_cb(from_idx):
            node = manager.IndexToNode(from_idx)
            return demands[node] if node < len(demands) else 0

        demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_idx, 0, capacities, True, 'Capacity'
        )

        # ── 距离维度 (路线均衡) ──
        routing.AddDimension(base_idx, 0, 10_000_000, True, 'Distance')
        routing.GetDimensionOrDie('Distance').SetGlobalSpanCostCoefficient(100)

        # ── 时间窗口 ──
        has_tw = (time_windows is not None and len(time_windows) == num_nodes)
        use_ml = (
            travel_time_predictor is not None
            and travel_time_predictor.is_trained
            and points is not None and len(points) == num_nodes
        )
        ml_hour = (sim_time_seconds or 36000) % 86400 // 3600
        ml_dow = ((sim_time_seconds or 36000) // 86400) % 7

        if has_tw:
            if use_ml:
                def time_cb(from_idx, to_idx):
                    fn, tn = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
                    return int(travel_time_predictor.predict(
                        points[fn], points[tn],
                        distance_matrix[fn][tn], ml_hour, ml_dow))
            elif has_tt:
                # 直接使用预计算的行程时间矩阵 (已含拥堵)
                def time_cb(from_idx, to_idx):
                    fn, tn = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
                    return int(travel_time_matrix[fn][tn])
            else:
                def time_cb(from_idx, to_idx):
                    fn, tn = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
                    return int(distance_matrix[fn][tn] / AVG_SPEED_KMH * 3600)

            time_idx = routing.RegisterTransitCallback(time_cb)

            tw_spans = [tw[1] - tw[0] for tw in time_windows if tw[1] < 86400]
            max_slack = max(max(tw_spans) if tw_spans else 7200, 7200)

            routing.AddDimension(time_idx, max_slack, 86400, True, 'Time')
            td = routing.GetDimensionOrDie('Time')

            # ── 软时间窗: 硬约束放宽 + 软边界惩罚 ──
            from config import settings
            sw_config = getattr(settings, 'SOFT_TIME_WINDOW_CONFIG', {})
            use_soft_tw = sw_config.get('enabled', True)
            hard_slack = sw_config.get('hard_bound_slack', 7200)
            early_penalty = sw_config.get('early_penalty_per_min', 100) / 60.0  # → 每秒
            late_penalty = sw_config.get('late_penalty_per_min', 500) / 60.0

            for node in range(num_nodes):
                idx = manager.NodeToIndex(node)
                if node == depot:
                    td.CumulVar(idx).SetRange(0, 86400)
                else:
                    tw = time_windows[node]
                    if use_soft_tw:
                        # 硬约束: 原始时间窗 ± hard_slack (确保始终有可行解)
                        td.CumulVar(idx).SetRange(
                            max(0, tw[0] - hard_slack),
                            min(tw[1] + hard_slack, 86400),
                        )
                        # 软约束: 在原始时间窗边界施加线性惩罚
                        if early_penalty > 0:
                            td.SetCumulVarSoftLowerBound(idx, tw[0], int(early_penalty))
                        if late_penalty > 0:
                            td.SetCumulVarSoftUpperBound(idx, tw[1], int(late_penalty))
                    else:
                        td.CumulVar(idx).SetRange(max(0, tw[0]), min(tw[1], 86400))
            for vid in range(num_vehicles):
                td.CumulVar(routing.Start(vid)).SetRange(0, 86400)

        # ── 求解 ──
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        # 自适应时间上限: 20点≈4s, 50点≈10s, 100点=20s
        params.time_limit.seconds = min(time_limit, max(1, num_nodes // 5))
        params.use_full_propagation = False

        solution = None
        try:
            solution = routing.SolveWithParameters(params)
        except Exception as e:
            logger.error(f"OR-Tools SolveWithParameters 异常: {e}")
            return None
        if not solution:
            return None

        # ── 提取路线 ──
        routes = []
        assignments = []   # 车型索引
        total_dist = 0.0
        visited = {depot}
        overflow = []

        for vid in range(num_vehicles):
            idx = routing.Start(vid)
            if solution.Value(routing.NextVar(idx)) == routing.End(vid):
                routes.append([depot])
                assignments.append(-1)
                continue

            route = []
            dist = 0.0
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                route.append(node)
                visited.add(node)
                nxt = solution.Value(routing.NextVar(idx))
                if not routing.IsEnd(nxt):
                    dist += distance_matrix[node][manager.IndexToNode(nxt)]
                idx = nxt
            if route[-1] != depot:
                route.append(depot)

            routes.append(route)
            assignments.append(types_flat[vid].name)  # 存储车型名
            total_dist += dist

        # 检测溢出
        all_nodes = set(range(num_nodes))
        overflow = sorted(all_nodes - visited)

        # 后处理: 保持路线不变, 优化车型分配 (避免稀疏场景下轮询车型选到高成本车)
        assignments = self._optimize_vehicle_assignments(routes, distance_matrix, demands)

        result = self._build_result(routes, assignments, total_dist, overflow, demands, distance_matrix)
        logger.info(
            f"OR-Tools: {result.num_active}车/{num_vehicles}候选, "
            f"{total_dist:.1f}km, 代价{result.total_cost_km_eq:.1f}"
            + (f", 溢出{len(overflow)}点" if overflow else "")
        )
        return result

    # ═══════════════════════════════════════════════
    # 贪心算法 (最佳适配 + 车型选择)
    # ═══════════════════════════════════════════════

    def greedy_vrp_solver(
        self, distance_matrix, demands, **kwargs
    ) -> UnifiedSolution:
        """
        贪心算法: 最近邻 + 最佳车型选择 + 溢出检测。

        对每个未分配节点, 优先尝试填满当前车辆; 若装不下,
        从异构车队中选择"能装下该订单的最经济车型"。
        达到车辆上限后仍有未分配节点 → 标记溢出。
        """
        num_nodes = len(distance_matrix)
        depot = self.depot_index
        unvisited = set(range(num_nodes)) - {depot}

        routes, assignments = [], []
        total_dist = 0.0

        # 可用车型按综合成本排序 (从便宜到贵)
        sorted_types = sorted(self.fleet.types,
                              key=lambda t: t.fixed_cost + t.cost_per_km * 5)

        while unvisited and len(routes) < self.fleet.max_total:
            # 智能车型选择: 综合考虑单点最大需求和全局容量需求
            max_demand = max(demands[n] for n in unvisited)
            total_remaining = sum(demands[n] for n in unvisited)
            vehicles_left = self.fleet.max_total - len(routes)
            # 每辆车至少需要承担的容量 = 总剩余需求 / 剩余车辆数
            per_vehicle_needed = total_remaining / max(vehicles_left, 1)
            target_capacity = max(max_demand, per_vehicle_needed)

            vt = None
            for t in sorted_types:
                if t.capacity >= target_capacity:
                    vt = t
                    break
            if vt is None:
                vt = sorted_types[-1]  # 兜底用最大车

            route = [depot]
            load = 0
            current = depot
            dist = 0.0

            # 首站: 选距离depot最远的未访问点 (区域划分, 避免外围点遗留)
            first_node = max(unvisited, key=lambda n: distance_matrix[depot][n])
            if load + demands[first_node] <= vt.capacity:
                route.append(first_node)
                load += demands[first_node]
                dist += distance_matrix[depot][first_node]
                unvisited.discard(first_node)
                current = first_node

            while unvisited:
                best_node, best_d = None, float('inf')
                for node in unvisited:
                    if load + demands[node] <= vt.capacity:
                        d = distance_matrix[current][node]
                        if d < best_d:
                            best_d = d
                            best_node = node

                if best_node is None:
                    break

                route.append(best_node)
                load += demands[best_node]
                dist += best_d
                unvisited.discard(best_node)
                current = best_node

            if len(route) > 1:
                route.append(depot)
                # 2-opt 局部优化: 消除最近邻产生的路径交叉
                route = self._two_opt_improve(route, distance_matrix)
                dist = sum(distance_matrix[route[k]][route[k + 1]]
                           for k in range(len(route) - 1))
                routes.append(route)
                assignments.append(vt.name)
                total_dist += dist
            else:
                break  # 无未访问节点了

        overflow = sorted(unvisited)

        result = self._build_result(routes, assignments, total_dist, overflow, demands, distance_matrix)
        logger.info(
            f"贪心: {result.num_active}车/{self.fleet.max_total}上限, "
            f"{total_dist:.1f}km" + (f", 溢出{len(overflow)}点" if overflow else "")
        )
        return result

    # ═══════════════════════════════════════════════
    # 聚类优先算法 (K-Means + 车型匹配)
    # ═══════════════════════════════════════════════

    def cluster_first_route_second(
        self, distance_matrix, demands, points=None, **kwargs
    ) -> UnifiedSolution:
        """
        聚类优先: K-Means 聚类 → 每簇匹配车型 → 簇内 TSP。

        聚类数 ≤ min(fleet.max_total, n_customers)。
        每簇根据总需求自动匹配最经济车型。
        """
        num_nodes = len(distance_matrix)
        depot = self.depot_index
        customers = [i for i in range(num_nodes) if i != depot]

        if not customers:
            return self._empty_result()

        n_customers = len(customers)
        total_demand = sum(demands[n] for n in customers)

        # ── 聚类数 = 需求量与车型容量匹配 ──
        max_cap = max(t.capacity for t in self.fleet.types)
        min_for_demand = max(1, int((total_demand + max_cap - 1) // max_cap))
        n_clusters = min(self.fleet.max_total, n_customers)
        n_clusters = max(n_clusters, min_for_demand)

        if points is not None and len(points) == num_nodes:
            coords = np.array([[float(p[0]), float(p[1])] for p in points])
            features = coords[customers]
        else:
            # 使用距离矩阵作为特征 (MDS近似)
            features = np.array([distance_matrix[i] for i in customers])

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        # ── 分组 ──
        clusters = [[] for _ in range(n_clusters)]
        for i, lbl in enumerate(labels):
            clusters[lbl].append(customers[i])

        # ── 每簇: 匹配车型 + 内部TSP ──
        routes, assignments = [], []
        total_dist = 0.0
        overflow = []

        for cluster_nodes in clusters:
            if not cluster_nodes:
                continue

            total_demand = sum(demands[n] for n in cluster_nodes)

            # 选择最经济车型 (基于实际簇大小估算距离)
            vt = self._best_type_for_load(total_demand, distance_matrix, cluster_nodes)
            if vt.capacity < total_demand:
                # 一辆车装不下 → 拆分为多车 (简单贪心)
                split_overflow, split_dist = self._split_cluster(
                    cluster_nodes, vt, distance_matrix, demands,
                    routes, assignments,
                )
                overflow.extend(split_overflow)
                total_dist += split_dist
                continue

            # 簇内最近邻TSP + 2-opt局部优化
            route, _ = self._tsp_nearest_neighbor(cluster_nodes, distance_matrix, depot)
            route = self._two_opt_improve(route, distance_matrix)
            dist = sum(distance_matrix[route[k]][route[k + 1]]
                       for k in range(len(route) - 1))
            routes.append(route)
            assignments.append(vt.name)
            total_dist += dist

        # 补齐占位
        while len(routes) < self.fleet.max_total:
            routes.append([depot])
            assignments.append(-1)

        result = self._build_result(routes, assignments, total_dist, overflow, demands, distance_matrix)
        logger.info(
            f"聚类优先: {n_clusters}簇, {result.num_active}车, "
            f"{total_dist:.1f}km" + (f", 溢出{len(overflow)}点" if overflow else "")
        )
        return result

    # ═══════════════════════════════════════════════
    # 2-opt 局部搜索 (消除路径交叉, 实际配送必备)
    # ═══════════════════════════════════════════════

    def _two_opt_improve(self, route: List[int], dist_mtx) -> List[int]:
        """
        对单条路线执行 2-opt 局部优化。

        重复尝试反转子路径 [i+1, j], 如果缩短总距离则采纳。
        消除"八字交叉"等纯最近邻贪心产生的次优结构。
        时间复杂度: O(n²) 每次迭代, n 为路线节点数。
        """
        if len(route) <= 4:  # depot-a-b-depot 以下无需优化
            return route

        improved = True
        best = list(route)

        while improved:
            improved = False
            # 不翻转 depot (索引 0 和 -1)
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    # 2-opt: 反转 best[i:j+1], 新边为 best[i-1]→best[j] 和 best[i]→best[j+1]
                    old_edges = (dist_mtx[best[i - 1]][best[i]] +
                                 dist_mtx[best[j]][best[j + 1]])
                    new_edges = (dist_mtx[best[i - 1]][best[j]] +
                                 dist_mtx[best[i]][best[j + 1]])
                    if new_edges < old_edges - 1e-9:
                        best[i:j + 1] = reversed(best[i:j + 1])
                        improved = True

        return best

    # ═══════════════════════════════════════════════
    # 内部辅助
    # ═══════════════════════════════════════════════

    def _best_type_for_load(self, total_load: float,
                             dist_mtx=None, nodes=None) -> VehicleType:
        """为给定总负载选择综合成本最低的车型.

        如果提供了 dist_mtx 和 nodes, 用实际簇大小估算路线距离;
        否则使用默认 8km 估算 (城市配送典型).
        """
        # 估算路线距离: 簇内 TSP 近似 ≈ 簇大小 × 平均边距离
        if dist_mtx is not None and nodes and len(nodes) >= 2:
            # 用簇内点对平均距离 × 点数 作为路线长度下界
            sample_edges = [dist_mtx[nodes[i]][nodes[j]]
                           for i in range(min(len(nodes), 5))
                           for j in range(i + 1, min(len(nodes), 5))]
            avg_edge = sum(sample_edges) / len(sample_edges) if sample_edges else 5.0
            est_dist = avg_edge * len(nodes) * 1.2  # 1.2x 修正因子 (TSP > 平均)
        else:
            est_dist = 8.0  # 默认城市配送单簇 ~8km

        best, best_cost = self.fleet.types[-1], float('inf')
        for vt in self.fleet.types:
            if vt.capacity >= total_load:
                cost = vt.fixed_cost + vt.cost_per_km * est_dist
                if cost < best_cost:
                    best_cost, best = cost, vt
        return best

    def _split_cluster(self, nodes, vt, dist_mtx, demands,
                       routes, assignments):
        """将超容量的簇拆分为多辆车, 返回 (overflow, added_dist)."""
        overflow = []
        remaining = sorted(nodes, key=lambda n: -demands[n])
        current_route = [self.depot_index]
        current_load = 0
        current = self.depot_index
        dist = 0.0
        added_dist = 0.0

        while remaining:
            best_n, best_d = None, float('inf')
            for n in remaining:
                if current_load + demands[n] <= vt.capacity:
                    d = dist_mtx[current][n]
                    if d < best_d:
                        best_d, best_n = d, n
            if best_n is None:
                # 装不下 → 发车, 新开一辆
                if len(current_route) > 1:
                    current_route.append(self.depot_index)
                    current_route = self._two_opt_improve(current_route, dist_mtx)
                    routes.append(current_route)
                    assignments.append(vt.name)
                    added_dist += dist
                if len(routes) >= self.fleet.max_total:
                    overflow.extend(remaining)
                    break
                current_route = [self.depot_index]
                current_load = 0
                current = self.depot_index
                dist = 0.0
                continue

            current_route.append(best_n)
            current_load += demands[best_n]
            dist += best_d
            remaining.remove(best_n)
            current = best_n

        if len(current_route) > 1 and len(routes) < self.fleet.max_total:
            current_route.append(self.depot_index)
            current_route = self._two_opt_improve(current_route, dist_mtx)
            routes.append(current_route)
            assignments.append(vt.name)
            added_dist += dist
        elif remaining:
            overflow.extend(remaining)

        return overflow, added_dist

    @staticmethod
    def _tsp_nearest_neighbor(nodes, dist_mtx, depot):
        """簇内最近邻TSP."""
        route = [depot]
        remaining = list(nodes)
        current = depot
        dist = 0.0
        while remaining:
            best = min(remaining, key=lambda n: dist_mtx[current][n])
            dist += dist_mtx[current][best]
            route.append(best)
            remaining.remove(best)
            current = best
        route.append(depot)
        dist += dist_mtx[current][depot]
        return route, dist

    def _optimize_vehicle_assignments(
        self, routes, distance_matrix, demands
    ) -> List:
        """
        后处理: 保持路线不变, 重新为每条路线匹配最优车型。

        对每条活跃路线, 计算实际距离和载重, 然后从异构车队中选择
        能装下该路线总需求且总成本最低的车型。

        成本 = 车辆固定成本 + 路线距离 × 车型每公里成本。
        这与贪心/聚类算法中「先建路线再选车」的解耦策略一致,
        避免 OR-Tools 轮询车型分配在稀疏场景下选到高成本车型。
        """
        assignments = []
        for route in routes:
            if len(route) <= 2:
                assignments.append(-1)
                continue

            dist = sum(
                distance_matrix[route[k]][route[k + 1]]
                for k in range(len(route) - 1)
            )
            load = sum(demands[n] for n in route if n != self.depot_index)

            best_vt = None
            best_cost = float('inf')
            for vt in self.fleet.types:
                if vt.capacity >= load:
                    cost = vt.fixed_cost + dist * vt.cost_per_km
                    if cost < best_cost:
                        best_cost = cost
                        best_vt = vt

            if best_vt is None:
                best_vt = self.fleet.types[-1]

            assignments.append(best_vt.name)

        return assignments

    def _build_result(
        self, routes: List, assignments: List,
        total_dist: float, overflow: List[int],
        demands: List, distance_matrix=None,
    ) -> UnifiedSolution:
        """构建统一返回格式."""
        name_to_idx = {t.name: i for i, t in enumerate(self.fleet.types)}
        norm_assignments = []
        for a in assignments:
            if isinstance(a, str):
                norm_assignments.append(name_to_idx.get(a, -1))
            elif isinstance(a, int):
                norm_assignments.append(a)
            else:
                norm_assignments.append(-1)

        while len(routes) < self.fleet.max_total:
            routes.append([self.depot_index])
            norm_assignments.append(-1)

        result = UnifiedSolution(
            routes=routes,
            vehicle_assignments=norm_assignments,
            fleet=self.fleet,
            total_distance_km=total_dist,
            is_overflow=len(overflow) > 0,
            overflow_nodes=overflow,
            distance_matrix=distance_matrix,
        )
        result.compute(demands=demands)
        return result

    def _empty_result(self) -> UnifiedSolution:
        routes = [[self.depot_index]]
        assigns = [-1]
        result = UnifiedSolution(
            routes=routes, vehicle_assignments=assigns,
            fleet=self.fleet, total_distance_km=0.0,
            is_overflow=False, overflow_nodes=[],
        )
        return result

    # ═══════════════════════════════════════════════
    # 辅助计算 (保持兼容)
    # ═══════════════════════════════════════════════

    def calculate_route_metrics(self, routes, distance_matrix, demands):
        if not routes:
            return {'total_distance': 0, 'total_demand': 0,
                    'num_vehicles_used': 0, 'routes_info': [],
                    'efficiency_score': 0}

        total_dist = 0
        total_demand = 0
        routes_info = []

        for i, route in enumerate(routes):
            d = sum(distance_matrix[route[j]][route[j + 1]]
                    for j in range(len(route) - 1))
            demand = sum(demands[n] for n in route if n != self.depot_index)
            stops = len([n for n in route if n != self.depot_index])
            routes_info.append({
                'route_id': i, 'nodes': route, 'distance': d,
                'demand': demand, 'num_stops': stops,
            })
            total_dist += d
            total_demand += demand

        return {
            'total_distance': total_dist, 'total_demand': total_demand,
            'num_vehicles_used': sum(1 for r in routes if len(r) > 1),
            'routes_info': routes_info,
            'efficiency_score': total_demand / total_dist if total_dist > 0 else 0,
        }

    def calculate_total_distance(self, routes, distance_matrix):
        if not routes:
            return 0
        return sum(
            sum(distance_matrix[route[i]][route[i + 1]]
                for i in range(len(route) - 1))
            for route in routes
        )


def convert_seconds_to_time(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
