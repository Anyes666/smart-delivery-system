import sys
import os
import logging
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from config import settings
from src.utils.data_processing import (
    load_points_csv,
    get_coordinates,
    get_demands,
    get_time_windows,
    get_priorities,
    get_ids,
    load_all_data,
    list_available_data_files
)
from src.algorithms.shortest_path import ShortestPathCalculator
from src.algorithms.vrp_solver import VRPSolver
from src.visualization.folium_maps import FoliumVisualizer
from src.visualization.plotly_animation import PlotlyAnimator

logger = logging.getLogger(__name__)


def _road_label():
    """根据配置返回路网标签字符串."""
    from config import settings
    provider = getattr(settings, 'ROAD_NETWORK_PROVIDER', 'osmnx')
    if provider == 'amap':
        return '高德路网'
    elif getattr(settings, 'USE_REAL_ROAD_NETWORK', True):
        return 'OSM路网'
    return '欧氏距离'


class DeliverySystem:
    def __init__(self, use_road_network: bool = None):
        """初始化配送系统

        :param use_road_network: 是否使用OSM真实路网。None则读取settings.USE_REAL_ROAD_NETWORK
        """
        self.use_road_network = (settings.USE_REAL_ROAD_NETWORK
                                 if use_road_network is None
                                 else use_road_network)
        # Auto-enable for Amap if key is configured
        provider = getattr(settings, 'ROAD_NETWORK_PROVIDER', 'osmnx')
        if provider == 'amap' and use_road_network is None:
            key = (getattr(settings, 'AMAP_CONFIG', {}) or {}).get('api_key', '')
            if key and key != 'YOUR_AMAP_KEY':
                self.use_road_network = True
        self.road_network = None
        self.vehicle_profile = None

        # Phase 2: Dynamic traffic modules (only with road network)
        self.congestion_engine = None
        self.traffic_lights = None
        self.time_rules = None
        self.traffic_lights_data = None  # Pre-generated traffic light data for all routes
        self._time_windows_relaxed = False  # Flag set when time windows were relaxed

        # Phase 1.1: ML行程时间预测
        self.travel_time_predictor = None  # TravelTimePredictor 实例
        self.travel_time_collector = None  # TravelTimeDataCollector 实例

        # 在线自适应行程时间预测 (替代硬编码交通惩罚)
        self.congestion_predictor = None  # AdaptiveCongestionPredictor 实例

        # Phase 3: Enhanced renderer — always available (degrades gracefully)
        self._init_enhanced_renderer()

        self.shortest_path_calculator = None
        self.vrp_solver = None
        self.folium_visualizer = FoliumVisualizer(
            center_point=settings.MAP_CENTER,
            zoom_start=settings.ZOOM_LEVEL
        )
        self.plotly_animator = PlotlyAnimator(
            animation_speed=settings.VISUALIZATION_CONFIG['animation_speed']
        )
        self.points = None
        self.distance_matrix = None
        self.demands = None
        self.time_windows = None
        self.priorities = None
        self.ids = None
        self.n_points = 0
        self.routes = None
        self.depot_index = settings.DEPOT_INDEX

        if self.use_road_network:
            self._init_road_network()

    def _init_enhanced_renderer(self):
        """Phase 3: Always init the enhanced renderer (degrades gracefully)."""
        from src.visualization.enhanced_map import EnhancedMapRenderer
        vc = settings.VISUALIZATION_CONFIG
        self.enhanced_renderer = EnhancedMapRenderer(
            road_network=None,  # wired later if road network loads
            congestion_engine=None,
            traffic_lights=None,
            time_rules=None,
            use_dark_theme=vc.get('use_dark_theme', False),
        )

    def _init_road_network(self):
        """Initialize road network based on configured provider."""
        provider = getattr(settings, 'ROAD_NETWORK_PROVIDER', 'osmnx')

        if provider == 'amap':
            self._init_amap_road_network()
        else:
            self._init_osmnx_road_network()

    def _init_osmnx_road_network(self):
        """初始化 OSMnx 路网。失败时回退到欧氏距离。"""
        print("=" * 40)
        print("[OSM] 正在下载 OSM 路网数据...")
        print("=" * 40)

        from src.map.road_network import RoadNetwork, VehicleProfile
        rn_config = settings.ROAD_NETWORK_CONFIG
        vp = settings.VEHICLE_PROFILE

        self.vehicle_profile = VehicleProfile(
            max_height_m=vp['max_height'],
            max_weight_t=vp['max_weight'],
            max_width_m=vp['max_width'],
            min_highway_class=vp['min_highway_class'],
            penalize_low_class_roads=vp['penalize_low_class_roads'],
        )

        self.road_network = RoadNetwork(cache_dir='.cache')

        try:
            if rn_config.get('use_bbox') and rn_config.get('bbox'):
                b = rn_config['bbox']
                print(f"  使用边界框: N={b[0]} S={b[1]} E={b[2]} W={b[3]}")
                self.road_network.download_by_bbox(
                    north=b[0], south=b[1], east=b[2], west=b[3],
                    network_type=rn_config['network_type'],
                    cache=rn_config['cache_graph'],
                )
                print(f"[OK] OSM 路网加载成功 (bbox)")
            else:
                self.road_network.download(
                    place_name=rn_config['place_name'],
                    network_type=rn_config['network_type'],
                    simplify=rn_config['simplify'],
                    retain_all=rn_config['retain_all'],
                    truncate_by_edge=rn_config.get('truncate_by_edge', True),
                    cache=rn_config['cache_graph'],
                )
                print(f"[OK] OSM 路网加载成功")

            self.road_network.filter_for_vehicle(self.vehicle_profile)
            info = self.road_network.info()
            print(f"  节点: {info.get('nodes','?')}, 边: {info.get('edges','?')}")

            self._init_dynamic_modules()
        except Exception as e:
            print(f"[WARN] OSM 下载失败: {e}")
            print("  回退到欧氏距离")
            self.use_road_network = False
            self.road_network = None

    def _init_amap_road_network(self):
        """初始化高德地图 Web API 路网。"""
        print("=" * 40)
        print("[高德] 正在初始化高德地图 API...")
        print("=" * 40)

        from src.map.amap_network import AmapRoadNetwork
        amap_cfg = settings.AMAP_CONFIG

        api_key = amap_cfg.get('api_key', '')
        if not api_key or api_key == 'YOUR_AMAP_KEY':
            print("[WARN] 高德 API Key 未配置")
            print("  请在 config/settings.py 中设置 AMAP_CONFIG['api_key']")
            print("  回退到欧氏距离")
            self.use_road_network = False
            self.road_network = None
            return

        try:
            self.road_network = AmapRoadNetwork(
                api_key=api_key,
                cache_dir=amap_cfg.get('cache_dir', '.cache/amap'),
                strategy=amap_cfg.get('strategy', 0),
                max_retries=amap_cfg.get('max_retries', 3),
                retry_delay=amap_cfg.get('retry_delay', 1.0),
                rate_limit_interval=amap_cfg.get('rate_limit_interval', 0.05),
            )
            self.vehicle_profile = None
            print(f"[OK] 高德地图 API 初始化成功 (策略={amap_cfg.get('strategy', 0)})")

            self.enhanced_renderer.road_network = self.road_network
            self.enhanced_renderer.has_road_network = True

            # Phase 1.1: 注入ML数据采集器 -> 每次API调用自动记录训练样本
            try:
                from src.ml.data_collector import TravelTimeDataCollector
                self.travel_time_collector = TravelTimeDataCollector()
                # 检查是否为模拟采集模式 (--fast)
                if getattr(settings, 'SIMULATED_COLLECTION', False):
                    sim_time = getattr(settings, 'DEFAULT_SIMULATION_TIME', 36000)
                    self.travel_time_collector.set_real_time_mode(False, sim_time)
                    print(f"  [INFO] 模拟采集模式: real_time=False, "
                          f"sim_time={sim_time//3600:02d}:{(sim_time%3600)//60:02d}")
                self.road_network._collector = self.travel_time_collector
                stats = self.travel_time_collector.stats()
                print(f"[OK] ML数据采集器就绪 (已有 {stats['total']} 条样本)")
            except Exception as e:
                print(f"[INFO] ML数据采集器未启用: {e}")

            # Phase 1.1: ML采集器已注入, 模型延迟到数据加载后初始化
            self.travel_time_predictor = None  # 将在 calculate_optimal_routes 中加载
            print("[INFO] ML数据采集器已启动, 模型在数据加载后自动初始化")
        except Exception as e:
            print(f"[WARN] 高德初始化失败: {e}")
            print("  回退到欧氏距离")
            self.use_road_network = False
            self.road_network = None

    def _init_dynamic_modules(self):
        """Phase 2: Initialize traffic simulation modules (OSMnx only)."""
        if getattr(settings, 'ROAD_NETWORK_PROVIDER', 'osmnx') == 'amap':
            return  # Amap handles traffic server-side, synthetic fallback in renderer
        if not self.road_network or self.road_network.graph is None:
            return

        print("=" * 40)
        print("[交通] 初始化动态交通模块...")
        print("=" * 40)

        try:
            from src.traffic.congestion_engine import CongestionEngine
            from src.traffic.traffic_lights import TrafficLightModel
            from src.traffic.time_rules import TimeBasedAccessRules

            cc = settings.CONGESTION_CONFIG
            tlc = settings.TRAFFIC_LIGHT_CONFIG

            self.congestion_engine = CongestionEngine(
                self.road_network.graph,
                rush_hour_multiplier=cc['rush_hour_multiplier'],
                random_seed=cc['random_seed'],
            )
            print(f"  拥堵引擎: {len(self.road_network.graph.edges())} 条边已建模")

            self.traffic_lights = TrafficLightModel(
                self.road_network.graph,
                cycle_length_seconds=tlc['cycle_length_seconds'],
                green_ratio=tlc['green_ratio'],
                yellow_seconds=tlc['yellow_seconds'],
            )
            lights_count = len(self.traffic_lights.get_intersections())
            print(f"  交通信号灯: {lights_count} 个受控路口")

            self.time_rules = TimeBasedAccessRules()
            print(f"  时间规则: {len(self.time_rules.rules)} 条限行规则")

            # Wire traffic modules into the already-created enhanced renderer
            self.enhanced_renderer.road_network = self.road_network
            self.enhanced_renderer.congestion_engine = self.congestion_engine
            self.enhanced_renderer.traffic_lights = self.traffic_lights
            self.enhanced_renderer.time_rules = self.time_rules
            self.enhanced_renderer.has_road_network = True
            self.enhanced_renderer.has_congestion = True
            self.enhanced_renderer.has_lights = True
            self.enhanced_renderer.has_time_rules = True

            print("[OK] 动态交通模块初始化完成")
        except Exception as e:
            print(f"[WARN] 动态模块初始化失败: {e}")
            self.congestion_engine = None
            self.traffic_lights = None
            self.time_rules = None

    def load_data(self, filepath: str = None, use_sample: bool = False, num_points: int = 10) -> bool:
        """
        加载数据
        """
        if use_sample:
            filepath = None  # 强制使用内置示例数据

        if filepath:
            self._data_filepath = filepath  # 记录原始路径, 用于输出文件命名
            print(f"从文件加载数据: {os.path.abspath(filepath)}")
            try:
                data_dict = load_all_data(filepath)
                
                self.points = data_dict['coords'] # [(x, y), ...]
                self.demands = data_dict['demands']
                self.time_windows = data_dict['time_windows']
                self.priorities = data_dict['priorities']
                self.ids = data_dict['ids']
                self.n_points = data_dict['n']
                
                sp_calc = ShortestPathCalculator(
                    self.points,
                    road_network=self.road_network if self.use_road_network else None
                )
                self.distance_matrix = sp_calc.distance_matrix

                print(f"成功加载 {self.n_points} 个点的数据。")
                print(f"距离矩阵形状: {self.distance_matrix.shape} ("
                      f"{_road_label()})")
                return True
            except FileNotFoundError as e:
                print(f"文件未找到: {e}")
                return False
            except ValueError as e:
                print(f"数据格式错误: {e}")
                return False
            except Exception as e:
                print(f"加载数据时发生未知错误: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            # 内置示例数据
            self._data_filepath = "示例数据"  # 用于输出文件命名
            print("使用内置示例数据...")
            sample_coords = [
                (121.4737, 31.2304),  # 配送中心
                (121.4888, 31.2387),
                (121.4602, 31.2249),
                (121.4950, 31.2195),
                (121.4523, 31.2456),
                (121.5023, 31.2312)
            ]
            sample_demands = [0, 2, 3, 1, 2, 4]
            
            self.points = sample_coords
            self.demands = sample_demands
            n = len(sample_coords)
            # 模拟真实配送场景: 仓库全天开放, 各配送点有不同时间窗
            self.time_windows = [
                (0, 86400),       # 配送中心 (全天)
                (28800, 43200),   # 8:00-12:00 上午配送
                (36000, 50400),   # 10:00-14:00 中午配送
                (28800, 43200),   # 8:00-12:00
                (43200, 57600),   # 12:00-16:00 下午配送
                (36000, 54000),   # 10:00-15:00
            ]
            self.priorities = [1] * n
            self.ids = list(range(n))
            self.n_points = n
            
            sp_calc = ShortestPathCalculator(
                self.points,
                road_network=self.road_network if self.use_road_network else None
            )
            self.distance_matrix = sp_calc.distance_matrix

            print(f"示例数据加载完成。距离矩阵形状: {self.distance_matrix.shape} "
                  f"({_road_label()})")
            return True
    
    def calculate_optimal_routes(self, num_vehicles: int = None, algorithm: str = 'ortools') -> bool:
        """
        计算最优配送路线
        """
        self._algorithm = algorithm  # 记录算法名, 用于输出文件命名

        if not self.points or self.distance_matrix is None:
            print("错误: 请先加载数据")
            return False
        
        if num_vehicles is None:
            num_vehicles = settings.DEFAULT_NUM_VEHICLES
        
        max_vehicles = max(1, len(self.points) - 1)
        num_vehicles = min(num_vehicles, max_vehicles)

        # ── 容量可行性检查 ──
        total_demand = sum(self.demands) if self.demands else 0
        total_capacity = num_vehicles * settings.VEHICLE_CAPACITY
        if total_demand > total_capacity:
            min_v = (total_demand + settings.VEHICLE_CAPACITY - 1) // settings.VEHICLE_CAPACITY
            min_v = min(min_v, max_vehicles)
            print(f"  [WARN] 总需求({total_demand}) > 总容量({total_capacity}), "
                  f"自动增加车辆: {num_vehicles} -> {min_v}")
            num_vehicles = min_v

        print(f"使用 {algorithm} 算法计算 {num_vehicles} 辆车的最优路线...")

        # Phase 1.1: 延迟加载ML模型 (此时 self.points 已就绪)
        if self.travel_time_predictor is None and self.use_road_network:
            try:
                from src.ml.travel_time_predictor import TravelTimePredictor, detect_city_from_points
                city = detect_city_from_points(self.points) if self.points else "Shanghai"
                self.travel_time_predictor = TravelTimePredictor.load_or_fallback(
                    city=city, points=self.points
                )
                if self.travel_time_predictor.is_trained:
                    m = self.travel_time_predictor.metrics
                    print(f"  [OK] ML行程时间模型 ({city}) RMSE={m.get('rmse_seconds','?')}s, "
                          f"vs定速提升{m.get('improvement_pct','?')}%")
                else:
                    print(f"  [INFO] {city} 模型未训练, 定速回退 (40km/h)")
            except Exception as e:
                print(f"  [INFO] ML模型加载失败: {e}")

        # Compute travel-time matrix from congestion models
        sim_time = getattr(settings, 'DEFAULT_SIMULATION_TIME', 36000)
        self.sim_time = sim_time  # 保存以便后续使用
        clean_dist = self.distance_matrix.copy()
        travel_time_sec = self._compute_travel_time_matrix(clean_dist, sim_time)

        from src.algorithms.vehicle_types import Fleet, VehicleType
        fleet = Fleet(
            types=[VehicleType(**t) for t in settings.VEHICLE_TYPES],
            max_total=num_vehicles,
        )
        self.vrp_solver = VRPSolver(fleet=fleet, depot_index=self.depot_index)

        if algorithm == 'ortools':
            result = self.vrp_solver.solve_with_ortools(
                distance_matrix=clean_dist.tolist(),
                demands=self.demands,
                time_limit=settings.TIME_LIMIT_SECONDS,
                time_windows=self.time_windows,
                travel_time_predictor=self.travel_time_predictor,
                points=self.points,
                sim_time_seconds=sim_time,
                travel_time_matrix=travel_time_sec.tolist(),
            )
            routes = result.routes if result else None
            # 仅在真正无解 (result is None) 时放宽约束重试
            # overflow 是容量不足, 更长时间搜索无法解决, 直接接受部分解
            if result is None:
                tw_info = self._check_time_window_feasibility() if self.time_windows else "未知"
                print(f"  [WARN] 无解 ({tw_info}), 放宽时间窗重试...")
                result = self.vrp_solver.solve_with_ortools(
                    distance_matrix=clean_dist.tolist(),
                    demands=self.demands,
                    time_limit=settings.TIME_LIMIT_SECONDS,
                    time_windows=None,
                    travel_time_predictor=self.travel_time_predictor,
                    points=self.points,
                    sim_time_seconds=sim_time,
                    travel_time_matrix=travel_time_sec.tolist(),
                )
                routes = result.routes if result else None
                if result is not None:
                    self._time_windows_relaxed = True
            elif result.is_overflow:
                print(f"  [INFO] {len(result.overflow_nodes)} 个点容量溢出, "
                      f"增加车辆数或调整需求可消除")
        elif algorithm == 'greedy':
            result = self.vrp_solver.greedy_vrp_solver(
                distance_matrix=clean_dist.tolist(),
                demands=self.demands,
            )
            routes = result.routes
        elif algorithm == 'cluster':
            result = self.vrp_solver.cluster_first_route_second(
                distance_matrix=clean_dist.tolist(),
                demands=self.demands,
                points=self.points,
            )
            routes = result.routes
        else:
            print(f"警告: 未知算法 '{algorithm}'，使用OR-Tools算法")
            result = self.vrp_solver.solve_with_ortools(
                distance_matrix=clean_dist.tolist(),
                demands=self.demands,
                time_limit=settings.TIME_LIMIT_SECONDS,
                time_windows=self.time_windows,
                travel_time_predictor=self.travel_time_predictor,
                points=self.points,
                sim_time_seconds=sim_time,
                travel_time_matrix=travel_time_sec.tolist(),
            )
            routes = result.routes if result else None
            if result is None:
                print("  -> 无解, 放宽约束重试...")
                result = self.vrp_solver.solve_with_ortools(
                    distance_matrix=clean_dist.tolist(),
                    demands=self.demands,
                    time_limit=settings.TIME_LIMIT_SECONDS,
                    time_windows=None,
                    travel_time_predictor=self.travel_time_predictor,
                    points=self.points,
                    sim_time_seconds=sim_time,
                    travel_time_matrix=travel_time_sec.tolist(),
                )
                routes = result.routes if result else None
                if result is not None:
                    self._time_windows_relaxed = True
            elif result.is_overflow:
                print(f"  [INFO] {len(result.overflow_nodes)} 个点容量溢出")

        if routes and self._validate_routes(routes):
            self.routes = routes
            self._vrp_result = result  # 保存 UnifiedSolution 用于显示
            metrics = self.vrp_solver.calculate_route_metrics(
                routes, self.distance_matrix.tolist(), self.demands)
            provider = getattr(settings, 'ROAD_NETWORK_PROVIDER', 'osmnx')
            dist_label = '高德路网' if provider == 'amap' else 'OSM路网' if self.use_road_network else '欧氏距离'

            # 异构车队显示
            if result and hasattr(result, 'vehicles_used') and result.vehicles_used:
                active_vehicles = [u for u in result.vehicles_used if u['is_active']]
                print(f"路线优化完成!")
                print(f"  总行驶距离: {result.total_distance_km:.2f} km  ({dist_label})")
                print(f"  车辆启用:   {len(active_vehicles)}/{result.num_available} 辆")
                print(f"  经济总代价: {result.total_cost_km_eq:.1f} km基准 (中型车=1.0×, 小车更便宜)")
                if result.is_overflow:
                    print(f"  [WARN] 溢出! 未分配节点: {result.overflow_nodes}")
                for u in active_vehicles:
                    print(f"  车辆{u['route_id']+1} ({u.get('vehicle_type','?')}): "
                          f"{u['stops']}站, {u['distance_km']:.1f}km, "
                          f"载重{u['load']}/{u['capacity']}, "
                          f"代价({u['fixed_cost_km']:.0f}+{u['running_cost_km']:.0f})km基准")
            else:
                vehicles_used = sum(1 for r in routes if len(r) > 1)
                total_dist = metrics['total_distance']
                print(f"路线优化完成!")
                print(f"  总行驶距离: {total_dist:.2f} km  ({dist_label})")
                print(f"  车辆启用:   {vehicles_used}/{len(routes)} 辆")
                for ri in metrics.get('routes_info', []):
                    if ri['num_stops'] == 0:
                        continue
                    print(f"  车辆 {ri['route_id']+1}: {ri['num_stops']} 站, {ri['distance']:.2f} km, "
                          f"载重 {ri['demand']}/{settings.VEHICLE_CAPACITY}")

            # Phase 2: Simulate traffic incidents (~10% probability per site)
            self.incidents = []
            import random, time as _time
            cc = settings.CONGESTION_CONFIG
            seed = cc.get('random_seed')
            if seed is None:
                seed = int(_time.time() * 1000) % (2**31)
            rng = random.Random(seed)
            prob = cc.get('accident_probability', 0.10)
            max_sites = cc.get('n_random_accidents', 2)

            if self.congestion_engine and max_sites > 0:
                # OSM mode: place on random graph edges
                for _ in range(max_sites):
                    if rng.random() > prob:
                        continue
                    try:
                        incidents = self.congestion_engine.simulate_accidents(1)
                        if incidents:
                            self.incidents.extend(incidents)
                    except Exception:
                        pass
            elif max_sites > 0:
                # Amap / non-OSM mode: place on route segments with actual polyline coords
                active_routes = [r for r in routes if len(r) > 2]
                for _ in range(max_sites):
                    if rng.random() > prob:
                        continue
                    if not active_routes:
                        break
                    route = rng.choice(active_routes)
                    seg_idx = rng.randint(0, len(route) - 2)
                    a, b = route[seg_idx], route[seg_idx + 1]
                    start_t = rng.randint(21600, 72000)
                    duration = rng.randint(1800, 7200)
                    mult = round(rng.uniform(3.0, 5.0), 1)

                    # Get Amap polyline coords for this segment
                    mid_latlon = None
                    polyline_coords = None  # full road path for drawing
                    if self.road_network and hasattr(self.road_network, 'get_driving_info'):
                        try:
                            is_lonlat = self.road_network._detect_coordinate_format(self.points)
                            if is_lonlat:
                                orig = (float(self.points[a][0]), float(self.points[a][1]))
                                dest = (float(self.points[b][0]), float(self.points[b][1]))
                            else:
                                orig = (float(self.points[a][1]), float(self.points[a][0]))
                                dest = (float(self.points[b][1]), float(self.points[b][0]))
                            info = self.road_network.get_driving_info(orig, dest)
                            poly = self.road_network.decode_polyline(info.get('polyline', ''))
                            if poly and len(poly) > 2:
                                mid = poly[len(poly) // 2]  # midpoint (lon, lat)
                                mid_latlon = (mid[1], mid[0])  # (lat, lon)
                                # Convert entire polyline to (lat, lon) for Folium
                                polyline_coords = [(p[1], p[0]) for p in poly]
                        except Exception:
                            pass

                    self.incidents.append({
                        'route_segment': (a, b),
                        'start': start_t, 'end': start_t + duration,
                        'multiplier': mult, 'type': 'accident',
                        'midpoint': mid_latlon,
                        'polyline_coords': polyline_coords,
                    })

            if self.incidents:
                print(f"[交通事故] 注入了 {len(self.incidents)} 个事故 (概率={prob:.0%}):")
                for inc in self.incidents:
                    sh, r = divmod(inc['start'], 3600); sm, _ = divmod(r, 60)
                    eh, r = divmod(inc['end'], 3600); em, _ = divmod(r, 60)
                    seg = inc.get('route_segment')
                    loc = inc.get('midpoint')
                    if seg:
                        loc_str = f"路段 {seg[0]}->{seg[1]}"
                        if loc:
                            loc_str += f" ({loc[0]:.4f}, {loc[1]:.4f})"
                        print(f"  - {loc_str}: {int(sh):02d}:{int(sm):02d} - "
                              f"{int(eh):02d}:{int(em):02d}, 拥堵 {inc['multiplier']}倍")
                    else:
                        hw = inc.get('highway', '?')
                        print(f"  - {hw}: {int(sh):02d}:{int(sm):02d} - "
                              f"{int(eh):02d}:{int(em):02d}, 拥堵 {inc['multiplier']}倍")

            return True
        
        print("路线计算失败或无效")
        return False

    def _check_time_window_feasibility(self) -> str:
        """
        诊断时间窗口为什么不可行。
        返回可读的诊断字符串。
        """
        sim_time = getattr(settings, 'DEFAULT_SIMULATION_TIME', 36000)
        parts = []
        h, m = divmod(int(sim_time), 3600)
        parts.append(f"模拟时间 {h:02d}:{m:02d}")

        if self.time_windows:
            passed = 0
            tight = 0
            for i, tw in enumerate(self.time_windows):
                if i == self.depot_index:
                    continue
                span = tw[1] - tw[0]
                if tw[1] <= sim_time:
                    passed += 1
                elif span < 3600:
                    tight += 1
            if passed > 0:
                parts.append(f"{passed}个点的时间窗已过")
            if tight > 0:
                parts.append(f"{tight}个点的时间窗不足1小时")

        # 检查容量是否足够
        total_demand = sum(self.demands) if self.demands else 0
        total_capacity = settings.VEHICLE_CAPACITY * settings.DEFAULT_NUM_VEHICLES
        if total_demand > total_capacity:
            parts.append(f"总需求({total_demand})>总容量({total_capacity})")

        return "; ".join(parts) if parts else "未知原因"

    def _validate_routes(self, routes: List[List[int]]) -> bool:
        """
        验证路线的有效性
        """
        if not routes:
            return False
        
        for route in routes:
            if not route or self.depot_index not in route:
                print(f"路线中缺少配送中心: {route}")
                return False
        
        all_visited = set()
        for route in routes:
            all_visited.update(route)
        
        if len(all_visited) < len(self.points):
            missing_points = set(range(len(self.points))) - all_visited
            depot_missing = self.depot_index in missing_points
            if depot_missing:
                print(f"错误: 配送中心未出现在路线中")
                return False
            print(f"警告: {len(missing_points)} 个点未被访问: {sorted(missing_points)} (容量不足, 请增加车辆)")
            # 允许部分分配结果通过 (溢出标记由 UnifiedSolution.is_overflow 传递)
        
        return True

    def _compute_travel_time_matrix(self, dist_matrix, sim_time_seconds):
        """双模型融合拥堵预测 → 行程时间矩阵 (秒)。

        广义成本语义:
          arc_cost = travel_time_sec × (cost_per_km × AVG_SPEED / 3600)
        即「时间价值」换算为货币成本，等价于经济学广义成本函数
        generalized_cost = α·distance + β·travel_time 在 α=0 时的特例。

        LightGBM (39特征, 批量训练, RMSE~60s) 提供基础乘数,
        自适应模型 (20特征, 在线SGD) 提供残差修正。
        两者加权融合: 70% LightGBM + 30% 自适应。

        Returns:
            np.ndarray: N×N 行程时间矩阵 (秒), 对角线为 0
        """
        import numpy as np
        from config import settings

        t = sim_time_seconds % 86400
        n = dist_matrix.shape[0]
        hour = int(t // 3600)
        # 使用模拟时间对应的星期几, 而非真实系统时间,
        # 保证模拟时间模式下的语义一致性
        sim_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        day_of_week = sim_date.weekday()
        travel_time_sec = np.zeros_like(dist_matrix)

        # ── 初始化/加载自适应预测器 ──
        adaptive_enabled = getattr(settings, 'ADAPTIVE_CONFIG', {}).get('enabled', True)
        if adaptive_enabled and self.congestion_predictor is None and self.points:
            try:
                from src.ml.travel_time_predictor import detect_city_from_points
                from src.adaptive.congestion_predictor import AdaptiveCongestionPredictor
                city = detect_city_from_points(self.points)
                self.congestion_predictor = AdaptiveCongestionPredictor.load_or_create(
                    city=city, points=self.points
                )
                if self.congestion_predictor.is_trained:
                    m = self.congestion_predictor.metrics
                    loss_str = f"{m['running_loss']:.4f}" if m['running_loss'] else "N/A"
                    print(f"  [OK] 自适应预测器 ({city}) "
                          f"updates={m['updates']} loss={loss_str}")
                else:
                    print(f"  [INFO] 自适应预测器 ({city}) 未训练, 在线学习模式")
            except Exception as e:
                print(f"  [WARN] 自适应预测加载失败: {e}")
                self.congestion_predictor = None

        # ── 模型可用性 ──
        has_lgb = (self.travel_time_predictor is not None
                   and self.travel_time_predictor.is_trained)
        has_adaptive = (self.congestion_predictor is not None
                        and self.congestion_predictor.is_trained)

        # ── 逐边预测拥堵乘数 ──
        # 与模型 sigmoid 输出范围 [0.5, 5.0] 对齐, 避免训练-推理不一致
        MULT_MIN, MULT_MAX = 0.5, 5.0
        all_factors = []
        lgb_factors = []
        adaptive_factors = []

        FALLBACK_SPEED = 40.0  # km/h 定速基线

        for i in range(n):
            for j in range(n):
                if i == j or dist_matrix[i][j] <= 0:
                    travel_time_sec[i][j] = 0.0
                    continue

                d_km = dist_matrix[i][j]

                # 尝试从 Amap 缓存获取 polyline (热缓存, O(1) 内存查找)
                polyline_str = ""
                if self.road_network and hasattr(self.road_network, 'cache'):
                    cached = self.road_network.cache.get(
                        "road_driving", self.points[i], self.points[j],
                        self.road_network.strategy,
                    )
                    if cached:
                        polyline_str = cached.get("polyline", "")

                # LightGBM 乘数: 预测行程时间 / 定速时间
                lgb_mult = 1.0
                if has_lgb:
                    try:
                        lgb_sec = self.travel_time_predictor.predict(
                            self.points[i], self.points[j],
                            d_km, hour, day_of_week,
                        )
                        lgb_mult = lgb_sec / max(d_km / FALLBACK_SPEED * 3600, 1.0)
                        lgb_mult = max(MULT_MIN, min(MULT_MAX, lgb_mult))
                    except Exception:
                        lgb_mult = 1.0

                # 自适应乘数
                adaptive_mult = 1.0
                if has_adaptive:
                    try:
                        adaptive_mult = self.congestion_predictor.predict(
                            self.points[i], self.points[j],
                            d_km, hour, day_of_week,
                            polyline_str=polyline_str,
                        )
                        adaptive_mult = max(MULT_MIN, min(MULT_MAX, adaptive_mult))
                    except Exception:
                        adaptive_mult = 1.0

                # ── 融合策略 ──
                if has_lgb and has_adaptive:
                    multiplier = 0.7 * lgb_mult + 0.3 * adaptive_mult
                elif has_lgb:
                    multiplier = lgb_mult
                elif has_adaptive:
                    multiplier = adaptive_mult
                else:
                    multiplier = 1.0

                # 行程时间 = 定速基准时间 × 拥堵乘数 (单位: 秒)
                travel_time_sec[i][j] = d_km / FALLBACK_SPEED * 3600 * multiplier
                all_factors.append(multiplier)
                if has_lgb:
                    lgb_factors.append(lgb_mult)
                if has_adaptive:
                    adaptive_factors.append(adaptive_mult)

        # ── 诊断输出 ──
        avg_factor = np.mean(all_factors) if all_factors else 1.0
        h, m = divmod(int(t), 3600)

        if has_lgb and has_adaptive:
            source = "LightGBM×70% + 自适应×30%"
        elif has_lgb:
            source = "LightGBM"
        elif has_adaptive:
            source = "自适应"
        else:
            source = "定速(无模型)"

        detail_parts = [f"平均系数 {avg_factor:.2f}x"]
        if lgb_factors:
            detail_parts.append(f"LGB={np.mean(lgb_factors):.2f}x")
        if adaptive_factors:
            detail_parts.append(f"Adaptive={np.mean(adaptive_factors):.2f}x")
        print(f"  拥堵预测: {source} (时间 {h:02d}:{m:02d}, {', '.join(detail_parts)})")

        # ── 在线更新: 用高德 API 真实数据更新自适应模型 ──
        if (self.congestion_predictor and self.travel_time_collector):
            try:
                from src.ml.travel_time_predictor import detect_city_from_points
                city = detect_city_from_points(self.points)
                new_samples = self.travel_time_collector.get_new_samples(city)
                if new_samples:
                    for s in new_samples:
                        self.congestion_predictor.update(
                            (s["origin_lon"], s["origin_lat"]),
                            (s["dest_lon"], s["dest_lat"]),
                            s["dist_km"], s["duration_seconds"],
                            s["hour"], s["day_of_week"],
                            polyline_str=s.get("polyline", ""),
                        )
                    self.congestion_predictor.save(city)
                    adaptive_loss = self.congestion_predictor.metrics['running_loss']
                    loss_str = f"{adaptive_loss:.4f}" if adaptive_loss else "N/A"
                    print(f"  [自适应] 在线更新: {len(new_samples)} 条样本, "
                          f"loss={loss_str}")
            except Exception:
                pass  # 在线更新失败不影响主流程

        return travel_time_sec

    def visualize_routes(self, output_dir: str = "output") -> Dict[str, str]:
        """
        可视化路线
        """
        if not self.routes or not self.points:
            print("错误: 请先计算路线")
            return {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 构建文件名前缀: 从 filepath 提取 CSV 文件名 (无扩展名), 兜底用 "demo"
        raw_path = getattr(self, '_data_filepath', 'demo')
        csv_stem = os.path.splitext(os.path.basename(raw_path))[0] if raw_path != 'demo' else 'demo'
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        algo_display = {
            'ortools': 'ORTools', 'greedy': 'Greedy', 'cluster': 'Cluster'
        }.get(getattr(self, '_algorithm', ''), getattr(self, '_algorithm', 'ortools'))

        files = {}
        
        try:
            # 0. 生成路口红绿灯数据 (高德 polyline -> 路口检测 -> 信号灯模拟)
            sim_time = getattr(settings, 'DEFAULT_SIMULATION_TIME', 36000)
            self.traffic_lights_data = None
            if self.use_road_network and self.road_network is not None:
                print("检测路口红绿灯...")
                try:
                    from src.traffic.route_traffic_lights import generate_route_traffic_lights
                    tld_cfg = getattr(settings, 'TRAFFIC_LIGHT_DETECTION', {})
                    seed = int(datetime.now().timestamp() * 1000) % (2**31)
                    self.traffic_lights_data = generate_route_traffic_lights(
                        routes=self.routes,
                        points=self.points,
                        road_network=self.road_network,
                        sim_time_seconds=sim_time,
                        min_spacing_m=tld_cfg.get('min_spacing_m', 800),
                        max_spacing_m=tld_cfg.get('max_spacing_m', 1500),
                        turn_threshold_deg=tld_cfg.get('turn_threshold_deg', 50),
                        max_lights_per_route=tld_cfg.get('max_lights_per_route', 15),
                        seed=seed,
                    )
                    s = self.traffic_lights_data['summary']
                    print(f"  检测到 {s['total_lights']} 个路口, "
                          f"{s['red_lights']} 个红灯, "
                          f"总等待 {s['total_wait_seconds']:.0f} 秒")
                    for vid, count in sorted(s.get('per_vehicle', {}).items()):
                        reds = s.get('per_vehicle_red', {}).get(vid, 0)
                        print(f"    V{vid+1}: {count} 个路口, {reds} 个红灯")
                except Exception as e:
                    print(f"  红绿灯检测失败: {e}")
                    import traceback
                    traceback.print_exc()

            # 1. 生成Folium地图
            print("生成Folium地图...")
            folium_points = self.points

            if self.enhanced_renderer is not None:
                # Phase 3: Digital twin rendering
                print("  使用数字孪生增强渲染器...")
                folium_map = self.enhanced_renderer.create_digital_twin_map(
                    routes=self.routes,
                    points=folium_points,
                    depot_index=self.depot_index,
                    point_names=[f"P{i}" for i in range(len(self.points))],
                    timestamp_seconds=sim_time,
                    title=f"数字孪生 - 配送路线 - {timestamp}{' [时间窗已放宽]' if self._time_windows_relaxed else ''}",
                    incidents=getattr(self, 'incidents', []),
                    distance_matrix=self.distance_matrix if self.distance_matrix is not None else None,
                    traffic_lights_data=self.traffic_lights_data,
                )
            else:
                # Phase 1: Standard rendering
                folium_map = self.folium_visualizer.create_complete_map(
                    points=folium_points,
                    routes=self.routes,
                    depot_index=self.depot_index,
                    point_names=[f"Point_{i}" for i in range(len(self.points))],
                    route_colors=settings.VISUALIZATION_CONFIG['route_colors'],
                    title=f"Delivery Routes - {timestamp}",
                )
            
            folium_filename = f"{output_dir}/folium-{csv_stem}-{algo_display}-{run_ts}.html"
            self.folium_visualizer.save_map(folium_map, folium_filename)
            files['folium_map'] = folium_filename
            
            # 2. 生成Plotly动画
            print("生成Plotly动画...")
            plotly_fig = self.plotly_animator.create_route_animation(
                points=self.points,
                routes=self.routes,
                point_names=[f"P{i}" for i in range(len(self.points))],
                title="VRP 配送路线动态演示",
                road_network=self.road_network if self.use_road_network else None,
            )
            
            plotly_filename = f"{output_dir}/plotly-{csv_stem}-{algo_display}-{run_ts}.html"
            self.plotly_animator.save_animation(plotly_fig, plotly_filename)
            files['plotly_animation'] = plotly_filename
            
            # 3. 生成指标仪表盘
            print("生成指标仪表盘...")
            metrics_fig = self.plotly_animator.create_metrics_dashboard(
                routes=self.routes,
                distance_matrix=self.distance_matrix.tolist(),
                demands=self.demands,
                title=f"Delivery Metrics Dashboard - {timestamp}",
                road_network=self.road_network if self.use_road_network else None,
                points=self.points,
                sim_time=sim_time,
                traffic_lights_data=self.traffic_lights_data,
            )
            
            metrics_filename = f"{output_dir}/metrics-{csv_stem}-{algo_display}-{run_ts}.html"
            metrics_fig.write_html(metrics_filename)
            files['metrics_dashboard'] = metrics_filename
            
            # 4. 保存结果到JSON
            print("保存结果数据...")
            result_data = {
                'timestamp': timestamp,
                'num_vehicles': len(self.routes),
                'num_points': len(self.points),
                'simulation_time': sim_time,
                'simulation_time_str': f"{sim_time//3600:02d}:{(sim_time%3600)//60:02d}",
                'routes': self.routes,
                'total_distance': self.vrp_solver.calculate_total_distance(
                    self.routes, self.distance_matrix.tolist()
                ),
                'demands': self.demands,
                'metrics': self.vrp_solver.calculate_route_metrics(
                    self.routes, self.distance_matrix.tolist(), self.demands
                ),
                'traffic_lights': self.traffic_lights_data,
                'time_windows_relaxed': self._time_windows_relaxed,
                'incidents': getattr(self, 'incidents', []),
            }
            
            result_filename = f"{output_dir}/result-{csv_stem}-{algo_display}-{run_ts}.json"
            with open(result_filename, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            files['results_json'] = result_filename
            
            print("可视化完成！生成的文件:")
            for name, path in files.items():
                print(f"  - {name}: {path}")
            
            return files
            
        except Exception as e:
            print(f"可视化过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_full_pipeline(self, filepath: str = None, num_vehicles: int = None, 
                         algorithm: str = 'ortools', output_dir: str = "output") -> bool:
        """
        运行完整流程
        """
        print("=" * 60)
        print("启动城市快递路径智能优化系统")
        print("=" * 60)
        
        # 1. 加载数据
        print("\n" + "-" * 40)
        print("1. 加载配送点数据")
        print("-" * 40)
        if self.use_road_network:
            provider = getattr(settings, 'ROAD_NETWORK_PROVIDER', 'osmnx')
            if provider == 'amap':
                print("  路径引擎: 高德地图 API")
            else:
                print("  路径引擎: OSM 路网")
        else:
            print("  路径引擎: 欧氏距离")
        if not self.load_data(filepath=filepath, use_sample=not filepath):
            return False
        
        print(f"数据摘要:")
        print(f"  - 总点数: {self.n_points}")
        print(f"  - 配送点数: {self.n_points - 1}")
        print(f"  - 总需求量: {sum(self.demands)}")
        if self.points:
            lons, lats = zip(*self.points)
            print(f"  - 经度范围: ({min(lons):.4f}, {max(lons):.4f})")
            print(f"  - 纬度范围: ({min(lats):.4f}, {max(lats):.4f})")
        
        # 2. 计算最优路线
        print("\n" + "-" * 40)
        print("2. 计算最优配送路线")
        print("-" * 40)
        if not self.calculate_optimal_routes(num_vehicles=num_vehicles, algorithm=algorithm):
            return False
        
        # 3. 可视化结果
        print("\n" + "-" * 40)
        print("3. 生成可视化结果")
        print("-" * 40)
        files = self.visualize_routes(output_dir=output_dir)
        
        if not files:
            return False
        
        print("\n" + "=" * 60)
        print("[OK] 系统运行完成!")
        print("=" * 60)
        
        return True

def main():
    """主函数"""
    print("检查可用数据文件...")
    available_files = list_available_data_files()
    print(f"找到 {len(available_files)} 个文件: {available_files}")

    # 使用纯文件名，load_all_data 内部会自动拼接 data/ 目录
    filepath = "校内游览.csv"

    # 创建系统实例并运行
    system = DeliverySystem()
    success = system.run_full_pipeline(
        filepath=filepath,
        num_vehicles=3,      # 示例：使用3辆车
        algorithm='ortools',
        output_dir="output"
    )
    
    if success:
        print("\n[OK] 系统成功运行! 请查看 output 目录中的结果文件。")
    else:
        print("\n[ERROR] 系统运行失败! 请检查错误信息。")

if __name__ == "__main__":
    main()