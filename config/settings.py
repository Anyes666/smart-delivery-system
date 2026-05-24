# 配置参数
import os

# ===================== AMAP (高德) WEB API CONFIG =====================
# API Key 优先级: 环境变量 AMAP_API_KEY > settings 中的硬编码值(仅开发用)
_amap_api_key = os.environ.get('AMAP_API_KEY', '')
if not _amap_api_key:
    _amap_api_key = ''  # 请设置环境变量 AMAP_API_KEY，或在此填入你的高德 API Key
AMAP_CONFIG = {
    'api_key': _amap_api_key,
    'base_url': 'https://restapi.amap.com/v3',
    'strategy': 0,                          # 0=fastest, 2=shortest, 5=no-highway
    'cache_dir': '.cache/amap',
    'max_retries': 3,
    'retry_delay': 1.0,
    'rate_limit_interval': 0.35,             # ~3 QPS (高德免费版并发上限)
    'assume_symmetric': True,
}
# ===================== 异构车队配置 (JT/T 1325-2020 城市绿色配送车辆) =====================
# 车型名称使用国标术语, 代码对标 JT/T 1325
VEHICLE_TYPES = [
    {'name': '微型封闭货车',  'capacity': 15,  'fixed_cost': 3000,  'cost_per_km': 500,  'code': 'V-MINI'},
    {'name': '轻型封闭货车',  'capacity': 30,  'fixed_cost': 5000,  'cost_per_km': 800,  'code': 'V-LIGHT'},
    {'name': '中型厢式货车',  'capacity': 60,  'fixed_cost': 10000, 'cost_per_km': 1000, 'code': 'V-MEDIUM'},
    {'name': '重型厢式货车',  'capacity': 120, 'fixed_cost': 18000, 'cost_per_km': 1200, 'code': 'V-HEAVY'},
]
MAX_VEHICLES = 20                   # 车队总数上限 (含所有车型)
DEFAULT_NUM_VEHICLES = 3            # 默认最大可用车辆数 (兼容旧版)
VEHICLE_CAPACITY = 60               # 默认单车容量 (兼容旧版)
VEHICLE_FIXED_COST = 10000          # 默认单车启用成本 (兼容旧版)
DEPOT_INDEX = 0
MAP_CENTER = (31.2304, 121.4737)  # (纬度, 经度) - 上海
ZOOM_LEVEL = 12
TIME_LIMIT_SECONDS = 30

# 算法参数
ALGORITHM_PARAMS = {
    'greedy': {
        'use_clustering': True,
        'cluster_method': 'kmeans'
    },
    'ortools': {
        'first_solution_strategy': 'PATH_CHEAPEST_ARC',
        'local_search_metaheuristic': 'GUIDED_LOCAL_SEARCH'
    }
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'route_colors': [
        '#E60000',  # 纯红
        '#FF00FF',  # 品红
        '#FF6600',  # 橙红
        '#CC00CC',  # 深紫
        '#0066FF',  # 深蓝
        '#CC0000',  # 暗红
        '#FF0099',  # 玫红
        '#9933FF',  # 紫罗兰
        '#FF3300',  # 朱红
        '#0099FF',  # 宝蓝
        '#CC3300',  # 铁锈红
        '#6600CC',  # 深紫罗兰
    ],
    'marker_size': 8,
    'line_width': 4,                         # Route polyline width (加粗提升可见度)
    'animation_speed': 500,  # 毫秒

    # Phase 3: Digital twin visualization
    'use_dark_theme': True,                 # Dark tile theme for Folium
    'show_congestion_heatmap': True,         # Overlay congestion colors on roads
    'show_traffic_lights': True,             # Show traffic light markers
    'show_time_rules': True,                 # Show time-restricted road overlays
    'road_following_vehicles': True,         # Vehicles follow actual road curves
    'heatmap_opacity': 0.55,                 # Congestion heatmap line opacity
    'route_line_opacity': 0.95,              # Route polyline opacity (max visibility)
}

# 地图瓦片与 Mapbox 配置
MAP_TILE = 'CartoDB positron'
DARK_TILE = 'CartoDB dark_matter'            # Phase 3: free dark tile (no API key)
MAPBOX_STYLE = 'carto-positron'
MAPBOX_TOKEN = None

# Tile provider selection
TILE_PROVIDER = 'amap'                       # 'cartodb' | 'amap'
# Amap tile URL — must use http:// (Amap servers don't have HTTPS certs
# for tile subdomains). {s} → 1,2,3,4, {x}/{y}/{z} → Leaflet tile coords.
AMAP_TILE_URL = (
    "http://webrd0{s}.is.autonavi.com/appmaptile"
    "?lang=zh_cn&size=1&scale=1&style=8"
    "&x={x}&y={y}&z={z}"
)
AMAP_TILE_SUBDOMAINS = ['1', '2', '3', '4']
AMAP_TILE_ATTR = 'Amap'

# ===================== ROAD NETWORK PROVIDER =====================
# 'osmnx' = local OpenStreetMap graph (requires osmnx, blocked in China)
# 'amap'  = high-level Amap Web API (requires API key, works in China)
ROAD_NETWORK_PROVIDER = 'amap'

# ===================== ROAD NETWORK CONFIG =====================
# Whether to use real road network routing instead of Euclidean distance.
USE_REAL_ROAD_NETWORK = True

# OSMnx download configuration
ROAD_NETWORK_CONFIG = {
    'place_name': 'Shanghai, China',      # OSMnx geocoding query (blocked in China → use bbox)
    'network_type': 'drive',             # 'drive' = motor-vehicle roads only
    'simplify': True,                    # Collapse intersection chains
    'retain_all': False,                 # Keep disconnected components
    'truncate_by_edge': True,            # Edge-based boundary truncation
    'cache_graph': True,                 # Pickle graph to disk for reuse
    'graph_cache_path': '.cache/road_graph.pkl',
    # Bounding box fallback (north, south, east, west) — bypasses geocoding
    # Shanghai center: 31.23, 121.47. Set to None to use place_name geocoding.
    'use_bbox': True,                    # Use bbox instead of geocoding (recommended in China)
    'bbox': (31.40, 31.10, 121.65, 121.30),  # Shanghai: N, S, E, W
}

# Vehicle profile for legality filtering
VEHICLE_PROFILE = {
    'max_height': 4.0,                   # metres; 0 = no restriction
    'max_weight': 10.0,                  # tonnes; 0 = no restriction
    'max_width': 2.5,                    # metres; 0 = no restriction
    'min_highway_class': 'residential',  # lowest allowed OSM highway class
    'penalize_low_class_roads': True,    # apply cost multiplier to low-class roads
}

# ===================== PHASE 2: DYNAMIC TRAFFIC =====================

# Congestion engine configuration
CONGESTION_CONFIG = {
    'rush_hour_multiplier': 2.0,             # Peak multiplier vs off-peak
    'n_random_accidents': 1,                 # Max incident sites to try per run
    'accident_probability': 0.10,            # ~10% chance per site (~1 in 10 runs)
    'random_seed': None,                     # None = truly random each run
}

# Traffic light simulation
TRAFFIC_LIGHT_CONFIG = {
    'enabled': True,                         # Enable traffic light visualization
    'cycle_length_seconds': 60,              # Full red-yellow-green cycle
    'green_ratio': 0.50,                     # Proportion of cycle that is green
    'yellow_seconds': 3,                     # Yellow transition duration
}

# Default simulation time (seconds since midnight)
DEFAULT_SIMULATION_TIME = 36000  # 10:00 AM
SIMULATED_COLLECTION = False     # 由 --fast 模式设为 True, 标记采集数据为模拟

# ===================== 在线自适应行程时间预测配置 =====================
ADAPTIVE_CONFIG = {
    'enabled': True,                  # 启用在线自适应预测 (替代硬编码曲线)
    'exploration_noise_start': 0.3,   # 初始探索噪声标准差
    'exploration_noise_min': 0.02,    # 最小探索噪声
    'exploration_decay': 0.999,       # 每次更新衰减因子
    'learning_rate': 0.001,           # SGD 学习率
    'hidden_sizes': [64, 32],         # 隐藏层神经元数
    'cache_dir': '.cache/adaptive',   # 模型缓存目录
}

# ===================== 路口红绿灯检测配置 =====================
TRAFFIC_LIGHT_DETECTION = {
    'min_spacing_m': 800,          # 等距采样最小间距 (米), 城市主干道路口间距
    'max_spacing_m': 1500,         # 等距采样最大间距 (米)
    'turn_threshold_deg': 50,      # 转弯检测阈值 (度), 只有显著转弯才视为路口
    'min_light_spacing_m': 250,    # 去重最小间距 (米), 同一大路口合并为一个灯
    'max_lights_per_route': 15,    # 每条路线最多红绿灯数, 防止超长路线过度密集
    'cycle_length_s': 60,          # 信号灯周期
    'green_ratio': 0.50,           # 绿灯占比
    'yellow_s': 3,                 # 黄灯时长
    'avg_speed_kmh': 40,           # 城市配送平均速度, 估算到达时间用
}

# ===================== PHASE 1.1: ML 行程时间预测 =====================
ML_TRAVEL_TIME_CONFIG = {
    'enabled': True,                          # 是否启用 ML 预测 (冷启动时自动回退定速)
    'min_samples_for_training': 500,          # 最少训练样本数
    'model_path': '.cache/ml/travel_time_lgb.pkl',
    'samples_path': '.cache/ml/travel_time_samples.jsonl',
    'auto_collect': True,                     # 是否被动采集训练数据
    'dedup_window_hours': 2,                 # 去重时间窗口 (小时)
    'collect_save_interval': 50,             # 每N条样本刷盘
}
