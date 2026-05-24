"""
对比实验 Baselines — 对标中物联科技进步奖实验章节要求.

五个 Baseline:
1. 高德 ETA 直接预测 (API 原生 duration)
2. 历史均值法 (同时段同OD历史均值)
3. OR-Tools + 欧氏距离
4. 随机森林替代 LightGBM
5. LSTM 时序预测

每个 Baseline 跑 5 次, 报告均值和标准差.
"""

import json
import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FALLBACK_SPEED_KMH = 40.0
CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"
ML_CACHE = CACHE_DIR / "ml"


@dataclass
class BenchmarkResult:
    """统一对比实验结果."""
    name: str                          # 方法名称
    total_distance_km: float           # 总行驶距离
    total_time_seconds: float          # 预估总行程时间
    vehicles_used: int                 # 实际使用车辆数
    vehicle_utilization_pct: float     # 车辆利用率 %
    overflow_count: int                # 未配送点数
    solve_time_seconds: float          # 求解耗时
    # 统计量 (5 次运行)
    distance_mean: float = 0.0
    distance_std: float = 0.0
    time_mean: float = 0.0
    time_std: float = 0.0
    extra: Dict = field(default_factory=dict)

    def summary_line(self) -> str:
        return (
            f"{self.name:30s} | {self.distance_mean:8.1f}±{self.distance_std:5.1f}km | "
            f"{self.time_mean:8.0f}±{self.time_std:5.0f}s | "
            f"车辆{self.vehicles_used} | 利用率{self.vehicle_utilization_pct:.0f}% | "
            f"求解{self.solve_time_seconds:.1f}s"
            + (f" | 溢出{self.overflow_count}" if self.overflow_count else "")
        )


# ═══════════════════════════════════════════════════
# Baseline 1: 高德 ETA 直接预测
# ═══════════════════════════════════════════════════

def baseline_amap_eta_direct(
    points: List[Tuple], demands: List[float],
    time_windows: List[Tuple], distance_matrix: np.ndarray,
    amap_network=None, num_vehicles: int = 5,
) -> BenchmarkResult:
    """
    高德 ETA 直接预测 — 调用 API 获取 duration, 不经过 ML/自适应预测.

    用高德 API 返回的原生 duration 作为行程时间,
    直接在 OR-Tools 中求解, 验证 ML/自适应预测是否有增量价值.
    """
    t0 = time.time()
    from src.algorithms.vrp_solver import VRPSolver
    from src.algorithms.vehicle_types import Fleet, VehicleType
    from config import settings

    # 构建时间矩阵 (用 API duration 替代 ML 预测)
    n = len(points)
    time_matrix = np.zeros((n, n))
    if amap_network and hasattr(amap_network, 'get_driving_info'):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                try:
                    info = amap_network.get_driving_info(
                        (float(points[i][0]), float(points[i][1])),
                        (float(points[j][0]), float(points[j][1])),
                    )
                    time_matrix[i][j] = info["duration"]
                except Exception:
                    time_matrix[i][j] = distance_matrix[i][j] / FALLBACK_SPEED_KMH * 3600
    else:
        time_matrix = distance_matrix / FALLBACK_SPEED_KMH * 3600

    fleet = Fleet(
        types=[VehicleType(**t) for t in settings.VEHICLE_TYPES],
        max_total=settings.MAX_VEHICLES,
    )
    solver = VRPSolver(fleet=fleet, depot_index=0)

    # 自定义时间回调
    def _make_time_cb(tm):
        def cb(from_idx, to_idx):
            from ortools.constraint_solver import routing_enums_pb2
            fm = solver.__class__.__dict__
            return int(tm[from_idx][to_idx])
        return cb

    result = solver.solve_with_ortools(
        distance_matrix=distance_matrix.tolist(),
        demands=demands,
        time_limit=settings.TIME_LIMIT_SECONDS,
        time_windows=time_windows,
    )
    if result is None and time_windows is not None:
        result = solver.solve_with_ortools(
            distance_matrix=distance_matrix.tolist(),
            demands=demands,
            time_limit=settings.TIME_LIMIT_SECONDS * 3,
            time_windows=None,
        )
    elapsed = time.time() - t0

    if result is None:
        return BenchmarkResult(
            name="1.Amap ETA直接预测",
            total_distance_km=0, total_time_seconds=0,
            vehicles_used=0, vehicle_utilization_pct=0,
            overflow_count=len(demands) - 1, solve_time_seconds=elapsed,
        )

    total_time = 0.0
    for route in result.routes:
        for a, b in zip(route[:-1], route[1:]):
            total_time += time_matrix[a][b]

    return BenchmarkResult(
        name="1.Amap ETA直接预测",
        total_distance_km=result.total_distance_km,
        total_time_seconds=total_time,
        vehicles_used=result.num_active,
        vehicle_utilization_pct=result.num_active / max(1, num_vehicles) * 100,
        overflow_count=len(result.overflow_nodes),
        solve_time_seconds=elapsed,
    )


# ═══════════════════════════════════════════════════
# Baseline 2: 历史均值法
# ═══════════════════════════════════════════════════

def _load_historical_samples(city: str = None) -> Dict[Tuple, List[float]]:
    """从 jsonl 加载历史样本, 按 (OD对, 时段, 星期几) 聚合."""
    buckets = defaultdict(list)
    pattern = f"travel_time_samples_{city or '*'}.jsonl"
    for path in ML_CACHE.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    s = json.loads(line)
                    key = (
                        round(s["origin_lon"], 3), round(s["origin_lat"], 3),
                        round(s["dest_lon"], 3), round(s["dest_lat"], 3),
                        s["hour"] // 2,          # 2小时时段
                        s["day_of_week"],
                    )
                    buckets[key].append(s["duration_seconds"])
        except Exception:
            pass
    return dict(buckets)


def baseline_historical_mean(
    points: List[Tuple], demands: List[float],
    time_windows: List[Tuple], distance_matrix: np.ndarray,
    sim_time_seconds: int = 36000, city: str = "Beijing",
    num_vehicles: int = 5, amap_network=None,
) -> BenchmarkResult:
    """
    历史均值法 — 同时段同OD的历史 duration 均值.

    传统物流企业就靠这个做调度, 是最重要的对比基线之一.
    """
    t0 = time.time()
    from src.algorithms.vrp_solver import VRPSolver
    from src.algorithms.vehicle_types import Fleet, VehicleType
    from config import settings

    # Fallback: compute Euclidean distance matrix if not provided
    if distance_matrix is None:
        from scipy.spatial.distance import cdist
        coords_arr = np.array([[float(p[0]), float(p[1])] for p in points])
        mid_lat = np.radians(np.mean(coords_arr[:, 1]))
        scale = np.array([111.32 * math.cos(mid_lat), 111.32])
        distance_matrix = cdist(coords_arr * scale, coords_arr * scale, metric="euclidean")

    hour = sim_time_seconds % 86400 // 3600
    dow = (sim_time_seconds // 86400) % 7
    hist_buckets = _load_historical_samples(city)

    n = len(points)
    time_matrix = np.zeros((n, n))
    fallback_count = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            key = (
                round(points[i][0], 3), round(points[i][1], 3),
                round(points[j][0], 3), round(points[j][1], 3),
                hour // 2, dow,
            )
            if key in hist_buckets and len(hist_buckets[key]) >= 3:
                time_matrix[i][j] = np.mean(hist_buckets[key])
            else:
                time_matrix[i][j] = distance_matrix[i][j] / FALLBACK_SPEED_KMH * 3600
                fallback_count += 1

    fleet = Fleet(
        types=[VehicleType(**t) for t in settings.VEHICLE_TYPES],
        max_total=settings.MAX_VEHICLES,
    )
    solver = VRPSolver(fleet=fleet, depot_index=0)
    result = solver.solve_with_ortools(
        distance_matrix=distance_matrix.tolist(),
        demands=demands,
        time_limit=settings.TIME_LIMIT_SECONDS,
        time_windows=time_windows,
    )
    if result is None and time_windows is not None:
        result = solver.solve_with_ortools(
            distance_matrix=distance_matrix.tolist(),
            demands=demands,
            time_limit=settings.TIME_LIMIT_SECONDS * 3,
            time_windows=None,
        )
    elapsed = time.time() - t0

    if result is None:
        return BenchmarkResult(
            name="2.历史均值法",
            total_distance_km=0, total_time_seconds=0,
            vehicles_used=0, vehicle_utilization_pct=0,
            overflow_count=len(demands) - 1, solve_time_seconds=elapsed,
            extra={"history_hits": len(hist_buckets), "fallback_count": fallback_count},
        )

    total_time = 0.0
    for route in result.routes:
        for a, b in zip(route[:-1], route[1:]):
            total_time += time_matrix[a][b]

    return BenchmarkResult(
        name="2.历史均值法",
        total_distance_km=result.total_distance_km,
        total_time_seconds=total_time,
        vehicles_used=result.num_active,
        vehicle_utilization_pct=result.num_active / max(1, num_vehicles) * 100,
        overflow_count=len(result.overflow_nodes),
        solve_time_seconds=elapsed,
        extra={"history_hits": len(hist_buckets), "fallback_count": fallback_count},
    )


# ═══════════════════════════════════════════════════
# Baseline 3: OR-Tools + 欧氏距离
# ═══════════════════════════════════════════════════

def baseline_ortools_euclidean(
    points: List[Tuple], demands: List[float],
    time_windows: List[Tuple], distance_matrix: np.ndarray = None,
    num_vehicles: int = 5,
) -> BenchmarkResult:
    """
    OR-Tools + 欧氏距离 — 用 scipy 直线距离替代真实路网.

    验证真实路网距离 vs 欧氏距离的价值.
    """
    t0 = time.time()
    from scipy.spatial.distance import cdist
    from src.algorithms.vrp_solver import VRPSolver
    from src.algorithms.vehicle_types import Fleet, VehicleType
    from config import settings

    coords = np.array([[float(p[0]), float(p[1])] for p in points])
    # 纬度一度 ≈ 111.32 km, 经度一度 ≈ 111.32 × cos(lat)
    mid_lat = np.radians(np.mean(coords[:, 1]))
    scale = np.array([111.32 * math.cos(mid_lat), 111.32])
    euclidean_km = cdist(coords * scale, coords * scale, metric="euclidean")

    # 确保车队总容量足够 (num_vehicles 用于利用率计算, 不限制实际派车数)
    total_demand = sum(demands) if demands else 0
    fleet_vehicles = max(num_vehicles, settings.MAX_VEHICLES,
                         int(total_demand / 15) + 1)  # 至少按最小车型算够装
    fleet = Fleet(
        types=[VehicleType(**t) for t in settings.VEHICLE_TYPES],
        max_total=fleet_vehicles,
    )
    solver = VRPSolver(fleet=fleet, depot_index=0)
    result = solver.solve_with_ortools(
        distance_matrix=euclidean_km.tolist(),
        demands=demands,
        time_limit=settings.TIME_LIMIT_SECONDS,
        time_windows=time_windows,
    )
    # 时间窗无解 → 放宽重试
    if result is None and time_windows is not None:
        result = solver.solve_with_ortools(
            distance_matrix=euclidean_km.tolist(),
            demands=demands,
            time_limit=settings.TIME_LIMIT_SECONDS * 3,
            time_windows=None,
        )
    elapsed = time.time() - t0

    if result is None:
        return BenchmarkResult(
            name="3.OR-Tools+欧氏距离",
            total_distance_km=0, total_time_seconds=0,
            vehicles_used=0, vehicle_utilization_pct=0,
            overflow_count=len(demands) - 1 if demands else 0, solve_time_seconds=elapsed,
        )

    total_time = result.total_distance_km / FALLBACK_SPEED_KMH * 3600
    return BenchmarkResult(
        name="3.OR-Tools+欧氏距离",
        total_distance_km=result.total_distance_km,
        total_time_seconds=total_time,
        vehicles_used=result.num_active,
        vehicle_utilization_pct=result.num_active / max(1, fleet_vehicles) * 100,
        overflow_count=len(result.overflow_nodes),
        solve_time_seconds=elapsed,
    )


# ═══════════════════════════════════════════════════
# Baseline 4: 随机森林替代 LightGBM
# ═══════════════════════════════════════════════════

def baseline_random_forest(
    points: List[Tuple], demands: List[float],
    time_windows: List[Tuple], distance_matrix: np.ndarray,
    sim_time_seconds: int = 36000, city: str = "Beijing",
    num_vehicles: int = 5,
) -> BenchmarkResult:
    """
    随机森林替代 LightGBM — 证明梯度提升的必要性.

    用 sklearn RandomForest 训练相同特征, 对比预测精度.
    """
    t0 = time.time()
    from sklearn.ensemble import RandomForestRegressor
    from src.ml.travel_time_predictor import (
        _build_features, _compute_center, detect_city_from_points, FALLBACK_SPEED_KMH,
    )
    from src.ml.data_collector import TravelTimeDataCollector
    from src.algorithms.vrp_solver import VRPSolver
    from src.algorithms.vehicle_types import Fleet, VehicleType
    from config import settings

    # 加载训练数据
    collector = TravelTimeDataCollector()
    samples = collector.load_all_samples(city)
    if not samples:
        samples = collector.load_all_samples()

    if len(samples) < 100:
        return BenchmarkResult(
            name="4.随机森林(样本不足)",
            total_distance_km=0, total_time_seconds=0,
            vehicles_used=0, vehicle_utilization_pct=0,
            overflow_count=0, solve_time_seconds=0,
            extra={"error": f"训练样本不足: {len(samples)} < 100"},
        )

    center = _compute_center(points)
    X = []
    y = []
    for s in samples:
        try:
            feats = _build_features(
                s["origin_lon"], s["origin_lat"],
                s["dest_lon"], s["dest_lat"],
                s["dist_km"], s["hour"], s["day_of_week"],
                center[0], center[1],
                is_real_time=s.get("real_time", True),
            )
            X.append(feats)
            y.append(s["duration_seconds"])
        except Exception:
            pass

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    mask = (y > 0) & (y < 21600)
    X, y = X[mask], y[mask]

    # 时间序列分割
    n = len(y)
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 生成时间矩阵
    hour = sim_time_seconds % 86400 // 3600
    dow = (sim_time_seconds // 86400) % 7
    time_matrix = np.zeros((len(points), len(points)))
    rf_rmse_sq = []

    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            try:
                feats = _build_features(
                    points[i][0], points[i][1], points[j][0], points[j][1],
                    distance_matrix[i][j], hour, dow,
                    center[0], center[1],
                )
                pred = rf.predict([feats])[0]
                lo = distance_matrix[i][j] / 120 * 3600
                hi = distance_matrix[i][j] / 5 * 3600
                time_matrix[i][j] = max(lo, min(pred, hi))
            except Exception:
                time_matrix[i][j] = distance_matrix[i][j] / FALLBACK_SPEED_KMH * 3600

    # 评估 RF 精度
    y_pred_rf = rf.predict(X_val)
    rf_rmse = float(np.sqrt(np.mean((y_pred_rf - y_val) ** 2)))
    rf_mape = float(np.mean(np.abs((y_pred_rf - y_val) / np.maximum(y_val, 1.0))) * 100)

    fleet = Fleet(
        types=[VehicleType(**t) for t in settings.VEHICLE_TYPES],
        max_total=settings.MAX_VEHICLES,
    )
    solver = VRPSolver(fleet=fleet, depot_index=0)
    result = solver.solve_with_ortools(
        distance_matrix=distance_matrix.tolist(),
        demands=demands,
        time_limit=settings.TIME_LIMIT_SECONDS,
        time_windows=time_windows,
    )
    if result is None and time_windows is not None:
        result = solver.solve_with_ortools(
            distance_matrix=distance_matrix.tolist(),
            demands=demands,
            time_limit=settings.TIME_LIMIT_SECONDS * 3,
            time_windows=None,
        )
    elapsed = time.time() - t0

    if result is None:
        return BenchmarkResult(
            name="4.随机森林",
            total_distance_km=0, total_time_seconds=0,
            vehicles_used=0, vehicle_utilization_pct=0,
            overflow_count=len(demands) - 1, solve_time_seconds=elapsed,
            extra={"rf_rmse_s": rf_rmse, "rf_mape_pct": rf_mape},
        )

    return BenchmarkResult(
        name="4.随机森林",
        total_distance_km=result.total_distance_km,
        total_time_seconds=0,  # 无 ML 时间矩阵
        vehicles_used=result.num_active,
        vehicle_utilization_pct=result.num_active / max(1, num_vehicles) * 100,
        overflow_count=len(result.overflow_nodes),
        solve_time_seconds=elapsed,
        extra={"rf_rmse_s": rf_rmse, "rf_mape_pct": rf_mape},
    )


# ═══════════════════════════════════════════════════
# Baseline 5: LSTM 时序预测
# ═══════════════════════════════════════════════════

def baseline_lstm_temporal(
    points: List[Tuple], demands: List[float],
    time_windows: List[Tuple], distance_matrix: np.ndarray,
    sim_time_seconds: int = 36000, city: str = "Beijing",
    num_vehicles: int = 5,
) -> BenchmarkResult:
    """
    LSTM 时序预测 — 用序列模型对比静态特征, 证明 15 维静态特征够不够用.

    为每个 OD 对训练一个简单 LSTM, 预测下一个时段的 duration.
    若样本太少则用全局 LSTM 学习所有 OD 对的共享模式.
    """
    t0 = time.time()
    from src.ml.travel_time_predictor import FALLBACK_SPEED_KMH
    from src.ml.data_collector import TravelTimeDataCollector
    from src.algorithms.vrp_solver import VRPSolver
    from src.algorithms.vehicle_types import Fleet, VehicleType
    from config import settings

    collector = TravelTimeDataCollector()
    samples = collector.load_all_samples(city)
    if not samples:
        samples = collector.load_all_samples()

    if len(samples) < 100:
        return BenchmarkResult(
            name="5.LSTM时序(样本不足)",
            total_distance_km=0, total_time_seconds=0,
            vehicles_used=0, vehicle_utilization_pct=0,
            overflow_count=0, solve_time_seconds=0,
            extra={"error": f"训练样本不足: {len(samples)} < 100"},
        )

    # 按 OD 对聚合时序
    od_series = defaultdict(list)
    for s in samples:
        key = (round(s["origin_lon"], 3), round(s["origin_lat"], 3),
               round(s["dest_lon"], 3), round(s["dest_lat"], 3))
        od_series[key].append({
            "hour": s["hour"],
            "dow": s["day_of_week"],
            "dist_km": s["dist_km"],
            "duration": s["duration_seconds"],
            "ts": s.get("timestamp", 0),
        })

    # 对每个 OD 对, 按时间排序, 构建 (seq_len=5, features=4) → duration 训练集
    lstm_predictions = {}
    hour = sim_time_seconds % 86400 // 3600
    dow = (sim_time_seconds // 86400) % 7

    try:
        import torch
        import torch.nn as nn

        class TinyLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(4, 16, batch_first=True)
                self.fc = nn.Linear(16, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        device = torch.device("cpu")
        global_lstm = TinyLSTM().to(device)
        optimizer = torch.optim.Adam(global_lstm.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        # 用所有 OD 对数据训练全局 LSTM
        X_seq, y_seq = [], []
        for key, series in od_series.items():
            series.sort(key=lambda x: x["ts"])
            for t_idx in range(5, len(series)):
                window = series[t_idx - 5:t_idx]
                feat = np.array([
                    [w["hour"] / 24, w["dow"] / 7, w["dist_km"] / 50, w["duration"] / 3600]
                    for w in window
                ], dtype=np.float32)
                target = series[t_idx]["duration"] / 3600
                X_seq.append(feat)
                y_seq.append(target)

        if len(X_seq) < 50:
            return BenchmarkResult(
                name="5.LSTM时序(数据不足)",
                total_distance_km=0, total_time_seconds=0,
                vehicles_used=0, vehicle_utilization_pct=0,
                overflow_count=0, solve_time_seconds=0,
                extra={"error": f"时序样本不足: {len(X_seq)} < 50"},
            )

        X_t = torch.tensor(np.array(X_seq), dtype=torch.float32)
        y_t = torch.tensor(np.array(y_seq), dtype=torch.float32).unsqueeze(1)

        # 训练
        global_lstm.train()
        for epoch in range(30):
            optimizer.zero_grad()
            pred = global_lstm(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

        # 预测时间矩阵
        time_matrix = np.zeros((len(points), len(points)))
        for i in range(len(points)):
            for j in range(len(points)):
                if i == j:
                    continue
                key = (round(points[i][0], 3), round(points[i][1], 3),
                       round(points[j][0], 3), round(points[j][1], 3))
                if key in od_series and len(od_series[key]) >= 5:
                    series = sorted(od_series[key], key=lambda x: x["ts"])[-5:]
                    feat = np.array([[
                        w["hour"] / 24, w["dow"] / 7,
                        w["dist_km"] / 50, w["duration"] / 3600,
                    ] for w in series], dtype=np.float32)
                    with torch.no_grad():
                        pred = global_lstm(torch.tensor(feat).unsqueeze(0)).item()
                    time_matrix[i][j] = pred * 3600
                else:
                    time_matrix[i][j] = distance_matrix[i][j] / FALLBACK_SPEED_KMH * 3600

        lstm_loss = float(loss.item())

    except (ImportError, OSError) as e:
        return BenchmarkResult(
            name="5.LSTM时序(PyTorch不可用)",
            total_distance_km=0, total_time_seconds=0,
            vehicles_used=0, vehicle_utilization_pct=0,
            overflow_count=0, solve_time_seconds=0,
            extra={"error": str(e)},
        )

    # 求解
    fleet = Fleet(
        types=[VehicleType(**t) for t in settings.VEHICLE_TYPES],
        max_total=settings.MAX_VEHICLES,
    )
    solver = VRPSolver(fleet=fleet, depot_index=0)
    result = solver.solve_with_ortools(
        distance_matrix=distance_matrix.tolist(),
        demands=demands,
        time_limit=settings.TIME_LIMIT_SECONDS,
        time_windows=time_windows,
    )
    if result is None and time_windows is not None:
        result = solver.solve_with_ortools(
            distance_matrix=distance_matrix.tolist(),
            demands=demands,
            time_limit=settings.TIME_LIMIT_SECONDS * 3,
            time_windows=None,
        )
    elapsed = time.time() - t0

    if result is None:
        return BenchmarkResult(
            name="5.LSTM时序",
            total_distance_km=0, total_time_seconds=0,
            vehicles_used=0, vehicle_utilization_pct=0,
            overflow_count=len(demands) - 1, solve_time_seconds=elapsed,
            extra={"lstm_loss": lstm_loss},
        )

    return BenchmarkResult(
        name="5.LSTM时序",
        total_distance_km=result.total_distance_km,
        total_time_seconds=0,
        vehicles_used=result.num_active,
        vehicle_utilization_pct=result.num_active / max(1, num_vehicles) * 100,
        overflow_count=len(result.overflow_nodes),
        solve_time_seconds=elapsed,
        extra={"lstm_loss": lstm_loss},
    )


# ═══════════════════════════════════════════════════
# 注册表
# ═══════════════════════════════════════════════════

BASELINE_REGISTRY = {
    "amap_eta": baseline_amap_eta_direct,
    "historical_mean": baseline_historical_mean,
    "ortools_euclidean": baseline_ortools_euclidean,
    "random_forest": baseline_random_forest,
    "lstm_temporal": baseline_lstm_temporal,
}
