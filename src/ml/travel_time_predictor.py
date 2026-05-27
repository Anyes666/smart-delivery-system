"""
Phase 1.1 — 路段行程时间预测 (LightGBM)

替代 VRP 求解器中 ``dist_km / 40km/h × 3600`` 的定速假设。

跨城市设计:
- 特征工程使用**相对城市中心的距离**, 而非绝对经纬度
  → 上海训练的模型对北京也有参考价值 (学到的是"距市中心Xkm→行程时间"规律)
- 按城市分模型文件, 自动检测 → 自动切换
- 冷启动时自动回退定速公式

数据来源: 高德 API get_driving_info() 返回的 duration 字段作为训练标签。
"""

import logging
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FALLBACK_SPEED_KMH = 40.0
MIN_SAMPLES_FOR_TRAINING = 500

MODEL_DIR = Path(__file__).parent.parent.parent / ".cache" / "ml"


# ═══════════════════════════════════════════════════
# 城市检测
# ═══════════════════════════════════════════════════

def _compute_center(points: List[Tuple]) -> Tuple[float, float]:
    """从点集计算几何中心 (lon, lat)."""
    if not points:
        return (121.47, 31.23)  # 默认上海
    lons = [float(p[0]) for p in points]
    lats = [float(p[1]) for p in points]
    return (sum(lons) / len(lons), sum(lats) / len(lats))


def _city_slug(center_lon: float, center_lat: float) -> str:
    """
    将城市中心坐标映射为城市标识。

    用 0.5°×0.5° 网格覆盖中国主要城市:
    - 北京 (116.4, 39.9)
    - 上海 (121.5, 31.2)
    - 广州 (113.3, 23.1)
    - 深圳 (114.1, 22.5)
    - 成都 (104.1, 30.6)
    - 武汉 (114.3, 30.6)
    - 杭州 (120.2, 30.3)
    - 南京 (118.8, 32.1)
    """
    # 主要城市经纬度范围
    KNOWN_CITIES = {
        (39.5, 40.5, 116.0, 117.0): "Beijing",
        (30.5, 31.8, 121.0, 122.0): "Shanghai",
        (22.5, 23.8, 112.8, 114.0): "Guangzhou",
        (22.0, 23.0, 113.5, 114.5): "Shenzhen",
        (30.0, 31.2, 103.5, 104.5): "Chengdu",
        (30.0, 31.0, 113.8, 114.8): "Wuhan",
        (29.5, 30.8, 119.5, 120.5): "Hangzhou",
        (31.5, 32.6, 118.0, 119.5): "Nanjing",
    }

    for (lat_lo, lat_hi, lon_lo, lon_hi), name in KNOWN_CITIES.items():
        if lat_lo <= center_lat <= lat_hi and lon_lo <= center_lon <= lon_hi:
            return name

    # 兜底: 用坐标哈希
    return f"city_{center_lon:.2f}_{center_lat:.2f}".replace(".", "p")


def detect_city_from_points(points: List[Tuple]) -> str:
    """从配送点自动检测城市名称。"""
    center = _compute_center(points)
    return _city_slug(*center)


# ═══════════════════════════════════════════════════
# 特征工程 (跨城市版本)
# ═══════════════════════════════════════════════════

def _haversine_km(lon1, lat1, lon2, lat2):
    """Haversine距离 (km)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _build_features(
    origin_lon: float, origin_lat: float,
    dest_lon: float, dest_lat: float,
    dist_km: float,
    hour: int,
    day_of_week: int,
    city_center_lon: float,
    city_center_lat: float,
    is_real_time: bool = True,
    polyline_str: str = "",
    month: int = 1,
) -> List[float]:
    """
    构建跨城市可迁移的特征向量 (39维).

    核心设计: 不用绝对经纬度, 用"距市中心距离"+"方位"编码空间位置。
    这样上海训练的"郊区→市区 5km"模式也能适用于北京。

    特征清单 (~37维):
    - 时间特征 (19): hour_sin/cos, dow_sin/cos, is_weekend, is_rush,
                     is_peak_am, is_peak_pm, is_noon, is_night,
                     month_sin, month_cos,
                     hour_norm, dow_norm (树模型可直接切分的连续值),
                     bucket_night/morning/noon/afternoon/evening (离散时段桶)
    - 距离特征 (5): dist_km, log_dist_km, dist_sq, dist_bucket_short,
                    dist_bucket_mid
    - 空间特征 (8): origin/dest距市中心距离及方位(sin/cos),
                    bearing_diff, crosses_center
    - 交互特征 (4): rush×dist, weekend×dist, night×dist, noon×dist
    - 道路等级代理 (3): polyline密度, 转弯率, 直行率
    """
    # ── 时间循环编码 ──
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    dow_sin = math.sin(2 * math.pi * day_of_week / 7)
    dow_cos = math.cos(2 * math.pi * day_of_week / 7)
    is_weekend = 1.0 if day_of_week >= 5 else 0.0
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)

    # ── 细化时段标记 ──
    is_rush = 1.0 if (7 <= hour < 10) or (16 <= hour < 19) else 0.0
    is_night = 1.0 if (hour >= 22 or hour < 6) else 0.0
    is_peak_am = 1.0 if 8 <= hour < 10 else 0.0
    is_peak_pm = 1.0 if 17 <= hour < 19 else 0.0
    is_noon = 1.0 if 12 <= hour < 14 else 0.0

    # ── 离散时段桶 (树模型友好) ──
    hour_norm = hour / 24.0  # 0~1 连续值, 树可自行切分
    dow_norm = day_of_week / 7.0
    bucket_night = 1.0 if 0 <= hour < 6 else 0.0
    bucket_morning = 1.0 if 6 <= hour < 11 else 0.0
    bucket_noon = 1.0 if 11 <= hour < 15 else 0.0
    bucket_afternoon = 1.0 if 15 <= hour < 19 else 0.0
    bucket_evening = 1.0 if 19 <= hour < 24 else 0.0

    # ── 空间特征: 相对城市中心 ──
    origin_center_dist = _haversine_km(origin_lon, origin_lat,
                                        city_center_lon, city_center_lat)
    dest_center_dist = _haversine_km(dest_lon, dest_lat,
                                      city_center_lon, city_center_lat)

    # 方位角 (sin/cos编码)
    origin_bearing = math.atan2(
        origin_lat - city_center_lat,
        origin_lon - city_center_lon
    )
    dest_bearing = math.atan2(
        dest_lat - city_center_lat,
        dest_lon - city_center_lon
    )

    # 起终点方位差 (反映穿越市中心程度)
    bearing_diff = abs(origin_bearing - dest_bearing)
    if bearing_diff > math.pi:
        bearing_diff = 2 * math.pi - bearing_diff

    # 起终点连线是否经过市中心附近
    dx = dest_lon - origin_lon
    dy = dest_lat - origin_lat
    cx = city_center_lon - origin_lon
    cy = city_center_lat - origin_lat
    dot = dx * cx + dy * cy
    projection = dot / max((dx * dx + dy * dy) * 111.32 ** 2, 0.01)
    crosses_center = 1.0 if 0.1 < projection < 0.9 else 0.0

    # ── 距离分桶 ──
    dist_bucket_short = 1.0 if dist_km < 2 else 0.0
    dist_bucket_mid = 1.0 if 2 <= dist_km < 5 else 0.0

    # ── 交互特征 ──
    rush_x_dist = is_rush * dist_km
    weekend_x_dist = is_weekend * dist_km
    night_x_dist = is_night * dist_km
    noon_x_dist = is_noon * dist_km

    # ── 道路等级代理特征 (从 polyline 坐标密度反推) ──
    road_class_proxy, turn_rate, straight_rate = 0.5, 0.0, 0.5
    if polyline_str:
        try:
            coords = []
            for pair in polyline_str.split(";"):
                pair = pair.strip()
                if pair:
                    parts = pair.split(",")
                    if len(parts) >= 2:
                        coords.append((float(parts[0]), float(parts[1])))
            if len(coords) >= 3:
                # 坐标密度: 点数/km → 高密度=城市道路(多弯), 低密度=快速路/高速
                density = len(coords) / max(dist_km, 0.1)
                # 高密度(>20点/km)=城市街道, 低密度(<5点/km)=高速/快速路
                road_class_proxy = min(1.0, density / 30.0)

                # 转弯率: 大角度转弯数 / 总点数
                turns = 0
                for k in range(1, len(coords) - 1):
                    x1, y1 = coords[k - 1]
                    x2, y2 = coords[k]
                    x3, y3 = coords[k + 1]
                    ang1 = math.atan2(y2 - y1, x2 - x1)
                    ang2 = math.atan2(y3 - y2, x3 - x2)
                    diff = abs(ang1 - ang2)
                    if diff > math.pi:
                        diff = 2 * math.pi - diff
                    if diff > math.radians(20):
                        turns += 1
                turn_rate = turns / max(len(coords), 1)
                straight_rate = 1.0 - turn_rate
        except Exception:
            pass

    # ── 非真实时间样本: 时段特征置零, 防止模型学错 ──
    if not is_real_time:
        hour_sin = hour_cos = dow_sin = dow_cos = 0.0
        is_weekend = is_rush = is_night = 0.0
        is_peak_am = is_peak_pm = is_noon = 0.0
        rush_x_dist = weekend_x_dist = night_x_dist = noon_x_dist = 0.0
        hour_norm = dow_norm = 0.0
        bucket_night = bucket_morning = bucket_noon = 0.0
        bucket_afternoon = bucket_evening = 0.0

    return [
        # ── 时间 (12+7=19) ──
        hour_sin, hour_cos, dow_sin, dow_cos,
        is_weekend, is_rush, is_night,
        is_peak_am, is_peak_pm, is_noon,
        month_sin, month_cos,
        hour_norm, dow_norm,
        bucket_night, bucket_morning, bucket_noon,
        bucket_afternoon, bucket_evening,
        # ── 距离 (5) ──
        dist_km, math.log1p(dist_km), dist_km ** 2,
        dist_bucket_short, dist_bucket_mid,
        # ── 空间 (8) ──
        origin_center_dist, dest_center_dist,
        math.sin(origin_bearing), math.cos(origin_bearing),
        math.sin(dest_bearing), math.cos(dest_bearing),
        crosses_center, bearing_diff / math.pi,
        # ── 交互 (4) ──
        rush_x_dist, weekend_x_dist,
        night_x_dist, noon_x_dist,
        # ── 道路等级代理 (3) ──
        road_class_proxy, turn_rate, straight_rate,
    ]


FEATURE_NAMES = [
    # 时间 (19)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_rush", "is_night",
    "is_peak_am", "is_peak_pm", "is_noon",
    "month_sin", "month_cos",
    "hour_norm", "dow_norm",
    "bucket_night", "bucket_morning", "bucket_noon",
    "bucket_afternoon", "bucket_evening",
    # 距离 (5)
    "dist_km", "log_dist_km", "dist_km_sq",
    "dist_bucket_short", "dist_bucket_mid",
    # 空间 (8)
    "origin_center_dist", "dest_center_dist",
    "origin_bearing_sin", "origin_bearing_cos",
    "dest_bearing_sin", "dest_bearing_cos",
    "crosses_center", "bearing_diff",
    # 交互 (4)
    "rush_x_dist", "weekend_x_dist",
    "night_x_dist", "noon_x_dist",
    # 道路等级代理 (3)
    "road_class_proxy", "turn_rate", "straight_rate",
]


# ═══════════════════════════════════════════════════
# 预测器
# ═══════════════════════════════════════════════════

class TravelTimePredictor:
    """
    LightGBM 行程时间预测器 (跨城市)。

    用法::

        # 自动检测城市
        city = detect_city_from_points(points)

        # 尝试加载该城市模型
        pred = TravelTimePredictor.load_or_fallback(city=city)
        seconds = pred.predict(origin, dest, dist_km, hour, day_of_week)

        # 训练
        pred = TravelTimePredictor(city_center=(121.5, 31.2))
        pred.train(samples)
        pred.save(city="Shanghai")
    """

    def __init__(self, city_center: Optional[Tuple[float, float]] = None):
        self._model = None
        self._trained = False
        self._n_samples = 0
        self._metrics: Dict = {}
        self._feature_importances: Dict[str, float] = {}

        # 城市中心 (lon, lat) — 用于特征归一化
        if city_center is None:
            city_center = (121.47, 31.23)  # 默认上海
        self.city_center_lon = float(city_center[0])
        self.city_center_lat = float(city_center[1])

    # ── 模型路径 ────────────────────────────────

    @staticmethod
    def _model_path(city: str) -> Path:
        return MODEL_DIR / f"travel_time_lgb_{city}.pkl"

    # ── 冷启动回退 ──────────────────────────────

    @staticmethod
    def fallback_predict(dist_km: float) -> float:
        return dist_km / FALLBACK_SPEED_KMH * 3600

    # ── 训练 ────────────────────────────────────

    def train(self, training_data: list) -> dict:
        if len(training_data) < MIN_SAMPLES_FOR_TRAINING:
            raise ValueError(
                f"训练数据不足: {len(training_data)} < {MIN_SAMPLES_FOR_TRAINING}."
            )

        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("pip install lightgbm")
            raise

        X_rows, y = [], []
        dkm_list = []  # 记录每个样本的距离, 用于分段评估
        for d in training_data:
            is_rt = d.get("real_time", True)
            ts = d.get("timestamp", 0)
            month = 1
            if ts > 0:
                from datetime import datetime
                month = datetime.fromtimestamp(ts).month
            feats = _build_features(
                d["origin_lon"], d["origin_lat"],
                d["dest_lon"], d["dest_lat"],
                d["dist_km"], d["hour"], d["day_of_week"],
                self.city_center_lon, self.city_center_lat,
                is_real_time=is_rt,
                polyline_str=d.get("polyline", ""),
                month=month,
            )
            X_rows.append(feats)
            y.append(d["duration_seconds"])
            dkm_list.append(d["dist_km"])

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        dkm = np.array(dkm_list, dtype=np.float32)

        # 剔除异常值
        mask = (y > 0) & (y < 21600)
        X, y, dkm = X[mask], y[mask], dkm[mask]
        logger.info(f"训练样本: {len(y)} (过滤后)")

        n = len(y)

        # ── 时间序列分割 (主验证方式) ──
        # 按时间戳排序, 前 80% 训练, 后 20% 验证
        # 这避免了随机打乱导致的时间泄露 (内插 vs 外推)
        timestamps = [d.get("timestamp", 0) for d in training_data]
        ts_arr = np.array(timestamps, dtype=np.float64)[mask]
        ts_order = np.argsort(ts_arr)
        X_ts = X[ts_order]
        y_ts = y[ts_order]
        dkm_ts = dkm[ts_order]

        n_ts = len(y_ts)
        ts_split = int(n_ts * 0.8)
        X_train_ts, y_train_ts = X_ts[:ts_split], y_ts[:ts_split]
        X_val_ts, y_val_ts = X_ts[ts_split:], y_ts[ts_split:]
        dkm_val_ts = dkm_ts[ts_split:]

        logger.info(f"时间序列分割: 训练 {len(y_train_ts)} | 验证 {len(y_val_ts)} "
                     f"(后{len(y_val_ts) / max(n_ts, 1) * 100:.0f}%)")

        train_ds = lgb.Dataset(X_train_ts, label=y_train_ts, feature_name=FEATURE_NAMES)
        val_ds = lgb.Dataset(X_val_ts, label=y_val_ts, feature_name=FEATURE_NAMES)

        params = {
            "objective": "regression", "metric": "rmse",
            "boosting_type": "gbdt", "num_leaves": 31,
            "learning_rate": 0.05, "feature_fraction": 0.9,
            "bagging_fraction": 0.8, "bagging_freq": 5,
            "min_data_in_leaf": 20, "verbose": -1, "seed": 42,
        }

        self._model = lgb.train(
            params, train_ds, valid_sets=[train_ds, val_ds],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(stopping_rounds=30),
                        lgb.log_evaluation(period=0)],
        )

        y_pred = self._model.predict(X_val_ts)
        rmse = float(np.sqrt(np.mean((y_pred - y_val_ts) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_val_ts)))
        mape = float(np.mean(np.abs((y_pred - y_val_ts) / np.maximum(y_val_ts, 1.0))) * 100)

        # ── 分距离段 MAPE (行业标准指标) ──
        mape_by_distance = self._compute_mape_by_distance(
            y_val_ts, y_pred, dkm_val_ts,
            buckets={"短途(<2km)": (0, 2), "中途(2-5km)": (2, 5), "长途(>5km)": (5, 999)},
        )

        # 与定速对比 (dist_km 在特征索引 19)
        fallback = np.array([self.fallback_predict(d) for d in X_val_ts[:, 19]])
        fallback_rmse = float(np.sqrt(np.mean((fallback - y_val_ts) ** 2)))

        self._metrics = {
            "rmse_seconds": round(rmse, 1),
            "mae_seconds": round(mae, 1),
            "mape_pct": round(mape, 1),
            "mape_by_distance": mape_by_distance,
            "fallback_rmse": round(fallback_rmse, 1),
            "improvement_pct": round((1 - rmse / max(fallback_rmse, 0.01)) * 100, 1),
            "n_samples": len(y),
            "n_features": X.shape[1],
            "split_method": "time_series",  # 标注分割方式
            "n_train": int(len(y_train_ts)),
            "n_val": int(len(y_val_ts)),
        }
        self._trained = True
        self._n_samples = len(y)
        self._feature_importances = dict(
            zip(FEATURE_NAMES,
                self._model.feature_importance(importance_type="gain"))
        )

        logger.info(f"LightGBM训练(时间序列分割): RMSE={rmse:.1f}s MAE={mae:.1f}s MAPE={mape:.1f}% "
                     f"vs定速提升{self._metrics['improvement_pct']}%")
        for bucket, info in mape_by_distance.items():
            logger.info(f"  {bucket}: MAPE={info['mape_pct']:.1f}% (n={info['count']})")
        return dict(self._metrics)

    @staticmethod
    def _compute_mape_by_distance(y_true, y_pred, dkm,
                                   buckets: dict) -> dict:
        """按距离段计算 MAPE."""
        result = {}
        for name, (lo, hi) in buckets.items():
            mask = (dkm >= lo) & (dkm < hi)
            if mask.sum() == 0:
                result[name] = {"mape_pct": 0.0, "count": 0}
                continue
            yt = y_true[mask]
            yp = y_pred[mask]
            m = float(np.mean(np.abs((yp - yt) / np.maximum(yt, 1.0))) * 100)
            result[name] = {"mape_pct": round(m, 1), "count": int(mask.sum())}
        return result

    # ── 预测 ────────────────────────────────────

    def predict(
        self, origin: Tuple[float, float], dest: Tuple[float, float],
        dist_km: float, hour: int, day_of_week: int,
        month: int = 1, polyline_str: str = "",
    ) -> float:
        if not self._trained or self._model is None:
            return self.fallback_predict(dist_km)

        feats = _build_features(
            origin[0], origin[1], dest[0], dest[1],
            dist_km, hour, day_of_week,
            self.city_center_lon, self.city_center_lat,
            polyline_str=polyline_str,
            month=month,
        )
        X = np.array([feats], dtype=np.float32)
        pred = float(self._model.predict(X)[0])
        lo = dist_km / 120.0 * 3600
        hi = dist_km / 5.0 * 3600
        return max(lo, min(pred, hi))

    # ── 持久化 ──────────────────────────────────

    def save(self, city: str = "default"):
        path = self._model_path(city)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self._model,
            "metrics": self._metrics,
            "n_samples": self._n_samples,
            "feature_importances": self._feature_importances,
            "city_center_lon": self.city_center_lon,
            "city_center_lat": self.city_center_lat,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"模型已保存: {path} ({self._n_samples}样本)")

    @classmethod
    def load(cls, city: str = "default"):
        path = cls._model_path(city)
        if not path.exists():
            raise FileNotFoundError(f"模型不存在: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = cls(city_center=(data["city_center_lon"], data["city_center_lat"]))
        inst._model = data["model"]
        inst._metrics = data.get("metrics", {})
        inst._n_samples = data.get("n_samples", 0)
        inst._feature_importances = data.get("feature_importances", {})
        inst._trained = True
        logger.info(f"模型已加载: {path} (RMSE={inst._metrics.get('rmse_seconds','?')}s)")
        return inst

    @classmethod
    def load_or_fallback(cls, city: str = "default", points: List[Tuple] = None):
        """
        智能加载: 按城市加载 → 无则回退。
        若 points 提供, 自动检测城市。
        """
        if points:
            city = detect_city_from_points(points)
        try:
            return cls.load(city)
        except FileNotFoundError:
            logger.info(f"城市 '{city}' 模型不存在, 回退定速 ({FALLBACK_SPEED_KMH}km/h)")
        except Exception as e:
            logger.warning(f"模型加载失败: {e}, 回退定速")

        # 返回未训练实例 (使用检测到的城市中心)
        if points:
            center = _compute_center(points)
        else:
            center = (121.47, 31.23)
        return cls(city_center=center)

    # ── 属性 ────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def metrics(self) -> dict:
        return dict(self._metrics)

    @property
    def feature_importance(self) -> dict:
        return dict(self._feature_importances)
