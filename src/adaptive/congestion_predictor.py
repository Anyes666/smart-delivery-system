"""
在线自适应行程时间预测器 — Online Adaptive Travel Time Prediction

替代硬编码交通惩罚曲线。每次高德 API 返回真实耗时后,
自动用 SGD 更新神经网络权重, 越跑越准。

方法: 神经网络回归模型 (23维输入 → 64 ReLU → 32 ReLU → 拥堵乘数输出),
      配合在线 SGD 更新 + 探索噪声实现持续自适应。
"""

import logging
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "adaptive"
FALLBACK_SPEED_KMH = 40.0


def _haversine_km(lon1, lat1, lon2, lat2):
    """Haversine 距离 (km)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _compute_center(points: List[Tuple]) -> Tuple[float, float]:
    """从点集计算几何中心."""
    if not points:
        return (121.47, 31.23)
    lons = [float(p[0]) for p in points]
    lats = [float(p[1]) for p in points]
    return (sum(lons) / len(lons), sum(lats) / len(lats))


def _build_features(
    origin_lon: float, origin_lat: float,
    dest_lon: float, dest_lat: float,
    dist_km: float, hour: int, day_of_week: int,
    city_center_lon: float, city_center_lat: float,
    polyline_str: str = "",
) -> np.ndarray:
    """构建 23 维特征向量, 用于拥堵预测 (含道路几何代理特征).

    与 LightGBM 的 _build_features 对齐空间特征和道路几何特征,
    确保两个模型共享关键信息维度。

    特征清单:
    - 时间循环 (5): hour_sin/cos, dow_sin/cos, is_weekend
    - 细化时段 (5): is_rush, is_night, is_peak_am, is_peak_pm, is_noon
    - 距离 (4): dist_km, log1p(dist_km), dist_bucket_short, dist_bucket_mid
    - 空间 (4): origin_center_dist, dest_center_dist, crosses_center, bearing_diff_norm
    - 交互 (2): rush_x_dist, night_x_dist
    - 道路几何代理 (3): road_class_proxy, turn_rate, straight_rate
    """
    # ── 时间循环编码 ──
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    dow_sin = math.sin(2 * math.pi * day_of_week / 7)
    dow_cos = math.cos(2 * math.pi * day_of_week / 7)
    is_weekend = 1.0 if day_of_week >= 5 else 0.0

    # ── 细化时段标记 ──
    is_rush = 1.0 if (7 <= hour < 10) or (16 <= hour < 19) else 0.0
    is_night = 1.0 if (hour >= 22 or hour < 6) else 0.0
    is_peak_am = 1.0 if 8 <= hour < 10 else 0.0
    is_peak_pm = 1.0 if 17 <= hour < 19 else 0.0
    is_noon = 1.0 if 12 <= hour < 14 else 0.0

    # ── 距离分桶 ──
    dist_bucket_short = 1.0 if dist_km < 2 else 0.0
    dist_bucket_mid = 1.0 if 2 <= dist_km < 5 else 0.0

    # ── 空间特征 ──
    origin_center_dist = _haversine_km(
        origin_lon, origin_lat, city_center_lon, city_center_lat)
    dest_center_dist = _haversine_km(
        dest_lon, dest_lat, city_center_lon, city_center_lat)

    # 路段是否穿过市中心
    dx = dest_lon - origin_lon
    dy = dest_lat - origin_lat
    cx = city_center_lon - origin_lon
    cy = city_center_lat - origin_lat
    dot = dx * cx + dy * cy
    # 分子分母均为 degree² 单位, 比例无量纲
    denom = max(dx * dx + dy * dy, 1e-8)
    projection = dot / denom
    crosses_center = 1.0 if 0.1 < projection < 0.9 else 0.0

    # 起终点方位差 (标准化)
    origin_bearing = math.atan2(origin_lat - city_center_lat,
                                origin_lon - city_center_lon)
    dest_bearing = math.atan2(dest_lat - city_center_lat,
                              dest_lon - city_center_lon)
    bearing_diff = abs(origin_bearing - dest_bearing)
    if bearing_diff > math.pi:
        bearing_diff = 2 * math.pi - bearing_diff

    # ── 交互特征 ──
    rush_x_dist = is_rush * dist_km
    night_x_dist = is_night * dist_km

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
                density = len(coords) / max(dist_km, 0.1)
                road_class_proxy = min(1.0, density / 30.0)

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

    return np.array([
        # 时间循环 (5)
        hour_sin, hour_cos, dow_sin, dow_cos, is_weekend,
        # 细化时段 (5)
        is_rush, is_night, is_peak_am, is_peak_pm, is_noon,
        # 距离 (4)
        dist_km, math.log1p(dist_km), dist_bucket_short, dist_bucket_mid,
        # 空间 (4)
        origin_center_dist, dest_center_dist, crosses_center,
        bearing_diff / math.pi,
        # 交互 (2)
        rush_x_dist, night_x_dist,
        # 道路几何代理 (3) — 与 LightGBM 特征对齐
        road_class_proxy, turn_rate, straight_rate,
    ], dtype=np.float32)


class AdaptiveCongestionPredictor:
    """在线自适应行程时间预测器 (神经网络回归 + 在线 SGD 更新).

    用法::

        pred = AdaptiveCongestionPredictor.load_or_create(city="Beijing", points=points)
        multiplier = pred.predict(origin, dest, dist_km, hour, dow)
        # ... 跑完高德 API 拿到真实 duration ...
        loss = pred.update(origin, dest, dist_km, duration_seconds, hour, dow)
        pred.save("Beijing")
    """

    def __init__(self, city_center: Optional[Tuple[float, float]] = None):
        if city_center is None:
            city_center = (116.40, 39.90)  # 默认北京
        self.city_center_lon = float(city_center[0])
        self.city_center_lat = float(city_center[1])

        # 从配置文件读取超参数 (含兜底默认值)
        try:
            from config.settings import ADAPTIVE_CONFIG
        except ImportError:
            ADAPTIVE_CONFIG = {}
        self._hidden_sizes = list(
            ADAPTIVE_CONFIG.get('hidden_sizes', [64, 32]))
        self._exploration_noise = float(
            ADAPTIVE_CONFIG.get('exploration_noise_start', 0.3))
        self._exploration_min = float(
            ADAPTIVE_CONFIG.get('exploration_noise_min', 0.02))
        self._exploration_decay = float(
            ADAPTIVE_CONFIG.get('exploration_decay', 0.999))
        self._lr = float(
            ADAPTIVE_CONFIG.get('learning_rate', 0.001))

        # 网络结构: 23 → hidden[0] → hidden[1] → 1
        self._input_dim = 23
        self._init_weights()

        # 特征归一化参数 (训练时更新)
        self._feat_mean = np.zeros(self._input_dim, dtype=np.float32)
        self._feat_std = np.ones(self._input_dim, dtype=np.float32)
        self._feat_frozen = False  # 首次批量训练后冻结

        # 在线学习状态
        self._update_count = 0
        self._running_loss = None

    def _init_weights(self):
        """He 初始化, 输出偏置初始化为 1.0 (定速基线)."""
        rng = np.random.RandomState(42)
        sizes = [self._input_dim] + self._hidden_sizes + [1]

        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            std = math.sqrt(2.0 / fan_in)
            self.weights.append(rng.randn(sizes[i], sizes[i + 1]).astype(np.float32) * std)
            self.biases.append(np.zeros(sizes[i + 1], dtype=np.float32))
        # 输出偏置 -2.0 → sigmoid(-2.0)≈0.119 → 缩放后≈1.04 (初始无拥堵假设)
        self.biases[-1][0] = -2.0

    # ═══════════════════════════════════════════════════
    # 特征归一化
    # ═══════════════════════════════════════════════════

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Z-score 归一化."""
        return (x - self._feat_mean) / np.maximum(self._feat_std, 1e-6)

    def fit_normalizer(self, samples: list):
        """从样本计算特征均值和标准差."""
        if self._feat_frozen or not samples:
            return
        feats = []
        for s in samples:
            f = _build_features(
                s["origin_lon"], s["origin_lat"],
                s["dest_lon"], s["dest_lat"],
                s["dist_km"], s["hour"], s["day_of_week"],
                self.city_center_lon, self.city_center_lat,
                s.get("polyline", ""),
            )
            feats.append(f)
        X = np.stack(feats)
        self._feat_mean = X.mean(axis=0).astype(np.float32)
        self._feat_std = X.std(axis=0).astype(np.float32)
        self._feat_std = np.maximum(self._feat_std, 1e-6)
        self._feat_frozen = True

    # ═══════════════════════════════════════════════════
    # 前向传播
    # ═══════════════════════════════════════════════════

    def _forward(self, x: np.ndarray) -> Tuple[float, dict]:
        """前向传播, 返回 (输出值, 中间变量缓存)."""
        x_norm = self._normalize(x)
        cache = {"a0": x_norm}
        a = x_norm
        for i, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            z = a @ W + b
            a = np.maximum(z, 0)  # ReLU
            cache[f"z{i+1}"] = z
            cache[f"a{i+1}"] = a

        # 输出层: sigmoid 缩放到 [0.5, 5.0] (对应 8~80 km/h)
        z_out = a @ self.weights[-1] + self.biases[-1]
        sigmoid_out = 1.0 / (1.0 + float(np.exp(-z_out)))
        y_out = 0.5 + 4.5 * sigmoid_out
        cache["z_out"] = z_out
        cache["sigmoid_out"] = sigmoid_out
        return y_out, cache

    # ═══════════════════════════════════════════════════
    # 反向传播 (SGD, batch_size=1)
    # ═══════════════════════════════════════════════════

    def _backward(self, cache: dict, y_pred: float, y_true: float):
        """单样本 SGD 反向传播 (含梯度裁剪 + sigmoid 输出层)."""
        lr = self._lr

        # MSE loss: L = (y_pred - y_true)^2,  dL/dy_pred = 2 * (y_pred - y_true)
        # sigmoid 缩放: y_pred = 0.5 + 4.5 * sigmoid(z_out)
        # dy_pred/dz_out = 4.5 * sigmoid(z_out) * (1 - sigmoid(z_out))
        sigmoid_out = float(cache["sigmoid_out"])
        d_loss = 2.0 * (y_pred - y_true)
        d_sigmoid = 4.5 * sigmoid_out * (1.0 - sigmoid_out)
        grad = d_loss * d_sigmoid
        grad = float(np.clip(grad, -5.0, 5.0))  # 梯度裁剪
        dz_out = np.array([[grad]], dtype=np.float32)

        # 最后一层
        a_prev = cache[f"a{len(self._hidden_sizes)}"].reshape(1, -1)
        dW_out = a_prev.T @ dz_out
        db_out = dz_out.flatten()

        self.weights[-1] -= lr * dW_out
        self.biases[-1] -= lr * db_out

        # 反向传播通过隐藏层
        da = dz_out @ self.weights[-1].T  # (1,1) @ (1,32) = (1,32)

        for layer_idx in range(len(self._hidden_sizes) - 1, -1, -1):
            z = cache[f"z{layer_idx + 1}"]  # (1, hidden_size)
            dz = da * (z > 0).astype(np.float32)  # ReLU derivative
            a_prev = cache[f"a{layer_idx}"].reshape(1, -1)  # (1, prev_size)

            dW = a_prev.T @ dz  # (prev_size, 1) @ (1, hidden_size) = (prev_size, hidden_size)
            db = dz.flatten()

            self.weights[layer_idx] -= lr * dW
            self.biases[layer_idx] -= lr * db

            if layer_idx > 0:
                da = dz @ self.weights[layer_idx].T  # (1,hidden) @ (hidden,prev) = (1,prev)

    # ═══════════════════════════════════════════════════
    # 预测
    # ═══════════════════════════════════════════════════

    def predict(
        self, origin: Tuple[float, float], dest: Tuple[float, float],
        dist_km: float, hour: int, day_of_week: int,
        polyline_str: str = "",
    ) -> float:
        """预测拥堵乘数 (1.0 = 40km/h 定速水平)."""
        x = _build_features(
            origin[0], origin[1], dest[0], dest[1],
            dist_km, hour, day_of_week,
            self.city_center_lon, self.city_center_lat,
            polyline_str,
        ).reshape(1, -1)

        y_pred, _ = self._forward(x)
        # sigmoid 已限定 [0.5, 5.0], 无需额外 clamp

        # 探索噪声: 训练初期加入高斯噪声以探索不同预测值的效果
        if self._update_count > 0 and random.random() < self._exploration_noise:
            noise = random.gauss(0, 0.3)
            y_pred = max(0.2, min(10.0, y_pred + noise))

        return y_pred

    # ═══════════════════════════════════════════════════
    # 在线更新
    # ═══════════════════════════════════════════════════

    def update(
        self, origin: Tuple[float, float], dest: Tuple[float, float],
        dist_km: float, duration_seconds: float,
        hour: int, day_of_week: int,
        polyline_str: str = "",
    ) -> float:
        """用高德 API 返回的真实值在线更新模型, 返回 loss."""
        # 真实拥堵乘数 = 实际耗时 / 定速耗时
        baseline_seconds = dist_km / FALLBACK_SPEED_KMH * 3600
        if baseline_seconds <= 0:
            return 0.0
        y_true = duration_seconds / baseline_seconds

        x = _build_features(
            origin[0], origin[1], dest[0], dest[1],
            dist_km, hour, day_of_week,
            self.city_center_lon, self.city_center_lat,
            polyline_str,
        ).reshape(1, -1)

        # 过滤异常训练目标: 极端值会导致梯度爆炸
        if y_true <= 0 or y_true > 20 or not np.isfinite(y_true):
            return 0.0

        y_pred, cache = self._forward(x)
        loss = float((y_pred - y_true) ** 2)

        # NaN/Inf 保护: 跳过无效更新, 防止权重损坏
        if not np.isfinite(loss) or loss > 1e6:
            return 0.0

        self._backward(cache, y_pred, y_true)
        self._update_count += 1

        # 衰减探索噪声
        self._exploration_noise = max(
            self._exploration_min,
            self._exploration_noise * self._exploration_decay,
        )

        # 平滑 loss
        if self._running_loss is None or not np.isfinite(self._running_loss):
            self._running_loss = loss
        else:
            self._running_loss = 0.95 * self._running_loss + 0.05 * loss

        return loss

    # ═══════════════════════════════════════════════════
    # 批量训练 (冷启动)
    # ═══════════════════════════════════════════════════

    def train_batch(self, samples: list, epochs: int = 10) -> dict:
        """从历史样本批量训练 (离线冷启动)."""
        self.fit_normalizer(samples)
        losses = []
        for epoch in range(epochs):
            random.shuffle(samples)
            epoch_losses = []
            for s in samples:
                loss = self.update(
                    (s["origin_lon"], s["origin_lat"]),
                    (s["dest_lon"], s["dest_lat"]),
                    s["dist_km"], s["duration_seconds"],
                    s["hour"], s["day_of_week"],
                )
                epoch_losses.append(loss)
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            if epoch % 3 == 0:
                logger.info(f"  epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  "
                           f"noise={self._exploration_noise:.3f}  updates={self._update_count}")

        if self._running_loss is None and losses:
            self._running_loss = losses[-1]

        return {
            "final_loss": losses[-1] if losses else float("nan"),
            "epoch_losses": losses,
            "updates": self._update_count,
            "exploration_noise": self._exploration_noise,
        }

    # ═══════════════════════════════════════════════════
    # 数据诊断 & 概念漂移检测
    # ═══════════════════════════════════════════════════

    def diagnose_data_sufficiency(self, samples: list) -> dict:
        """诊断训练数据是否充分覆盖时段×区域×距离组合.

        专家评审要求: 10个时段 × 5个区域 × 5种距离档 = 250个最小组合.
        每组最少30个样本 → 7500样本是最低要求.

        Returns:
            {
                "total_samples": int,
                "unique_days": int,
                "hour_buckets": {bucket: count},
                "region_buckets": {region: count},
                "distance_buckets": {bucket: count},
                "combinations": int,        # 实际覆盖的组合数
                "min_required": 7500,
                "sufficiency_pct": float,   # 样本充足度 %
                "underpowered_combos": int, # 低于30样本的组合数
                "verdict": str
            }
        """
        if not samples:
            return {"verdict": "无样本数据", "total_samples": 0}

        # 时段分桶 (2小时间隔)
        hour_buckets = defaultdict(int)
        # 区域分桶 (相对市中心距离: 近/中/远 三层 + 方位四象限)
        region_buckets = defaultdict(int)
        # 距离分桶
        dist_buckets = defaultdict(int)
        # 唯一自然日
        unique_days = set()
        # 组合计数: (hour_bucket, region, dist_bucket)
        combo_counts = defaultdict(int)

        for s in samples:
            h = s.get("hour", 0)
            hb = h // 2  # 12个2小时时段
            hour_buckets[hb] += 1

            dist = s.get("dist_km", 0)
            if dist < 2:
                db = "short"
            elif dist < 5:
                db = "mid"
            elif dist < 10:
                db = "long"
            else:
                db = "very_long"
            dist_buckets[db] += 1

            # 区域: 相对市中心方位
            olon, olat = s.get("origin_lon", 0), s.get("origin_lat", 0)
            dlon, dlat = s.get("dest_lon", 0), s.get("dest_lat", 0)
            mid_lon = (olon + dlon) / 2
            mid_lat = (olat + dlat) / 2
            dr = _haversine_km(mid_lon, mid_lat,
                               self.city_center_lon, self.city_center_lat)
            if dr < 5:
                region = "core"
            elif dr < 15:
                region = "urban"
            else:
                region = "suburb"
            region_buckets[region] += 1

            # 组合
            combo = (hb, region, db)
            combo_counts[combo] += 1

            # 唯一日
            ts = s.get("timestamp", 0)
            if ts > 0:
                from datetime import datetime
                day_str = datetime.fromtimestamp(ts).strftime("%Y%m%d")
                unique_days.add(day_str)

        total = len(samples)
        n_combos = len(combo_counts)
        # 理论上覆盖的组合数: 12时段 × 5区域 × 4距离档 = 240
        max_combos = 12 * 5 * 4
        underpowered = sum(1 for c in combo_counts.values() if c < 30)

        sufficiency = min(100.0, total / 7500 * 100)

        if total < 1000:
            verdict = "严重不足: 样本量远低于最低要求(7500), 预测结论不可靠"
        elif underpowered > n_combos * 0.5:
            verdict = f"数据薄弱: {underpowered}/{n_combos} 组合不足30样本, 泛化能力存疑"
        elif underpowered > 0:
            verdict = f"部分不足: {underpowered}/{n_combos} 组合低于30样本, 需补全"
        else:
            verdict = "数据充分: 所有组合均满足统计推断最低要求"

        return {
            "total_samples": total,
            "unique_days": len(unique_days),
            "hour_buckets": dict(sorted(hour_buckets.items())),
            "region_buckets": dict(region_buckets),
            "distance_buckets": dict(dist_buckets),
            "combinations": n_combos,
            "max_combinations": max_combos,
            "min_required": 7500,
            "sufficiency_pct": round(sufficiency, 1),
            "underpowered_combos": underpowered,
            "verdict": verdict,
        }

    def detect_concept_drift(self, recent_samples: list,
                              threshold_multiplier: float = 2.0) -> dict:
        """检测概念漂移: 近期样本的预测误差是否显著高于历史基线.

        若误差超出阈值 (默认 2x 历史 RMSE), 说明交通模式已变化
        (如节假日、修路、季节更换), 需要提高探索率重新适应.

        Returns:
            {
                "drift_detected": bool,
                "recent_loss": float,
                "historical_loss": float,
                "loss_ratio": float,
                "action": str
            }
        """
        if not recent_samples or self._running_loss is None:
            return {"drift_detected": False, "action": "无基线或样本, 跳过检测"}

        losses = []
        for s in recent_samples:
            baseline = s["dist_km"] / FALLBACK_SPEED_KMH * 3600
            if baseline <= 0:
                continue
            y_true = s["duration_seconds"] / baseline

            x = _build_features(
                s["origin_lon"], s["origin_lat"],
                s["dest_lon"], s["dest_lat"],
                s["dist_km"], s["hour"], s["day_of_week"],
                self.city_center_lon, self.city_center_lat,
            ).reshape(1, -1)

            y_pred, _ = self._forward(x)
            losses.append(float((y_pred - y_true) ** 2))

        if not losses:
            return {"drift_detected": False, "action": "无有效样本"}

        recent_loss = float(np.mean(losses))
        hist_loss = float(self._running_loss)
        ratio = recent_loss / max(hist_loss, 1e-6)

        if ratio > threshold_multiplier:
            # 漂移检测: 提高探索噪声以重新适应
            old_noise = self._exploration_noise
            self._exploration_noise = min(0.5, old_noise * 3.0)
            return {
                "drift_detected": True,
                "recent_loss": round(recent_loss, 4),
                "historical_loss": round(hist_loss, 4),
                "loss_ratio": round(ratio, 1),
                "action": (f"概念漂移! loss比率={ratio:.1f}x, "
                          f"探索噪声 {old_noise:.3f}→{self._exploration_noise:.3f} "
                          f"(重新适应新交通模式)"),
            }

        return {
            "drift_detected": False,
            "recent_loss": round(recent_loss, 4),
            "historical_loss": round(hist_loss, 4),
            "loss_ratio": round(ratio, 1),
            "action": f"无漂移 (loss比率={ratio:.1f}x, 阈值={threshold_multiplier}x)",
        }

    # ═══════════════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _model_path(city: str) -> Path:
        return CACHE_DIR / f"congestion_{city}.pkl"

    def save(self, city: str):
        path = self._model_path(city)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": self.weights,
            "biases": self.biases,
            "update_count": self._update_count,
            "exploration_noise": self._exploration_noise,
            "running_loss": self._running_loss,
            "lr": self._lr,
            "city_center_lon": self.city_center_lon,
            "city_center_lat": self.city_center_lat,
            "feat_mean": self._feat_mean,
            "feat_std": self._feat_std,
            "feat_frozen": self._feat_frozen,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, city: str):
        path = cls._model_path(city)
        if not path.exists():
            raise FileNotFoundError(f"自适应预测模型不存在: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = cls(city_center=(data["city_center_lon"], data["city_center_lat"]))
        inst.weights = data["weights"]
        inst.biases = data["biases"]
        inst._update_count = data.get("update_count", 0)
        # 兼容旧字段名 epsilon → exploration_noise
        inst._exploration_noise = data.get(
            "exploration_noise", data.get("epsilon", 0.3)
        )
        inst._running_loss = data.get("running_loss", None)
        if inst._running_loss is not None and not np.isfinite(inst._running_loss):
            inst._running_loss = None
        inst._lr = data.get("lr", 0.0003)
        inst._feat_mean = data.get("feat_mean", np.zeros(inst._input_dim, dtype=np.float32))
        inst._feat_std = data.get("feat_std", np.ones(inst._input_dim, dtype=np.float32))
        inst._feat_frozen = data.get("feat_frozen", False)
        logger.info(f"自适应预测模型已加载: {path} (updates={inst._update_count}, "
                     f"loss={inst._running_loss})")
        return inst

    @classmethod
    def load_or_create(cls, city: str = "Beijing",
                       points: List[Tuple] = None) -> "AdaptiveCongestionPredictor":
        """智能加载: 有模型则加载, 无则创建空白实例."""
        if points:
            center = _compute_center(points)
        else:
            center = (116.40, 39.90)
        try:
            return cls.load(city)
        except (FileNotFoundError, Exception):
            logger.info(f"自适应预测模型 '{city}' 不存在, 创建空白实例 (将在线学习)")
            return cls(city_center=center)

    # ═══════════════════════════════════════════════════
    # 属性
    # ═══════════════════════════════════════════════════

    @property
    def is_trained(self) -> bool:
        return self._update_count > 0

    @property
    def metrics(self) -> dict:
        return {
            "updates": self._update_count,
            "running_loss": self._running_loss,
            "exploration_noise": self._exploration_noise,
            "city_center": (self.city_center_lon, self.city_center_lat),
        }
