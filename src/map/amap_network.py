"""
Road network via Amap (高德) Web API.

Replaces OSMnx local graph with 高德 HTTP API calls:
- Geocoding: address → coordinates
- Driving direction: origin → destination → distance + polyline
- Distance matrix: batch direction API calls with caching + symmetry

Interface-compatible with RoadNetwork (osmnx) so EnhancedMapRenderer
works without modification.
"""

import hashlib
import logging
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────────

class AmapCache:
    """Two-level cache: in-memory + disk pickle."""

    def __init__(self, cache_dir: str = ".cache/amap"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: Dict[str, dict] = {}
        self._stats = {"memory_hits": 0, "disk_hits": 0, "misses": 0}
        self._last_cleanup = 0.0
        self._cleanup_interval = 600  # 每10分钟最多扫一次

    def _make_key(self, *args) -> str:
        raw = "|".join(str(a) for a in args)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, *args) -> Optional[dict]:
        key = self._make_key(*args)
        # Memory
        if key in self._memory:
            entry = self._memory[key]
            if time.time() - entry["ts"] < entry["ttl"]:
                self._stats["memory_hits"] += 1
                return entry["data"]
        # Disk
        disk_path = self.cache_dir / f"{key}.pkl"
        if disk_path.exists():
            try:
                with open(disk_path, "rb") as f:
                    entry = pickle.load(f)
                if time.time() - entry["ts"] < entry["ttl"]:
                    self._stats["disk_hits"] += 1
                    self._memory[key] = entry
                    return entry["data"]
                else:
                    disk_path.unlink()  # 过期，删除
            except Exception:
                disk_path.unlink(missing_ok=True)
        self._stats["misses"] += 1
        return None

    def set(self, *args, data, ttl=3600):
        key = self._make_key(*args)
        entry = {"data": data, "ts": time.time(), "ttl": ttl}
        self._memory[key] = entry
        try:
            with open(self.cache_dir / f"{key}.pkl", "wb") as f:
                pickle.dump(entry, f)
        except Exception:
            pass
        if time.time() - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired()

    def _cleanup_expired(self):
        """扫一遍磁盘，删除所有过期的 .pkl 缓存文件。最多每10分钟触发一次。"""
        self._last_cleanup = time.time()
        now = time.time()
        removed = 0
        for f in self.cache_dir.glob("*.pkl"):
            try:
                with open(f, "rb") as fh:
                    entry = pickle.load(fh)
                if now - entry.get("ts", 0) > entry.get("ttl", 3600):
                    f.unlink()
                    removed += 1
            except Exception:
                f.unlink(missing_ok=True)
                removed += 1
        if removed:
            logger.info(f"缓存清理: 删除 {removed} 个过期文件")

    def stats(self) -> Dict:
        return dict(self._stats)


# ──────────────────────────────────────────────────
# AmapRoadNetwork
# ──────────────────────────────────────────────────

class AmapRoadNetwork:
    """
    Road network powered by Amap (高德) Web API.

    Usage::

        rn = AmapRoadNetwork(api_key="YOUR_KEY")
        dist = rn.compute_distance_matrix(points)
        geom = rn.get_route_geometry(route, points)
    """

    BASE_URL = "https://restapi.amap.com/v3"

    def __init__(
        self,
        api_key: str,
        cache_dir: str = ".cache/amap",
        strategy: int = 0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_interval: float = 0.05,
    ):
        if not api_key or api_key == "YOUR_AMAP_KEY":
            raise ValueError("Invalid Amap API key. Set AMAP_CONFIG['api_key'] in settings.py")

        self.api_key = api_key
        self.strategy = strategy
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_interval = rate_limit_interval

        # Compatibility with EnhancedMapRenderer
        self.graph = None
        self.graph_projected = None
        self._vehicle_profile = None

        self.cache = AmapCache(cache_dir)
        self._session = requests.Session()
        self._last_request_time = 0.0
        self._rate_lock = threading.Lock()  # 线程安全的限频锁

        # Phase 1.1: 被动数据采集器 (ML训练数据)
        self._collector = None  # 由外部注入 TravelTimeDataCollector

        logger.info(f"AmapRoadNetwork initialized (strategy={strategy})")

    # ── HTTP core ───────────────────────────────────

    def _throttle(self):
        """线程安全限频: 保证两次API调用起始间隔 ≥ rate_limit_interval."""
        with self._rate_lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit_interval:
                time.sleep(self.rate_limit_interval - elapsed)
            self._last_request_time = time.time()

    def _request(self, endpoint: str, params: dict) -> dict:
        params["key"] = self.api_key
        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(1, self.max_retries + 2):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except requests.Timeout:
                if attempt <= self.max_retries:
                    wait = self.retry_delay * (2 ** (attempt - 1))
                    logger.debug(f"Timeout, retry {attempt}/{self.max_retries} in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"API timeout after {self.max_retries} retries: {endpoint}")
            except requests.RequestException as e:
                if attempt <= self.max_retries:
                    wait = self.retry_delay * (2 ** (attempt - 1))
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"API request failed: {e}") from e

            if data.get("status") != "1":
                info = data.get("info", "unknown error")
                if info in ("INVALID_USER_KEY", "USERKEY_PLAT_NOMATCH"):
                    raise RuntimeError(f"Amap API error: {info}. Check your API key.")
                if attempt <= self.max_retries:
                    # Rate limit: wait longer
                    if "CUQPS" in info or "LIMIT" in info:
                        wait = self.retry_delay * (3 ** attempt)
                        time.sleep(wait)
                    continue
                raise RuntimeError(f"Amap API error: {info}")

            return data

        raise RuntimeError(f"API request failed after {self.max_retries} retries")

    # ── geocoding ───────────────────────────────────

    def geocode(self, address: str, city: str = "") -> Tuple[float, float]:
        """Forward geocode: address → (lon, lat)."""
        cached = self.cache.get("road_geocode", address, city)
        if cached:
            return tuple(cached)

        params = {"address": address}
        if city:
            params["city"] = city

        data = self._request("/geocode/geo", params)
        geos = data.get("geocodes", [])
        if not geos:
            raise RuntimeError(f"No geocode result for '{address}'")

        location = geos[0].get("location", "")
        lon_str, lat_str = location.split(",")
        result = (float(lon_str), float(lat_str))
        self.cache.set("road_geocode", address, city, data=result, ttl=604800)
        return result

    def reverse_geocode(self, lon: float, lat: float) -> dict:
        """Reverse geocode: (lon, lat) → address info."""
        cached = self.cache.get("road_regeo", lon, lat)
        if cached:
            return cached

        params = {"location": f"{lon},{lat}", "extensions": "base"}
        data = self._request("/geocode/regeo", params)
        result = data.get("regeocode", {})
        self.cache.set("road_regeo", lon, lat, data=result, ttl=604800)
        return result

    # ── driving direction ───────────────────────────

    def get_driving_info(
        self, origin: Tuple[float, float], destination: Tuple[float, float]
    ) -> dict:
        """
        Get driving route between two points.

        Returns: {"distance": float (metres), "duration": float (seconds),
                  "polyline": str, "strategy": int}
        """
        cached = self.cache.get("road_driving", origin, destination, self.strategy)
        if cached:
            self._record_sample(origin, destination, cached)
            return cached

        params = {
            "origin": f"{origin[0]},{origin[1]}",
            "destination": f"{destination[0]},{destination[1]}",
            "strategy": str(self.strategy),
            "extensions": "all",
        }
        data = self._request("/direction/driving", params)

        route = data.get("route", {})
        paths = route.get("paths", [])
        if not paths:
            raise RuntimeError(f"No route between {origin} and {destination}")

        path = paths[0]
        poly = path.get("polyline", "")
        # Route-level polyline may be empty; extract from step-level polylines instead
        if not poly:
            step_polys = []
            for step in path.get("steps", []):
                sp = step.get("polyline", "")
                if sp:
                    step_polys.append(sp)
            poly = ";".join(step_polys)

        result = {
            "distance": float(path.get("distance", 0)),
            "duration": float(path.get("duration", 0)),
            "polyline": poly,
            "strategy": path.get("strategy", ""),
        }
        self.cache.set("road_driving", origin, destination, self.strategy, data=result, ttl=3600)
        self._record_sample(origin, destination, result)
        return result

    def _record_sample(self, origin, destination, result):
        """Phase 1.1: 被动采集训练样本 (缓存命中和API调用均记录)."""
        if self._collector is None:
            return
        try:
            self._collector.record(
                origin_lon=origin[0], origin_lat=origin[1],
                dest_lon=destination[0], dest_lat=destination[1],
                dist_km=result["distance"] / 1000.0,
                duration_seconds=result["duration"],
                polyline=result.get("polyline", ""),
            )
        except Exception:
            pass

    # ── polyline decoding ───────────────────────────

    @staticmethod
    def _smooth_polyline(coords: List[Tuple[float, float]], iterations: int = 2) -> List[Tuple[float, float]]:
        """
        Smooth a polyline using Chaikin's corner-cutting algorithm.

        Each iteration replaces every sharp corner with two points at 1/4 and 3/4
        along the segment, producing a smoother curve. 2 iterations is enough
        to make routes look natural without oversmoothing.
        """
        if len(coords) < 3:
            return coords
        pts = list(coords)
        for _ in range(iterations):
            new_pts = [pts[0]]
            for i in range(len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i + 1]
                qx = 0.75 * x1 + 0.25 * x2
                qy = 0.75 * y1 + 0.25 * y2
                rx = 0.25 * x1 + 0.75 * x2
                ry = 0.25 * y1 + 0.75 * y2
                new_pts.append((qx, qy))
                new_pts.append((rx, ry))
            new_pts.append(pts[-1])
            pts = new_pts
        return pts

    @staticmethod
    def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
        """
        Decode 高德 polyline string → list of (lon, lat) coordinate pairs.

        高德 format: "lon1,lat1;lon2,lat2;..."  (semicolon-separated)
        """
        if not polyline_str:
            return []
        coords = []
        for pair in polyline_str.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            parts = pair.split(",")
            if len(parts) >= 2:
                coords.append((float(parts[0]), float(parts[1])))
        return coords

    # ── distance matrix ─────────────────────────────

    def compute_distance_matrix(
        self, points: List[Tuple], profile=None, points_are_lonlat: bool = None,
        max_workers: int = 6,
    ) -> np.ndarray:
        """
        Compute N×N distance matrix via 高德 direction API (并行).

        Uses symmetry: only queries (i, j) for i < j, copies to (j, i).
        Multiple threads share a rate-limit lock so the API is never flooded.
        Returns distances in kilometres.
        """
        n = len(points)
        if points_are_lonlat is None:
            points_are_lonlat = self._detect_coordinate_format(points)

        if points_are_lonlat:
            lonlat_pts = [(float(p[0]), float(p[1])) for p in points]
        else:
            lonlat_pts = [(float(p[1]), float(p[0])) for p in points]

        dist_matrix = np.zeros((n, n), dtype=np.float64)
        total_pairs = n * (n - 1) // 2

        # 生成所有 (i, j) 对
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        logger.info(f"Distance matrix: {total_pairs} pairs via Amap "
                     f"(parallel, {max_workers} workers, "
                     f"rate_limit={self.rate_limit_interval}s)")

        failed_pairs = []
        done_count = [0]  # 用 list 以在闭包中可修改
        done_lock = threading.Lock()

        def _fetch_one(i, j):
            """获取一对点之间的距离 (线程安全)."""
            try:
                info = self.get_driving_info(lonlat_pts[i], lonlat_pts[j])
                km = info["distance"] / 1000.0
                return (i, j, km, None)
            except Exception as e:
                # 等待后重试一次
                try:
                    time.sleep(3.0)
                    info = self.get_driving_info(lonlat_pts[i], lonlat_pts[j])
                    km = info["distance"] / 1000.0
                    return (i, j, km, None)
                except Exception:
                    return (i, j, 0.0, e)

        if total_pairs <= 4 or max_workers <= 1:
            # 小规模单线程, 省去线程开销
            for i, j in pairs:
                i_, j_, km, err = _fetch_one(i, j)
                if err:
                    failed_pairs.append((i_, j_))
                else:
                    dist_matrix[i_][j_] = km
                    dist_matrix[j_][i_] = km
                done_count[0] += 1
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_fetch_one, i, j): (i, j) for i, j in pairs}
                for future in as_completed(futures):
                    i_, j_, km, err = future.result()
                    if err:
                        failed_pairs.append((i_, j_))
                        logger.warning(f"API failed for ({i_},{j_}): {err}")
                    else:
                        dist_matrix[i_][j_] = km
                        dist_matrix[j_][i_] = km

                    with done_lock:
                        done_count[0] += 1
                        if total_pairs > 20 and done_count[0] % max(1, total_pairs // 5) == 0:
                            pct = done_count[0] * 100 // total_pairs
                            logger.info(f"  Distance matrix: {done_count[0]}/{total_pairs} ({pct}%)")

        if failed_pairs:
            logger.warning(
                f"高德API: {len(failed_pairs)} 对点使用欧氏距离回退: {failed_pairs[:5]}..."
            )
            for i, j in failed_pairs:
                fallback_km = self._euclidean_fallback(lonlat_pts[i], lonlat_pts[j])
                dist_matrix[i][j] = fallback_km
                dist_matrix[j][i] = fallback_km

        stats = self.cache.stats()
        logger.info(
            f"Distance matrix complete. Cache: {stats['memory_hits']} mem hits, "
            f"{stats['disk_hits']} disk hits, {stats['misses']} API calls."
        )
        return dist_matrix

    # ── route geometry ──────────────────────────────

    def get_route_geometry(
        self, route: List[int], points: List[Tuple],
        points_are_lonlat: bool = None,
    ) -> List[Tuple[float, float]]:
        """
        Convert a VRP route into a road-following coordinate list.

        Returns (lat, lon) tuples for Folium PolyLine rendering.
        """
        if not route or len(route) < 2:
            return []

        if points_are_lonlat is None:
            points_are_lonlat = self._detect_coordinate_format(points)

        if points_are_lonlat:
            lonlat_pts = [(float(p[0]), float(p[1])) for p in points]
        else:
            lonlat_pts = [(float(p[1]), float(p[0])) for p in points]

        # Prepend the exact depot coordinate to ensure connection to the marker
        if points_are_lonlat:
            depot_latlon = (float(points[route[0]][1]), float(points[route[0]][0]))
        else:
            depot_latlon = (float(points[route[0]][0]), float(points[route[0]][1]))

        all_coords: List[Tuple[float, float]] = [depot_latlon]

        for idx in range(len(route) - 1):
            src = route[idx]
            tgt = route[idx + 1]
            if src == tgt:
                continue

            src_latlon = (float(points[src][1]), float(points[src][0])) if points_are_lonlat else (float(points[src][0]), float(points[src][1]))
            tgt_latlon = (float(points[tgt][1]), float(points[tgt][0])) if points_are_lonlat else (float(points[tgt][0]), float(points[tgt][1]))

            info = self.get_driving_info(lonlat_pts[src], lonlat_pts[tgt])
            poly = self.decode_polyline(info["polyline"])
            if poly and len(poly) >= 2:
                # Apply Bezier smoothing to polyline
                smoothed = self._smooth_polyline(poly)
                for lon, lat in smoothed:
                    all_coords.append((lat, lon))
            else:
                # Direct connection as last resort
                all_coords.append(tgt_latlon)

        # Append exact depot coordinate at the end
        if len(route) >= 2:
            last = route[-1]
            if points_are_lonlat:
                all_coords.append((float(points[last][1]), float(points[last][0])))
            else:
                all_coords.append((float(points[last][0]), float(points[last][1])))

        return all_coords

    # ── compatibility stubs ─────────────────────────

    def filter_for_vehicle(self, profile) -> None:
        """No-op: Amap server handles road restrictions."""
        self._vehicle_profile = profile
        logger.info("Amap: vehicle filtering delegated to server-side routing.")

    def download(self, *args, **kwargs):
        """No-op: Amap has no local graph to download."""
        logger.info("Amap: no local graph download needed.")

    def info(self) -> Dict:
        return {
            "provider": "amap",
            "api_key": self.api_key[:4] + "****" + self.api_key[-4:] if len(self.api_key) > 8 else "****",
            "strategy": self.strategy,
            "cache_stats": self.cache.stats(),
        }

    # ── cost estimation & fallback ──────────────────

    # 各城市道路绕行系数 (道路距离/直线距离的经验值)
    CITY_DETOUR_FACTORS = {
        "Beijing": 1.42,
        "Shanghai": 1.38,
        "Guangzhou": 1.35,
        "Shenzhen": 1.32,
        "Chengdu": 1.40,
        "Wuhan": 1.36,
        "Hangzhou": 1.37,
        "Nanjing": 1.39,
        "default": 1.35,
    }

    @staticmethod
    def estimate_api_cost(n_points: int, qps: float = 3.0,
                          price_per_call: float = 0.1) -> dict:
        """预估给定规模下的高德 API 调用次数/耗时/费用.

        距离矩阵: n × (n-1) / 2 次方向 API 调用
        路径几何: 每条路线额外调用 n-1 次

        Args:
            n_points: 配送点数量
            qps: API 每秒配额
            price_per_call: 超出免费额度后每次调用费用(元)

        Returns:
            {
                "distance_matrix_calls": int,
                "route_geometry_calls": int,
                "total_api_calls": int,
                "estimated_time_seconds": float,
                "estimated_cost_yuan": float,
                "free_tier_enough": bool,
                "recommendation": str,
            }
        """
        dm_calls = n_points * (n_points - 1) // 2
        route_calls = n_points - 1  # 最坏情况每辆车单独路段
        total = dm_calls + route_calls

        # 免费额度: 高德 Web API 日调用量 5000 次/天 (个人开发者)
        free_daily = 5000
        billable = max(0, total - free_daily)
        cost = billable * price_per_call

        time_s = total / max(qps, 0.1)

        if total <= free_daily:
            rec = f"免费额度内 ({total}/{free_daily} 次), 安全"
        elif n_points <= 30:
            rec = f"个人开发者额度够用, 建议分时段调用 (间隔≥0.3s)"
        elif n_points <= 50:
            rec = f"建议使用企业版 API (100 QPS), 或启用缓存复用"
        else:
            rec = f"必须使用企业版 + 缓存预热, 否则单次优化费用 ¥{cost:.0f}+"

        return {
            "distance_matrix_calls": dm_calls,
            "route_geometry_calls": route_calls,
            "total_api_calls": total,
            "free_daily_quota": free_daily,
            "estimated_time_seconds": round(time_s, 1),
            "estimated_cost_yuan": round(cost, 1),
            "free_tier_enough": total <= free_daily,
            "recommendation": rec,
        }

    @classmethod
    def fallback_distance(cls, p1: Tuple[float, float], p2: Tuple[float, float],
                          city: str = "default", detour: float = None) -> float:
        """三级回退: 按城市道路系数估算道路距离.

        回退优先级:
          1. 历史缓存数据 + 时段修正
          2. 城市道路系数 × 欧氏距离
          3. 通用绕行系数 × 欧氏距离

        Args:
            p1, p2: (lon, lat) 坐标
            city: 城市名, 用于选择绕行系数
            detour: 手动指定绕行系数, None则自动选择

        Returns:
            估算道路距离 (km)
        """
        import numpy as np
        lon1, lat1 = float(p1[0]), float(p1[1])
        lon2, lat2 = float(p2[0]), float(p2[1])
        mid_lat = np.radians((lat1 + lat2) / 2.0)
        dlat = (lat2 - lat1) * 111.32
        dlon = (lon2 - lon1) * 111.32 * np.cos(mid_lat)
        euclidean = np.sqrt(dlat ** 2 + dlon ** 2)

        if detour is None:
            detour = cls.CITY_DETOUR_FACTORS.get(city,
                      cls.CITY_DETOUR_FACTORS["default"])

        return euclidean * detour

    # ── helpers ─────────────────────────────────────

    @staticmethod
    def _detect_coordinate_format(points: List[Tuple]) -> bool:
        from src.utils.geo_utils import is_lonlat_format
        return is_lonlat_format(points)

    @staticmethod
    def _euclidean_fallback(
        p1: Tuple[float, float], p2: Tuple[float, float],
        city: str = "default",
    ) -> float:
        """Approximate road distance as Euclidean × city detour factor."""
        import numpy as np
        detour = AmapRoadNetwork.CITY_DETOUR_FACTORS.get(city, 1.35)
        lon1, lat1 = float(p1[0]), float(p1[1])
        lon2, lat2 = float(p2[0]), float(p2[1])
        mid_lat = np.radians((lat1 + lat2) / 2.0)
        dlat = (lat2 - lat1) * 111.32
        dlon = (lon2 - lon1) * 111.32 * np.cos(mid_lat)
        return np.sqrt(dlat ** 2 + dlon ** 2) * detour

