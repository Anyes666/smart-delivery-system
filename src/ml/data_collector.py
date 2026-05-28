"""
Phase 1.1 — 行程时间数据采集器 (跨城市)

被动采集: 挂载到 AmapRoadNetwork, 每次 get_driving_info() 自动记录样本。
按城市分文件存储, 支持上海/北京/广州等自动切换。
"""

import atexit
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / ".cache" / "ml"


def _city_from_coords(lon: float, lat: float) -> str:
    """从坐标推断城市名 (简单网格)."""
    from src.ml.travel_time_predictor import _city_slug
    return _city_slug(lon, lat)


class TravelTimeDataCollector:
    """
    被动数据采集器 — 挂载到 AmapRoadNetwork, 按城市分文件。

    用法::

        collector = TravelTimeDataCollector()
        rn._collector = collector
        # 每次 get_driving_info() 自动记录
    """

    def __init__(self, auto_save: bool = True, save_interval: int = 20):
        self._samples: Dict[str, List[dict]] = {}
        self._auto_save = auto_save
        self._save_interval = save_interval
        self._save_counter = 0
        self._dedup: Dict[str, set] = {}
        self._adaptive_trained_count: Dict[str, int] = {}
        self._lock = threading.RLock()  # 可重入: _flush() 可从 record() 内调用
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        atexit.register(self._flush)
        self.real_time_mode = True
        self.simulation_time = None

    def set_real_time_mode(self, mode: bool, sim_time_seconds: int = None):
        """设置采集模式: True=真实时间(默认), False=模拟/快速 (可传入模拟时间)."""
        self.real_time_mode = mode
        if not mode and sim_time_seconds is not None:
            self.simulation_time = sim_time_seconds

    @staticmethod
    def _samples_path(city: str) -> Path:
        return DATA_DIR / f"travel_time_samples_{city}.jsonl"

    def record(
        self,
        origin_lon: float, origin_lat: float,
        dest_lon: float, dest_lat: float,
        dist_km: float, duration_seconds: float,
        timestamp: Optional[float] = None,
        real_time: Optional[bool] = None,
        polyline: str = "",
    ):
        if real_time is None:
            real_time = self.real_time_mode
        if duration_seconds <= 0 or dist_km <= 0.01:
            return

        if timestamp is not None:
            ts = timestamp
        elif not self.real_time_mode and self.simulation_time is not None:
            today = datetime.now().date()
            h = self.simulation_time // 3600
            m = (self.simulation_time % 3600) // 60
            s = self.simulation_time % 60
            ts = datetime(today.year, today.month, today.day, h, m, s).timestamp()
        else:
            ts = time.time()
        dt = datetime.fromtimestamp(ts)

        # 推断城市
        city = _city_from_coords(
            (origin_lon + dest_lon) / 2,
            (origin_lat + dest_lat) / 2,
        )

        # 去重 & 写入 — 加锁保护, 因为 amap_network 用 ThreadPoolExecutor
        # 并发调用 get_driving_info → _record_sample → record()
        dedup_key = (
            round(origin_lon, 3), round(origin_lat, 3),
            round(dest_lon, 3), round(dest_lat, 3),
            dt.strftime("%Y%m%d"), dt.hour // 2,
        )
        with self._lock:
            self._dedup.setdefault(city, set())
            if dedup_key in self._dedup[city]:
                return
            self._dedup[city].add(dedup_key)

            sample = {
                "origin_lon": round(origin_lon, 6),
                "origin_lat": round(origin_lat, 6),
                "dest_lon": round(dest_lon, 6),
                "dest_lat": round(dest_lat, 6),
                "dist_km": round(dist_km, 3),
                "duration_seconds": round(duration_seconds, 1),
                "hour": dt.hour,
                "day_of_week": dt.weekday(),
                "timestamp": ts,
                "city": city,
                "real_time": real_time,
                "polyline": polyline,
            }
            self._samples.setdefault(city, []).append(sample)
            self._save_counter += 1

            if self._auto_save and self._save_counter >= self._save_interval:
                self._flush()
                self._save_counter = 0

    def _flush(self):
        """安全刷盘, 支持 atexit 和 __del__ 双重调用."""
        try:
            with self._lock:
                for city, samples in list(self._samples.items()):
                    if not samples:
                        continue
                    try:
                        path = self._samples_path(city)
                        with open(path, "a", encoding="utf-8") as f:
                            for s in samples:
                                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                        self._samples[city].clear()
                    except Exception:
                        pass
        except Exception:
            pass

    def stats(self, city: str = None) -> Dict:
        result = {"cities": {}}
        total = 0
        for c in self._list_cities():
            n = self._count_file(c)
            mem = len(self._samples.get(c, []))
            result["cities"][c] = {"on_disk": n, "in_memory": mem, "total": n + mem}
            total += n + mem
        result["total"] = total
        if city:
            result["city_stats"] = result["cities"].get(city, {"total": 0})
        return result

    def _list_cities(self) -> List[str]:
        cities = set()
        if DATA_DIR.exists():
            for f in DATA_DIR.glob("travel_time_samples_*.jsonl"):
                name = f.stem.replace("travel_time_samples_", "")
                cities.add(name)
        for c in self._samples:
            cities.add(c)
        return sorted(cities)

    @staticmethod
    def _count_file(city: str) -> int:
        path = DATA_DIR / f"travel_time_samples_{city}.jsonl"
        if not path.exists():
            return 0
        try:
            return sum(1 for _ in open(path, "r", encoding="utf-8"))
        except Exception:
            return 0

    @staticmethod
    def load_all_samples(city: str = None) -> List[dict]:
        """加载训练样本。若 city=None 则加载全部城市。"""
        samples = []
        pattern = f"travel_time_samples_{city or '*'}.jsonl"
        paths = list(DATA_DIR.glob(pattern))
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            samples.append(json.loads(line))
            except Exception as e:
                logger.error(f"加载失败 {path}: {e}")
        logger.info(f"已加载 {len(samples)} 条样本 (city={city or 'all'})")
        return samples

    def get_new_samples(self, city: str = None) -> List[dict]:
        """返回上次自适应训练后新采集的样本 (增量查询)."""
        if city is None:
            # 从文件检测城市
            cities = self._list_cities()
            if not cities:
                return []
            city = cities[0]

        all_samples = self.load_all_samples(city)
        trained = self._adaptive_trained_count.get(city, 0)
        new = all_samples[trained:]
        if new:
            self._adaptive_trained_count[city] = len(all_samples)
        return new

    def __del__(self):
        try:
            self._flush()
        except Exception:
            pass
