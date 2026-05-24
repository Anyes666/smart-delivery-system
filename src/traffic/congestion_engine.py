"""
Simulates time-varying traffic congestion on OSM road segments.

No external API calls. Congestion is modeled procedurally based on:
- Time of day (rush hour peaks at 8:00-9:30 and 17:00-18:30)
- Road class (primary/secondary/residential have different base congestion)
- Random incidents (accidents, construction) with configurable probability
"""

import random
import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class CongestionEngine:
    """
    Simulates time-varying traffic congestion on OSM road segments.

    Usage::

        engine = CongestionEngine(graph, random_seed=42)
        multiplier = engine.get_congestion(u, v, key, 36000)  # 10:00 AM
        engine.simulate_accidents(n_incidents=3)
    """

    # Base multipliers per highway class (1.0 = ideal free flow)
    _BASE_CONGESTION = {
        'motorway': 0.85, 'motorway_link': 0.85,
        'trunk': 0.90, 'trunk_link': 0.90,
        'primary': 0.95, 'primary_link': 0.95,
        'secondary': 1.00,
        'tertiary': 1.10, 'tertiary_link': 1.10,
        'residential': 1.30,
        'service': 1.50,
        'living_street': 1.60,
        'unclassified': 1.30,
    }

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        rush_hour_multiplier: float = 2.0,
        base_congestion: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None,
    ):
        self.graph = graph
        self.rush_hour_multiplier = rush_hour_multiplier
        self.base_congestion = base_congestion or self._BASE_CONGESTION.copy()
        self._rng = random.Random(random_seed)

        # Pre-compute base multipliers for all edges
        self._edge_base: Dict[Tuple[int, int, int], float] = {}
        for u, v, k, data in graph.edges(keys=True, data=True):
            hw = data.get('highway')
            if isinstance(hw, list):
                hw = hw[0]
            self._edge_base[(u, v, k)] = self.base_congestion.get(
                hw, 1.30
            )

        # Active incidents: (u, v, k) -> (start_time, end_time, multiplier)
        self._incidents: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}

        # Per-edge random noise (-0.1 to +0.1) for natural variation
        self._edge_noise: Dict[Tuple[int, int, int], float] = {}
        for key in self._edge_base:
            self._edge_noise[key] = self._rng.uniform(-0.1, 0.1)

        logger.info(
            f"CongestionEngine initialized: {len(self._edge_base)} edges, "
            f"rush_hour_multiplier={rush_hour_multiplier}"
        )

    # ── time-of-day curve ───────────────────────────

    @staticmethod
    def _time_of_day_factor(timestamp_seconds: int) -> float:
        """
        Time-of-day congestion factor (0.7-2.0 range).

        - 00:00-05:00: 0.70 (night, low traffic)
        - 05:00-07:00: ramp 0.70→1.20
        - 07:00-09:30: peak AM, 1.20→2.00→1.50
        - 09:30-16:30: midday, 1.00-1.30
        - 16:30-18:30: peak PM, 1.30→2.00→1.50
        - 18:30-20:00: ramp down 1.50→0.90
        - 20:00-24:00: 0.90→0.80
        """
        t = timestamp_seconds % 86400  # wrap to 24h
        if t < 18000:        # 00:00-05:00
            return 0.70
        elif t < 25200:      # 05:00-07:00
            return 0.70 + (t - 18000) / 7200 * 0.50
        elif t < 34200:      # 07:00-09:30
            frac = (t - 25200) / 9000
            # Parabolic peak: rises to 2.0 at center (8:15), falls to 1.5
            return 1.20 + frac * (1 - frac) * 3.2
        elif t < 59400:      # 09:30-16:30
            return 1.00 + (t - 34200) / 25200 * 0.25
        elif t < 66600:      # 16:30-18:30
            frac = (t - 59400) / 7200
            return 1.30 + frac * (1 - frac) * 2.8
        elif t < 72000:      # 18:30-20:00
            return 1.50 - (t - 66600) / 5400 * 0.60
        else:                # 20:00-24:00
            return 0.90 - (t - 72000) / 14400 * 0.10

    # ── congestion query ────────────────────────────

    def get_congestion_multiplier(
        self, u: int, v: int, key: int, timestamp_seconds: int
    ) -> float:
        """
        Return congestion multiplier for an edge at the given time.

        Multiplier 1.0 = free flow (no delay).
        Combines: time-of-day + road class base + active incidents + noise.
        """
        base = self._edge_base.get((u, v, key), 1.30)
        tod = self._time_of_day_factor(timestamp_seconds)
        noise = self._edge_noise.get((u, v, key), 0.0)

        multiplier = base * tod * self.rush_hour_multiplier / 2.0 + noise

        # Active incidents
        if (u, v, key) in self._incidents:
            start, end, inc_mult = self._incidents[(u, v, key)]
            if start <= timestamp_seconds <= end:
                multiplier *= inc_mult

        return max(0.5, round(multiplier, 3))

    # ── incident simulation ─────────────────────────

    def simulate_accidents(self, n_incidents: int = 2) -> List[Dict]:
        """
        Randomly place accidents/construction events on the graph.

        Each incident affects a random edge for 30-120 minutes with a
        3.0-5.0x congestion boost.
        """
        edges = list(self._edge_base.keys())
        if len(edges) == 0:
            return []

        chosen = self._rng.sample(
            edges, min(n_incidents, len(edges))
        )
        incidents = []
        for u, v, k in chosen:
            start = self._rng.randint(0, 86400)
            duration = self._rng.randint(1800, 7200)  # 30-120 min
            end = start + duration
            mult = round(self._rng.uniform(3.0, 5.0), 1)
            self._incidents[(u, v, k)] = (start, end, mult)
            data = self.graph.get_edge_data(u, v, k)
            highway = data.get('highway', '?') if data else '?'
            incidents.append({
                'u': u, 'v': v, 'key': k,
                'start': start, 'end': end, 'multiplier': mult,
                'highway': highway,
            })
            logger.info(
                f"Incident on {highway} edge ({u},{v},{k}): "
                f"{self._format_time(start)}-{self._format_time(end)}, {mult}x"
            )
        return incidents

    def add_incident(
        self, u: int, v: int, key: int,
        start_time: int, duration_seconds: int,
        multiplier: float = 4.0,
    ) -> None:
        """Manually add a traffic incident (for testing)."""
        self._incidents[(u, v, key)] = (
            start_time, start_time + duration_seconds, multiplier
        )

    def clear_incidents(self) -> None:
        """Remove all simulated incidents."""
        self._incidents.clear()

    # ── heatmap support ─────────────────────────────

    def get_congested_edges(
        self, timestamp_seconds: int, threshold: float = 1.5
    ) -> List[Tuple[int, int, int, float]]:
        """
        Return all edges with congestion multiplier above threshold.

        Used by the heatmap visualization to know which edges to color.
        """
        result = []
        for (u, v, k) in self._edge_base:
            mult = self.get_congestion_multiplier(u, v, k, timestamp_seconds)
            if mult >= threshold:
                result.append((u, v, k, mult))
        return result

    def get_incident_edges(self) -> List[Dict]:
        """Return all active incident details."""
        return [
            {
                'u': u, 'v': v, 'key': k,
                'start': start, 'end': end,
                'multiplier': mult,
            }
            for (u, v, k), (start, end, mult) in self._incidents.items()
        ]

    @staticmethod
    def _format_time(seconds: int) -> str:
        h, r = divmod(seconds % 86400, 3600)
        m, _ = divmod(r, 60)
        return f"{h:02d}:{m:02d}"
