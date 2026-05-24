"""
Simulates red/yellow/green traffic light cycles at major intersections.

Lights are placed at nodes where:
- At least 3 edges meet (crossroads, T-junctions)
- At least one incident edge is primary class or higher

Each light has a periodic cycle: green (50%) → yellow (3s) → red (47%).
Phase offsets are randomized so lights are not synchronized.
"""

import math
import random
import logging
from typing import Dict, List, Optional

import networkx as nx

logger = logging.getLogger(__name__)

# Minimum spacing between traffic lights in metres (filters dense urban grids)
_MIN_INTERSECTION_SPACING_M = 300


class TrafficLightModel:
    """
    Simulates traffic light cycles at major intersections.

    Usage::

        model = TrafficLightModel(graph)
        state = model.get_light_state(node_id, 36000)  # 10:00 AM → 'green'
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        cycle_length_seconds: int = 60,
        green_ratio: float = 0.50,
        yellow_seconds: int = 3,
        random_seed: Optional[int] = None,
    ):
        self.graph = graph
        self.cycle_length = cycle_length_seconds
        self.green_duration = int(cycle_length_seconds * green_ratio)
        self.yellow_duration = yellow_seconds
        self.red_duration = (
            cycle_length_seconds - self.green_duration - yellow_seconds
        )

        rng = random.Random(random_seed)

        # Auto-detect intersections controlled by traffic lights
        self._intersections: Dict[int, float] = {}  # node_id → phase_offset
        self._detect_intersections(rng)

        logger.info(
            f"TrafficLightModel: {len(self._intersections)} intersections, "
            f"cycle={cycle_length_seconds}s, green={self.green_duration}s, "
            f"yellow={yellow_seconds}s, red={self.red_duration}s"
        )

    def _detect_intersections(self, rng: random.Random) -> None:
        """
        Detect nodes that qualify as controlled intersections.

        Criteria: degree >= 3 AND at least one incident edge is primary+
        Then filter by minimum spacing to avoid unrealistically dense lights.
        """
        candidates = []

        for node_id in self.graph.nodes():
            in_edges = list(self.graph.in_edges(node_id))
            out_edges = list(self.graph.out_edges(node_id))
            degree = len(set((e[0], e[1]) for e in in_edges + out_edges))

            if degree < 3:
                continue

            has_major_road = False
            for u, v, data in list(self.graph.in_edges(node_id, data=True)) + \
                              list(self.graph.out_edges(node_id, data=True)):
                hw = data.get('highway')
                if isinstance(hw, list):
                    hw = hw[0]
                if hw in ('motorway', 'motorway_link', 'trunk', 'trunk_link',
                          'primary', 'primary_link', 'secondary'):
                    has_major_road = True
                    break

            if has_major_road:
                lat = self.graph.nodes[node_id].get('y', 0)
                lon = self.graph.nodes[node_id].get('x', 0)
                candidates.append((node_id, lat, lon))

        # Spatial filter: keep only candidates spaced >= _MIN_INTERSECTION_SPACING_M apart
        kept = []
        for node_id, lat, lon in candidates:
            too_close = False
            for _, klat, klon in kept:
                dlat = lat - klat
                dlon = lon - klon
                # Approximate metres from degrees at mid-latitudes
                dist_m = math.sqrt((dlat * 111320) ** 2 + (dlon * 111320 * math.cos(math.radians(lat))) ** 2)
                if dist_m < _MIN_INTERSECTION_SPACING_M:
                    too_close = True
                    break
            if not too_close:
                kept.append((node_id, lat, lon))
                self._intersections[node_id] = rng.uniform(0, self.cycle_length)

    def get_light_state(self, node_id: int, timestamp_seconds: int) -> str:
        """
        Return traffic light state at the given time.

        Returns 'green', 'yellow', or 'red'.
        If the node is not a controlled intersection, returns 'green'
        (free pass — no light at minor intersections).
        """
        if node_id not in self._intersections:
            return 'green'

        offset = self._intersections[node_id]
        t = (timestamp_seconds + offset) % self.cycle_length

        if t < self.green_duration:
            return 'green'
        elif t < self.green_duration + self.yellow_duration:
            return 'yellow'
        else:
            return 'red'

    def get_intersections(self) -> List[int]:
        """Return all node IDs that have simulated traffic lights."""
        return list(self._intersections.keys())

    def is_intersection(self, node_id: int) -> bool:
        """Check if a node has a simulated traffic light."""
        return node_id in self._intersections

    def get_light_snapshot(
        self, timestamp_seconds: int
    ) -> Dict[int, str]:
        """
        Get the state of ALL traffic lights at a given time.

        Returns dict {node_id: 'green'|'yellow'|'red'}.
        """
        return {
            nid: self.get_light_state(nid, timestamp_seconds)
            for nid in self._intersections
        }
