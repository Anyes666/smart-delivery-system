"""
Legal road-edge filter for vehicle physical constraints.

Validates OSM edge attributes (maxheight, maxweight, maxwidth, highway class)
against a VehicleProfile to determine if a delivery truck can traverse each edge.
"""

import numpy as np
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LegalityMask:
    """Stateless utility for checking road-edge legality for a given vehicle."""

    HIGHWAY_CLASS_RANKS = {
        'motorway': 1, 'motorway_link': 1,
        'trunk': 2, 'trunk_link': 2,
        'primary': 3, 'primary_link': 3,
        'secondary': 4, 'secondary_link': 4,
        'tertiary': 5, 'tertiary_link': 5,
        'residential': 6,
        'service': 7,
        'living_street': 8,
        'unclassified': 9,
        'track': 10,
        'path': 11,
        'footway': 12,
        'pedestrian': 12,
        'cycleway': 13,
    }

    # Cost multiplier: heavier penalty = less desirable for truck routing
    _ROAD_PENALTIES = {
        'motorway': 1.0, 'motorway_link': 1.0,
        'trunk': 1.0, 'trunk_link': 1.0,
        'primary': 1.0, 'primary_link': 1.0,
        'secondary': 1.0, 'secondary_link': 1.0,
        'tertiary': 1.0, 'tertiary_link': 1.0,
        'residential': 1.5,
        'service': 2.5,
        'living_street': 3.0,
        'unclassified': 3.0,
        'track': 5.0,
        'path': 10.0,
        'footway': 10.0,
        'pedestrian': 10.0,
        'cycleway': 10.0,
    }

    @staticmethod
    def parse_osm_restriction(tag_value: Optional[str]) -> Optional[float]:
        """
        Parse OSM restriction tags (maxheight, maxweight, maxwidth) to numeric value.

        Handles: "4.0", "4", "4 m", "4.5 t", "default", "none", None, NaN.
        Returns None when no actionable restriction exists.
        """
        if tag_value is None:
            return None
        if isinstance(tag_value, float) and np.isnan(tag_value):
            return None
        s = str(tag_value).strip().lower()
        if s in ('', 'default', 'none', 'no_sign', 'n/a', 'restricted'):
            return None
        m = re.match(r'^(\d+(?:\.\d+)?)\s*(m|t|ton|ft)?$', s)
        if m:
            return float(m.group(1))
        logger.debug(f"Unparseable OSM restriction value: '{tag_value}'")
        return None

    @classmethod
    def is_edge_legal(cls, edge_data: dict, profile) -> bool:
        """
        Check whether a single OSM edge can be traversed by the given vehicle.

        Returns True if the edge is legal, False if it should be removed from the graph.
        """
        # 1. Height constraint
        if profile.max_height_m > 0:
            h_val = cls.parse_osm_restriction(edge_data.get('maxheight'))
            if h_val is not None and h_val < profile.max_height_m:
                return False

        # 2. Weight constraint
        if profile.max_weight_t > 0:
            w_val = cls.parse_osm_restriction(edge_data.get('maxweight'))
            if w_val is not None and w_val < profile.max_weight_t:
                return False

        # 3. Width constraint
        if profile.max_width_m > 0:
            wi_val = cls.parse_osm_restriction(edge_data.get('maxwidth'))
            if wi_val is not None and wi_val < profile.max_width_m:
                return False

        # 4. Highway class filter
        highway_tag = edge_data.get('highway')
        if highway_tag:
            hw = highway_tag[0] if isinstance(highway_tag, list) else highway_tag
            edge_rank = cls.HIGHWAY_CLASS_RANKS.get(hw, 99)
            min_rank = cls.HIGHWAY_CLASS_RANKS.get(profile.min_highway_class, 6)
            if edge_rank > min_rank:
                return False

        return True

    @classmethod
    def get_road_penalty(cls, highway_tag: Optional[str]) -> float:
        """
        Return cost multiplier for a road class (used to adjust edge weight).

        Motorway/trunk/primary: 1.0 (no penalty, preferred for trucks).
        Residential: 1.5 (slight penalty — narrow, pedestrians).
        Service: 2.5 (significant penalty — tight turns, low speed).
        Track/path: 5.0-10.0 (essentially avoided unless destination is there).
        """
        if not highway_tag:
            return 3.0
        hw = highway_tag[0] if isinstance(highway_tag, list) else highway_tag
        return cls._ROAD_PENALTIES.get(hw, 3.0)

    @classmethod
    def apply_road_penalties(cls, graph, profile) -> None:
        """
        Modify edge 'length' in graph by multiplying with road-class penalty.

        This ensures Dijkstra prefers main roads over narrow alleys for trucks.
        Only run this if profile.penalize_low_class_roads is True.
        Mutates the graph in-place.
        """
        if not getattr(profile, 'penalize_low_class_roads', True):
            return
        for u, v, k, data in graph.edges(keys=True, data=True):
            highway = data.get('highway')
            penalty = cls.get_road_penalty(highway)
            if penalty > 1.0 and 'length' in data:
                data['length'] = data['length'] * penalty
        logger.info(f"Applied road-class penalties to graph edges.")
