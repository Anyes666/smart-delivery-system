"""
Time-dependent road access restrictions for delivery vehicles.

Examples:
- Trucks banned on primary/secondary roads during rush hour (7:00-9:30, 16:30-18:30)
- Trucks banned on residential/living_street at night (22:00-06:00, noise)
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TimeBasedAccessRules:
    """
    Time-dependent road access restrictions for delivery vehicles.

    These rules do NOT mutate the graph. They provide check methods that
    the routing/visualization layer calls at query time.

    Usage::

        rules = TimeBasedAccessRules()
        if rules.is_edge_restricted(edge_data, 28800):  # 8:00 AM
            ...  # avoid this edge
    """

    DEFAULT_RULES = [
        {
            'name': 'rush_hour_truck_ban',
            'highway_classes': ['primary', 'primary_link',
                               'secondary', 'secondary_link'],
            'time_ranges': [(25200, 34200), (59400, 66600)],
            # 7:00-9:30, 16:30-18:30
            'restriction_type': 'ban_truck',
        },
        {
            'name': 'night_noise_zone',
            'highway_classes': ['residential', 'living_street', 'service'],
            'time_ranges': [(79200, 86400), (0, 21600)],
            # 22:00-06:00
            'restriction_type': 'ban_truck',
        },
    ]

    def __init__(self, rules: Optional[List[Dict]] = None):
        self.rules = rules or self.DEFAULT_RULES
        logger.info(
            f"TimeBasedAccessRules initialized with {len(self.rules)} rule(s)"
        )

    def is_edge_restricted(self, edge_data: dict, timestamp_seconds: int) -> bool:
        """
        Check whether an OSM edge is restricted at the given time.

        Returns True if the edge should be avoided.
        """
        hw_tag = edge_data.get('highway')
        if not hw_tag:
            return False
        hw = hw_tag[0] if isinstance(hw_tag, list) else hw_tag
        t = timestamp_seconds % 86400  # wrap to 24h in seconds

        for rule in self.rules:
            if hw not in rule['highway_classes']:
                continue
            for start, end in rule['time_ranges']:
                if start <= t <= end:
                    return True
        return False

    def get_active_restrictions(
        self, timestamp_seconds: int
    ) -> List[Dict]:
        """
        Return all rules active at the given time.
        Used for visualization legend / info panel.
        """
        t = timestamp_seconds % 86400
        active = []
        for rule in self.rules:
            for start, end in rule['time_ranges']:
                if start <= t <= end:
                    active.append(rule)
                    break
        return active

    def get_restricted_edges(
        self, graph, timestamp_seconds: int
    ) -> List[tuple]:
        """
        Scan the graph and return all edges restricted at the given time.

        Returns list of (u, v, key, rule_name) tuples.
        """
        restricted = []
        for u, v, k, data in graph.edges(keys=True, data=True):
            if self.is_edge_restricted(data, timestamp_seconds):
                hw_tag = data.get('highway')
                hw = hw_tag[0] if isinstance(hw_tag, list) else hw_tag
                # Find matching rule name
                t = timestamp_seconds % 86400
                rule_name = 'unknown'
                for rule in self.rules:
                    if hw in rule['highway_classes']:
                        for start, end in rule['time_ranges']:
                            if start <= t <= end:
                                rule_name = rule['name']
                                break
                restricted.append((u, v, k, rule_name))
        return restricted

    @staticmethod
    def format_time(seconds: int) -> str:
        h, r = divmod(seconds % 86400, 3600)
        m, _ = divmod(r, 60)
        return f"{h:02d}:{m:02d}"

    @classmethod
    def rule_description(cls, rule: Dict) -> str:
        """Human-readable description of a restriction rule."""
        classes = ', '.join(rule['highway_classes'])
        times = ', '.join(
            f"{cls.format_time(s)}-{cls.format_time(e)}"
            for s, e in rule['time_ranges']
        )
        return f"[{rule['name']}] {rule['restriction_type']} on {classes} at {times}"
