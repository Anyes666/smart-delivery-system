"""
Amap (high-level) map tile support for Folium.

Provides Amap tile URLs and helper to create a Folium map with Amap tiles.
Integrated into EnhancedMapRenderer via config/settings.py TILE_PROVIDER.
"""

import folium
from typing import List, Tuple
from config import settings


class AmapMapVisualizer:
    """Helper: create Folium maps with Amap (high level) tiles."""

    # Amap tile URLs — use http:// (Amap tile servers lack HTTPS certs)
    TILE_LIGHT = (
        "http://webrd0{s}.is.autonavi.com/appmaptile"
        "?lang=zh_cn&size=1&scale=1&style=8"
        "&x={x}&y={y}&z={z}"
    )
    TILE_DARK = (
        "http://webrd0{s}.is.autonavi.com/appmaptile"
        "?lang=zh_cn&size=1&scale=1&style=6"
        "&x={x}&y={y}&z={z}"
    )
    SUBDOMAINS = ['1', '2', '3', '4']
    ATTR = "Amap"

    @classmethod
    def get_tile_url(cls, style: str = "light") -> str:
        return cls.TILE_DARK if style == "dark" else cls.TILE_LIGHT

    @classmethod
    def create_base_map(
        cls,
        points: List[Tuple] = None,
        center: Tuple[float, float] = None,
        zoom_start: int = None,
        dark: bool = False,
    ) -> folium.Map:
        """
        Create a Folium map with Amap tiles, centered on points or center.
        """
        tile_url = cls.get_tile_url("dark" if dark else "light")
        zoom = zoom_start or settings.ZOOM_LEVEL

        if center is None and points:
            if AmapMapVisualizer._detect_lonlat(points):
                lats = [float(p[1]) for p in points]
                lons = [float(p[0]) for p in points]
                center = (sum(lats) / len(lats), sum(lons) / len(lons))
            else:
                center = (
                    sum(float(p[0]) for p in points) / len(points),
                    sum(float(p[1]) for p in points) / len(points),
                )
        elif center is None:
            center = settings.MAP_CENTER

        # Create map with no default tiles, add Amap TileLayer explicitly
        m = folium.Map(
            location=center, zoom_start=zoom,
            tiles=None, attr=cls.ATTR,
        )
        folium.TileLayer(
            tiles=tile_url,
            attr=cls.ATTR,
            name='Amap',
            subdomains=cls.SUBDOMAINS,
            max_zoom=18,
            min_zoom=3,
        ).add_to(m)

        try:
            folium.plugins.Fullscreen(
                position="topright", title="Fullscreen", title_cancel="Exit",
                force_separate_button=True,
            ).add_to(m)
        except Exception:
            pass

        return m

    @staticmethod
    def _detect_lonlat(points: List[Tuple]) -> bool:
        from src.utils.geo_utils import is_lonlat_format
        return is_lonlat_format(points)
