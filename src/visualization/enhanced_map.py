"""
Phase 3 Digital Twin Map Renderer.

Produces Folium maps with:
- 贴路行驶 车辆路线 (curved, using OSM edge geometry)
- 拥堵热力图 overlay (color-coded road segments)
- Traffic light markers at controlled intersections
- Time-restricted road overlays (dashed lines)
- Dark tile theme (free CartoDB dark_matter)
- LayerControl for toggling overlays

All features degrade gracefully when dependencies are unavailable.
"""

import logging
from typing import Dict, List, Optional, Tuple

import folium
from folium import LayerControl, FeatureGroup

from config import settings

logger = logging.getLogger(__name__)


class EnhancedMapRenderer:
    """
    Phase 3 digital twin renderer for VRP delivery routes.

    Usage::

        renderer = EnhancedMapRenderer(
            road_network=rn,
            congestion_engine=ce,
            traffic_lights=tl,
            time_rules=tr,
            use_dark_theme=True,
        )
        folium_map = renderer.create_digital_twin_map(
            routes=vrp_routes,
            points=all_points,
            depot_index=0,
            timestamp_seconds=36000,
            title="数字孪生 - 10:00 AM",
        )
    """

    def __init__(
        self,
        road_network=None,
        congestion_engine=None,
        traffic_lights=None,
        time_rules=None,
        use_dark_theme: bool = False,
    ):
        self.road_network = road_network
        self.congestion_engine = congestion_engine
        self.traffic_lights = traffic_lights
        self.time_rules = time_rules
        self.use_dark_theme = use_dark_theme

        cfg = settings.VISUALIZATION_CONFIG
        self.show_heatmap = cfg.get('show_congestion_heatmap', True)
        self.show_lights = cfg.get('show_traffic_lights', True)
        self.show_restrictions = cfg.get('show_time_rules', True)
        self.road_following = cfg.get('road_following_vehicles', True)
        self.heatmap_opacity = cfg.get('heatmap_opacity', 0.55)
        self.route_opacity = cfg.get('route_line_opacity', 0.85)
        self.line_width = cfg.get('line_width', 5)
        self.route_colors = cfg.get('route_colors', [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        ])

        self.has_road_network = (
            road_network is not None
            and (road_network.graph is not None
                 or hasattr(road_network, 'get_route_geometry'))
        )
        self.has_congestion = congestion_engine is not None
        self.has_lights = traffic_lights is not None
        self.has_time_rules = time_rules is not None

        logger.info(
            f"EnhancedMapRenderer: road_network={self.has_road_network}, "
            f"congestion={self.has_congestion}, traffic_lights={self.has_lights}, "
            f"time_rules={self.has_time_rules}, dark_theme={use_dark_theme}"
        )

    # ── main entry ──────────────────────────────────

    def create_digital_twin_map(
        self,
        routes: List[List[int]],
        points: List[Tuple],
        depot_index: int = 0,
        point_names: Optional[List[str]] = None,
        timestamp_seconds: int = None,
        title: str = "数字孪生配送路线",
        incidents: Optional[List[Dict]] = None,
        distance_matrix = None,
        traffic_lights_data: Optional[Dict] = None,
    ) -> folium.Map:
        """
        Create the complete digital twin Folium map.

        :param routes: VRP routes as list of point-index lists.
        :param points: All (x, y) coordinate pairs.
        :param depot_index: Index of the depot/warehouse point.
        :param point_names: Optional labels per point.
        :param timestamp_seconds: Simulation time (seconds since midnight).
        :param title: Map title displayed in overlay header.
        :param traffic_lights_data: Pre-generated traffic light data from route_traffic_lights.
        :returns: ``folium.Map`` with all layers.
        """
        if timestamp_seconds is None:
            timestamp_seconds = settings.DEFAULT_SIMULATION_TIME

        # 1. Base map
        m = self._create_base_map(points)

        # 2. Delivery points (always shown)
        self._add_delivery_points(m, points, depot_index, point_names)

        # 3. Routes — one toggleable layer per vehicle
        self._add_route_polylines(
            m, routes, points, depot_index, timestamp_seconds
        )

        # 4. Congestion heatmap — follows road polylines
        if self.show_heatmap:
            heat_group = FeatureGroup(name="交通拥堵", show=False)
            self._add_synthetic_congestion(heat_group, routes, points, timestamp_seconds)
            heat_group.add_to(m)

        # 5. Traffic lights — prefer pre-generated data, fall back to synthetic
        if self.show_lights:
            light_group = FeatureGroup(name="路口红绿灯", show=True)
            if traffic_lights_data and traffic_lights_data.get('lights'):
                self._add_route_traffic_lights(light_group, traffic_lights_data)
            elif self.has_lights and self.has_road_network:
                self._add_traffic_light_markers(light_group, timestamp_seconds)
            else:
                self._add_synthetic_traffic_lights(light_group, routes, points, timestamp_seconds, distance_matrix)
            light_group.add_to(m)

        # 6. Time-restricted roads (OSM only, or synthetic time overlay on routes)
        if self.show_restrictions:
            restrict_group = FeatureGroup(name="Time Restrictions", show=False)
            if self.has_time_rules and self.has_road_network:
                self._add_restricted_road_overlay(restrict_group, timestamp_seconds)
            else:
                self._add_synthetic_restrictions(restrict_group, routes, points, timestamp_seconds)
            restrict_group.add_to(m)

        # 7. Accident incidents (if any)
        if incidents:
            accident_group = FeatureGroup(name="Accidents / Incidents", show=True)
            self._add_accident_layer(accident_group, incidents, points)
            accident_group.add_to(m)

        # 8. Layer control
        LayerControl(collapsed=False).add_to(m)

        # 8. Title overlay
        self._add_title_overlay(m, title, routes, points, timestamp_seconds)

        # 9. 图例
        self._add_legend(m)

        return m

    # ── base map ─────────────────────────────────────

    def _create_base_map(self, points: List[Tuple]) -> folium.Map:
        center = settings.MAP_CENTER  # (lat, lon)

        # If we have actual points, center on them
        if points:
            if self._detect_lonlat(points):
                lats = [float(p[1]) for p in points]
                lons = [float(p[0]) for p in points]
                center = (
                    sum(lats) / len(lats),
                    sum(lons) / len(lons),
                )
            else:
                center = (
                    sum(float(p[0]) for p in points) / len(points),
                    sum(float(p[1]) for p in points) / len(points),
                )

        provider = getattr(settings, 'TILE_PROVIDER', 'cartodb')

        if provider == 'amap':
            # Create map with no default tiles, then add Amap TileLayer explicitly
            m = folium.Map(
                location=center,
                zoom_start=settings.ZOOM_LEVEL,
                tiles=None,
                attr=getattr(settings, 'AMAP_TILE_ATTR', 'Amap'),
            )
            subdomains = getattr(settings, 'AMAP_TILE_SUBDOMAINS', ['1', '2', '3', '4'])
            folium.TileLayer(
                tiles=settings.AMAP_TILE_URL,
                attr=getattr(settings, 'AMAP_TILE_ATTR', 'Amap'),
                name='Amap',
                subdomains=subdomains,
                max_zoom=18,
                min_zoom=3,
            ).add_to(m)
        else:
            tile = settings.DARK_TILE if self.use_dark_theme else settings.MAP_TILE
            m = folium.Map(
                location=center,
                zoom_start=settings.ZOOM_LEVEL,
                tiles=tile,
                attr='Map data (C) OpenStreetMap contributors',
            )

        try:
            folium.plugins.Fullscreen(
                position='topright',
                title='全屏',
                title_cancel='退出全屏',
                force_separate_button=True,
            ).add_to(m)
        except Exception:
            pass

        return m

    # ── delivery points ─────────────────────────────

    def _add_delivery_points(
        self,
        m: folium.Map,
        points: List[Tuple],
        depot_index: int,
        point_names: Optional[List[str]],
    ) -> None:
        """Add depot (red star) and customer (blue dots) markers."""
        from folium.plugins import MarkerCluster
        cluster = MarkerCluster(name="配送点").add_to(m)

        is_lonlat = self._detect_lonlat(points)

        for i, p in enumerate(points):
            if is_lonlat:
                lon, lat = float(p[0]), float(p[1])
            else:
                lat, lon = float(p[0]), float(p[1])

            label = point_names[i] if point_names and i < len(point_names) else f"P{i}"

            if i == depot_index:
                folium.Marker(
                    [lat, lon],
                    popup=f"<b>Distribution Center</b><br>{label}",
                    icon=folium.Icon(color='red', icon='home', prefix='fa'),
                    tooltip='Depot',
                ).add_to(cluster)
            else:
                folium.CircleMarker(
                    [lat, lon],
                    radius=5,
                    color='#1f77b4',
                    fill=True,
                    fill_color='#1f77b4',
                    fill_opacity=0.7,
                    popup=f"<b>Point {i}</b><br>{label}",
                    tooltip=label,
                ).add_to(cluster)

    # ── route polylines ─────────────────────────────

    def _add_route_polylines(
        self,
        m: folium.Map,
        routes: List[List[int]],
        points: List[Tuple],
        depot_index: int,
        timestamp_seconds: int,
    ) -> None:
        """
        Add 贴路行驶 route Polylines.

        When road_network is available, uses OSM edge geometry for smooth curves.
        Otherwise falls back to straight lines between waypoints.
        """
        is_lonlat = self._detect_lonlat(points)

        for vid, route in enumerate(routes):
            if not route or len(route) < 2:
                continue

            # Per-vehicle toggleable layer
            v_group = FeatureGroup(name=f"V{vid+1}", show=True)
            color = self.route_colors[vid % len(self.route_colors)]

            # Try 贴路行驶 geometry
            if self.has_road_network and self.road_following:
                try:
                    geom = self.road_network.get_route_geometry(
                        route, points, points_are_lonlat=is_lonlat
                    )
                    if geom and len(geom) >= 2:
                        folium.PolyLine(
                            geom,
                            color=color,
                            weight=self.line_width,
                            opacity=self.route_opacity,
                            popup=f"Vehicle {vid+1}<br>Stops: {len(route)}<br>Road-following",
                        ).add_to(v_group)
                        self._add_route_stops(v_group, route, points, is_lonlat, color, vid)
                        v_group.add_to(m)
                        continue
                except Exception as e:
                    logger.debug(f"Road geometry failed for vehicle {vid}: {e}")

            # Fallback: straight line — only used when Amap/road geometry fails
            # Draw as dashed line to clearly distinguish from road-following routes
            if is_lonlat:
                coords = [(float(p[1]), float(p[0])) for p in (points[i] for i in route)]
            else:
                coords = [(float(p[0]), float(p[1])) for p in (points[i] for i in route)]

            folium.PolyLine(
                coords,
                color=color,
                weight=self.line_width,
                opacity=self.route_opacity,
                dash_array='5, 5' if not self.has_road_network else None,
                popup=f"Vehicle {vid + 1} (直线)<br>Stops: {len(route)}",
            ).add_to(v_group)

            self._add_route_stops(v_group, route, points, is_lonlat, color, vid)
            v_group.add_to(m)

    def _add_route_stops(
        self,
        group: FeatureGroup,
        route: List[int],
        points: List[Tuple],
        is_lonlat: bool,
        color: str,
        vid: int,
    ) -> None:
        """Add numbered stop markers along the route."""
        for seq, pt_idx in enumerate(route):
            if is_lonlat:
                lon, lat = float(points[pt_idx][0]), float(points[pt_idx][1])
            else:
                lat, lon = float(points[pt_idx][0]), float(points[pt_idx][1])
            if seq == 0:
                continue  # skip depot (already shown)
            folium.CircleMarker(
                [lat, lon],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                tooltip=f"V{vid+1} Stop {seq}",
            ).add_to(group)

    # ── 拥堵热力图 ──────────────────────────

    def _add_congestion_heatmap(
        self,
        group: FeatureGroup,
        timestamp_seconds: int,
    ) -> None:
        """
        Add color-coded PolyLines for congested road segments.

        Colors by 拥堵路段 multiplier:
        - Yellow: 1.3-1.8 (moderate)
        - Orange: 1.8-2.5 (heavy)
        - Red: > 2.5 (severe / accident)
        """
        if not self.has_road_network:
            return

        G = self.road_network.graph
        if G is None:
            return

        edges = self.congestion_engine.get_congested_edges(
            timestamp_seconds, threshold=1.3
        )
        logger.info(f"Rendering {len(edges)} congested edges for heatmap.")

        for u, v, k, mult in edges:
            color = self._congestion_color(mult)
            edge_data = G.get_edge_data(u, v, k)
            if edge_data is None:
                continue

            data = edge_data[0] if isinstance(edge_data, dict) and 0 in edge_data else edge_data
            geometry = data.get('geometry')

            if geometry is not None and hasattr(geometry, 'coords'):
                coords = [(c[1], c[0]) for c in geometry.coords]
            else:
                nu = G.nodes[u]
                nv = G.nodes[v]
                coords = [(nu['y'], nu['x']), (nv['y'], nv['x'])]

            if len(coords) < 2:
                continue

            folium.PolyLine(
                coords,
                color=color,
                weight=2.5,
                opacity=self.heatmap_opacity,
                tooltip=f"拥堵路段: {mult:.1f}x",
            ).add_to(group)

    @staticmethod
    def _congestion_color(multiplier: float) -> str:
        if multiplier < 1.3:
            return '#2ca02c'  # green
        elif multiplier < 1.8:
            return '#ffcc00'  # yellow
        elif multiplier < 2.5:
            return '#ff7f0e'  # orange
        elif multiplier < 4.0:
            return '#d62728'  # red
        else:
            return '#8b0000'  # dark red (accident)

    # ── traffic lights ──────────────────────────────

    def _add_traffic_light_markers(
        self,
        group: FeatureGroup,
        timestamp_seconds: int,
    ) -> None:
        """
        Add CircleMarkers at controlled intersections, colored by light state.
        """
        if not self.has_road_network:
            return

        G = self.road_network.graph
        if G is None:
            return

        for node_id in self.traffic_lights.get_intersections():
            state = self.traffic_lights.get_light_state(node_id, timestamp_seconds)
            node_data = G.nodes[node_id]
            lat, lon = node_data['y'], node_data['x']
            state_colors = {'green': '#2ca02c', 'yellow': '#ffcc00', 'red': '#d62728'}

            folium.CircleMarker(
                [lat, lon],
                radius=6,
                color=state_colors.get(state, '#888888'),
                fill=True,
                fill_color=state_colors.get(state, '#888888'),
                fill_opacity=0.85,
                weight=1,
                tooltip=f"Traffic Light: {state}",
            ).add_to(group)

    # ── time restrictions ───────────────────────────

    def _add_restricted_road_overlay(
        self,
        group: FeatureGroup,
        timestamp_seconds: int,
    ) -> None:
        """
        Add dashed red overlays on time-restricted road segments.
        """
        if not self.has_road_network:
            return

        G = self.road_network.graph
        if G is None:
            return

        restricted = self.time_rules.get_restricted_edges(G, timestamp_seconds)
        logger.info(f"Rendering {len(restricted)} restricted edges for time overlay.")

        for u, v, k, rule_name in restricted:
            edge_data = G.get_edge_data(u, v, k)
            if edge_data is None:
                continue
            data = edge_data[0] if isinstance(edge_data, dict) and 0 in edge_data else edge_data
            geometry = data.get('geometry')

            if geometry is not None and hasattr(geometry, 'coords'):
                coords = [(c[1], c[0]) for c in geometry.coords]
            else:
                nu = G.nodes[u]
                nv = G.nodes[v]
                coords = [(nu['y'], nu['x']), (nv['y'], nv['x'])]

            if len(coords) < 2:
                continue

            folium.PolyLine(
                coords,
                color='#d62728',
                weight=2,
                opacity=0.55,
                dash_array='8, 8',
                tooltip=f"Restricted: {rule_name}",
            ).add_to(group)

    # ── synthetic (no-OSM) traffic rendering ─────────

    def _add_synthetic_congestion(
        self,
        group: FeatureGroup,
        routes: List[List[int]],
        points: List[Tuple],
        timestamp_seconds: int,
    ) -> None:
        """
        Congestion heatmap following actual road polylines from Amap.
        Renders each road segment as a colored polyline matching the routes.
        """
        import random
        rng = random.Random(timestamp_seconds // 300)
        tod = self._time_of_day_factor(timestamp_seconds)

        for vid, route in enumerate(routes):
            if not route or len(route) < 2:
                continue

            # Get road polyline for the route
            geom = None
            if self.has_road_network and self.road_network:
                try:
                    geom = self.road_network.get_route_geometry(route, points)
                except Exception:
                    pass

            if geom and len(geom) >= 2:
                # Break polyline into colored segments
                chunk = max(1, len(geom) // 20)
                for i in range(0, len(geom) - 1, chunk):
                    end = min(i + chunk + 1, len(geom))
                    seg = geom[i:end]
                    noise = rng.uniform(-0.15, 0.15)
                    mult = tod + noise
                    color = self._congestion_color(mult)
                    folium.PolyLine(
                        seg, color=color, weight=5,
                        opacity=self.heatmap_opacity,
                        tooltip=f"V{vid+1}: {mult:.1f}x",
                    ).add_to(group)
            else:
                # Fallback: between waypoints
                is_lonlat = self._detect_lonlat(points)
                for i in range(len(route) - 1):
                    p1 = points[route[i]]; p2 = points[route[i+1]]
                    if is_lonlat:
                        c1 = (float(p1[1]), float(p1[0]))
                        c2 = (float(p2[1]), float(p2[0]))
                    else:
                        c1 = (float(p1[0]), float(p1[1]))
                        c2 = (float(p2[0]), float(p2[1]))
                    noise = rng.uniform(-0.15, 0.15)
                    mult = tod + noise
                    folium.PolyLine(
                        [c1, c2], color=self._congestion_color(mult),
                        weight=5, opacity=self.heatmap_opacity,
                        tooltip=f"V{vid+1}: {mult:.1f}x",
                    ).add_to(group)

    def _add_synthetic_traffic_lights(
        self,
        group: FeatureGroup,
        routes: List[List[int]],
        points: List[Tuple],
        timestamp_seconds: int,
        distance_matrix = None,
    ) -> None:
        """
        Traffic lights at real road intersections extracted from Amap polylines.

        Finds points where driving polylines from different routes intersect
        (within 30m proximity). These are real road junctions, not delivery stops.
        """
        import math
        CYCLE = 60; GREEN = 30; YELLOW = 3

        # Step 1: Collect all polyline coordinates from all routes
        all_polylines = []  # [(vid, [(lat, lon), ...])]
        for vid, route in enumerate(routes):
            if not route or len(route) < 2:
                continue
            try:
                if self.has_road_network and self.road_network:
                    coords = self.road_network.get_route_geometry(route, points)
                    if coords and len(coords) >= 2:
                        all_polylines.append((vid, coords))
            except Exception:
                pass

        # Step 2: Find intersection points (crossing/rejoining) between polyline sets
        intersections = []  # [(lat, lon, route_count)]
        for i in range(len(all_polylines)):
            for j in range(i + 1, len(all_polylines)):
                vid_a, poly_a = all_polylines[i]
                vid_b, poly_b = all_polylines[j]
                for pa in poly_a:
                    for pb in poly_b:
                        dlat = pa[0] - pb[0]
                        dlon = pa[1] - pb[1]
                        dist_deg = math.sqrt(dlat**2 + dlon**2)
                        if dist_deg < 0.0003:  # ~30m at mid-latitudes
                            mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
                            intersections.append(mid)
                            break
                    else:
                        continue
                    break

        # Step 3: Deduplicate nearby intersections
        unique = []
        for pt in intersections:
            is_dup = False
            for u in unique:
                d = math.sqrt((pt[0] - u[0])**2 + (pt[1] - u[1])**2)
                if d < 0.0005:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(pt)

        # Step 4: Place traffic lights at intersections
        max_lights = min(len(unique), 12)
        state_colors = {'green': '#2ca02c', 'yellow': '#ffcc00', 'red': '#d62728'}
        state_cn = {'green': '绿', 'yellow': '黄', 'red': '红'}
        for k, (lat, lon) in enumerate(unique[:max_lights]):
            offset = (k * 23 + 7) % CYCLE
            t = (timestamp_seconds + offset) % CYCLE
            if t < GREEN:
                state = 'green'
            elif t < GREEN + YELLOW:
                state = 'yellow'
            else:
                state = 'red'

            folium.CircleMarker(
                [lat, lon],
                radius=6,
                color=state_colors[state],
                fill=True,
                fill_color=state_colors[state],
                fill_opacity=0.9,
                weight=2,
                tooltip=f'路口信号灯 #{k+1}: {state_cn[state]}灯',
            ).add_to(group)

    def _add_route_traffic_lights(
        self,
        group: FeatureGroup,
        traffic_lights_data: Dict,
    ) -> None:
        """
        渲染预生成的路口红绿灯数据 (来自 route_traffic_lights 模块)。

        每个路口的 CircleMarker:
        - 绿色 = 到达时绿灯, 无需等待
        - 黄色 = 到达时黄灯
        - 红色 = 到达时红灯, 显示等待秒数
        """
        state_colors = {'green': '#2ca02c', 'yellow': '#ffcc00', 'red': '#d62728'}
        state_cn = {'green': '绿', 'yellow': '黄', 'red': '红'}

        lights = traffic_lights_data.get('lights', [])
        summary = traffic_lights_data.get('summary', {})

        for lt in lights:
            lat, lon = lt['lat'], lt['lon']
            state = lt['state']
            wait_s = lt['wait_seconds']
            vid = lt['vehicle_id'] + 1

            tooltip_parts = [f'V{vid} 路口 #{lt["id"]+1}']
            tooltip_parts.append(f'{state_cn.get(state, state)}灯')
            if wait_s > 0:
                tooltip_parts.append(f'等待 {wait_s:.0f}秒')
            tooltip_parts.append(
                f'距起点 {lt["position_m"]:.0f}m'
            )

            folium.CircleMarker(
                [lat, lon],
                radius=7,
                color=state_colors.get(state, '#888888'),
                fill=True,
                fill_color=state_colors.get(state, '#888888'),
                fill_opacity=0.9,
                weight=2,
                tooltip=' | '.join(tooltip_parts),
            ).add_to(group)

        # 汇总标签
        if summary:
            total = summary.get('total_lights', len(lights))
            reds = summary.get('red_lights', 0)
            wait = summary.get('total_wait_seconds', 0)
            logger.info(
                f"红绿灯 Folium 渲染: {total} 个路口, "
                f"{reds} 个红灯, 总等待 {wait:.0f} 秒"
            )

    def _add_synthetic_restrictions(
        self,
        group: FeatureGroup,
        routes: List[List[int]],
        points: List[Tuple],
        timestamp_seconds: int,
    ) -> None:
        """
        Add time-restriction overlays on route segments without OSM.

        During rush hour (7:00-9:30, 16:30-18:30), marks the first segment
        of each route as potentially restricted (simulating urban truck bans).
        """
        t = timestamp_seconds % 86400
        in_rush_hour = (25200 <= t <= 34200) or (59400 <= t <= 66600)
        if not in_rush_hour:
            return

        is_lonlat = self._detect_lonlat(points)
        for vid, route in enumerate(routes):
            if not route or len(route) < 2:
                continue
            for i in range(len(route) - 1):
                p1 = points[route[i]]
                p2 = points[route[i + 1]]
                if is_lonlat:
                    c1 = (float(p1[1]), float(p1[0]))
                    c2 = (float(p2[1]), float(p2[0]))
                else:
                    c1 = (float(p1[0]), float(p1[1]))
                    c2 = (float(p2[0]), float(p2[1]))
                folium.PolyLine(
                    [c1, c2],
                    color='#d62728',
                    weight=2,
                    opacity=0.45,
                    dash_array='8, 8',
                    tooltip=f"Rush hour restriction ({self._fmt_time(timestamp_seconds)})",
                ).add_to(group)

    # ── accident layer ───────────────────────────────

    def _add_accident_layer(
        self,
        group: FeatureGroup,
        incidents: List[Dict],
        points: List[Tuple],
    ) -> None:
        """
        Add accident/incident markers and highlighted segments to the map.

        Each incident gets:
        - A warning icon marker placed directly on the route line
        - The affected road segment highlighted in bright red
        - Popup showing time window and delay multiplier
        """
        is_lonlat = self._detect_lonlat(points)
        for inc in incidents:
            mult = inc.get('multiplier', 4.0)
            start_t = inc.get('start', 0)
            end_t = inc.get('end', 0)

            # Build the polyline path and description
            polyline_path = None
            a, b = None, None

            if 'polyline_coords' in inc and inc['polyline_coords']:
                # Full road polyline from Amap — follows real road shape
                polyline_path = inc['polyline_coords']
                a, b = inc.get('route_segment', (None, None))
                desc = f"路段 {a}->{b}" if a is not None else "Unknown"
            elif 'route_segment' in inc:
                # Straight line between route points
                a, b = inc['route_segment']
                if 0 <= a < len(points) and 0 <= b < len(points):
                    if is_lonlat:
                        c1 = (float(points[a][1]), float(points[a][0]))
                        c2 = (float(points[b][1]), float(points[b][0]))
                    else:
                        c1 = (float(points[a][0]), float(points[a][1]))
                        c2 = (float(points[b][0]), float(points[b][1]))
                    polyline_path = [c1, c2]
                desc = f"路段 {a}->{b}"
            elif 'midpoint' in inc and inc['midpoint']:
                # Have midpoint but no segment — draw a short line through midpoint
                mid = inc['midpoint']
                polyline_path = [(mid[0] - 0.0001, mid[1] - 0.0001),
                                 (mid[0] + 0.0001, mid[1] + 0.0001)]
                desc = "Unknown"
            else:
                # OSM edge incident — use depot as fallback
                if is_lonlat:
                    c1 = (float(points[0][1]), float(points[0][0]))
                else:
                    c1 = (float(points[0][0]), float(points[0][1]))
                polyline_path = [c1, c1]
                desc = f"Edge ({inc.get('u','?')},{inc.get('v','?')})"

            # Derive marker position FROM the drawn polyline — guarantees icon is on the line
            mid_idx = len(polyline_path) // 2
            mid_lat, mid_lon = polyline_path[mid_idx]

            sh, r = divmod(int(start_t), 3600); sm, _ = divmod(r, 60)
            eh, r = divmod(int(end_t), 3600); em, _ = divmod(r, 60)
            popup = (
                f"<b>交通事故!</b><br>"
                f"类型: {inc.get('type', 'accident')}<br>"
                f"时间: {int(sh):02d}:{int(sm):02d} - {int(eh):02d}:{int(em):02d}<br>"
                f"延迟: {mult}x<br>"
                f"位置: {desc}"
            )

            # Draw the affected road segment in bright red
            folium.PolyLine(
                polyline_path,
                color='#ff0000',
                weight=6,
                opacity=0.7,
                popup=popup,
                tooltip=f"事故: {mult}x 拥堵",
            ).add_to(group)

            # Warning icon placed at midpoint of the drawn polyline
            folium.Marker(
                [mid_lat, mid_lon],
                icon=folium.Icon(color='darkred', icon='warning-sign', prefix='fa'),
                popup=popup,
                tooltip=f"交通事故 ({mult}x)",
            ).add_to(group)

    # ── overlays ─────────────────────────────────────

    def _add_title_overlay(
        self,
        m: folium.Map,
        title: str,
        routes: List[List[int]],
        points: List[Tuple],
        timestamp_seconds: int,
    ) -> None:
        h, r = divmod(timestamp_seconds % 86400, 3600)
        minutes, _ = divmod(r, 60)
        time_str = f"{h:02d}:{minutes:02d}"

        active_routes = sum(1 for r in routes if len(r) > 1)
        title_html = f'''
            <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                        z-index: 9999; background: rgba(0,0,0,0.75); padding: 10px 24px;
                        border-radius: 6px; box-shadow: 0 0 16px rgba(0,180,255,0.3); text-align: center;">
                <h3 style="margin:0; color: #00d4ff; font-family: monospace;">{title}</h3>
                <p style="margin:4px 0 0 0; color: #aaa; font-size: 13px;">
                    车辆: {active_routes} | Stops: {len(points)-1} | 模拟时间: {time_str}
                    {' | 暗色主题' if self.use_dark_theme else ''}
                </p>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

    def _add_legend(self, m: folium.Map) -> None:
        """Add a legend panel to the map."""
        legend_html = f'''
        <div style="position: fixed; bottom: 20px; right: 20px; width: 180px;
                    border:1px solid {'#555' if self.use_dark_theme else '#ccc'};
                    z-index:9999; font-size:11px;
                    background-color:{'rgba(30,30,30,0.9)' if self.use_dark_theme else 'rgba(255,255,255,0.92)'};
                    color:{'#ccc' if self.use_dark_theme else '#333'};
                    padding: 10px; border-radius: 5px;">
            <b>图例</b><br>
            <i style="background:red; width:12px; height:12px; display:inline-block; margin:4px 4px 0 0;"></i> 配送中心<br>
            <i style="background:#1f77b4; width:12px; height:12px; display:inline-block; margin:4px 4px 0 0;"></i> 配送点<br>
            <i style="background:#1f77b4; width:20px; height:2px; display:inline-block; margin:0 4px 0 0;"></i> 车辆路线<br>
            <i style="background:#2ca02c; width:20px; height:3px; display:inline-block; margin:0 4px 0 0;"></i> 畅通<br>
            <i style="background:#ffcc00; width:20px; height:3px; display:inline-block; margin:0 4px 0 0;"></i> 缓行<br>
            <i style="background:#ff7f0e; width:20px; height:3px; display:inline-block; margin:0 4px 0 0;"></i> 拥堵<br>
            <i style="background:#d62728; width:20px; height:3px; display:inline-block; margin:0 4px 0 0;"></i> 严重拥堵<br>
            <i style="background:#2ca02c; width:10px; height:10px; border-radius:50%; display:inline-block; margin:4px 4px 0 0;"></i> 绿灯路口<br>
            <i style="background:#d62728; width:10px; height:10px; border-radius:50%; display:inline-block; margin:4px 4px 0 0;"></i> 红灯路口
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

    # ── helpers ──────────────────────────────────────

    @staticmethod
    def _detect_lonlat(points: List[Tuple]) -> bool:
        from src.utils.geo_utils import is_lonlat_format
        return is_lonlat_format(points)

    @staticmethod
    def _time_of_day_factor(timestamp_seconds: int) -> float:
        """Time-of-day 拥堵路段 factor (0.7-2.0 range, peaks at 8:00 and 17:00)."""
        t = timestamp_seconds % 86400
        if t < 18000:
            return 0.70
        elif t < 25200:
            return 0.70 + (t - 18000) / 7200 * 0.50
        elif t < 34200:
            frac = (t - 25200) / 9000
            return 1.20 + frac * (1 - frac) * 3.2
        elif t < 59400:
            return 1.00 + (t - 34200) / 25200 * 0.25
        elif t < 66600:
            frac = (t - 59400) / 7200
            return 1.30 + frac * (1 - frac) * 2.8
        elif t < 72000:
            return 1.50 - (t - 66600) / 5400 * 0.60
        else:
            return 0.90 - (t - 72000) / 14400 * 0.10

    @staticmethod
    def _fmt_time(seconds: int) -> str:
        h, r = divmod(seconds % 86400, 3600)
        m, _ = divmod(r, 60)
        return f"{h:02d}:{m:02d}"



