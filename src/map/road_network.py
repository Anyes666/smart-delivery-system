"""
Real road-network routing layer for the delivery system.

Replaces Euclidean straight-line distances with OSM-based road network distances.
Uses OSMnx to download OpenStreetMap data and NetworkX for shortest-path computation.

One-way streets are enforced automatically — OSMnx creates a MultiDiGraph where
one-way streets have only a single directed edge in the allowed direction.
"""

import os
import hashlib
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import networkx as nx

from .legal_mask import LegalityMask

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# VehicleProfile
# ──────────────────────────────────────────────

@dataclass
class VehicleProfile:
    """Physical constraints and preferences of a delivery vehicle."""

    max_height_m: float = 4.0       # 0 = no restriction
    max_weight_t: float = 10.0      # 0 = no restriction
    max_width_m: float = 2.5        # 0 = no restriction
    min_highway_class: str = 'residential'  # inclusive lower bound
    penalize_low_class_roads: bool = True


# ──────────────────────────────────────────────
# RoadNetwork
# ──────────────────────────────────────────────

class RoadNetwork:
    """
    Wraps an OSMnx road graph and provides real-road distance matrices.

    Usage::

        rn = RoadNetwork()
        rn.download("Shanghai, China")
        rn.filter_for_vehicle(profile)
        dist_mat = rn.compute_distance_matrix(points, profile)
    """

    def __init__(self, cache_dir: Union[str, Path] = ".cache"):
        self.graph: Optional[nx.MultiDiGraph] = None
        self.graph_projected: Optional[nx.MultiDiGraph] = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._vehicle_profile: Optional[VehicleProfile] = None

    # ── download ────────────────────────────────────

    def download(self,
                 place_name: str = "Shanghai, China",
                 network_type: str = "drive",
                 simplify: bool = True,
                 retain_all: bool = False,
                 truncate_by_edge: bool = True,
                 cache: bool = True) -> nx.MultiDiGraph:
        """
        Download road network from OpenStreetMap via OSMnx.

        On first call this hits the Nominatim geocoding API + Overpass API.
        Subsequent calls with ``cache=True`` load from a local pickle file.

        *If you are in China*, Nominatim may be blocked. Use
        ``download_by_bbox()`` instead with explicit lat/lon bounds.
        """
        safe_name = place_name.replace(', ', '_').replace(' ', '_')
        cache_path = self.cache_dir / f"road_graph_{safe_name}.pkl"

        if cache and cache_path.exists():
            logger.info(f"Loading cached road graph: {cache_path}")
            try:
                self.graph = nx.read_gpickle(str(cache_path))
                logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes, "
                            f"{self.graph.number_of_edges()} edges.")
                return self.graph
            except Exception as e:
                logger.warning(f"Failed to load cached graph: {e}. Re-downloading.")

        logger.info(f"Downloading road network for '{place_name}' ...")
        t0 = time.time()

        import osmnx as ox
        ox.settings.use_cache = True
        ox.settings.log_console = False
        # Shorter timeout for Nominatim (default 180s is way too long)
        ox.settings.timeout = 30

        try:
            self.graph = ox.graph_from_place(
                place_name,
                network_type=network_type,
                simplify=simplify,
                retain_all=retain_all,
                truncate_by_edge=truncate_by_edge,
            )
        except Exception as e:
            ox.settings.timeout = 180  # restore default
            raise RuntimeError(
                f"OSM download failed for '{place_name}': {e}\n\n"
                "If you are in mainland China, Nominatim geocoding is often blocked.\n"
                "Use RoadNetwork.download_by_bbox(north, south, east, west) instead,\n"
                "or pass --bbox 31.40,31.10,121.65,121.30 via CLI.\n"
                "Example for central Shanghai: download_by_bbox(31.40, 31.10, 121.65, 121.30)"
            ) from e

        elapsed = time.time() - t0
        logger.info(f"Downloaded {self.graph.number_of_nodes()} nodes, "
                    f"{self.graph.number_of_edges()} edges in {elapsed:.1f}s.")

        if cache:
            nx.write_gpickle(self.graph, str(cache_path))
            logger.info(f"Saved graph to {cache_path}")

        return self.graph

    def download_by_bbox(self,
                         north: float, south: float,
                         east: float, west: float,
                         network_type: str = "drive",
                         cache: bool = True) -> nx.MultiDiGraph:
        """Download road network for a specific bounding box."""
        bbox_key = f"{north:.3f}_{south:.3f}_{east:.3f}_{west:.3f}"
        cache_path = self.cache_dir / f"road_graph_bbox_{bbox_key}.pkl"

        if cache and cache_path.exists():
            logger.info(f"Loading cached road graph: {cache_path}")
            try:
                self.graph = nx.read_gpickle(str(cache_path))
                return self.graph
            except Exception:
                pass

        import osmnx as ox
        ox.settings.use_cache = True
        ox.settings.log_console = False

        self.graph = ox.graph_from_bbox(
            north=north, south=south, east=east, west=west,
            network_type=network_type,
        )
        if cache:
            nx.write_gpickle(self.graph, str(cache_path))
        return self.graph

    # ── vehicle filtering ───────────────────────────

    def filter_for_vehicle(self, profile: VehicleProfile) -> nx.MultiDiGraph:
        """
        Filter graph edges to only those legal for the vehicle.

        Removes edges with height/weight/width restrictions the vehicle exceeds,
        and edges whose highway class is below the minimum allowed rank.

        Also projects the graph to UTM for accurate distance calculation.
        Returns a new graph (does not mutate ``self.graph``).
        """
        self._vehicle_profile = profile
        logger.info(f"Filtering graph for vehicle profile: "
                    f"max_h={profile.max_height_m}m, "
                    f"max_wt={profile.max_weight_t}t, "
                    f"max_wd={profile.max_width_m}m, "
                    f"min_class={profile.min_highway_class}")

        G = self.graph.copy()

        edges_to_remove = []
        for u, v, k, data in G.edges(keys=True, data=True):
            if not LegalityMask.is_edge_legal(data, profile):
                edges_to_remove.append((u, v, k))

        G.remove_edges_from(edges_to_remove)
        removed = len(edges_to_remove)
        remaining = G.number_of_edges()
        logger.info(f"Removed {removed} illegal edges; {remaining} edges remain.")

        if remaining == 0:
            logger.warning("All edges were removed by the vehicle filter! "
                           "Check vehicle profile constraints. Using unfiltered graph.")
            G = self.graph.copy()

        # Apply road-class penalties if enabled
        if profile.penalize_low_class_roads:
            LegalityMask.apply_road_penalties(G, profile)

        # Project to UTM for accurate distance
        import osmnx as ox
        self.graph_projected = ox.project_graph(G)
        logger.info("Graph projected to UTM for distance calculation.")
        return G

    # ── coordinate detection ────────────────────────

    @staticmethod
    def _detect_coordinate_format(points: List[Tuple]) -> bool:
        from src.utils.geo_utils import is_lonlat_format
        return is_lonlat_format(points)

    # ── nearest node mapping ────────────────────────

    def _map_points_to_nodes(self,
                             points: List[Tuple],
                             points_are_lonlat: bool = True) -> List[int]:
        """
        Map each delivery point to its nearest graph node.

        Returns a list of OSM node IDs, same length as ``points``.
        Uses ``ox.distance.nearest_nodes`` with the unprojected graph.
        """
        import osmnx as ox

        if self.graph is None:
            raise RuntimeError("Road network not loaded. Call download() first.")

        if points_are_lonlat:
            lons = [float(p[0]) for p in points]
            lats = [float(p[1]) for p in points]
        else:
            lats = [float(p[0]) for p in points]
            lons = [float(p[1]) for p in points]

        # Batch nearest-node query
        X = np.array(lons)
        Y = np.array(lats)
        node_ids = ox.distance.nearest_nodes(self.graph, X, Y, return_dist=False)

        result = []
        for i, nid in enumerate(node_ids):
            if isinstance(nid, float) and np.isnan(nid):
                logger.warning(f"Point {i} ({paths[i] if 'paths' in dir() else lons[i]},"
                               f"{lats[i]}) — no nearest OSM node. Using nearest valid node in graph.")
                # Fallback: find closest graph node by brute-force searching a subset
                nid = self._fallback_nearest_node(lons[i], lats[i])
            result.append(int(nid))
        return result

    def _fallback_nearest_node(self, lon: float, lat: float) -> int:
        """
        Find the closest graph node when OSMnx returns NaN.
        Searches a random sample of graph nodes; fast enough for sparse failures.
        """
        min_dist = float('inf')
        best_node = None
        # Sample ~5000 nodes to find approximate nearest
        import random
        nodes = list(self.graph.nodes(data=True))
        sample = random.sample(nodes, min(5000, len(nodes))) if len(nodes) > 5000 else nodes
        for node_id, data in sample:
            nx_lon = data.get('x', 0)
            ny_lat = data.get('y', 0)
            dist = (lon - nx_lon) ** 2 + (lat - ny_lat) ** 2
            if dist < min_dist:
                min_dist = dist
                best_node = node_id
        logger.info(f"Fallback nearest node at distance sqrt({min_dist:.6f}) deg.")
        return best_node if best_node is not None else 0

    # ── distance matrix ─────────────────────────────

    def compute_distance_matrix(self,
                                points: List[Tuple],
                                profile: Optional[VehicleProfile] = None,
                                points_are_lonlat: bool = None) -> np.ndarray:
        """
        Compute an N×N distance matrix using road-network shortest paths.

        Returns distances in **kilometres** (float64).

        :param points:  List of ``(x, y)`` coordinate pairs.
        :param profile: Optional VehicleProfile for edge filtering.
        :param points_are_lonlat:
            ``True`` → first column is longitude;
            ``False`` → first column is latitude;
            ``None`` → auto-detect.
        :returns:  ``np.ndarray`` of shape (N, N).
        """
        if self.graph is None:
            raise RuntimeError("Road network not loaded. Call download() first.")

        # Determine coordinate format
        if points_are_lonlat is None:
            points_are_lonlat = self._detect_coordinate_format(points)
        logger.info(f"Coordinate format: {'(lon, lat)' if points_are_lonlat else '(lat, lon)'}")

        # Choose graph: use projected one if available (accurate distances in metres)
        G = self.graph_projected if self.graph_projected is not None else self.graph

        # Apply vehicle filtering if needed
        if profile is not None and profile != self._vehicle_profile:
            self.filter_for_vehicle(profile)

        # Map points to graph nodes
        node_ids = self._map_points_to_nodes(points, points_are_lonlat)
        n = len(points)

        # Deduplicate: run SSSP only from unique source nodes
        unique_nodes = list(dict.fromkeys(node_ids))  # preserves order, removes dupes
        logger.info(f"Computing distances: {n} points → {len(unique_nodes)} unique graph nodes.")

        dist_matrix = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(dist_matrix, 0.0)

        edge_weight_attr = 'length'  # OSMnx projected graph has 'length' in metres

        for src_node in unique_nodes:
            src_indices = [i for i, nid in enumerate(node_ids) if nid == src_node]

            try:
                lengths = nx.single_source_dijkstra_path_length(
                    G, src_node, weight=edge_weight_attr
                )
            except nx.NetworkXNoPath:
                lengths = {}

            for i in src_indices:
                for j in range(n):
                    if i == j:
                        continue
                    target_node = node_ids[j]
                    if target_node in lengths:
                        dist_matrix[i][j] = lengths[target_node] / 1000.0  # m → km
                    else:
                        # Unreachable on graph — use Euclidean fallback
                        fallback = self._euclidean_fallback(points[i], points[j])
                        dist_matrix[i][j] = fallback
                        logger.debug(f"Unreachable path from {i} to {j}; "
                                     f"Euclidean fallback = {fallback:.2f} km")

        unreachable_count = int(np.sum(np.isinf(dist_matrix)))
        if unreachable_count > 0:
            logger.warning(f"{unreachable_count} cell(s) still inf after fallback. "
                           f"Check graph connectivity.")

        return dist_matrix

    # ── path extraction for visualization ────────────

    def get_route_geometry(self,
                           route: List[int],
                           points: List[Tuple],
                           points_are_lonlat: bool = None) -> List[Tuple[float, float]]:
        """
        Convert a VRP route (list of point indices) into a road-following
        sequence of (lat, lon) coordinates by extracting OSM edge geometries.

        Each consecutive pair of VRP waypoints expands into the full
        OSM shortest-path node sequence with edge geometry interpolation,
        producing a smooth polyline that follows actual road curves.

        :param route: Ordered list of point indices, e.g. [0, 3, 7, 2, 0].
        :param points: All (x, y) coordinate pairs matching the indices.
        :param points_are_lonlat: Coordinate format; None to auto-detect.
        :returns: List of (lat, lon) tuples tracing actual road centerlines.
        :raises ValueError: if graph is not loaded.
        """
        if self.graph is None:
            raise RuntimeError("Road network not loaded. Call download() first.")
        if not route or len(route) < 2:
            return []

        if points_are_lonlat is None:
            points_are_lonlat = self._detect_coordinate_format(points)

        # Map all points to OSM nodes (cached for consistency)
        if not hasattr(self, '_cached_point_nodes') or self._cached_point_nodes is None:
            self._cached_point_nodes = self._map_points_to_nodes(points, points_are_lonlat)
            self._cached_point_coords = points
        elif self._cached_point_coords is not points:
            self._cached_point_nodes = self._map_points_to_nodes(points, points_are_lonlat)
            self._cached_point_coords = points

        node_ids = self._cached_point_nodes
        G = self.graph  # unprojected graph for lat/lon coordinates

        all_coords: List[Tuple[float, float]] = []

        for idx in range(len(route) - 1):
            src_point_idx = route[idx]
            tgt_point_idx = route[idx + 1]
            src_node = node_ids[src_point_idx]
            tgt_node = node_ids[tgt_point_idx]

            if src_node == tgt_node:
                continue

            try:
                path_nodes = nx.shortest_path(G, src_node, tgt_node, weight='length')
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Fallback: direct line between the two points
                p_src = points[src_point_idx]
                p_tgt = points[tgt_point_idx]
                if points_are_lonlat:
                    all_coords.append((float(p_src[1]), float(p_src[0])))
                    all_coords.append((float(p_tgt[1]), float(p_tgt[0])))
                else:
                    all_coords.append((float(p_src[0]), float(p_src[1])))
                    all_coords.append((float(p_tgt[0]), float(p_tgt[1])))
                continue

            # Extract edge geometries along the path
            for pi in range(len(path_nodes) - 1):
                u = path_nodes[pi]
                v = path_nodes[pi + 1]
                # OSMnx MultiDiGraph: get edge data (pick edge 0 if multiple)
                edge_data = G.get_edge_data(u, v)
                if edge_data is None:
                    # Try reverse direction
                    edge_data = G.get_edge_data(v, u)
                if edge_data is None:
                    continue

                data = edge_data[0] if isinstance(edge_data, dict) and 0 in edge_data else edge_data

                geometry = data.get('geometry')
                if geometry is not None and hasattr(geometry, 'coords'):
                    # shapely LineString: coords are (lon, lat) → convert to (lat, lon)
                    for c in geometry.coords:
                        all_coords.append((c[1], c[0]))
                else:
                    # No geometry: use node coordinates
                    for node in (u, v):
                        nd = G.nodes[node]
                        all_coords.append((nd['y'], nd['x']))

            # Remove duplicate junction point between consecutive segments
            # (the last coord of previous segment == first coord of next segment)

        # Deduplicate consecutive identical coordinates
        deduped = []
        for c in all_coords:
            if not deduped or abs(c[0] - deduped[-1][0]) > 1e-7 or abs(c[1] - deduped[-1][1]) > 1e-7:
                deduped.append(c)
        return deduped

    # ── helpers ─────────────────────────────────────

    @staticmethod
    def _euclidean_fallback(p1: Tuple[float, float],
                            p2: Tuple[float, float],
                            detour_factor: float = 1.3) -> float:
        """
        Approximate road distance as Euclidean × typical detour ratio.

        Uses a rough spherical-Earth distance formula for lat/lon pairs.
        Assumes input is ``(lon, lat)``.
        """
        lon1, lat1 = float(p1[0]), float(p1[1])
        lon2, lat2 = float(p2[0]), float(p2[1])
        mid_lat = np.radians((lat1 + lat2) / 2.0)
        dlat = (lat2 - lat1) * 111.32          # km per degree latitude
        dlon = (lon2 - lon1) * 111.32 * np.cos(mid_lat)  # km per degree longitude
        euclidean_km = np.sqrt(dlat ** 2 + dlon ** 2)
        return euclidean_km * detour_factor

    # ── serialization ───────────────────────────────

    def save_graph(self, path: Union[str, Path]) -> None:
        """Save the raw OSM graph to a pickle file."""
        if self.graph is None:
            raise RuntimeError("No graph loaded.")
        nx.write_gpickle(self.graph, str(path))
        logger.info(f"Saved graph to {path}")

    def load_graph(self, path: Union[str, Path]) -> nx.MultiDiGraph:
        """Load a previously saved OSM graph from pickle."""
        self.graph = nx.read_gpickle(str(path))
        logger.info(f"Loaded graph: {self.graph.number_of_nodes()} nodes, "
                    f"{self.graph.number_of_edges()} edges.")
        return self.graph

    # ── info ────────────────────────────────────────

    def info(self) -> Dict:
        """Return a summary dict describing the current graph state."""
        if self.graph is None:
            return {'status': 'not loaded'}
        G = self.graph
        edge_types = {}
        for u, v, k, data in G.edges(keys=True, data=True):
            hw = data.get('highway')
            if isinstance(hw, list):
                hw = hw[0]
            edge_types[hw] = edge_types.get(hw, 0) + 1
        return {
            'status': 'loaded',
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'is_directed': G.is_directed(),
            'projected': self.graph_projected is not None,
            'vehicle_profile': self._vehicle_profile,
            'edge_types': dict(sorted(edge_types.items(), key=lambda x: -x[1])),
        }
