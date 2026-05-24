import logging
import networkx as nx
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

class ShortestPathCalculator:
    def __init__(self, points, road_network: Optional['RoadNetwork'] = None):
        """
        初始化最短路径计算器
        :param points: 配送点坐标列表 [(x1, y1), (x2, y2), ...]
        :param road_network: 可选 RoadNetwork 实例，提供真实路网距离
        """
        self.points = points
        self.n = len(points)
        self.road_network = road_network
        self.distance_matrix = self._calculate_distance_matrix()

    def _calculate_distance_matrix(self):
        """
        计算距离矩阵。
        如果提供了 road_network，使用真实路网距离（公里）；
        否则回退到欧氏距离。
        """
        if self.road_network is not None:
            try:
                logger.info("使用 OSM 真实路网计算距离矩阵...")
                self._road_matrix = self.road_network.compute_distance_matrix(
                    self.points,
                    profile=self.road_network._vehicle_profile,
                )
                return self._road_matrix
            except Exception as e:
                logger.warning(f"OSM 路网路由失败: {e}，回退到欧氏距离。")

        import math
        n = len(self.points)
        dist_mtx = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                lon1, lat1 = float(self.points[i][0]), float(self.points[i][1])
                lon2, lat2 = float(self.points[j][0]), float(self.points[j][1])
                R = 6371.0
                phi1, phi2 = math.radians(lat1), math.radians(lat2)
                dphi = math.radians(lat2 - lat1)
                dlambda = math.radians(lon2 - lon1)
                a = (math.sin(dphi / 2) ** 2 +
                     math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
                km = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                dist_mtx[i][j] = km
                dist_mtx[j][i] = km
        return dist_mtx
    
    def dijkstra_path(self, start_idx, end_idx):
        """Dijkstra算法计算单源最短路径"""
        G = nx.Graph()
        
        # 添加节点
        for i in range(self.n):
            G.add_node(i)
        
        # 添加边（基于距离矩阵）
        for i in range(self.n):
            for j in range(i + 1, self.n):
                G.add_edge(i, j, weight=self.distance_matrix[i][j])
        
        try:
            path = nx.dijkstra_path(G, start_idx, end_idx, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return None
    
    def floyd_warshall_all_pairs(self):
        """Floyd-Warshall算法计算所有点对最短路径"""
        G = nx.Graph()
        
        # 添加节点和边
        for i in range(self.n):
            G.add_node(i)
            for j in range(i + 1, self.n):
                G.add_edge(i, j, weight=self.distance_matrix[i][j])
        
        # 计算所有点对最短路径
        paths = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
        distances = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
        
        return paths, distances
    
    def calculate_tsp_path(self, start_idx=0):
        """使用贪心算法解决TSP问题（单车辆情况）"""
        unvisited = set(range(self.n))
        unvisited.remove(start_idx)
        current = start_idx
        path = [start_idx]
        
        while unvisited:
            next_node = min(unvisited, key=lambda x: self.distance_matrix[current][x])
            path.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        
        # 返回起点
        path.append(start_idx)
        return path
    
    def get_distance(self, i, j):
        """获取两点间距离"""
        return self.distance_matrix[i][j]
    
    def get_path_distance(self, path):
        """计算路径总距离"""
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += self.distance_matrix[path[i]][path[i + 1]]
        return total_distance