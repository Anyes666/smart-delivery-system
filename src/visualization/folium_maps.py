import folium
from folium.plugins import MarkerCluster, PolyLineTextPath
import numpy as np
from config import settings as cfg

class FoliumVisualizer:
    def __init__(self, center_point=(30, 105), zoom_start=5):
        """
        初始化Folium可视化器
        :param center_point: 初始地图中心 (纬度, 经度) - 仅作为默认值
        :param zoom_start: 初始缩放级别
        """
        self.center_point = center_point
        self.zoom_start = zoom_start
        # 12 种高对比度深色调，在高德地图白底道路上清晰可见
        self.colors = [
            '#E60000',  # 纯红
            '#FF00FF',  # 品红
            '#FF6600',  # 橙红
            '#CC00CC',  # 深紫
            '#0066FF',  # 深蓝
            '#CC0000',  # 暗红
            '#FF0099',  # 玫红
            '#9933FF',  # 紫罗兰
            '#FF3300',  # 朱红
            '#0099FF',  # 宝蓝
            '#CC3300',  # 铁锈红
            '#6600CC',  # 深紫罗兰
        ]

    def _generate_colors(self, n):
        """从高对比度色板中取 n 种颜色，不足则循环。"""
        if n <= 0:
            return []
        return [self.colors[i % len(self.colors)] for i in range(n)]
    
    def _normalize_points(self, points):
        """
        标准化点为 (lon, lat) 格式。能自动检测输入是 (lat, lon) 还是 (lon, lat)。
        策略：纬度范围 [-90, 90]，经度范围 [-180, 180]。
        若第一列出现超出 [-90, 90] 的值，则第一列必为经度，数据已是 (lon, lat)。
        """
        if not points:
            return []
        # 检查第一列是否有超出纬度范围的值（> 90 或 < -90），若有则必为经度
        first_is_lon = False
        for p in points[:min(20, len(points))]:
            try:
                a = float(p[0])
            except Exception:
                continue
            if abs(a) > 90:
                first_is_lon = True
                break
        if first_is_lon:
            return [(float(p[0]), float(p[1])) for p in points]
        # 第一列全在 [-90, 90] 内，可能是 (lat, lon)，需翻转
        return [(float(p[1]), float(p[0])) for p in points]
    
    def _spread_duplicate_points(self, points):
        """
        对完全相同坐标的点做确定性微偏移，返回新点列表（lon, lat）长度不变。
        这样在 Plotly 中能看到重叠点的数量，同时保持原始索引对应关系。
        """
        if not points:
            return []
        from math import cos, sin, pi
        # 统计相同坐标的组
        groups = {}
        for idx, p in enumerate(points):
            key = (round(float(p[0]), 6), round(float(p[1]), 6))
            groups.setdefault(key, []).append(idx)
        n = len(points)
        base = 0.05 / max(1, n**0.5)  # degree尺度，随点数调整
        new_points = [tuple(p) for p in points]
        for key, idxs in groups.items():
            m = len(idxs)
            if m == 1:
                continue
            for j, idx in enumerate(idxs):
                angle = 2 * pi * j / m
                # 尺度随组内序号略增，保证不完全重合
                r = base * (0.5 + 0.5 * (j / max(1, m - 1)))
                dx = r * cos(angle)
                dy = r * sin(angle)
                lon0, lat0 = float(points[idx][0]), float(points[idx][1])
                new_points[idx] = (lon0 + dx, lat0 + dy)
        return new_points
    
    def create_base_map(self):
        """创建基础地图"""
        m = folium.Map(
            location=self.center_point,
            zoom_start=self.zoom_start,
            tiles=cfg.MAP_TILE,
            attr=f'Map data © {cfg.MAP_TILE}'
        )
        
        # 添加全屏控件
        try:
            folium.plugins.Fullscreen(
                position='topright',
                title='全屏',
                title_cancel='退出全屏',
                force_separate_button=True
            ).add_to(m)
        except (ImportError, AttributeError):
            pass

        return m
    
    def add_delivery_points(self, m, points, depot_index=0, point_names=None):
        """
        添加配送点标记
        :param m: Folium地图对象
        :param points: 点坐标列表 [(lon1, lat1), (lon2, lat2), ...]
        :param depot_index: 配送中心索引
        :param point_names: 点名称列表
        """
        marker_cluster = MarkerCluster().add_to(m)
        
        for i, point in enumerate(points):
            # 确保 point 是 (lon, lat) 格式
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                lon, lat = point[0], point[1]
            else:
                continue
                
            # 处理名称显示
            popup_text = f"点 {i}"
            if point_names and i < len(point_names):
                popup_text = point_names[i]
            
            if i == depot_index:
                # 配送中心（红色标记）
                folium.Marker(
                    [lat, lon],
                    popup=f"<b>配送中心</b><br>{popup_text}",
                    icon=folium.Icon(color='red', icon='home', prefix='fa'),
                    tooltip='配送中心'
                ).add_to(marker_cluster)
            else:
                # 普通配送点
                folium.Marker(
                    [lat, lon],
                    popup=f"<b>配送点 {i}</b><br>{popup_text}",
                    icon=folium.Icon(color='blue', icon='info-sign'),
                    tooltip=popup_text
                ).add_to(marker_cluster)
        
        return m
    
    def add_routes(self, m, routes, points, route_colors=None):
        """
        添加配送路线
        :param m: Folium地图对象
        :param routes: 路线列表，每个路线是点索引列表
        :param points: 点坐标列表 [(lon1, lat1), (lon2, lat2), ...]
        :param route_colors: 路线颜色列表
        """
        if route_colors is None:
            # 为每辆车生成足够多的颜色以避免重复
            route_colors = self._generate_colors(len(routes)) if routes else self.colors
        
        for vehicle_id, route in enumerate(routes):
            if not route or len(route) < 2:
                continue
            
            # 转换坐标为 (lat, lon) 供 Folium 使用
            route_coords = [(points[i][1], points[i][0]) for i in route]
            
            color = route_colors[vehicle_id % len(route_colors)]
            
            # 创建路线 PolyLine
            # smooth_factor=0 减少曲线平滑度，让线条更贴合实际路径
            polyline = folium.PolyLine(
                route_coords,
                color=color,
                weight=4,
                opacity=0.8,
                smooth_factor=1.0,
                popup=f'车辆 {vehicle_id + 1}<br>站点数: {len(route)}'
            )
            polyline.add_to(m)
            
            # 尝试添加箭头（如果插件可用）
            try:
                PolyLineTextPath(
                    polyline,
                    text=' ➤ ',
                    repeat=True,
                    offset=6,
                    attributes={'fill': color, 'font-weight': 'bold', 'font-size': '12px'}
                ).add_to(m)
            except (ImportError, AttributeError):
                pass

            # 添加站点序号标记
            for idx, point_idx in enumerate(route):
                if point_idx < len(points):
                    lat, lon = points[point_idx][1], points[point_idx][0]
                    # 只在非配送中心点或特定逻辑下显示序号，避免遮挡
                    if idx > 0: 
                        folium.Marker(
                            [lat, lon],
                            icon=folium.DivIcon(
                                html=f'<div style="font-weight: bold; color: {color}; '
                                     f'background-color: white; border: 1px solid {color}; '
                                     f'border-radius: 50%; width: 20px; height: 20px; '
                                     f'display: flex; align-items: center; justify-content: center;'
                                     f'font-size: 12px; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);">{idx}</div>'
                            ),
                            tooltip=f'第 {idx} 站'
                        ).add_to(m)
        
        return m
    
    def create_complete_map(self, points, routes, depot_index=0, point_names=None, route_colors=None, title="配送路线优化"):
        """
        创建完整地图（修复了参数缺失和地图自适应问题）
        """
        # 1. 创建基础地图
        m = self.create_base_map()
        
        # 标准化点为 (lon, lat)
        norm_points = self._normalize_points(points)
        # 对重复点做微偏移以便在可视化上区分重叠点（保持索引一致）
        disp_points = self._spread_duplicate_points(norm_points)
        
        # 2. 添加图层（使用偏移后的点）
        self.add_delivery_points(m, disp_points, depot_index, point_names)
        self.add_routes(m, routes, disp_points, route_colors)
        
        # 3. 智能调整视野：根据所有点的边界自动缩放
        if disp_points:
            # 提取所有 lat, lon
            lats = [p[1] for p in disp_points]
            lons = [p[0] for p in disp_points]
            # 计算边界
            bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
            # 增加一点 padding 防止点贴边
            m.fit_bounds(bounds, padding=(20, 20))
        
        # 4. 添加标题 (使用 MacroElement 避免干扰地图逻辑)
        title_html = f'''
            <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); 
                        z-index: 9999; background: white; padding: 10px 20px; 
                        border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.2); text-align: center;">
                <h3 style="margin:0; color: #333;">{title}</h3>
                <p style="margin:5px 0 0 0; color: #666; font-size: 14px;">
                    车辆数: {len(routes)} | 配送点数: {len(points)-1}
                </p>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # 5. 添加简单图例
        legend_html = '''
        <div style="position: fixed; bottom: 20px; right: 20px; width: 160px; height: auto; 
                    border:1px solid grey; z-index:9999; font-size:12px;
                    background-color:white; opacity: 0.9; padding: 10px; border-radius: 5px;">
            <b>图例</b><br>
            <i style="background:red; width:12px; height:12px; display:inline-block; margin: 5px;"></i> 配送中心<br>
            <i style="background:blue; width:12px; height:12px; display:inline-block; margin: 5px;"></i> 配送点<br>
            <i style="background:blue; width:20px; height:2px; display:inline-block; margin: 5px;"></i> 车辆路线
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def save_map(self, m, filename="delivery_map.html"):
        """保存地图到指定文件，确保目录存在"""
        import os
        # 如果传入的是完整路径，则直接使用；否则作为相对路径保存
        dirpath = os.path.dirname(filename)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
            filepath = filename
        else:
            filepath = filename
        m.save(filepath)
        print(f"[OK] 地图已生成: {filepath}")
        return filepath