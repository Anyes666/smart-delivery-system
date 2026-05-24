import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

class PlotlyAnimator:
    def __init__(self, animation_speed=1000):
        """
        初始化Plotly动画器
        :param animation_speed: 每一帧动画的持续时间（毫秒）
        """
        self.animation_speed = animation_speed
        # 高对比度深色调，用于地图路线（保持路线可见性）
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

        # 仪表盘专用商务配色 — 统一蓝色系 + 强调色
        self.DASH_BLUE = '#2563eb'
        self.DASH_ORANGE = '#f97316'
        self.DASH_GREEN = '#10b981'
        self.DASH_RED = '#ef4444'
        self.DASH_YELLOW = '#eab308'
        self.DASH_GRAY = '#6b7280'
        self.DASH_BLUE_GRADIENT = [
            '#eff6ff', '#dbeafe', '#bfdbfe', '#93c5fd',
            '#60a5fa', '#3b82f6', '#2563eb', '#1d4ed8', '#1e40af',
        ]

    def _generate_colors(self, n):
        """从高对比度色板中取 n 种颜色，不足则循环。"""
        if n <= 0:
            return []
        palette = self.colors
        return [palette[i % len(palette)] for i in range(n)]

    def _normalize_points(self, points):
        """
        标准化点为 (lon, lat) 格式。使用共享坐标检测函数。
        """
        if not points:
            return []
        from src.utils.geo_utils import is_lonlat_format
        if is_lonlat_format(points):
            return [(float(p[0]), float(p[1])) for p in points]
        return [(float(p[1]), float(p[0])) for p in points]

    @staticmethod
    def _simplify_coords(xs, ys, max_points=250):
        """下采样坐标数组到 max_points，保持路线形状平滑。"""
        n = len(xs)
        if n <= max_points:
            return xs, ys
        step = (n - 1) / (max_points - 1)
        new_xs, new_ys = [xs[0]], [ys[0]]
        for i in range(1, max_points - 1):
            idx = i * step
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            new_xs.append(xs[lo] * (1 - frac) + xs[hi] * frac)
            new_ys.append(ys[lo] * (1 - frac) + ys[hi] * frac)
        new_xs.append(xs[-1])
        new_ys.append(ys[-1])
        return new_xs, new_ys

    def _spread_duplicate_points(self, points):
        """
        对完全相同坐标的点做确定性微偏移，返回新点列表（lon, lat）长度不变。
        """
        if not points:
            return []
        from math import cos, sin, pi
        groups = {}
        for idx, p in enumerate(points):
            key = (round(float(p[0]), 6), round(float(p[1]), 6))
            groups.setdefault(key, []).append(idx)
        n = len(points)
        base = 0.05 / max(1, n**0.5)
        new_points = [tuple(p) for p in points]
        for key, idxs in groups.items():
            m = len(idxs)
            if m == 1:
                continue
            for j, idx in enumerate(idxs):
                angle = 2 * pi * j / m
                r = base * (0.5 + 0.5 * (j / max(1, m - 1)))
                dx = r * cos(angle)
                dy = r * sin(angle)
                lon0, lat0 = float(points[idx][0]), float(points[idx][1])
                new_points[idx] = (lon0 + dx, lat0 + dy)
        return new_points

    def create_route_animation(self, points, routes, point_names=None,
                               title="VRP 配送路线动态演示", road_network=None):
        """
        创建配送过程动画（每辆车同时出发，已走轨迹为颜色线，未走为灰色虚线）
        :param points: 所有点的坐标列表 [(x1, y1), (x2, y2), ...] (lon, lat) 或 (x,y)
        :param routes: 路线列表，例如 [[0, 1, 3, 0], [0, 2, 4, 0]]
        :param point_names: 点的名称列表
        :param title: 图表标题
        :return: Plotly Figure对象
        """
        import os

        # 1. 数据预处理
        norm_points = self._normalize_points(points)
        # 对重复点做微偏移以便在可视化上区分重叠点（保持索引一致）
        disp_points = self._spread_duplicate_points(norm_points)
        x_coords = [point[0] for point in disp_points]  # lon
        y_coords = [point[1] for point in disp_points]  # lat

        # Pre-compute road-following segment polylines if road_network available
        route_seg_polys = {}  # {vid: {seg_idx: [(lon,lat), ...]}}
        if road_network is not None:
            try:
                for vid, route in enumerate(routes):
                    if not route or len(route) < 2:
                        continue
                    seg_polys = {}
                    for idx in range(len(route) - 1):
                        seg_route = [route[idx], route[idx+1]]
                        try:
                            geom = road_network.get_route_geometry(seg_route, points)
                            if geom:
                                # geom is [(lat,lon), ...] → convert to [(lon,lat), ...]
                                lonlat = [(g[1], g[0]) for g in geom]
                                seg_polys[idx] = lonlat
                        except Exception:
                            # fallback: straight line between the two points
                            a, b = route[idx], route[idx+1]
                            seg_polys[idx] = [(x_coords[a], y_coords[a]), (x_coords[b], y_coords[b])]
                    route_seg_polys[vid] = seg_polys
            except Exception:
                pass

        use_poly = len(route_seg_polys) > 0

        # 2. 创建基础图表布局
        from config import settings as cfg

        # 根据配置或环境变量决定是否使用 Mapbox（需要 MAPBOX_TOKEN）
        token = cfg.MAPBOX_TOKEN or os.environ.get('MAPBOX_TOKEN')
        use_mapbox = bool(token)

        # 准备标签（用于 hover）
        labels = [f'P{i}' for i in range(1, len(points))] if point_names is None else (point_names[1:] if point_names else None)

        if use_mapbox:
            # 使用 Mapbox / CartoDB Positron 样式绘制动画（若无 token 则使用 'open-street-map'）
            style = cfg.MAPBOX_STYLE if token else 'open-street-map'
            fig = go.Figure()

            # 构造动画帧（Mapbox 模式，与非 mapbox 路径相同逻辑）
            frames = []
            node_steps = [max(0, len(route) - 1) for route in routes]
            colors = self._generate_colors(len(routes)) if routes else self.colors

            # Helper: background
            def _mb_bg():
                bg = []
                bg.append(go.Scattermapbox(lon=[x_coords[0]], lat=[y_coords[0]], mode='markers',
                    marker=dict(size=12, color='red', symbol='star'), hoverinfo='skip', showlegend=False))
                if len(x_coords) > 1:
                    bg.append(go.Scattermapbox(lon=x_coords[1:], lat=y_coords[1:], mode='markers',
                        marker=dict(size=4, color='blue', opacity=0.6), showlegend=False, hoverinfo='skip'))
                return bg

            def _mb_depot(vid, color, label='待出发'):
                return go.Scattermapbox(lon=[x_coords[0]], lat=[y_coords[0]], mode='markers',
                    marker=dict(size=10, color=color, symbol='circle', line=dict(color='black', width=1)),
                    hovertext=[f'Vehicle {vid+1} ({label})'], hoverinfo='text', showlegend=False)

            # frame 0: all vehicles parked at depot
            f0 = _mb_bg()
            for vid, vroute in enumerate(routes):
                f0.append(_mb_depot(vid, colors[vid % len(colors)], '待出发'))
            frames.append(go.Frame(data=f0, name='0'))

            frame_count = 1
            for active_vid, route in enumerate(routes):
                if not route or len(route) < 2:
                    continue
                for step_idx in range(1, len(route)):
                    frame_data = _mb_bg()
                    for vid, vroute in enumerate(routes):
                        color = colors[vid % len(colors)]
                        if not vroute or len(vroute) < 2:
                            frame_data.append(_mb_depot(vid, color, '停留'))
                            continue
                        if vid < active_vid:
                            nodes = vroute[:]
                            ll = [x_coords[i] for i in nodes]
                            lt = [y_coords[i] for i in nodes]
                            if len(ll) >= 2:
                                frame_data.append(go.Scattermapbox(lon=ll, lat=lt, mode='lines',
                                    line=dict(width=2, color=color), showlegend=False, hoverinfo='skip'))
                            frame_data.append(_mb_depot(vid, color, '已完成'))
                        elif vid == active_vid:
                            visited = vroute[:step_idx+1]
                            lv = [x_coords[i] for i in visited]
                            tv = [y_coords[i] for i in visited]
                            if len(lv) >= 2:
                                frame_data.append(go.Scattermapbox(lon=lv, lat=tv, mode='lines',
                                    line=dict(width=2, color=color), showlegend=False, hoverinfo='skip'))
                            frame_data.append(go.Scattermapbox(lon=[lv[-1]], lat=[tv[-1]], mode='markers',
                                marker=dict(size=10, color=color, symbol='circle', line=dict(color='black', width=1)),
                                hovertext=[f'Vehicle {vid+1}\nNode {vroute[step_idx]}'], hoverinfo='text', showlegend=False))
                        else:
                            frame_data.append(_mb_depot(vid, color, '待出发'))
                    frames.append(go.Frame(data=frame_data, name=f'{frame_count}'))
                    frame_count += 1

            # 最终帧
            ff = _mb_bg()
            for vid, vroute in enumerate(routes):
                color = colors[vid % len(colors)]
                if not vroute:
                    continue
                ll = [x_coords[i] for i in vroute]
                lt = [y_coords[i] for i in vroute]
                if len(ll) >= 2:
                    ff.append(go.Scattermapbox(lon=ll, lat=lt, mode='lines',
                        line=dict(width=2, color=color), showlegend=False, hoverinfo='skip'))
                ff.append(_mb_depot(vid, color, '已完成'))
            frames.append(go.Frame(data=ff, name=f'{frame_count}'))

            # 归一化帧长度
            max_traces = max(len(f.data) for f in frames) if frames else 0
            normalized_frames = []
            for f in frames:
                data_list = list(f.data)
                while len(data_list) < max_traces:
                    data_list.append(go.Scattermapbox(lon=[], lat=[], mode='markers', marker=dict(size=0), showlegend=False, hoverinfo='skip'))
                normalized_frames.append(go.Frame(data=data_list, name=f.name))

            fig.frames = normalized_frames
            if fig.frames:
                fig.data = []
                for tr in fig.frames[0].data:
                    fig.add_trace(tr)

            center = dict(lat=sum(y_coords) / len(y_coords), lon=sum(x_coords) / len(x_coords)) if points else dict(lat=0, lon=0)
            fig.update_layout(mapbox=dict(accesstoken=token if token else None, style=style, center=center,
                                          zoom=cfg.ZOOM_LEVEL if hasattr(cfg, 'ZOOM_LEVEL') else 5),
                                  title=title, margin=dict(l=0, r=0, t=50, b=0))

            # 动画控件
            steps = [dict(method='animate', args=[[f.name], dict(mode='immediate', frame=dict(duration=self.animation_speed, redraw=True), transition=dict(duration=0))], label=str(i)) for i, f in enumerate(fig.frames)]
            sliders = [dict(active=0, pad={"t": 50}, steps=steps, x=0.02, y=0, len=0.96)]
            fig.update_layout(updatemenus=[dict(type='buttons', showactive=False, y=1.15, x=0.0, xanchor='left', yanchor='top', buttons=[
                dict(label='播放', method='animate', args=[None, dict(frame=dict(duration=self.animation_speed, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                dict(label='暂停', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
            ])], sliders=sliders, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

            return fig

        # 否则使用原有基于笛卡尔坐标的动画实现
        fig = go.Figure()

        # 3. 绘制静态背景（所有点）
        # 配送中心 (红色星形)
        fig.add_trace(go.Scatter(
            x=[x_coords[0]],
            y=[y_coords[0]],
            mode='markers', marker=dict(size=15, color='red', symbol='star'),
            hoverinfo='text', showlegend=True
        ))

        # 普通配送点 (蓝色圆点)
        if len(x_coords) > 1:
            fig.add_trace(go.Scatter(
                x=x_coords[1:],
                y=y_coords[1:],
                mode='markers',
                marker=dict(size=6, color='blue', symbol='circle', opacity=0.7),
                hovertext=labels,
                hoverinfo='text',
                name='配送点',
                showlegend=True
            ))

        # 4. 创建动画帧（顺序模式：一辆车走完再下一辆；每一步对应一个节点）
        frames = []

        # 每辆车按节点移动（每帧代表访问下一个节点）
        node_steps = [max(0, len(route) - 1) for route in routes]
        total_frames = sum(node_steps)

        # 为每辆车生成颜色
        colors = self._generate_colors(len(routes)) if routes else self.colors

        # Helper: build background traces (depot star + customer markers)
        def _bg_traces():
            bg = []
            bg.append(go.Scatter(
                x=[x_coords[0]], y=[y_coords[0]],
                mode='markers', marker=dict(size=12, color='red', symbol='star'),
                hoverinfo='skip', showlegend=False
            ))
            if len(x_coords) > 1:
                bg.append(go.Scatter(
                    x=x_coords[1:], y=y_coords[1:], mode='markers',
                    marker=dict(size=4, color='blue', symbol='circle', opacity=0.6),
                    showlegend=False, hoverinfo='skip'
                ))
            return bg

        # Helper: vehicle marker at depot
        def _vehicle_at_depot(vid, color, label='(待出发)'):
            return go.Scatter(
                x=[x_coords[0]], y=[y_coords[0]], mode='markers',
                marker=dict(size=10, color=color, symbol='circle',
                            line=dict(color='black', width=1)),
                hovertext=[f'Vehicle {vid+1} {label}'], hoverinfo='text',
                showlegend=False
            )

        # frame 0: all vehicles parked at depot, no movement
        frame0_data = _bg_traces()
        for vid, vroute in enumerate(routes):
            color = colors[vid % len(colors)]
            frame0_data.append(_vehicle_at_depot(vid, color, '(待出发)'))
        frames.append(go.Frame(data=frame0_data, name='0'))

        # Helper: get polyline coords for a route up to a given segment index
        def _get_route_coords(vid, up_to_seg=None):
            """Return (xs, ys) for vehicle vid's full or partial route polyline."""
            if use_poly and vid in route_seg_polys:
                segs = route_seg_polys[vid]
                xs, ys = [], []
                max_seg = up_to_seg if up_to_seg is not None else max(segs.keys())
                for si in sorted(segs.keys()):
                    if si > max_seg:
                        break
                    for lon, lat in segs[si]:
                        xs.append(lon)
                        ys.append(lat)
                # 下采样：高德polyline坐标太密会导致路线呈点状，简化到250点以内
                return self._simplify_coords(xs, ys, max_points=250)
            # Fallback: straight line from point coordinates
            route_for = routes[vid] if vid < len(routes) else []
            if up_to_seg is not None:
                pts = route_for[:up_to_seg+2]
            else:
                pts = route_for
            return [x_coords[i] for i in pts], [y_coords[i] for i in pts]

        # Sequential frames: vehicle 0 finishes, then vehicle 1, etc.
        frame_count = 1
        for active_vid, route in enumerate(routes):
            if not route or len(route) < 2:
                continue

            for step_idx in range(1, len(route)):
                frame_data = _bg_traces()

                for vid, vroute in enumerate(routes):
                    color = colors[vid % len(colors)]

                    if not vroute or len(vroute) < 2:
                        frame_data.append(_vehicle_at_depot(vid, color, '(停留)'))
                        continue

                    if vid < active_vid:
                        # Completed: solid line + parked at depot (no dot markers)
                        fx, fy = _get_route_coords(vid)
                        if len(fx) >= 2:
                            frame_data.append(go.Scatter(x=fx, y=fy, mode='lines',
                                line=dict(width=4, color=color),
                                name=f'Vehicle {vid+1}', showlegend=False, hoverinfo='skip'))
                        frame_data.append(_vehicle_at_depot(vid, color, '(已完成)'))

                    elif vid == active_vid:
                        # Active: solid line up to step_idx-1 (no dot markers)
                        fx, fy = _get_route_coords(vid, up_to_seg=step_idx-1)
                        if len(fx) >= 2:
                            frame_data.append(go.Scatter(x=fx, y=fy, mode='lines',
                                line=dict(width=4, color=color),
                                name=f'Vehicle {vid+1}', showlegend=False, hoverinfo='skip'))
                        # Current position marker only
                        cx, cy = fx[-1], fy[-1]
                        frame_data.append(go.Scatter(x=[cx], y=[cy], mode='markers',
                            marker=dict(size=10, color=color, symbol='circle',
                                        line=dict(color='black', width=1)),
                            hovertext=[f'Vehicle {vid+1}\nNode {vroute[step_idx]}'],
                            hoverinfo='text', showlegend=False))

                    else:
                        frame_data.append(_vehicle_at_depot(vid, color, '(待出发)'))

                frames.append(go.Frame(data=frame_data, name=f'{frame_count}'))
                frame_count += 1

        # 最终帧：所有车辆均完成，显示完整轨迹并回到配送中心
        final_frame_data = []
        final_frame_data.append(go.Scatter(x=[x_coords[0]], y=[y_coords[0]], mode='markers+text',
                                          marker=dict(size=15, color='red', symbol='star'), text=['配送中心'],
                                          textposition='top center', showlegend=False, hoverinfo='skip'))
        if len(x_coords) > 1:
            final_frame_data.append(go.Scatter(x=x_coords[1:], y=y_coords[1:], mode='markers',
                                              marker=dict(size=6, color='blue', symbol='circle', opacity=0.7), hovertext=labels,
                                              hoverinfo='text', showlegend=False))

        for vid, vroute in enumerate(routes):
            color = colors[vid % len(colors)]
            if not vroute:
                continue
            fx, fy = _get_route_coords(vid)
            if len(fx) >= 2:
                final_frame_data.append(go.Scatter(x=fx, y=fy, mode='lines', line=dict(width=4, color=color),
                                                  name=f'车{vid+1}', showlegend=False, hoverinfo='skip'))
            final_frame_data.append(go.Scatter(x=[x_coords[0]], y=[y_coords[0]], mode='markers',
                                              marker=dict(size=10, color=color, symbol='circle', line=dict(color='black', width=1)),
                                              hovertext=[f'车辆 {vid+1} (已完成)'], hoverinfo='text', showlegend=False))

        frames.append(go.Frame(data=final_frame_data, name=f'{frame_count}'))

        # 归一化：确保所有帧都有相同数量的 traces（避免因为不同帧 trace 数量不一致导致后续车辆不显示）
        max_traces = max(len(f.data) for f in frames) if frames else 0
        normalized_frames = []
        for f in frames:
            data_list = list(f.data)
            # 用空散点补齐
            while len(data_list) < max_traces:
                data_list.append(go.Scatter(x=[], y=[], mode='markers', marker=dict(size=0), showlegend=False, hoverinfo='skip'))
            normalized_frames.append(go.Frame(data=data_list, name=f.name))

        fig.frames = normalized_frames

        # 将第一帧的数据设为初始 traces，保证动画时轨迹/点不会被覆盖或丢失
        if fig.frames:
            # 重置 fig.data 为第一帧的 traces（frames 中已包含静态背景），确保每帧绘制一致的 trace 结构
            fig.data = []
            for tr in fig.frames[0].data:
                fig.add_trace(tr)

        # 动画控件与布局
        steps = [dict(method='animate', args=[[f.name], dict(mode='immediate', frame=dict(duration=self.animation_speed, redraw=True), transition=dict(duration=0))], label=str(i)) for i, f in enumerate(fig.frames)]
        sliders = [dict(active=0, pad={"t": 50}, steps=steps, x=0.02, y=0, len=0.96)]

        fig.update_layout(
            title=title,
            xaxis=dict(title='经度'), yaxis=dict(title='纬度'),
            updatemenus=[dict(type='buttons', showactive=False, y=1.15, x=0.0, xanchor='left', yanchor='top', buttons=[
                dict(label='播放', method='animate', args=[None, dict(frame=dict(duration=self.animation_speed, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                dict(label='暂停', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
            ])],
            sliders=sliders,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig


    def create_metrics_dashboard(self, routes, distance_matrix, demands,
                                  title="配送指标仪表盘",
                                  road_network=None, points=None,
                                  sim_time=None,
                                  traffic_lights_data=None):
        """
        Comprehensive metrics dashboard with professional business styling.
        """
        import random as _random
        AVG_SPEED_KMH = 40.0
        CYCLE_LENGTH = 60
        GREEN_DUR = 30
        YELLOW_DUR = 3

        # ── color aliases ──
        BLUE = self.DASH_BLUE
        ORANGE = self.DASH_ORANGE
        GREEN = self.DASH_GREEN
        RED = self.DASH_RED
        YELLOW = self.DASH_YELLOW
        GRAY = self.DASH_GRAY

        # ── compute per-vehicle metrics ──
        vehicle_data = []
        total_distance = 0
        total_drive_s = 0
        total_wait_s = 0
        total_red_lights = 0

        lights_by_vehicle = {}
        if traffic_lights_data and traffic_lights_data.get('lights'):
            for lt in traffic_lights_data['lights']:
                vid = lt['vehicle_id']
                lights_by_vehicle.setdefault(vid, []).append(lt)

        for vid, route in enumerate(routes):
            if not route or len(route) < 2:
                vehicle_data.append({'id': vid, 'segments': [], 'lights': [],
                                     'dist': 0, 'drive_s': 0, 'wait_s': 0,
                                     'load': 0, 'stops': 0, 'red_count': 0})
                continue

            v_dist = 0.0
            v_drive_s = 0
            v_wait_s = 0
            v_red = 0
            segs = []
            lights = []

            for i in range(len(route) - 1):
                a, b = route[i], route[i + 1]
                seg_dist = distance_matrix[a][b]
                seg_drive_s = seg_dist / AVG_SPEED_KMH * 3600
                v_dist += seg_dist
                v_drive_s += seg_drive_s
                segs.append({'from': a, 'to': b, 'dist_km': seg_dist, 'drive_s': seg_drive_s})

            if vid in lights_by_vehicle:
                for lt in lights_by_vehicle[vid]:
                    wait_s = lt['wait_seconds']
                    lights.append({
                        'at': f"路口#{lt['id']+1}",
                        'arrival_s': lt['arrival_offset_s'],
                        'wait_s': wait_s,
                        'position_m': lt['position_m'],
                        'state': lt['state'],
                    })
                    if wait_s > 0:
                        v_red += 1
                        v_wait_s += wait_s
            else:
                arrival_offset = 0
                for i in range(1, len(route)):
                    node = route[i]
                    if node == 0:
                        continue
                    arrival_offset += segs[i-1]['drive_s'] if i-1 < len(segs) else segs[-1]['drive_s']
                    arrival_t = (sim_time or 36000) + arrival_offset
                    offset = (node * 17 + 3) % CYCLE_LENGTH
                    phase_t = (arrival_t + offset) % CYCLE_LENGTH
                    wait_s = 0
                    if phase_t >= GREEN_DUR + YELLOW_DUR:
                        wait_s = CYCLE_LENGTH - phase_t
                    elif phase_t >= GREEN_DUR:
                        wait_s = (GREEN_DUR + YELLOW_DUR) - phase_t
                    if wait_s > 0:
                        v_red += 1
                        v_wait_s += wait_s
                        lights.append({'at': node, 'arrival_s': arrival_offset, 'wait_s': wait_s})

            v_load = sum(demands[j] for j in route if j != 0)
            v_stops = len([j for j in route if j != 0])

            vehicle_data.append({
                'id': vid, 'segments': segs, 'lights': lights,
                'dist': v_dist, 'drive_s': v_drive_s, 'wait_s': v_wait_s,
                'load': v_load, 'stops': v_stops, 'red_count': v_red,
            })
            total_distance += v_dist
            total_drive_s += v_drive_s
            total_wait_s += v_wait_s
            total_red_lights += v_red

        active_vehicles = [v for v in vehicle_data if v['stops'] > 0]
        if active_vehicles:
            makespan_vehicle = max(active_vehicles, key=lambda v: v['drive_s'] + v['wait_s'])
            total_time_s = makespan_vehicle['drive_s'] + makespan_vehicle['wait_s']
        else:
            total_time_s = 0

        # ── build dashboard (3 rows × 2 cols) ──
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '配送完成时间', '路线均衡度',
                '每车耗时 (排序)', '每车距离与载重',
                '路段明细', '红灯等待汇总',
            ),
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "xy", "secondary_y": True}],
                [{"type": "table"}, {"type": "bar"}],
            ],
            row_heights=[0.28, 0.36, 0.36],
            vertical_spacing=0.10,
            horizontal_spacing=0.06,
        )

        # ── Row 1, Col 1: Total time gauge ──
        total_min = total_time_s / 60
        gauge_max = max(total_min * 1.5, 10)
        mv = makespan_vehicle if active_vehicles else None
        max_drive_min = (mv['drive_s'] / 60) if mv else 0
        max_wait_min = (mv['wait_s'] / 60) if mv else 0
        max_v_id = f'V{mv["id"]+1}' if mv else ''

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=total_min,
            title={'text': f"<b>配送完成时间</b><br><span style='font-size:12px;color:{BLUE}'>{max_v_id} 驾驶 {max_drive_min:.0f} 分</span>  |  <span style='font-size:12px;color:{ORANGE}'>等待 {max_wait_min:.0f} 分</span>  |  <span style='font-size:12px;color:{RED}'>红灯 {total_red_lights} 个</span>", 'font': {'size': 13}},
            number={'suffix': " 分", 'font': {'size': 36, 'color': '#1e293b'}},
            gauge={
                'axis': {'range': [0, gauge_max], 'tickfont': {'size': 11, 'color': GRAY}},
                'bar': {'color': BLUE, 'thickness': 0.15},
                'bgcolor': '#f8fafc',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, gauge_max * 0.4], 'color': 'rgba(37,99,235,0.08)'},
                    {'range': [gauge_max * 0.4, gauge_max * 0.85], 'color': 'rgba(249,115,22,0.08)'},
                    {'range': [gauge_max * 0.85, gauge_max], 'color': 'rgba(239,68,68,0.08)'},
                ],
                'threshold': {
                    'line': {'color': GREEN, 'width': 2},
                    'thickness': 0.8,
                    'value': gauge_max * 0.6,
                },
            },
        ), row=1, col=1)

        # ── Row 1, Col 2: Balance gauge ──
        dists = [v['dist'] for v in active_vehicles]
        if len(dists) >= 2 and max(dists) > 0:
            balance = (sum(dists) / len(dists)) / max(dists) * 100
        else:
            balance = 100
        if balance >= 80:
            balance_color = GREEN
            balance_label = '优秀'
        elif balance >= 60:
            balance_color = YELLOW
            balance_label = '一般'
        else:
            balance_color = RED
            balance_label = '较差'

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=balance,
            title={'text': f"<b>路线均衡度</b><br><span style='font-size:12px;color:{GRAY}'>均值距离 / 最大距离 · 状态: </span><span style='font-size:12px;color:{balance_color}'>{balance_label}</span>", 'font': {'size': 13}},
            number={'suffix': " %", 'font': {'size': 36, 'color': '#1e293b'}},
            gauge={
                'axis': {'range': [0, 100], 'tickfont': {'size': 11, 'color': GRAY}},
                'bar': {'color': balance_color, 'thickness': 0.15},
                'bgcolor': '#f8fafc',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 60], 'color': 'rgba(239,68,68,0.08)'},
                    {'range': [60, 80], 'color': 'rgba(234,179,8,0.08)'},
                    {'range': [80, 100], 'color': 'rgba(16,185,129,0.08)'},
                ],
                'threshold': {
                    'line': {'color': '#1e293b', 'width': 3},
                    'thickness': 0.8,
                    'value': 80,
                },
            },
        ), row=1, col=2)

        # ── Row 2, Col 1: Per-vehicle time — sorted by total descending ──
        sorted_v = sorted(active_vehicles, key=lambda v: v['drive_s'] + v['wait_s'], reverse=True)
        v_names_sorted = [f'V{v["id"]+1}' for v in sorted_v]
        drive_sorted = [v['drive_s'] / 60 for v in sorted_v]
        wait_sorted = [v['wait_s'] / 60 for v in sorted_v]

        fig.add_trace(go.Bar(
            name='驾驶时间',
            x=v_names_sorted, y=drive_sorted,
            marker=dict(color=BLUE, opacity=0.85),
            text=[f'{d:.0f}分' if d > 0 else '' for d in drive_sorted],
            textposition='auto',
            textfont=dict(size=10, color='white'),
            hovertemplate='%{x}<br>驾驶: %{y:.1f} 分<extra></extra>',
        ), row=2, col=1)

        fig.add_trace(go.Bar(
            name='红灯等待',
            x=v_names_sorted, y=wait_sorted,
            marker=dict(color=ORANGE, opacity=0.85),
            text=[f'{w:.0f}分' if w > 0 else '' for w in wait_sorted],
            textposition='auto',
            textfont=dict(size=10),
            hovertemplate='%{x}<br>等待: %{y:.1f} 分<extra></extra>',
        ), row=2, col=1)

        # ── Row 2, Col 2: Distance bars + load line (dual y-axis) ──
        # Sort by distance descending for consistency
        sorted_v2 = sorted(active_vehicles, key=lambda v: v['dist'], reverse=True)
        v_names_sorted2 = [f'V{v["id"]+1}' for v in sorted_v2]

        fig.add_trace(go.Bar(
            name='距离',
            x=v_names_sorted2,
            y=[v['dist'] for v in sorted_v2],
            marker=dict(color=BLUE, opacity=0.7),
            text=[f'{v["dist"]:.1f}' for v in sorted_v2],
            textposition='outside',
            textfont=dict(size=10, color='#1e293b'),
            hovertemplate='%{x}<br>距离: %{y:.2f} km<extra></extra>',
        ), row=2, col=2)

        fig.add_trace(go.Scatter(
            name='载重',
            x=v_names_sorted2,
            y=[v['load'] for v in sorted_v2],
            mode='lines+markers+text',
            marker=dict(size=10, color=ORANGE, symbol='circle'),
            line=dict(color=ORANGE, width=2.5),
            text=[f'{v["load"]}' for v in sorted_v2],
            textposition='top center',
            textfont=dict(size=10, color=ORANGE),
            hovertemplate='%{x}<br>载重: %{y}<extra></extra>',
        ), row=2, col=2, secondary_y=True)

        # ── Row 3, Col 1: Segment detail table (light theme, better contrast) ──
        seg_rows = []
        for v in active_vehicles:
            for i, s in enumerate(v['segments']):
                m, s_ = divmod(int(s['drive_s']), 60)
                seg_rows.append([
                    f'V{v["id"]+1}',
                    f'{s["from"]} → {s["to"]}',
                    f'{s["dist_km"]:.2f} km',
                    f'{m}分{s_:02d}秒',
                ])

        header_color = '#1e293b'
        cell_fill_odd = '#f8fafc'
        cell_fill_even = '#ffffff'

        if seg_rows:
            header_vals = ['车辆', '路段', '距离', '驾驶时间']
            # Create alternating row colors for readability
            cell_colors = [cell_fill_odd if i % 2 == 0 else cell_fill_even for i in range(len(seg_rows))]
            cols = list(zip(*seg_rows))

            fig.add_trace(go.Table(
                header=dict(
                    values=[f'<b>{h}</b>' for h in header_vals],
                    fill_color=header_color,
                    font=dict(color='white', size=11),
                    align='center',
                    height=32,
                    line=dict(color='#e2e8f0', width=1),
                ),
                cells=dict(
                    values=cols,
                    fill_color=[cell_colors],
                    font=dict(color='#334155', size=10),
                    align='center',
                    height=26,
                    line=dict(color='#e2e8f0', width=0.5),
                ),
            ), row=3, col=1)

        # ── Row 3, Col 2: Red light wait summary — clean, grouped by vehicle ──
        light_labels = []
        light_waits = []
        light_colors_list = []
        light_hover = []

        # Sort lights by wait time descending, take top 30 to avoid overcrowding
        all_lights_flat = []
        for v in vehicle_data:
            for li in v['lights']:
                all_lights_flat.append((v['id'], li))

        all_lights_flat.sort(key=lambda x: x[1]['wait_s'], reverse=True)
        max_lights = min(len(all_lights_flat), 30)

        for vid, li in all_lights_flat[:max_lights]:
            node_label = li['at']
            wait_s = int(li['wait_s'])
            if isinstance(node_label, str):
                lbl = f'V{vid+1}-{node_label}'
            else:
                lbl = f'V{vid+1}-站点{node_label}'
            light_labels.append(lbl)
            light_waits.append(wait_s)
            light_colors_list.append(RED if wait_s > 15 else YELLOW)
            arr_m, arr_s = divmod(int(li['arrival_s']), 60)
            pos_info = f'<br>距起点: {li["position_m"]:.0f}m' if 'position_m' in li else ''
            light_hover.append(
                f'V{vid+1} 到达{node_label}<br>'
                f'到达: +{arr_m}分{arr_s}秒<br>'
                f'等待: {wait_s}秒{pos_info}'
            )

        if light_labels:
            fig.add_trace(go.Bar(
                x=light_labels, y=light_waits,
                marker=dict(color=light_colors_list, opacity=0.85),
                text=[f'{w}秒' if w > 0 else '' for w in light_waits],
                textposition='auto',
                textfont=dict(size=9),
                name='红灯等待',
                hovertext=light_hover,
                hoverinfo='text',
            ), row=3, col=2)
        else:
            fig.add_trace(go.Bar(
                x=['未遇到红灯'], y=[0],
                marker=dict(color=GREEN, opacity=0.7),
                text=['0秒'], textposition='auto',
                textfont=dict(size=10),
                hoverinfo='skip',
            ), row=3, col=2)

        # ── Global layout ──
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation='h',
                y=1.06, x=0.5, xanchor='center',
                font=dict(size=11, color='#475569'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#e2e8f0',
                borderwidth=1,
            ),
            height=1250, width=1200,
            margin=dict(l=60, r=40, t=70, b=60),
            barmode='stack',
            font=dict(size=11, color='#475569'),
            paper_bgcolor='white',
            plot_bgcolor='#f8fafc',
        )

        # Unified annotation styling
        fig.update_annotations(font=dict(size=13, color='#1e293b'), yshift=-12)

        # Axis styling
        fig.update_xaxes(tickangle=-30, tickfont=dict(size=10, color='#475569'),
                         gridcolor='#e2e8f0', row=2, col=1)
        fig.update_xaxes(tickangle=-30, tickfont=dict(size=10, color='#475569'),
                         gridcolor='#e2e8f0', row=2, col=2)

        fig.update_yaxes(title_text='分钟', title_font=dict(size=11, color=GRAY),
                         tickfont=dict(size=10, color='#475569'),
                         gridcolor='#e2e8f0', row=2, col=1)
        fig.update_yaxes(title_text='距离 (km)', title_font=dict(size=11, color=GRAY),
                         tickfont=dict(size=10, color='#475569'),
                         gridcolor='#e2e8f0', row=2, col=2)
        # Secondary y-axis for load on row 2 col 2
        fig.update_yaxes(title_text='载重', title_font=dict(size=11, color=ORANGE),
                         tickfont=dict(size=10, color=ORANGE),
                         side='right', row=2, col=2, secondary_y=True)
        fig.update_yaxes(title_text='等待 (秒)', title_font=dict(size=11, color=GRAY),
                         tickfont=dict(size=10, color='#475569'),
                         gridcolor='#e2e8f0', row=3, col=2)
        fig.update_xaxes(tickangle=-30, tickfont=dict(size=9, color='#475569'),
                         row=3, col=2)

        return fig

    def save_animation(self, fig, filename="route_animation.html"):
        """保存动画到HTML文件，确保目录存在"""
        import os
        dirpath = os.path.dirname(filename)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        # 使用 CDN 方式引入 plotlyjs 减小文件体积
        fig.write_html(filename, include_plotlyjs='cdn')
        print(f"动画已保存到: {filename}")
        return filename

    def show_animation(self, fig):
        """显示动画"""
        fig.show()