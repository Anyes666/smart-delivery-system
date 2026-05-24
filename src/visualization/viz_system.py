"""
VRP Visualization System
完整的可视化系统，支持路线动画、指标仪表盘、优先级热力图和时间窗口分析
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import logging
import os
from typing import List, Dict, Tuple, Optional, Any
from IPython.display import display, HTML

class PlotlyAnimator:
    """Plotly动画器 - 用于创建VRP路线动画"""
    
    def __init__(self, animation_speed: int = 500):
        """
        初始化Plotly动画器
        
        :param animation_speed: 动画速度（毫秒）
        """
        self.animation_speed = animation_speed
        self.logger = logging.getLogger(__name__)
    
    def create_route_animation(self, points_df: pd.DataFrame, routes: List[List[int]], 
                              title: str = "配送路线动画", show_time_windows: bool = False,
                              distance_matrix: Optional[np.ndarray] = None) -> go.Figure:
        """
        创建路线动画
        
        :param points_df: 包含点数据的DataFrame (必须包含x, y列)
        :param routes: 路线列表 [[0,2,5,0], [0,3,4,0], ...]
        :param title: 图表标题
        :param show_time_windows: 是否显示时间窗口信息
        :param distance_matrix: 距离矩阵（用于计算到达时间）
        :return: Plotly Figure对象
        """
        try:
            # 从DataFrame提取坐标
            if 'x' not in points_df.columns or 'y' not in points_df.columns:
                self.logger.error("❌ DataFrame缺少必需的列: x, y")
                return None
            
            x_coords = points_df['x'].values.tolist()
            y_coords = points_df['y'].values.tolist()
            
            # 获取点名称
            point_names = self._get_point_names(points_df)
            
            # 确定配送中心索引（默认第一行或id=1）
            depot_idx = self._find_depot_index(points_df)
            depot_x, depot_y = x_coords[depot_idx], y_coords[depot_idx]
            
            # 创建基础图表
            fig = go.Figure()
            
            # 添加所有配送点（排除配送中心）
            customer_indices = [i for i in range(len(points_df)) if i != depot_idx]
            customer_x = [x_coords[i] for i in customer_indices]
            customer_y = [y_coords[i] for i in customer_indices]
            customer_names = [point_names[i] for i in customer_indices]
            
            fig.add_trace(go.Scatter(
                x=customer_x, y=customer_y,
                mode='markers+text',
                marker=dict(size=12, color='blue', symbol='circle'),
                text=customer_names,
                textposition="top center",
                name='配送点',
                hoverinfo='text',
                hovertext=[f'{name}<br>坐标: ({x:.4f}, {y:.4f})' 
                          for name, x, y in zip(customer_names, customer_x, customer_y)]
            ))
            
            # 添加配送中心
            fig.add_trace(go.Scatter(
                x=[depot_x], y=[depot_y],
                mode='markers+text',
                marker=dict(size=18, color='red', symbol='star'),
                text=[point_names[depot_idx]],
                textposition="top center",
                name='配送中心',
                hoverinfo='text',
                hovertext=[f'{point_names[depot_idx]}<br>坐标: ({depot_x:.4f}, {depot_y:.4f})']
            ))
            
            # 创建动画帧
            frames = []
            max_steps = max(len(route) for route in routes) if routes else 0
            
            # 为每个时间步创建帧
            for step in range(1, max_steps + 1):
                frame_data = []
                
                # 添加静态点（配送点和配送中心）
                frame_data.append(go.Scatter(
                    x=customer_x, y=customer_y,
                    mode='markers+text',
                    marker=dict(size=12, color='blue', symbol='circle'),
                    text=customer_names,
                    textposition="top center",
                    name='配送点',
                    hoverinfo='text'
                ))
                
                frame_data.append(go.Scatter(
                    x=[depot_x], y=[depot_y],
                    mode='markers+text',
                    marker=dict(size=18, color='red', symbol='star'),
                    text=[point_names[depot_idx]],
                    textposition="top center",
                    name='配送中心',
                    hoverinfo='text'
                ))
                
                # 为每条路线添加动态线
                for vehicle_id, route in enumerate(routes):
                    if step <= len(route):
                        # 获取当前步的路线点
                        current_route = route[:min(step + 1, len(route))]
                        route_x = [x_coords[i] for i in current_route]
                        route_y = [y_coords[i] for i in current_route]
                        
                        # 生成颜色
                        color = self._get_vehicle_color(vehicle_id)
                        
                        frame_data.append(go.Scatter(
                            x=route_x,
                            y=route_y,
                            mode='lines+markers',
                            line=dict(width=3, color=color),
                            marker=dict(size=8, color=color, symbol='diamond'),
                            name=f'车辆 {vehicle_id + 1}',
                            hoverinfo='text',
                            hovertext=[f'车辆 {vehicle_id + 1}<br>路线: {" -> ".join(str(i) for i in current_route)}']
                        ))
                
                frame = go.Frame(data=frame_data, name=str(step))
                frames.append(frame)
            
            fig.frames = frames
            
            # 添加动画控件
            fig.update_layout(
                title=title,
                xaxis_title='经度 (X)',
                yaxis_title='纬度 (Y)',
                showlegend=True,
                hovermode='closest',
                updatemenus=[dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="▶️ 播放",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": self.animation_speed, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 100, "easing": "quadratic-in-out"},
                                "mode": "immediate"
                            }]
                        ),
                        dict(
                            label="⏸️ 暂停",
                            method="animate",
                            args=[[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }]
                        )
                    ],
                    direction="left",
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.1,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                )],
                sliders=[dict(
                    steps=[dict(
                        method="animate",
                        args=[[f.name], {
                            "frame": {"duration": 100, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }],
                        label=f.name
                    ) for f in fig.frames],
                    active=0,
                    transition={"duration": 0},
                    x=0,
                    y=0,
                    currentvalue={"prefix": "步骤: "},
                    len=0.9,
                    xanchor="left",
                    yanchor="bottom"
                )]
            )
            
            # 设置坐标轴范围（添加边距）
            x_min, x_max = min(x_coords) - 0.05, max(x_coords) + 0.05
            y_min, y_max = min(y_coords) - 0.05, max(y_coords) + 0.05
            fig.update_xaxes(range=[x_min, x_max])
            fig.update_yaxes(range=[y_min, y_max])
            
            # 如果需要显示时间窗口，添加注释
            if show_time_windows and distance_matrix is not None:
                fig = self._add_time_window_annotations(fig, points_df, routes, distance_matrix, depot_idx)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"❌ 创建路线动画时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def create_metrics_dashboard(self, routes: List[List[int]], distance_matrix: np.ndarray, 
                                 demands: List[float], title: str = "配送指标仪表盘") -> go.Figure:
        """
        创建指标仪表盘
        
        :param routes: 路线列表
        :param distance_matrix: 距离矩阵
        :param demands: 需求量列表
        :param title: 仪表盘标题
        :return: Plotly Figure对象
        """
        try:
            # 计算指标
            total_distance = 0
            route_distances = []
            route_loads = []
            route_points = []
            
            for route in routes:
                route_distance = 0
                route_load = sum(demands[i] for i in route if i < len(demands))
                num_points = len(route) - 2  # 排除配送中心（起点和终点）
                
                for i in range(len(route) - 1):
                    if route[i] < len(distance_matrix) and route[i + 1] < len(distance_matrix):
                        route_distance += distance_matrix[route[i]][route[i + 1]]
                
                total_distance += route_distance
                route_distances.append(route_distance)
                route_loads.append(route_load)
                route_points.append(num_points)
            
            avg_distance = total_distance / len(routes) if routes else 0
            avg_load = sum(route_loads) / len(route_loads) if route_loads else 0
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('总行驶距离', '各车辆路线距离', '车辆负载情况', '路线效率'),
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "pie"}, {"type": "indicator"}]]
            )
            
            # 1. 总行驶距离 (仪表盘)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=total_distance,
                    title={'text': "总距离 (km)"},
                    gauge={'axis': {'range': [None, total_distance * 1.5]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, total_distance * 0.3], 'color': "lightgreen"},
                               {'range': [total_distance * 0.3, total_distance * 0.7], 'color': "lightyellow"},
                               {'range': [total_distance * 0.7, total_distance * 1.5], 'color': "lightcoral"}]}
                ),
                row=1, col=1
            )
            
            # 2. 各车辆路线距离 (柱状图)
            vehicle_names = [f'车辆 {i+1}' for i in range(len(route_distances))]
            colors = [self._get_vehicle_color(i) for i in range(len(route_distances))]
            
            fig.add_trace(
                go.Bar(
                    x=vehicle_names,
                    y=route_distances,
                    name='路线距离',
                    marker_color=colors,
                    text=[f'{d:.2f} km' for d in route_distances],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # 3. 车辆负载情况 (饼图)
            fig.add_trace(
                go.Pie(
                    labels=vehicle_names,
                    values=route_loads,
                    name='车辆负载',
                    hole=0.4,
                    textinfo='label+percent',
                    hoverinfo='label+value+percent',
                    marker=dict(colors=colors)
                ),
                row=2, col=1
            )
            
            # 4. 路线效率 (仪表盘)
            if max(route_distances) > 0:
                efficiency = min(1.0, avg_distance / max(route_distances))
            else:
                efficiency = 0
            
            efficiency_color = "green" if efficiency > 0.8 else "yellow" if efficiency > 0.6 else "red"
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=efficiency * 100,
                    title={'text': "路线均衡度 (%)"},
                    delta={'reference': 80, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': efficiency_color},
                        'steps': [
                            {'range': [0, 60], 'color': "rgba(255,0,0,0.3)"},
                            {'range': [60, 80], 'color': "rgba(255,255,0,0.3)"},
                            {'range': [80, 100], 'color': "rgba(0,255,0,0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text=title,
                showlegend=False,
                height=700,
                width=1000,
                font=dict(size=12)
            )
            
            # 更新子图布局
            fig.update_xaxes(title_text="车辆", row=1, col=2)
            fig.update_yaxes(title_text="距离 (km)", row=1, col=2)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"❌ 创建指标仪表盘时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def create_priority_heatmap(self, points_df: pd.DataFrame, 
                               title: str = "客户优先级热力图") -> Optional[go.Figure]:
        """
        创建客户优先级热力图
        
        :param points_df: 包含点数据的DataFrame
        :param title: 图表标题
        :return: Plotly Figure对象
        """
        try:
            if 'priority' not in points_df.columns:
                self.logger.warning("⚠️ DataFrame中没有priority列，无法创建优先级热力图")
                return None
            
            fig = go.Figure()
            
            # 过滤掉配送中心（假设id=1或第一行）
            depot_idx = self._find_depot_index(points_df)
            customer_mask = points_df.index != depot_idx
            
            customer_points = points_df[customer_mask]
            
            if len(customer_points) == 0:
                self.logger.warning("⚠️ 没有客户点数据")
                return None
            
            # 创建优先级到颜色的映射
            priority_colors = {
                1: 'red',      # 最高优先级 - 红色
                2: 'orange',   # 高优先级 - 橙色
                3: 'yellow',   # 中优先级 - 黄色
                4: 'lightgreen', # 低优先级 - 浅绿
                5: 'green'     # 最低优先级 - 绿色
            }
            
            # 为每个点分配颜色
            point_colors = []
            for idx, row in customer_points.iterrows():
                priority = int(row['priority']) if pd.notna(row['priority']) else 3
                point_colors.append(priority_colors.get(priority, 'gray'))
            
            # 获取点名称
            point_names = self._get_point_names(customer_points)
            
            # 添加散点图
            fig.add_trace(go.Scatter(
                x=customer_points['x'],
                y=customer_points['y'],
                mode='markers+text',
                marker=dict(
                    size=18,
                    color=point_colors,
                    symbol='circle',
                    line=dict(width=2, color='black')
                ),
                text=point_names,
                textposition="top center",
                hoverinfo='text',
                hovertext=[f'{name}<br>优先级: {int(p) if pd.notna(p) else "N/A"}<br>坐标: ({x:.4f}, {y:.4f})' 
                          for name, p, x, y in zip(point_names, 
                                                  customer_points['priority'], 
                                                  customer_points['x'], 
                                                  customer_points['y'])]
            ))
            
            # 添加配送中心
            depot = points_df.iloc[[depot_idx]]
            fig.add_trace(go.Scatter(
                x=depot['x'],
                y=depot['y'],
                mode='markers+text',
                marker=dict(size=25, color='black', symbol='star', line=dict(width=3, color='white')),
                text=[self._get_point_names(points_df)[depot_idx]],
                textposition="top center",
                name='配送中心',
                hoverinfo='text',
                hovertext=[f'配送中心<br>坐标: ({depot["x"].values[0]:.4f}, {depot["y"].values[0]:.4f})']
            ))
            
            # 添加图例说明
            legend_items = []
            for priority, color in sorted(priority_colors.items()):
                priority_text = {
                    1: '最高',
                    2: '高',
                    3: '中',
                    4: '低',
                    5: '最低'
                }.get(priority, str(priority))
                
                legend_items.append(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=15, color=color, symbol='circle'),
                    name=f'优先级 {priority} ({priority_text})',
                    showlegend=True
                ))
            
            for item in legend_items:
                fig.add_trace(item)
            
            fig.update_layout(
                title=title,
                xaxis_title='经度',
                yaxis_title='纬度',
                showlegend=True,
                hovermode='closest',
                height=600,
                width=900,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02
                )
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"❌ 创建优先级热力图时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def create_time_window_analysis(self, points_df: pd.DataFrame, routes: List[List[int]], 
                                   distance_matrix: np.ndarray, 
                                   title: str = "时间窗口分析") -> Optional[go.Figure]:
        """
        创建时间窗口分析图
        
        :param points_df: 包含点数据的DataFrame
        :param routes: 路线列表
        :param distance_matrix: 距离矩阵
        :param title: 图表标题
        :return: Plotly Figure对象
        """
        try:
            if 'time_window_start' not in points_df.columns or 'time_window_end' not in points_df.columns:
                self.logger.warning("⚠️ DataFrame中缺少时间窗口列")
                return None
            
            fig = go.Figure()
            
            # 计算每个点的到达时间
            all_arrivals = []
            all_departures = []
            all_time_windows = []
            all_vehicle_ids = []
            all_point_names = []
            all_status = []  # 'on_time', 'early', 'late'
            
            for vehicle_id, route in enumerate(routes):
                current_time = 0  # 从0秒开始
                
                for i in range(len(route) - 1):
                    from_idx = route[i]
                    to_idx = route[i + 1]
                    
                    # 跳过配送中心
                    if to_idx == self._find_depot_index(points_df):
                        continue
                    
                    # 计算行驶时间（假设平均速度40km/h）
                    if from_idx < len(distance_matrix) and to_idx < len(distance_matrix):
                        distance = distance_matrix[from_idx][to_idx]
                        travel_time = (distance / 40) * 3600  # 转换为秒
                    else:
                        travel_time = 0
                    
                    current_time += travel_time
                    arrival_time = current_time
                    
                    # 服务时间（假设5分钟）
                    service_time = 300
                    departure_time = arrival_time + service_time
                    
                    # 获取时间窗口
                    tw_start = points_df.iloc[to_idx]['time_window_start']
                    tw_end = points_df.iloc[to_idx]['time_window_end']
                    
                    # 判断是否准时
                    if arrival_time < tw_start:
                        status = 'early'
                        color = 'orange'
                    elif arrival_time <= tw_end:
                        status = 'on_time'
                        color = 'green'
                    else:
                        status = 'late'
                        color = 'red'
                    
                    all_arrivals.append(arrival_time)
                    all_departures.append(departure_time)
                    all_time_windows.append((tw_start, tw_end))
                    all_vehicle_ids.append(vehicle_id)
                    all_point_names.append(self._get_point_names(points_df)[to_idx])
                    all_status.append(status)
            
            if not all_arrivals:
                self.logger.warning("⚠️ 没有可分析的时间窗口数据")
                return None
            
            # 创建甘特图
            for i, (arrival, departure, tw, vehicle_id, name, status) in enumerate(
                zip(all_arrivals, all_departures, all_time_windows, all_vehicle_ids, all_point_names, all_status)):
                
                tw_start, tw_end = tw
                
                # 时间窗口背景
                fig.add_trace(go.Scatter(
                    x=[tw_start, tw_end, tw_end, tw_start, tw_start],
                    y=[i+0.3, i+0.3, i-0.3, i-0.3, i+0.3],
                    fill='toself',
                    fillcolor='rgba(200,200,200,0.3)',
                    line=dict(color='rgba(200,200,200,0.5)', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # 实际到达和离开
                color = {'on_time': 'green', 'early': 'orange', 'late': 'red'}[status]
                fig.add_trace(go.Scatter(
                    x=[arrival, departure],
                    y=[i, i],
                    mode='markers+lines',
                    marker=dict(size=12, color=color, symbol='circle'),
                    line=dict(color=color, width=3),
                    name=f'{name} (车辆 {vehicle_id+1})',
                    hoverinfo='text',
                    hovertext=f'{name}<br>车辆: {vehicle_id+1}<br>到达: {self._format_time(arrival)}<br>离开: {self._format_time(departure)}<br>时间窗口: {self._format_time(tw_start)}-{self._format_time(tw_end)}<br>状态: {status}'
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title='时间',
                yaxis_title='客户点',
                showlegend=False,
                hovermode='closest',
                height=max(400, len(all_arrivals) * 40),
                width=1000,
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(all_point_names))),
                    ticktext=all_point_names,
                    autorange='reversed'
                ),
                xaxis=dict(
                    tickmode='array',
                    tickvals=[0, 14400, 28800, 43200, 57600, 72000, 86400],
                    ticktext=['0:00', '4:00', '8:00', '12:00', '16:00', '20:00', '24:00']
                )
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"❌ 创建时间窗口分析时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _add_time_window_annotations(self, fig: go.Figure, points_df: pd.DataFrame, 
                                     routes: List[List[int]], distance_matrix: np.ndarray,
                                     depot_idx: int) -> go.Figure:
        """为路线动画添加时间窗口注释"""
        try:
            if 'time_window_start' not in points_df.columns or 'time_window_end' not in points_df.columns:
                return fig
            
            # 计算每个点的到达时间（使用第一条完整路线）
            if not routes:
                return fig
            
            route = routes[0]  # 使用第一条路线
            current_time = 0
            
            for i in range(len(route) - 1):
                from_idx = route[i]
                to_idx = route[i + 1]
                
                if to_idx == depot_idx:
                    continue
                
                # 计算行驶时间
                if from_idx < len(distance_matrix) and to_idx < len(distance_matrix):
                    distance = distance_matrix[from_idx][to_idx]
                    travel_time = (distance / 40) * 3600
                else:
                    travel_time = 0
                
                current_time += travel_time
                arrival_time = current_time
                
                # 获取时间窗口
                tw_start = points_df.iloc[to_idx]['time_window_start']
                tw_end = points_df.iloc[to_idx]['time_window_end']
                
                # 判断状态
                if tw_start <= arrival_time <= tw_end:
                    status = '✅ 准时'
                    color = 'green'
                elif arrival_time < tw_start:
                    status = '⚠️ 过早'
                    color = 'orange'
                else:
                    status = '❌ 迟到'
                    color = 'red'
                
                # 添加注释
                fig.add_annotation(
                    x=points_df.iloc[to_idx]['x'],
                    y=points_df.iloc[to_idx]['y'],
                    text=f'{status}<br>到达: {self._format_time(arrival_time)}',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    bgcolor=f'rgba(255,255,255,0.9)',
                    bordercolor=color,
                    borderwidth=2,
                    font=dict(color='black', size=9)
                )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"添加时间窗口注释时出错: {str(e)}")
            return fig
    
    def _get_point_names(self, points_df: pd.DataFrame) -> List[str]:
        """获取点名称列表"""
        if 'name' in points_df.columns and points_df['name'].notna().all():
            return points_df['name'].values.tolist()
        elif 'id' in points_df.columns:
            return [f'ID_{int(id)}' for id in points_df['id']]
        else:
            return [f'点_{i}' for i in range(len(points_df))]
    
    def _find_depot_index(self, points_df: pd.DataFrame) -> int:
        """查找配送中心索引"""
        if 'id' in points_df.columns and 1 in points_df['id'].values:
            return points_df[points_df['id'] == 1].index[0]
        return 0  # 默认第一行
    
    def _get_vehicle_color(self, vehicle_id: int) -> str:
        """获取车辆颜色"""
        colors = ['blue', 'green', 'purple', 'orange', 'red', 'cyan', 'magenta', 'brown', 'pink', 'lime']
        return colors[vehicle_id % len(colors)]
    
    def _format_time(self, seconds: float) -> str:
        """将秒数格式化为HH:MM"""
        if seconds < 0:
            seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours:02d}:{minutes:02d}"
    
    def save_animation(self, fig: go.Figure, filename: str = "route_animation.html") -> str:
        """保存动画到HTML文件"""
        try:
            fig.write_html(filename)
            self.logger.info(f"✅ 动画已保存到: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"❌ 保存动画时出错: {str(e)}")
            return None
    
    def show_animation(self, fig: go.Figure):
        """显示动画"""
        try:
            fig.show()
        except Exception as e:
            self.logger.error(f"❌ 显示动画时出错: {str(e)}")


class VRPVisualizationSystem:
    """VRP可视化系统 - 集成所有可视化功能"""
    
    def __init__(self, animation_speed: int = 500):
        """
        初始化VRP可视化系统
        
        :param animation_speed: 动画速度（毫秒）
        """
        self.animator = PlotlyAnimator(animation_speed)
        self.logger = logging.getLogger(__name__)
    
    def create_complete_visualization(self, points_df: pd.DataFrame, routes: List[List[int]], 
                                     distance_matrix: np.ndarray, demands: List[float],
                                     algorithm_name: str = "VRP") -> Optional[Dict[str, go.Figure]]:
        """
        创建完整的VRP可视化
        
        :param points_df: 包含点数据的DataFrame
        :param routes: 路线列表
        :param distance_matrix: 距离矩阵
        :param demands: 需求量列表
        :param algorithm_name: 算法名称
        :return: 包含所有可视化图表的字典
        """
        try:
            self.logger.info("🎨 开始创建可视化...")
            
            visualizations = {}
            
            # 1. 路线动画
            self.logger.info("  📊 创建路线动画...")
            route_fig = self.animator.create_route_animation(
                points_df, routes, 
                title=f"{algorithm_name} 算法配送路线动画",
                show_time_windows=True,
                distance_matrix=distance_matrix
            )
            if route_fig:
                visualizations['route_animation'] = route_fig
                self.logger.info("  ✅ 路线动画创建成功")
            
            # 2. 指标仪表盘
            self.logger.info("  📊 创建指标仪表盘...")
            metrics_fig = self.animator.create_metrics_dashboard(
                routes, distance_matrix, demands,
                title=f"{algorithm_name} 算法配送指标"
            )
            if metrics_fig:
                visualizations['metrics_dashboard'] = metrics_fig
                self.logger.info("  ✅ 指标仪表盘创建成功")
            
            # 3. 优先级热力图
            if 'priority' in points_df.columns:
                self.logger.info("  📊 创建优先级热力图...")
                priority_fig = self.animator.create_priority_heatmap(
                    points_df, title="客户优先级分布"
                )
                if priority_fig:
                    visualizations['priority_heatmap'] = priority_fig
                    self.logger.info("  ✅ 优先级热力图创建成功")
            
            # 4. 时间窗口分析
            if 'time_window_start' in points_df.columns and 'time_window_end' in points_df.columns:
                self.logger.info("  📊 创建时间窗口分析...")
                time_window_fig = self.animator.create_time_window_analysis(
                    points_df, routes, distance_matrix,
                    title="配送时间窗口分析"
                )
                if time_window_fig:
                    visualizations['time_window_analysis'] = time_window_fig
                    self.logger.info("  ✅ 时间窗口分析创建成功")
            
            self.logger.info(f"🎉 可视化创建完成！共创建 {len(visualizations)} 个图表")
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"❌ 创建可视化时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def save_all_visualizations(self, visualizations: Dict[str, go.Figure], 
                               output_dir: str = 'output') -> Dict[str, str]:
        """
        保存所有可视化结果到HTML文件
        
        :param visualizations: 可视化图表字典
        :param output_dir: 输出目录
        :return: 保存的文件路径字典
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            results = {}
            
            # 保存路线动画
            if 'route_animation' in visualizations:
                route_file = os.path.join(output_dir, 'route_animation.html')
                self.animator.save_animation(visualizations['route_animation'], route_file)
                results['route_animation'] = route_file
            
            # 保存指标仪表盘
            if 'metrics_dashboard' in visualizations:
                metrics_file = os.path.join(output_dir, 'metrics_dashboard.html')
                self.animator.save_animation(visualizations['metrics_dashboard'], metrics_file)
                results['metrics_dashboard'] = metrics_file
            
            # 保存优先级热力图
            if 'priority_heatmap' in visualizations:
                priority_file = os.path.join(output_dir, 'priority_heatmap.html')
                self.animator.save_animation(visualizations['priority_heatmap'], priority_file)
                results['priority_heatmap'] = priority_file
            
            # 保存时间窗口分析
            if 'time_window_analysis' in visualizations:
                time_file = os.path.join(output_dir, 'time_window_analysis.html')
                self.animator.save_animation(visualizations['time_window_analysis'], time_file)
                results['time_window_analysis'] = time_file
            
            self.logger.info(f"✅ 所有可视化已保存到: {output_dir}")
            for name, file_path in results.items():
                self.logger.info(f"   - {name}: {os.path.basename(file_path)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 保存可视化时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def show_interactive_dashboard(self, visualizations: Dict[str, go.Figure]):
        """
        在Jupyter notebook中显示交互式仪表盘
        
        :param visualizations: 可视化图表字典
        """
        try:
            print("\n" + "="*60)
            print("📊 VRP 交互式可视化仪表盘")
            print("="*60)
            
            # 显示路线动画
            if 'route_animation' in visualizations:
                print("\n🎬 配送路线动画:")
                display(visualizations['route_animation'])
            
            # 显示指标仪表盘
            if 'metrics_dashboard' in visualizations:
                print("\n📈 配送指标仪表盘:")
                display(visualizations['metrics_dashboard'])
            
            # 显示优先级热力图
            if 'priority_heatmap' in visualizations:
                print("\n⭐ 客户优先级分布:")
                display(visualizations['priority_heatmap'])
            
            # 显示时间窗口分析
            if 'time_window_analysis' in visualizations:
                print("\n⏰ 配送时间窗口分析:")
                display(visualizations['time_window_analysis'])
            
            print("\n" + "="*60)
            print("💡 提示: 点击图表可以交互查看详细信息")
            print("="*60 + "\n")
            
        except Exception as e:
            self.logger.error(f"❌ 显示仪表盘时出错: {str(e)}")
            print(f"错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 测试 VRP Visualization System")
    print("="*60)
    
    # 创建示例数据
    np.random.seed(42)
    
    # 创建示例DataFrame
    data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'x': [121.47 + np.random.uniform(-0.1, 0.1) for _ in range(10)],
        'y': [31.23 + np.random.uniform(-0.1, 0.1) for _ in range(10)],
        'demand': [0, 15, 12, 18, 10, 20, 14, 16, 22, 25],
        'time_window_start': [0, 28800, 32400, 36000, 39600, 28800, 32400, 36000, 28800, 32400],
        'time_window_end': [86400, 32400, 36000, 39600, 43200, 36000, 39600, 43200, 32400, 36000],
        'priority': [0, 1, 2, 1, 3, 2, 1, 3, 2, 1]
    }
    
    points_df = pd.DataFrame(data)
    
    # 创建示例路线
    routes = [
        [0, 1, 4, 7, 0],  # 车辆1
        [0, 2, 5, 8, 0],  # 车辆2
        [0, 3, 6, 9, 0]   # 车辆3
    ]
    
    # 创建示例距离矩阵
    n = len(points_df)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((points_df.iloc[i]['x'] - points_df.iloc[j]['x'])**2 + 
                          (points_df.iloc[i]['y'] - points_df.iloc[j]['y'])**2) * 100
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    
    demands = points_df['demand'].tolist()
    
    # 创建可视化系统
    print("\n1️⃣ 创建可视化系统...")
    viz_system = VRPVisualizationSystem(animation_speed=300)
    
    # 创建完整可视化
    print("2️⃣ 创建完整可视化...")
    visualizations = viz_system.create_complete_visualization(
        points_df=points_df,
        routes=routes,
        distance_matrix=distance_matrix,
        demands=demands,
        algorithm_name="测试算法"
    )
    
    if visualizations:
        print("3️⃣ 保存可视化结果...")
        results = viz_system.save_all_visualizations(visualizations, output_dir='test_output')
        
        print("\n✅ 测试完成！")
        print(f"   共创建 {len(visualizations)} 个图表")
        print(f"   保存到目录: test_output/")
        
        # 显示文件列表
        print("\n📁 生成的文件:")
        for name, file_path in results.items():
            print(f"   - {name}: {os.path.basename(file_path)}")
    else:
        print("❌ 可视化创建失败")
    
    print("\n🎉 测试完成！")