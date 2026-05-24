"""
智能配送系统 — Streamlit 交互界面

用法:
    streamlit run app.py
"""

import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════
st.set_page_config(
    page_title="智能配送优化系统",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a73e8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.3rem;
    }
    .route-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1a73e8;
    }
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.8rem;
        padding: 2rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════
# Session State 初始化
# ═══════════════════════════════════════
for key, default in {
    "optimization_done": False,
    "result": None,
    "city": "Beijing",
    "map_html": None,
    "model_status": None,
    "data_summary": None,
    "routes_detail": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════
# 核心逻辑 (从 demo.py 接入)
# ═══════════════════════════════════════

def get_data_summary(filepath: str) -> dict:
    """解析上传的 CSV 并返回统计摘要."""
    from src.utils.data_processing import load_all_data
    data = load_all_data(filepath)
    coords = data["coords"]
    demands = data["demands"]
    time_windows = data["time_windows"]
    priorities = data["priorities"]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    from src.ml.travel_time_predictor import detect_city_from_points
    city = detect_city_from_points(coords)
    return {
        "filename": Path(filepath).name,
        "n_points": len(coords),
        "n_customers": len(coords) - 1,
        "total_demand": sum(demands),
        "n_time_windows": sum(1 for tw in time_windows if tw != (0, 86400)),
        "n_priorities": sum(1 for p in priorities if p > 0),
        "lon_range": (min(lons), max(lons)),
        "lat_range": (min(lats), max(lats)),
        "area_km2": (max(lons) - min(lons)) * 111.32 * (max(lats) - min(lats)) * 111.32,
        "city": city,
        "coords": coords,
        "demands": demands,
    }


def get_model_status():
    """获取 AI 模型状态."""
    ml_status = None
    data_status = None

    try:
        from src.ml.travel_time_predictor import TravelTimePredictor
        city = st.session_state.get("city", "Beijing")
        points = st.session_state.get("data_summary", {}).get("coords", None)
        ml = TravelTimePredictor.load_or_fallback(city=city, points=points)
        if ml.is_trained:
            m = ml.metrics
            ml_status = {
                "trained": True,
                "rmse": m.get("rmse_seconds", "?"),
                "mae": m.get("mae_seconds", "?"),
                "mape": m.get("mape_pct", "?"),
                "n_samples": m.get("n_samples", "?"),
                "improvement": m.get("improvement_pct", "?"),
            }
        else:
            ml_status = {"trained": False}
    except Exception:
        ml_status = {"trained": False, "error": True}

    try:
        from src.adaptive.congestion_predictor import AdaptiveCongestionPredictor
        city = st.session_state.get("city", "Beijing")
        adaptive = AdaptiveCongestionPredictor.load_or_create(city=city)
        if adaptive.is_trained:
            m = adaptive.metrics
            adaptive_status = {
                "trained": True,
                "updates": m["updates"],
                "loss": f"{m['running_loss']:.4f}" if m["running_loss"] else "N/A",
            }
        else:
            adaptive_status = {"trained": False}
    except Exception:
        adaptive_status = {"trained": False, "error": True}

    try:
        from src.ml.data_collector import TravelTimeDataCollector
        col_stats = TravelTimeDataCollector().stats()
        city = st.session_state.get("city", "Beijing")
        data_status = {
            "total": col_stats.get("total", 0),
            "city_total": col_stats.get("cities", {}).get(city, {}).get("total", 0),
        }
    except Exception:
        data_status = {"total": 0, "city_total": 0}

    return {"ml": ml_status, "adaptive": adaptive_status, "data": data_status}


def run_optimization(filepath: str, num_vehicles: int, algorithm: str,
                     sim_time_seconds: int = 36000) -> dict:
    """运行一次完整优化并返回结果."""
    import logging
    logging.disable(logging.CRITICAL)

    from config import settings
    from main import DeliverySystem

    settings.DEFAULT_SIMULATION_TIME = sim_time_seconds

    system = DeliverySystem(use_road_network=True)
    success = system.run_full_pipeline(
        filepath=filepath,
        num_vehicles=num_vehicles,
        algorithm=algorithm,
        output_dir='output',
    )

    if success and hasattr(system, '_vrp_result'):
        r = system._vrp_result
        n_active = sum(1 for u in r.vehicles_used
                       if isinstance(u, dict) and u.get('is_active'))

        # 读最新 Folium 地图
        map_html = None
        map_files = sorted(Path("output").glob("folium-*.html"),
                           key=lambda x: x.stat().st_mtime, reverse=True)
        if map_files:
            map_html = map_files[0].read_text(encoding="utf-8")

        # 路线详情
        routes_detail = []
        for u in r.vehicles_used:
            if isinstance(u, dict) and u.get('is_active'):
                routes_detail.append({
                    "route_id": u["route_id"] + 1,
                    "vehicle_type": u.get("vehicle_type", "?"),
                    "stops": u["stops"],
                    "distance_km": round(u["distance_km"], 1),
                    "load": u["load"],
                    "capacity": u["capacity"],
                    "fixed_cost_km": u["fixed_cost_km"],
                    "running_cost_km": round(u["running_cost_km"], 1),
                })

        return {
            "success": True,
            "total_distance_km": round(r.total_distance_km, 1),
            "total_cost_km_eq": round(r.total_cost_km_eq, 1),
            "vehicles_used": n_active,
            "is_overflow": r.is_overflow,
            "overflow_nodes": r.overflow_nodes,
            "map_html": map_html,
            "routes_detail": routes_detail,
        }
    return {"success": False, "error": "优化失败"}


# ═══════════════════════════════════════
# 侧边栏 — 配置面板
# ═══════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ 配置面板")

    # ── 数据源 ──
    st.markdown("### 📂 数据源")
    data_option = st.radio("选择方式", ["使用现有文件", "上传 CSV"], horizontal=True,
                           label_visibility="collapsed")

    if data_option == "上传 CSV":
        uploaded = st.file_uploader("上传配送点 CSV", type=["csv"],
                                     help="必须包含 id,x,y,demand,time_window_start,time_window_end,priority 列")
        if uploaded:
            tmp_dir = Path(tempfile.gettempdir()) / "smart_delivery"
            tmp_dir.mkdir(exist_ok=True)
            tmp_path = tmp_dir / uploaded.name
            tmp_path.write_bytes(uploaded.getvalue())
            data_filepath = str(tmp_path)
        else:
            data_filepath = None
            st.info("请上传 CSV 文件")
    else:
        data_dir = PROJECT_ROOT / "data"
        csv_files = sorted([f.name for f in data_dir.glob("*.csv")])
        data_filepath = st.selectbox("选择数据文件", csv_files,
                                      index=csv_files.index("my_points1.csv")
                                      if "my_points1.csv" in csv_files else 0)

    # ── 参数 ──
    st.markdown("### 🎛️ 优化参数")
    num_vehicles = st.slider("最大车辆数", 3, 30, 5)
    algorithm = st.selectbox("求解算法",
                              ["ortools", "greedy", "cluster"],
                              format_func=lambda x: {
                                  "ortools": "OR-Tools (推荐)",
                                  "greedy": "贪心算法",
                                  "cluster": "聚类优先"
                              }[x])

    use_real_time = st.checkbox("使用当前真实时间", value=True,
                                 help="勾选则用系统当前时间, 否则用模拟时间")

    if use_real_time:
        now = datetime.now()
        sim_time_seconds = now.hour * 3600 + now.minute * 60 + now.second
        st.caption(f"当前模拟时间: {now.strftime('%H:%M:%S')}")
    else:
        custom_time = st.time_input("设置模拟时间", value=datetime.strptime("10:00", "%H:%M").time(),
                                     help="选择模拟的时间点, 用于拥堵预测和时间窗判断")
        sim_time_seconds = custom_time.hour * 3600 + custom_time.minute * 60
        st.caption(f"模拟时间: {custom_time.strftime('%H:%M')}")

    st.session_state.sim_time_seconds = sim_time_seconds

    st.markdown("---")

    # ── 预加载数据 ──
    if data_filepath:
        try:
            summary = get_data_summary(data_filepath)
            st.session_state.data_summary = summary
            st.session_state.city = summary["city"]
        except Exception as e:
            st.error(f"数据加载失败: {e}")
            st.session_state.data_summary = None

    # ── 运行按钮 ──
    st.markdown("### 🚀 运行")
    run_btn = st.button("开始优化", type="primary", use_container_width=True,
                        disabled=(data_filepath is None))

    st.markdown("---")
    st.caption("高德 API Key 需在 `config/settings.py` 配置")
    st.caption(f"© 2026 智能配送系统 v2.0")


# ═══════════════════════════════════════
# 主界面
# ═══════════════════════════════════════

st.markdown('<p class="main-header">智能配送优化系统</p>', unsafe_allow_html=True)
st.caption("基于高德真实路况 + LightGBM 行程预测 + 在线自适应拥堵学习")

# 响应运行按钮
if run_btn and data_filepath:
    with st.spinner("运行中... 正在调用高德 API 获取真实路网距离..."):
        t0 = time.time()
        result = run_optimization(data_filepath, num_vehicles, algorithm,
                                  sim_time_seconds=st.session_state.get("sim_time_seconds", 36000))
        elapsed = time.time() - t0

    if result["success"]:
        st.session_state.optimization_done = True
        st.session_state.result = result
        st.session_state.map_html = result.get("map_html")
        st.session_state.routes_detail = result.get("routes_detail")
        st.session_state.model_status = get_model_status()
        st.success(f"优化完成! 耗时 {elapsed:.1f} 秒")
    else:
        st.error(f"优化失败: {result.get('error', '未知错误')}")
        st.session_state.optimization_done = False

# ═══════════════════════════════════════
# 标签页
# ═══════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 数据概览", "🚚 路线结果", "🗺️ 交互地图", "🤖 AI 模型"]
)

# ── Tab 1: 数据概览 ──
with tab1:
    if st.session_state.data_summary:
        s = st.session_state.data_summary
        cols = st.columns(5)
        metrics = [
            ("配送点总数", s["n_points"], "个"),
            ("客户数", s["n_customers"], "个"),
            ("总需求量", s["total_demand"], "单位"),
            ("有时间窗", s["n_time_windows"], "单"),
            ("覆盖面积", f"{s['area_km2']:.1f}", "km²"),
        ]
        for col, (label, val, unit) in zip(cols, metrics):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label} ({unit})</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("检测城市", s["city"])
        c2.metric("经度范围", f"{s['lon_range'][0]:.4f} ~ {s['lon_range'][1]:.4f}")
        c3.metric("纬度范围", f"{s['lat_range'][0]:.4f} ~ {s['lat_range'][1]:.4f}")

        c1.metric("高优先级", f"{s['n_priorities']} 单")
    else:
        st.info("请在侧边栏选择或上传数据文件")

# ── Tab 2: 路线结果 ──
with tab2:
    if not st.session_state.optimization_done:
        st.info("点击侧边栏「开始优化」查看结果")
    else:
        r = st.session_state.result
        rd = st.session_state.routes_detail or []

        cols = st.columns(5)
        metrics2 = [
            ("总行驶距离", f"{r['total_distance_km']:.1f}", "km"),
            ("经济总代价", f"{r['total_cost_km_eq']:.1f}", "km基准"),
            ("使用车辆", str(r["vehicles_used"]), "辆"),
            ("是否溢出", "是" if r.get("is_overflow") else "否",
             f"({r.get('overflow_nodes', 0)}单)" if r.get("overflow_nodes") else ""),
            ("路线数", str(len(rd)), "条"),
        ]
        for col, (label, val, unit) in zip(cols, metrics2):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label} ({unit})</div>
                </div>""", unsafe_allow_html=True)

        st.caption("经济总代价 = Σ(固定成本 + 距离 × 车型费率)。中型车为基准(1.0×)，"
                   "微型车 0.5×、轻型车 0.8×、重型车 1.2×。"
                   "代价 < 距离说明多用了便宜小车，这是正常的。")

        st.markdown("---")
        st.subheader("车辆路线详情")

        for route in rd:
            icon = {"微型封闭货车": "🔵", "轻型封闭货车": "🟢", "中型厢式货车": "🟡", "重型厢式货车": "🔴"}.get(route["vehicle_type"], "⚪")
            pct = route["load"] / route["capacity"] * 100 if route["capacity"] else 0
            st.markdown(f"""
            <div class="route-card">
                <b>{icon} 车辆 {route['route_id']}</b> — {route['vehicle_type']} —
                {route['stops']} 站 | {route['distance_km']} km |
                载重 {route['load']}/{route['capacity']} ({pct:.0f}%) |
                固定 {route['fixed_cost_km']} + 运行 {route['running_cost_km']} = {route['fixed_cost_km'] + route['running_cost_km']:.1f} km基准
            </div>
            """, unsafe_allow_html=True)

# ── Tab 3: 交互地图 ──
with tab3:
    if not st.session_state.optimization_done:
        st.info("点击侧边栏「开始优化」生成地图")
    elif st.session_state.map_html:
        st.components.v1.html(st.session_state.map_html, height=600, scrolling=True)

# ── Tab 4: AI 模型 ──
with tab4:
    st.subheader("模型状态")

    if st.button("🔄 刷新模型状态"):
        st.session_state.model_status = get_model_status()

    ms = st.session_state.model_status
    if ms is None:
        ms = get_model_status()
        st.session_state.model_status = ms

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### LightGBM 行程预测")
        ml = ms.get("ml", {})
        if ml.get("trained"):
            st.success(f"已训练 ({ml.get('n_samples', '?')} 样本)")
            st.metric("RMSE", f"{ml.get('rmse', '?')} 秒")
            st.metric("MAE", f"{ml.get('mae', '?')} 秒")
            st.metric("vs 定速提升", f"{ml.get('improvement', '?')}%")
        elif ml.get("error"):
            st.error("加载失败")
        else:
            st.warning("未训练 — 需要 500+ 条样本")

    with c2:
        st.markdown("#### 在线自适应预测器")
        adaptive = ms.get("adaptive", {})
        if adaptive.get("trained"):
            st.success(f"在线学习中 ({adaptive.get('updates', '?')} 次更新)")
            st.metric("Running Loss", adaptive.get("loss", "N/A"))
            st.caption("越跑越准, 无需手动干预")
        elif adaptive.get("error"):
            st.error("加载失败")
        else:
            st.warning("未训练 — 运行系统后自动学习")

    with c3:
        st.markdown("#### 训练数据")
        d = ms.get("data", {})
        city = st.session_state.get("city", "?")
        st.metric("总样本", f"{d.get('total', 0)} 条")
        st.metric(f"{city} 样本", f"{d.get('city_total', 0)} 条")
        need = max(0, 500 - d.get("city_total", 0))
        if need > 0:
            st.caption(f"还需 {need} 条可训练 LightGBM")
        else:
            st.success("可训练 LightGBM")

    st.markdown("---")
    st.caption("收集更多时段数据: `python collect_all_day.py -d my_points1.csv --hours 8,10,12,14,16,18,20,22`")

# ═══════════════════════════════════════
# 页脚
# ═══════════════════════════════════════
st.markdown('<div class="footer">Smart Delivery System v2.0 — 物流设计大赛 | 数学建模大赛 | 交通科技大赛</div>',
            unsafe_allow_html=True)
