# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Web interface (primary entry point)
streamlit run app.py

# CLI mode — all flags: -d DATA -v NUM_VEHICLES -a ALGO [--amap-key KEY] [--real-time] [--dark-theme]
python scripts/run_system.py -d data/校内游览.csv -v 5 -a ortools

# One-shot demo (--no-viz for speed, --compare to run all 3 algorithms)
python scripts/demo.py

# Train LightGBM travel time model
python scripts/train_travel_time.py --train --city Beijing

# Train adaptive congestion predictor from collected samples
python scripts/train_adaptive.py --city Beijing --epochs 100

# Collect all-day training data (run overnight, 8:00-22:00)
python scripts/collect_all_day.py --city Beijing --interval 30

# Warm distance-matrix cache before class (~10 min before demo)
python scripts/warm_cache.py

# Overfitting diagnostic for adaptive model
python scripts/_check_overfit.py

# Generate Word report for course submission
python scripts/generate_report.py
```

Python interpreter: `D:\python\python.exe`. No virtual environment needed — dependencies are installed globally. Python >= 3.9 with ortools, lightgbm, streamlit, folium, plotly, and others in `requirements.txt`.

## Architecture

The system solves **heterogeneous-fleet Capacitated VRP** through a three-layer optimization pipeline applied sequentially to a distance matrix:

```
CSV delivery points → Amap real road distances → LightGBM travel time → Adaptive congestion multiplier → VRP solver → 3-stage visualization
```

### Layer 1 — Distance (map layer)
`src/map/amap_network.py` calls Amap Web API to build an N×N real-road distance matrix (km). Uses threading (6 workers), symmetry optimization (N(N-1)/2 calls), and two-level cache (memory LRU + disk pickle). Rate-limited to ~3 QPS (0.35s interval). Falls back to Haversine distance in `src/algorithms/shortest_path.py` if the API is unavailable. `ROAD_NETWORK_PROVIDER` switch (`'amap'` | `'osmnx'`) controls which backend is used.

### Layer 2 — Travel Time (ML layer)
`src/ml/travel_time_predictor.py` uses LightGBM (39 features across 6 categories) to predict point-to-point travel time in seconds. Features include sin/cos cyclic time encoding, discrete hour/day buckets, relative distance from city center, and road geometry proxies. Falls back to `dist_km / 40 km/h × 3600` when untrained. Training data is collected passively via `src/ml/data_collector.py` and stored as JSONL per city in `.cache/ml/`.

### Layer 3 — Congestion (adaptive layer)
`src/adaptive/congestion_predictor.py` uses a 10→64→32→1 neural network with online SGD. Predicts a **multiplier** applied to the distance matrix (`clamp 0.8–2.0`). Learns from every real API feedback sample via single-sample SGD with gradient clipping.

**Dual-model fusion** (`main.py:_apply_traffic_penalties`): 70% LightGBM + 30% Adaptive for travel time prediction. The adaptive model captures time-of-day, day-of-week, and spatial congestion patterns that the static LightGBM model misses.

### VRP Solver (three algorithms, one interface)
`src/algorithms/vrp_solver.py` — all three algorithms return `UnifiedSolution` (defined in `src/algorithms/vehicle_types.py`):

| Algorithm | Method | Best for |
|-----------|--------|----------|
| `ortools` | PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH | Best quality, heterogeneous fleet, time windows |
| `greedy` | Farthest-first seed + nearest-neighbor fill + 2-opt | Fast, practical zone-based routes |
| `cluster` | Demand-aware KMeans + per-cluster TSP + 2-opt | Geographically grouped points |

The fleet has 4 vehicle types per JT/T 1325-2020: micro (15), light (30), medium (60), heavy (120). Each has independent capacity, fixed cost, and per-km cost. OR-Tools trades off "more small vehicles vs fewer large vehicles" automatically.

### Key classes and data flow

- **`DeliverySystem`** (`main.py`): Central orchestrator. `run_full_pipeline()` calls `load_data()` → `calculate_optimal_routes()` → `visualize_routes()`.
- **`AmapRoadNetwork`** (`src/map/amap_network.py`): Wraps Amap API. `compute_distance_matrix()` returns N×N numpy array in km. Also provides `get_route_geometry()` for polyline-based visualization and `decode_polyline()` for road-following paths.
- **`VRPSolver`** (`src/algorithms/vrp_solver.py`): Accepts a `Fleet` and `distance_matrix`. The `Fleet.max_total` is the hard cap on vehicles used — set from the user's slider in `app.py` → `main.py:calculate_optimal_routes()`.
- **`UnifiedSolution`** (`src/algorithms/vehicle_types.py`): Dataclass with `.routes`, `.vehicle_assignments`, `.total_distance_km`, `.is_overflow`, `.overflow_nodes`. Call `.compute(demands)` to populate `.vehicles_used` details.

### Traffic & congestion system

- **`src/traffic/congestion_engine.py`** — `CongestionEngine`: models time-varying congestion on OSM edges using road-class base multipliers (motorway 0.85x → residential 1.30x → service 1.50x), time-of-day curves (AM peak 8:00-9:30 at 2.0x, PM peak 16:30-18:30 at 2.0x), per-edge noise, and random accidents (30-120 min, 3.0-5.0x). Key methods: `get_congestion_multiplier()`, `simulate_accidents()`, `get_congested_edges()`.
- **`src/traffic/traffic_lights.py`** — `TrafficLightModel`: places traffic lights at OSM intersections (nodes with ≥3 edges, ≥1 primary-class). Cycle: green 50% → yellow 3s → red 47%, randomized phase offsets.
- **`src/traffic/route_traffic_lights.py`** — `generate_route_traffic_lights()`: detects lights from Amap polylines (no OSM needed) by detecting turns >50°, sampling at 800-1500m spacing, deduplicating within 250m.
- **`src/traffic/time_rules.py`** — `TimeBasedAccessRules`: models truck restrictions during rush hours.

### Visualization pipeline (3-stage output)

`DeliverySystem.visualize_routes()` produces three HTML files in `output/`:

1. **Folium Digital Twin Map** (`src/visualization/enhanced_map.py`, `EnhancedMapRenderer.create_digital_twin_map()`): Interactive map with 9 layers — base tiles (Amap or CartoDB dark), delivery points with MarkerCluster, road-following route polylines per vehicle, congestion heatmap (green→red), traffic light markers, time-restricted road overlays (dashed red), accident incident markers with warning icons, LayerControl toggle, and a legend. Falls back to `src/visualization/folium_maps.py` (Phase-1 renderer) when enhanced renderer unavailable.

2. **Plotly Route Animation** (`src/visualization/plotly_animation.py`, `PlotlyAnimator.create_route_animation()`): Animated HTML with sequential vehicle movement, frame-by-frame slider, play/pause controls, road-following geometry when available.

3. **Metrics Dashboard** (`src/visualization/plotly_animation.py`, `PlotlyAnimator.create_metrics_dashboard()`): 3×2 subplot grid — total time gauge (drive + wait + red-light count), route balance gauge (with status label), per-vehicle time stacked bar (sorted descending, blue drive + orange wait), per-vehicle distance bars + load line (dual y-axis), segment detail table (light theme, alternating rows), red-light wait summary (sorted, top 30, color-coded). Professional business palette: blue (#2563eb) primary, orange (#f97316) accent.

### Streamlit app (`app.py`)

Four-tab web interface:
- **Tab 1 — Data Overview**: metric cards (total points, customers, demand, time-window orders, coverage area), city detection, coordinate ranges.
- **Tab 2 — Route Results**: metric cards (distance, cost, vehicles used, overflow, route count), per-vehicle detail cards with load/capacity bars and cost breakdown.
- **Tab 3 — Interactive Map**: embeds the Folium HTML via `st.components.v1.html()`.
- **Tab 4 — AI Model Status**: LightGBM metrics (RMSE, MAE, improvement), adaptive predictor state (running loss), training data counts.

Sidebar: data source selector, vehicle count slider, algorithm picker, simulation time controls, run button → instantiates `DeliverySystem.run_full_pipeline()`.

### Configuration hierarchy

`config/settings.py` is the single source of truth. API keys: environment variable `AMAP_API_KEY` takes precedence over the hardcoded fallback. Key config sections:

| Section | Controls |
|---------|----------|
| `VEHICLE_TYPES` | 4 fleet types with capacity, fixed cost, per-km cost per JT/T 1325-2020 |
| `AMAP_CONFIG` | API key, routing strategy (0=fastest/2=shortest/5=no-highway), rate limit (0.35s), cache dir |
| `ALGORITHM_PARAMS` | Greedy clustering, OR-Tools first-solution and local-search strategies |
| `CONGESTION_CONFIG` | Rush-hour multiplier (2.0), accident count/probability, random seed |
| `TRAFFIC_LIGHT_CONFIG` | Cycle length (60s), green ratio (50%), yellow duration (3s) |
| `TRAFFIC_LIGHT_DETECTION` | Amap polyline-based params: spacing (800-1500m), turn threshold (50°), max per route (15) |
| `ADAPTIVE_CONFIG` | Online NN: exploration noise (0.3→0.02 decay), learning rate (0.001), hidden layers [64,32] |
| `ML_TRAVEL_TIME_CONFIG` | Min samples for training (500), auto-collect, dedup window (2h), save interval (50) |
| `TILE_PROVIDER` | 'amap' (requires HTTP tiles) vs 'cartodb' (CartoDB dark_matter, free) |

### Data and caching

- Input CSVs in `data/` — must have `lon,lat,demand` columns (optionally `time_window_start,time_window_end`).
- `.cache/` stores road network graphs, Amap API responses, ML models, and adaptive predictor weights. Safe to delete — everything rebuilds on next run.
- ML training data collected via `src/ml/data_collector.py`, stored as JSONL per city in `.cache/ml/`.
- Output files in `output/`: `folium-*.html`, `plotly-*.html`, `metrics-*.html`, `result-*.json`.
