# 智能物流配送系统 — Smart Delivery System

基于异构车队与实时路网的智能配送路径优化系统，支持**高德地图 API** 真实道路距离、**LightGBM** 行程时间预测、**在线自适应神经网络**拥堵预测，以及 **OR-Tools** VRP 求解器。

## 功能特性

- **三层优化管线**：距离矩阵 → ML 行程时间预测 → 自适应拥堵乘数 → VRP 求解
- **异构车队**：4 种车型（微/轻/中/重型封闭货车），对标 JT/T 1325-2020 标准
- **真实路网**：高德 Web API 构建 N×N 距离矩阵，支持道路跟随可视化
- **ML 预测**：LightGBM 39 维特征，点对点行程时间预测
- **在线自适应**：20→64→32→1 神经网络，SGD 在线学习，边跑边准
- **交通模拟**：早晚高峰拥堵、随机事故、红绿灯周期检测
- **三维可视化**：Folium 数字孪生地图 + Plotly 车流动画 + 业务指标仪表盘
- **Web 界面**：Streamlit 交互式操作界面

## 环境要求

- Python >= 3.9
- 高德地图 Web API Key（[免费申请](https://lbs.amap.com/)）

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置高德 API Key（二选一）
# 方式 A: 环境变量
set AMAP_API_KEY=你的高德Key    # Windows
export AMAP_API_KEY=你的高德Key  # Linux/Mac

# 方式 B: 直接写入配置文件
# 编辑 config/settings.py，将 _amap_api_key = '' 改为你的 Key

# 3. 启动 Web 界面
streamlit run app.py

# 4. 或使用 CLI 模式
python scripts/run_system.py -d data/校内游览.csv -v 5 -a ortools

# 5. 或一键演示（三算法对比）
python scripts/demo.py --compare
```

## 项目结构

```
├── app.py                      # Streamlit Web 界面（4 页签）
├── main.py                     # 核心调度引擎 DeliverySystem
├── config/
│   └── settings.py             # 所有配置参数（API、车型、算法、可视化）
├── src/
│   ├── adaptive/               # 在线自适应神经网络预测器
│   ├── algorithms/             # VRP 求解器（OR-Tools / Greedy / Cluster）
│   ├── benchmark/              # 基准测试与公开数据集
│   ├── map/                    # 高德 API / OSM 路网封装
│   ├── ml/                     # LightGBM 行程时间预测 + 数据采集
│   ├── traffic/                # 拥堵引擎 / 红绿灯 / 限行规则
│   ├── utils/                  # 数据处理 / 坐标转换 / 国标常量
│   └── visualization/          # Folium 地图 / Plotly 动画 / 仪表盘
├── scripts/
│   ├── run_system.py           # CLI 命令行入口
│   ├── demo.py                 # 一键演示（三算法对比）
│   ├── train_travel_time.py    # 训练 LightGBM 模型
│   ├── train_adaptive.py       # 训练自适应预测器
│   ├── collect_all_day.py      # 全天数据采集 (8:00-22:00)
│   └── warm_cache.py           # 预热距离缓存
└── data/                       # 配送数据 CSV
    ├── 校内游览.csv
    ├── 西直门外卖店.csv
    └── 北京各个区快递配送.csv
```

## VRP 算法对比

| 算法 | 方法 | 特点 |
|------|------|------|
| **OR-Tools** | PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH | 最优解质量，支持异构车队 |
| **Greedy** | 最远种子 + 最近邻 + 2-opt | 速度快，路线呈放射状分区 |
| **Cluster** | KMeans 聚类 + 簇内 TSP + 2-opt | 地理分区明确，适合区域配送 |

## 可视化输出

每次运行生成三份 HTML 报告在 `output/` 目录：

1. **`folium-*.html`** — 交互式数字孪生地图（9 图层的路网可视化）
2. **`plotly-*.html`** — 配送过程动画（车辆移动轨迹）
3. **`metrics-*.html`** — 业务指标仪表盘（配送完成时间、均衡度、每车耗时等）

## 使用步骤

1. 在 `app.py` 侧边栏选择数据源和车辆数
2. 选择算法（推荐 OR-Tools）
3. 点击"开始优化"运行
4. 查看结果卡片、交互地图和指标仪表盘

## 技术栈

- OR-Tools — 运筹优化引擎
- LightGBM — 梯度提升树回归
- Plotly + Folium — 交互式可视化
- Streamlit — Web 应用框架
- NumPy / Pandas / Scikit-learn — 数据处理

## License

仅供课程学习与学术评审使用。
