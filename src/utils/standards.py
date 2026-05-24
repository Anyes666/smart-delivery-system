"""
标准化工具模块 — 对标国家物流标准体系.

参考标准:
- GB/T 22263-2008 物流信息系统技术规范
- GB/T 37378-2019 物流设施设备编码
- GB/T 18354-2021 物流术语
- GB/T 28583-2012 快递服务网络运输信息管理
- JT/T 1325-2020 城市绿色配送车辆选型
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════
# JT/T 1325-2020 城市绿色配送车辆分类
# ═══════════════════════════════════════════════════

NATIONAL_VEHICLE_TYPES = {
    "微型封闭货车": {
        "code": "V-MINI",
        "max_load_t": 1.8,
        "typical_capacity": 15,
        "length_m": (2.5, 4.5),
        "energy_types": ["electric", "petrol"],
        "applicable_scenarios": ["社区配送", "快递末端", "便利店补货"],
    },
    "轻型封闭货车": {
        "code": "V-LIGHT",
        "max_load_t": 4.5,
        "typical_capacity": 30,
        "length_m": (4.5, 6.0),
        "energy_types": ["electric", "petrol", "diesel"],
        "applicable_scenarios": ["城市配送", "商超配送", "生鲜冷链"],
    },
    "中型厢式货车": {
        "code": "V-MEDIUM",
        "max_load_t": 12.0,
        "typical_capacity": 60,
        "length_m": (6.0, 9.0),
        "energy_types": ["diesel", "electric"],
        "applicable_scenarios": ["区域配送", "城际运输", "仓储转运"],
    },
    "重型厢式货车": {
        "code": "V-HEAVY",
        "max_load_t": 25.0,
        "typical_capacity": 120,
        "length_m": (9.0, 12.0),
        "energy_types": ["diesel", "LNG"],
        "applicable_scenarios": ["干线运输", "大宗物资", "跨城配送"],
    },
}


def validate_vehicle_type(vehicle_type: Dict) -> Tuple[bool, str]:
    """验证车型定义是否符合 JT/T 1325-2020.

    Returns:
        (is_valid, message)
    """
    name = vehicle_type.get("name", "")
    if name not in NATIONAL_VEHICLE_TYPES:
        return False, (
            f"车型名称 '{name}' 不在国标分类中. "
            f"可用类型: {list(NATIONAL_VEHICLE_TYPES.keys())}"
        )
    spec = NATIONAL_VEHICLE_TYPES[name]
    capacity = vehicle_type.get("capacity", 0)
    typical = spec["typical_capacity"]
    lower = typical * 0.3
    upper = typical * 2.5
    if capacity < lower or capacity > upper:
        return False, (
            f"车型 '{name}' 载重 {capacity} 偏离国标典型值 {typical} "
            f"(合理范围 {lower:.0f}-{upper:.0f})"
        )
    return True, "OK"


# ═══════════════════════════════════════════════════
# GB/T 22263-2008 数据字典验证
# ═══════════════════════════════════════════════════

REQUIRED_INPUT_FIELDS = {
    "点编号": {"type": "int", "description": "配送点唯一标识符", "nullable": False},
    "经度": {"type": "float", "unit": "度(°)", "range": (73.0, 135.0), "nullable": False},
    "纬度": {"type": "float", "unit": "度(°)", "range": (18.0, 54.0), "nullable": False},
    "需求量": {"type": "float", "unit": "件/吨", "range": (0, 100000), "nullable": False},
    "时间窗开始": {"type": "int", "unit": "秒(当日)", "range": (0, 86400), "nullable": True},
    "时间窗结束": {"type": "int", "unit": "秒(当日)", "range": (0, 86400), "nullable": True},
    "优先级": {"type": "int", "range": (0, 10), "nullable": True},
}

REQUIRED_OUTPUT_FIELDS = {
    "路线编号": {"type": "str", "description": "唯一路线标识", "standard": "GB/T 28583"},
    "车辆编号": {"type": "str", "description": "执行车辆标识", "standard": "GB/T 37378"},
    "车型代码": {"type": "str", "description": "JT/T 1325 车型代码"},
    "途经点序列": {"type": "list[int]", "description": "配送点访问顺序"},
    "总行驶里程": {"type": "float", "unit": "km"},
    "总行程时间": {"type": "float", "unit": "s"},
    "装载率": {"type": "float", "unit": "%"},
}


def validate_data_dict(input_schema: Dict, output_schema: Dict = None) -> Tuple[bool, List[str]]:
    """验证数据字典是否覆盖 GB/T 22263 要求的字段.

    Returns:
        (is_valid, missing_fields)
    """
    missing = []
    for field, spec in REQUIRED_INPUT_FIELDS.items():
        if spec["nullable"]:
            continue
        if field not in input_schema:
            missing.append(f"输入缺少必要字段: {field} ({spec['description']})")
    return len(missing) == 0, missing


def format_output_standard(
    routes: List[List[int]],
    demands: List[float],
    distance_matrix: List[List[float]],
    vehicle_types: List[str] = None,
    depot_index: int = 0,
) -> List[Dict]:
    """按 GB/T 28583-2012 格式化路线输出.

    Returns:
        标准化的路线列表, 每条路线包含运单级别的字段.
    """
    output = []
    for vid, route in enumerate(routes):
        if len(route) <= 2:
            continue  # 空路线

        stops = [n for n in route if n != depot_index]
        if not stops:
            continue

        # 计算路线指标
        total_dist = sum(
            distance_matrix[route[j]][route[j + 1]]
            for j in range(len(route) - 1)
        )
        total_load = sum(demands[n] for n in stops)

        waybill = {
            "运单号": f"SDS-{datetime.now():%Y%m%d}-V{vid + 1:02d}",
            "路线编号": f"R-{vid + 1:04d}",
            "车辆编号": f"V-{vid + 1:04d}",
            "车型代码": vehicle_types[vid] if vehicle_types and vid < len(vehicle_types) else "V-LIGHT",
            "配送中心点编号": depot_index,
            "途经点序列": [int(n) for n in stops],
            "停靠站数": len(stops),
            "总行驶里程_km": round(total_dist, 2),
            "总装载量": round(total_load, 2),
            "装载率_pct": round(total_load / 60 * 100, 1),  # 默认中型车 60 单位
            "预计出发时间": "08:00:00",
            "预计返回时间": "12:00:00",
            "数据标准": "GB/T 28583-2012",
        }
        output.append(waybill)
    return output


# ═══════════════════════════════════════════════════
# GB/T 18354-2021 术语检查
# ═══════════════════════════════════════════════════

TERM_MAP = {
    # 非标准 → 标准
    "配送中心": "配送中心",  # GB/T 18354 定义: 从事配送业务且具有完善信息网络的场所或组织
    "仓库": "仓储设施",
    "中转站": "中转节点",
    "快递员": "快递服务人员",
    "货物": "物品",
    "载重": "装载质量",
    "路线": "配送路径",
    "车辆": "配送车辆",
}

DEPRECATED_TERMS = {
    "运货汽车": "配送车辆",
    "送货员": "快递服务人员",
    "货物量": "装载质量",
}


def check_terminology(text: str) -> List[Tuple[str, str, str]]:
    """检查文本中使用的术语是否符合 GB/T 18354-2021.

    Returns:
        [(deprecated_term, suggested_term, context), ...]
    """
    warnings = []
    for deprecated, suggested in DEPRECATED_TERMS.items():
        if deprecated in text:
            warnings.append((deprecated, suggested, text[max(0, text.find(deprecated) - 20):text.find(deprecated) + len(deprecated) + 20]))
    return warnings


# ═══════════════════════════════════════════════════
# 设施编码 (GB/T 37378-2019)
# ═══════════════════════════════════════════════════

def encode_facility(facility_type: str, city_code: str, serial: int) -> str:
    """按 GB/T 37378-2019 生成设施编码.

    Args:
        facility_type: "DC"=配送中心, "WH"=仓库, "TS"=中转站
        city_code: 城市行政区划代码 (6位)
        serial: 流水号 (4位)

    Returns:
        设施编码字符串, 格式: F-{type}{city}{serial}
    """
    return f"F-{facility_type}{city_code}{serial:04d}"


def encode_vehicle(vehicle_type_code: str, city_code: str, serial: int) -> str:
    """按 JT/T 1325 生成车辆编码.

    Returns:
        车辆编码字符串, 格式: V-{type}{city}{serial}
    """
    return f"V-{vehicle_type_code}{city_code}{serial:04d}"


def encode_shipment(date_str: str, city_code: str, serial: int) -> str:
    """按 GB/T 28583 生成运单编码.

    Returns:
        运单编码字符串, 格式: S-{date}{city}{serial}
    """
    return f"S-{date_str}-{city_code}-{serial:06d}"
