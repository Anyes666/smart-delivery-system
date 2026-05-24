# src/utils/data_processing.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Union
import hashlib
import pickle
from functools import wraps

# --- 配置日志 ---
logger = logging.getLogger(__name__)

# 定义数据根目录（相对于项目根目录）
PROJECT_ROOT = Path(__file__).parent.parent.parent  # smart-delivery-system/
DATA_DIR = PROJECT_ROOT / "data"

# --- 数据缓存设置 ---
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

def cache_result(func):
    """装饰器：缓存函数执行结果到磁盘"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 生成缓存键（基于函数名、参数）
        key_str = f"{func.__name__}_{str(args)}_{str(sorted(kwargs.items()))}"
        cache_key = hashlib.md5(key_str.encode()).hexdigest()
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        
        # 检查缓存是否存在且有效
        if cache_file.exists():
            try:
                logger.info(f"从缓存加载 {func.__name__} 的结果...")
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                logger.info(f"缓存命中: {func.__name__}")
                return result
            except Exception as e:
                logger.warning(f"缓存加载失败，重新计算: {e}")
                cache_file.unlink(missing_ok=True) # 删除损坏的缓存
        
        # 执行原函数
        logger.info(f"执行 {func.__name__} 并准备缓存结果...")
        result = func(*args, **kwargs)
        
        # 保存到缓存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            logger.info(f"结果已缓存至: {cache_file}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
            
        return result
    return wrapper

def validate_dataframe(df: pd.DataFrame, required_columns: List[str], source_name: str = "CSV"):
    """
    验证 DataFrame 是否包含必要的列。
    
    Args:
        df (pd.DataFrame): 待验证的 DataFrame
        required_columns (List[str]): 必需的列名列表
        source_name (str): 数据源名称，用于错误信息
        
    Raises:
        ValueError: 如果缺少必要列
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{source_name} 缺少必要的列: {missing_cols}. "
            f"现有列: {list(df.columns)}"
        )

@cache_result
def load_points_csv(filename: str = "points.csv") -> List[Dict]:
    """
    从 data/ 目录下加载指定 CSV 文件，返回点列表（字典形式）。
    结果会被缓存，下次调用相同参数时直接返回缓存值。
    缓存会自动检测 CSV 文件修改时间并失效。

    Args:
        filename (str): CSV 文件名，默认为 'points.csv'

    Returns:
        List[Dict]: 每个元素是 {'id': ..., 'x': ..., 'y': ..., 'demand': ..., ...}

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: CSV 格式错误或缺少必要列
    """
    # 智能路径: 若文件直接存在则用它, 否则拼接到 data/ 目录
    filepath = Path(filename)
    if not filepath.exists():
        filepath = DATA_DIR / filename
    logger.info(f"尝试加载数据文件: {filepath.absolute()}")

    # 检查缓存是否因 CSV 文件更新而过期
    key_str = f"load_points_csv_{(str(filename),)}_{{}}"
    cache_key = hashlib.md5(key_str.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    if cache_file.exists() and filepath.exists():
        try:
            csv_mtime = filepath.stat().st_mtime
            cache_mtime = cache_file.stat().st_mtime
            if csv_mtime > cache_mtime + 1.0:  # CSV 比缓存新 (1s容差)
                logger.info(f"CSV 文件已更新, 缓存失效: {filename}")
                cache_file.unlink(missing_ok=True)
        except Exception:
            pass
    
    if not filepath.exists():
        error_msg = f"数据文件未找到: {filepath.absolute()}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"成功读取 CSV 文件: {filepath.name}, 形状: {df.shape}")
    except Exception as e:
        error_msg = f"读取 CSV 文件失败: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e

    # 验证必需的列
    required_cols = ['id', 'x', 'y', 'demand', 'time_window_start', 'time_window_end', 'priority']
    validate_dataframe(df, required_cols, f"CSV文件 '{filename}'")

    # Helper: parse time fields — integer seconds OR "HH:MM" string → seconds
    def _parse_time(val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            if pd.isna(val):
                return None
            return int(val)
        s = str(val).strip()
        if ':' in s:
            parts = s.split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60
        return int(float(s))

    # 数据类型转换和基本验证
    try:
        df['id'] = df['id'].astype(int)
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df['demand'] = pd.to_numeric(df['demand'], errors='coerce')
        df['time_window_start'] = df['time_window_start'].apply(_parse_time)
        df['time_window_end'] = df['time_window_end'].apply(_parse_time)
        df['priority'] = pd.to_numeric(df['priority'], errors='coerce').astype(int)
    except Exception as e:
        error_msg = f"数据类型转换失败: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e

    # 检查是否有无效值 (NaN)
    invalid_rows = df.isnull().any(axis=1)
    if invalid_rows.any():
        invalid_indices = df[invalid_rows].index.tolist()
        error_msg = f"CSV 文件中发现无效数据 (NaN) 行索引: {invalid_indices}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 检查时间窗口是否合理 (start <= end)
    invalid_time_windows = df['time_window_start'] > df['time_window_end']
    if invalid_time_windows.any():
        invalid_ids = df[invalid_time_windows]['id'].tolist()
        error_msg = f"时间窗口不合理 (start > end) 的点 ID: {invalid_ids}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 检查 demand 是否非负
    if (df['demand'] < 0).any():
        neg_demand_ids = df[df['demand'] < 0]['id'].tolist()
        error_msg = f"需求量为负数的点 ID: {neg_demand_ids}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"数据验证通过，共加载 {len(df)} 个有效点。")
    return df.to_dict('records')

def get_coordinates(points: List[Dict]) -> List[Tuple[float, float]]:
    """
    从点列表中提取 (x, y) 坐标对。
    
    Args:
        points (List[Dict]): 点数据列表
        
    Returns:
        List[Tuple[float, float]]: (x, y) 坐标元组列表
    """
    if not points:
        logger.warning("输入点列表为空，返回空坐标列表。")
        return []
    
    coords = []
    for i, p in enumerate(points):
        try:
            x = float(p['x'])
            y = float(p['y'])
            coords.append((x, y))
        except (ValueError, KeyError) as e:
            logger.error(f"第 {i} 个点数据错误: {p}, 错误: {e}")
            raise ValueError(f"无法解析第 {i} 个点的坐标: {e}") from e
    logger.info(f"成功提取 {len(coords)} 个坐标对。")
    return coords

def get_demands(points: List[Dict]) -> List[int]:
    """
    提取需求量列表。
    
    Args:
        points (List[Dict]): 点数据列表
        
    Returns:
        List[int]: 需求量列表
    """
    if not points:
        logger.warning("输入点列表为空，返回空需求列表。")
        return []
    
    demands = []
    for i, p in enumerate(points):
        try:
            demand = int(p['demand'])
            demands.append(demand)
        except (ValueError, KeyError) as e:
            logger.error(f"第 {i} 个点需求量错误: {p}, 错误: {e}")
            raise ValueError(f"无法解析第 {i} 个点的需求量: {e}") from e
    logger.info(f"成功提取 {len(demands)} 个需求值。")
    return demands

def get_time_windows(points: List[Dict]) -> List[Tuple[int, int]]:
    """
    提取时间窗口列表。
    
    Args:
        points (List[Dict]): 点数据列表
        
    Returns:
        List[Tuple[int, int]]: (start, end) 时间窗口元组列表
    """
    if not points:
        logger.warning("输入点列表为空，返回空时间窗口列表。")
        return []
    
    windows = []
    for i, p in enumerate(points):
        try:
            start = int(p['time_window_start'])
            end = int(p['time_window_end'])
            windows.append((start, end))
        except (ValueError, KeyError) as e:
            logger.error(f"第 {i} 个点时间窗口错误: {p}, 错误: {e}")
            raise ValueError(f"无法解析第 {i} 个点的时间窗口: {e}") from e
    logger.info(f"成功提取 {len(windows)} 个时间窗口。")
    return windows

def get_priorities(points: List[Dict]) -> List[int]:
    """
    提取优先级列表。
    
    Args:
        points (List[Dict]): 点数据列表
        
    Returns:
        List[int]: 优先级列表
    """
    if not points:
        logger.warning("输入点列表为空，返回空优先级列表。")
        return []
    
    priorities = []
    for i, p in enumerate(points):
        try:
            priority = int(p['priority'])
            priorities.append(priority)
        except (ValueError, KeyError) as e:
            logger.error(f"第 {i} 个点优先级错误: {p}, 错误: {e}")
            raise ValueError(f"无法解析第 {i} 个点的优先级: {e}") from e
    logger.info(f"成功提取 {len(priorities)} 个优先级。")
    return priorities

def get_ids(points: List[Dict]) -> List[int]:
    """
    提取 ID 列表。
    
    Args:
        points (List[Dict]): 点数据列表
        
    Returns:
        List[int]: ID 列表
    """
    if not points:
        logger.warning("输入点列表为空，返回空 ID 列表。")
        return []
    
    ids = []
    for i, p in enumerate(points):
        try:
            pid = int(p['id'])
            ids.append(pid)
        except (ValueError, KeyError) as e:
            logger.error(f"第 {i} 个点 ID 错误: {p}, 错误: {e}")
            raise ValueError(f"无法解析第 {i} 个点的 ID: {e}") from e
    logger.info(f"成功提取 {len(ids)} 个 ID。")
    return ids

# --- 便捷函数 ---

def load_all_data(filename: str = "points.csv") -> Dict:
    """
    一次性加载所有数据并按结构返回。
    
    Args:
        filename (str): CSV 文件名
        
    Returns:
        Dict: 包含 'points', 'coords', 'demands', 'time_windows', 'priorities', 'ids', 'n' 的字典
    """
    points = load_points_csv(filename)
    return {
        'points': points,
        'coords': get_coordinates(points),
        'demands': get_demands(points),
        'time_windows': get_time_windows(points),
        'priorities': get_priorities(points),
        'ids': get_ids(points),
        'n': len(points)
    }

# --- 工具函数：获取可用数据文件列表 ---

def list_available_data_files() -> List[str]:
    """
    列出 data/ 目录下所有 .csv 文件。
    
    Returns:
        List[str]: CSV 文件名列表
    """
    files = list(DATA_DIR.glob("*.csv"))
    names = [f.name for f in files]
    logger.info(f"在 {DATA_DIR} 目录下找到 {len(names)} 个 CSV 文件: {names}")
    return names

if __name__ == "__main__":
    # --- 示例用法 ---
    import sys
    # 设置日志级别为 INFO 以便查看详细过程
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 尝试列出可用文件
    available_files = list_available_data_files()
    print("可用的 CSV 文件:", available_files)
    
    # 尝试加载默认文件
    try:
        filename = "provincial_capitals.csv" # 或 "points.csv"
        if filename not in available_files:
            print(f"警告: 文件 {filename} 不存在于 data/ 目录下。")
            filename = available_files[0] if available_files else "points.csv"
            print(f"使用第一个可用文件: {filename}")
        
        print(f"\n--- 加载文件: {filename} ---")
        data = load_all_data(filename)
        
        print(f"加载点数量: {data['n']}")
        print("前3个坐标:", data['coords'][:3])
        print("前3个需求:", data['demands'][:3])
        print("前3个时间窗口:", data['time_windows'][:3])
        print("前3个优先级:", data['priorities'][:3])
        print("前3个ID:", data['ids'][:3])
        
    except Exception as e:
        print(f"加载失败: {e}", file=sys.stderr)
        sys.exit(1)