"""
工具函数模块
"""
import os
from datetime import datetime, timedelta
from typing import Optional


def ensure_dir(directory: str) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def format_date(date: datetime, fmt: str = "%Y%m%d") -> str:
    """
    格式化日期为字符串
    
    Args:
        date: 日期对象
        fmt: 日期格式，默认为YYYYMMDD
        
    Returns:
        格式化后的日期字符串
    """
    return date.strftime(fmt)


def parse_date(date_str: str, fmt: str = "%Y%m%d") -> datetime:
    """
    解析日期字符串为datetime对象
    
    Args:
        date_str: 日期字符串
        fmt: 日期格式，默认为YYYYMMDD
        
    Returns:
        datetime对象
    """
    return datetime.strptime(date_str, fmt)


def get_date_range(days: int = 365) -> tuple[str, str]:
    """
    获取日期范围（从今天往前推指定天数）
    
    Args:
        days: 往前推的天数，默认365天
        
    Returns:
        (start_date, end_date) 元组，格式为YYYYMMDD
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return format_date(start_date), format_date(end_date)


def get_recent_days_date_range(days: int = 7) -> tuple[str, str]:
    """
    获取最近N天的日期范围
    
    Args:
        days: 天数，默认7天
        
    Returns:
        (start_date, end_date) 元组，格式为YYYYMMDD
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return format_date(start_date), format_date(end_date)


def get_trade_days(start_date: str, end_date: str) -> list[str]:
    """
    获取交易日列表（排除周末和节假日）
    注意：这是一个简化版本，实际应该使用交易日历API
    
    Args:
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        
    Returns:
        交易日列表，格式为YYYYMMDD
    """
    start = parse_date(start_date)
    end = parse_date(end_date)
    trade_days = []
    
    current = start
    while current <= end:
        # 排除周末（周六=5, 周日=6）
        if current.weekday() < 5:
            trade_days.append(format_date(current))
        current += timedelta(days=1)
    
    return trade_days

