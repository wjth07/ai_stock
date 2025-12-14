#!/usr/bin/env python
"""
补充最近一周的每日数据
"""
import sys
import os
from datetime import datetime, timedelta
from src.data_fetcher import StockDataFetcher
from src.utils import get_recent_days_date_range

def main():
    """主函数"""
    print("=" * 60)
    print("补充最近一周的每日数据")
    print("=" * 60)
    
    # 获取最近7天的日期范围
    start_date, end_date = get_recent_days_date_range(days=7)
    print(f"\n日期范围: {start_date} 至 {end_date}")
    
    # 初始化数据获取器
    fetcher = StockDataFetcher()
    
    # 获取已有数据的股票代码
    daily_dir = "data/daily"
    if os.path.exists(daily_dir):
        existing_files = [f for f in os.listdir(daily_dir) if f.endswith('_daily.csv')]
        stock_codes = [f.replace('_daily.csv', '') for f in existing_files]
        print(f"\n找到 {len(stock_codes)} 只已有数据的股票")
        print(f"股票代码: {', '.join(stock_codes[:10])}{'...' if len(stock_codes) > 10 else ''}")
    else:
        print("\n未找到已有数据，将获取全部A股数据")
        stock_codes = None
    
    # 获取数据
    print(f"\n开始获取最近一周的每日数据...")
    fetcher.fetch_all_daily_data(
        stock_codes=stock_codes,
        start_date=start_date,
        end_date=end_date
    )
    
    print("\n" + "=" * 60)
    print("数据更新完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

