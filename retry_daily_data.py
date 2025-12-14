#!/usr/bin/env python
"""
重试获取每日数据（带详细日志）
"""
import os
import sys
import time
from datetime import datetime, timedelta

# 禁用代理
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'

import akshare as ak
import pandas as pd
from src.data_fetcher import StockDataFetcher

def main():
    """主函数"""
    print("=" * 60)
    print("重试获取每日数据")
    print("=" * 60)
    print(f"\n当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 获取最近3天的数据
    from src.utils import get_recent_days_date_range
    start_date, end_date = get_recent_days_date_range(days=3)
    print(f"日期范围: {start_date} 至 {end_date}")
    
    fetcher = StockDataFetcher(use_proxy=False)
    
    # 获取已有数据的股票代码
    daily_dir = "data/daily"
    if os.path.exists(daily_dir):
        existing_files = [f for f in os.listdir(daily_dir) if f.endswith('_daily.csv')]
        stock_codes = [f.replace('_daily.csv', '') for f in existing_files]
        print(f"\n找到 {len(stock_codes)} 只股票: {', '.join(stock_codes)}")
    else:
        print("\n未找到已有数据")
        return
    
    print(f"\n开始获取数据（最多重试3次，每次间隔10秒）...")
    success_count = 0
    
    for stock_code in stock_codes:
        for attempt in range(3):
            try:
                if attempt > 0:
                    print(f"\n  [{stock_code}] 第 {attempt + 1} 次重试...")
                    time.sleep(10)
                else:
                    print(f"\n  [{stock_code}] 第 1 次尝试...")
                
                df = fetcher.fetch_daily_data(stock_code, start_date, end_date)
                
                if df is not None and not df.empty:
                    # 合并数据
                    filepath = os.path.join(fetcher.daily_dir, f"{stock_code}_daily.csv")
                    if os.path.exists(filepath):
                        existing_df = pd.read_csv(filepath)
                        existing_df['日期'] = pd.to_datetime(existing_df['日期'])
                        df['日期'] = pd.to_datetime(df['日期'])
                        
                        combined = pd.concat([existing_df, df]).drop_duplicates(subset=['日期'], keep='last')
                        combined = combined.sort_values('日期').reset_index(drop=True)
                        combined['日期'] = combined['日期'].dt.strftime('%Y-%m-%d')
                        combined.to_csv(filepath, index=False, encoding='utf-8-sig')
                        print(f"  ✓ 成功！新增 {len(df)} 条记录")
                    else:
                        fetcher.save_daily_data(stock_code, df)
                        print(f"  ✓ 成功！保存 {len(df)} 条记录")
                    
                    success_count += 1
                    break
                else:
                    if attempt < 2:
                        print(f"  ✗ 无数据，将重试...")
                    else:
                        print(f"  ✗ 最终失败：无数据返回")
            except Exception as e:
                error_msg = str(e)[:100]
                if attempt < 2:
                    print(f"  ✗ 失败: {error_msg}... 将重试")
                else:
                    print(f"  ✗ 最终失败: {error_msg}")
        
        time.sleep(2)  # 股票之间的间隔
    
    print("\n" + "=" * 60)
    print(f"完成！成功: {success_count}/{len(stock_codes)} 只")
    if success_count == 0:
        print("\n建议：")
        print("1. API服务器可能暂时不可用，请稍后重试")
        print("2. 可以尝试使用分钟级数据（当前可用）")
        print("3. 或者明天再试（可能今天的数据还没更新）")
    print("=" * 60)

if __name__ == "__main__":
    main()

