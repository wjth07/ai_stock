#!/usr/bin/env python
"""
补充今天的数据（带重试机制）
"""
import os
import sys
import time
from datetime import datetime

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

def update_stock_data(stock_code, start_date, end_date, max_retries=3):
    """更新单只股票数据，带重试"""
    fetcher = StockDataFetcher()
    
    for attempt in range(max_retries):
        try:
            print(f"  尝试 {attempt + 1}/{max_retries}: {stock_code}...")
            df = fetcher.fetch_daily_data(stock_code, start_date, end_date)
            
            if df is not None and not df.empty:
                # 读取现有数据
                filepath = os.path.join(fetcher.daily_dir, f"{stock_code}_daily.csv")
                if os.path.exists(filepath):
                    existing_df = pd.read_csv(filepath)
                    existing_df['日期'] = pd.to_datetime(existing_df['日期'])
                    df['日期'] = pd.to_datetime(df['日期'])
                    
                    # 合并数据，去重
                    combined = pd.concat([existing_df, df]).drop_duplicates(subset=['日期'], keep='last')
                    combined = combined.sort_values('日期').reset_index(drop=True)
                    combined['日期'] = combined['日期'].dt.strftime('%Y-%m-%d')
                    combined.to_csv(filepath, index=False, encoding='utf-8-sig')
                    print(f"  ✓ {stock_code}: 已更新，新增 {len(df)} 条记录")
                else:
                    fetcher.save_daily_data(stock_code, df)
                    print(f"  ✓ {stock_code}: 已保存 {len(df)} 条记录")
                return True
            else:
                print(f"  ✗ {stock_code}: 无数据返回")
                return False
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"  ✗ 失败: {str(e)[:100]}... 等待 {wait_time} 秒后重试")
                time.sleep(wait_time)
            else:
                print(f"  ✗ {stock_code}: 最终失败 - {str(e)[:100]}")
                return False
    
    return False

def main():
    """主函数"""
    print("=" * 60)
    print("补充今天的每日数据")
    print("=" * 60)
    
    today = datetime.now().strftime("%Y%m%d")
    # 获取最近3天的数据，以防今天数据还没更新
    from datetime import timedelta
    start_date = (datetime.now() - timedelta(days=3)).strftime("%Y%m%d")
    end_date = today
    
    print(f"\n日期范围: {start_date} 至 {end_date}")
    
    # 获取已有数据的股票代码
    daily_dir = "data/daily"
    if os.path.exists(daily_dir):
        existing_files = [f for f in os.listdir(daily_dir) if f.endswith('_daily.csv')]
        stock_codes = [f.replace('_daily.csv', '') for f in existing_files]
        print(f"\n找到 {len(stock_codes)} 只已有数据的股票")
        print(f"股票代码: {', '.join(stock_codes)}")
    else:
        print("\n未找到已有数据")
        return
    
    # 更新每只股票的数据
    print(f"\n开始更新数据...")
    success_count = 0
    fail_count = 0
    
    for stock_code in stock_codes:
        if update_stock_data(stock_code, start_date, end_date):
            success_count += 1
        else:
            fail_count += 1
        time.sleep(1)  # 请求间隔
    
    print("\n" + "=" * 60)
    print(f"更新完成！成功: {success_count} 只，失败: {fail_count} 只")
    print("=" * 60)

if __name__ == "__main__":
    main()

