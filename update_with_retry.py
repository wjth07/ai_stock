#!/usr/bin/env python
"""
补充数据（带重试和更长超时）
"""
import os
import sys
import time
from datetime import datetime, timedelta

# 禁用代理（VPN可能导致连接问题）
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'

import akshare as ak
import pandas as pd
from src.data_fetcher import StockDataFetcher

def update_stock_with_retry(stock_code, start_date, end_date, max_retries=5):
    """更新股票数据，带多次重试和更长等待"""
    fetcher = StockDataFetcher(use_proxy=False)
    
    for attempt in range(max_retries):
        try:
            wait_before = min(attempt * 3, 10)  # 最多等待10秒
            if wait_before > 0:
                print(f"  等待 {wait_before} 秒后重试...")
                time.sleep(wait_before)
            
            print(f"  尝试 {attempt + 1}/{max_retries}: {stock_code}...")
            
            # 直接使用akshare，设置更长超时
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period='daily',
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust=''
            )
            
            if df is not None and not df.empty:
                # 读取现有数据并合并
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
                if attempt < max_retries - 1:
                    continue
                return False
                
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                print(f"  ✗ 失败: {error_msg[:80]}... 将重试")
            else:
                print(f"  ✗ {stock_code}: 最终失败 - {error_msg[:80]}")
    
    return False

def main():
    """主函数"""
    print("=" * 60)
    print("补充最近一周的每日数据（带重试）")
    print("=" * 60)
    print("\n注意：如果VPN导致连接问题，建议：")
    print("1. 检查Clash Verge是否正常运行")
    print("2. 尝试在Clash中设置直连规则（绕过代理）")
    print("3. 或者暂时关闭VPN后重试")
    print("=" * 60)
    
    # 获取最近7天的日期范围
    from src.utils import get_recent_days_date_range
    start_date, end_date = get_recent_days_date_range(days=7)
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
    print(f"\n开始更新数据（最多重试5次）...")
    success_count = 0
    fail_count = 0
    
    for stock_code in stock_codes:
        if update_stock_with_retry(stock_code, start_date, end_date):
            success_count += 1
        else:
            fail_count += 1
        time.sleep(2)  # 请求间隔
    
    print("\n" + "=" * 60)
    print(f"更新完成！成功: {success_count} 只，失败: {fail_count} 只")
    if fail_count > 0:
        print("\n如果全部失败，可能的原因：")
        print("1. VPN代理配置问题 - 尝试在Clash中设置直连规则")
        print("2. 网络连接问题 - 检查网络连接")
        print("3. API服务器暂时不可用 - 稍后重试")
    print("=" * 60)

if __name__ == "__main__":
    main()

