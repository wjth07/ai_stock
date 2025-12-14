#!/usr/bin/env python
"""
获取全部A股最近一周的每日数据（带重试机制）
"""
from asyncore import file_dispatcher
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
from tqdm import tqdm
from src.data_fetcher import StockDataFetcher
from src.utils import get_recent_days_date_range

def fetch_stock_with_retry(fetcher, stock_code, start_date, end_date, max_retries=3):
    """
    获取单只股票数据，带重试
    
    Returns:
        (success: bool, df: DataFrame or None)
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = min(attempt * 2, 5)  # 最多等待5秒
                time.sleep(wait_time)
            
            df = fetcher.fetch_daily_data(stock_code, start_date, end_date)
            
            if df is not None and not df.empty:
                return True, df
            else:
                if attempt < max_retries - 1:
                    continue
                return False, None
                
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            return False, None
    
    return False, None

def merge_and_save(fetcher, stock_code, new_df):
    """合并新数据到现有文件"""
    filepath = os.path.join(fetcher.daily_dir, f"{stock_code}_daily.csv")
    print(filepath)
    
    if os.path.exists(filepath):
        # 读取现有数据
        existing_df = pd.read_csv(filepath)
        existing_df['日期'] = pd.to_datetime(existing_df['日期'])
        new_df['日期'] = pd.to_datetime(new_df['日期'])
        
        # 合并数据，去重
        combined = pd.concat([existing_df, new_df]).drop_duplicates(subset=['日期'], keep='last')
        combined = combined.sort_values('日期').reset_index(drop=True)
        combined['日期'] = combined['日期'].dt.strftime('%Y-%m-%d')
        combined.to_csv(filepath, index=False, encoding='utf-8-sig')
        return len(new_df)
    else:
        # 新文件
        fetcher.save_daily_data(stock_code, new_df)
        return len(new_df)

def main():
    """主函数"""
    print("=" * 70)
    print("获取全部A股最近一周的每日数据")
    print("=" * 70)
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 获取最近7天的日期范围
    start_date, end_date = get_recent_days_date_range(days=7)
    print(f"日期范围: {start_date} 至 {end_date}")
    
    # 初始化数据获取器
    fetcher = StockDataFetcher(use_proxy=False)
    
    # 获取全部A股股票列表
    print("\n正在获取A股股票列表...")
    try:
        stock_list = fetcher.get_stock_list()
        stock_codes = stock_list['code'].tolist()
        print(f"✓ 成功获取 {len(stock_codes)} 只A股股票")
    except Exception as e:
        print(f"✗ 获取股票列表失败: {e}")
        return
    
    # 统计信息
    success_count = 0
    fail_count = 0
    skip_count = 0
    total_new_records = 0
    
    # 检查已有数据
    daily_dir = fetcher.daily_dir
    existing_files = set()
    if os.path.exists(daily_dir):
        existing_files = {f.replace('_daily.csv', '') for f in os.listdir(daily_dir) if f.endswith('_daily.csv')}
        print(f"\n已有数据文件: {len(existing_files)} 个")
    
    print(f"\n开始批量下载数据（每只股票最多重试3次）...")
    print("=" * 70)
    
    # 批量下载
    consecutive_failures = 0
    max_consecutive_failures = 10  # 连续失败10次后暂停
    
    for stock_code in tqdm(stock_codes, desc="下载进度", unit="只"):
        try:
            # 尝试获取数据
            success, df = fetch_stock_with_retry(fetcher, stock_code, start_date, end_date, max_retries=3)
            
            if success and df is not None:
                # 保存或合并数据
                new_records = merge_and_save(fetcher, stock_code, df)
                total_new_records += new_records
                success_count += 1
                consecutive_failures = 0  # 重置连续失败计数
                
                # 每成功10只股票输出一次进度
                if success_count % 1 == 0:
                    tqdm.write(f"✓ 已成功下载 {success_count} 只股票，失败 {fail_count} 只")
            else:
                fail_count += 1
                consecutive_failures += 1
                
                # 如果连续失败太多，输出警告
                if consecutive_failures >= max_consecutive_failures:
                    tqdm.write(f"\n⚠️  连续失败 {consecutive_failures} 次，API可能暂时不可用")
                    tqdm.write(f"   当前进度: 成功 {success_count} 只，失败 {fail_count} 只")
                    tqdm.write(f"   建议: 可以按 Ctrl+C 暂停，稍后重试")
                    consecutive_failures = 0  # 重置计数，继续尝试
            
            # 请求间隔，避免API限流
            time.sleep(fetcher.request_delay)
            
        except KeyboardInterrupt:
            print("\n\n用户中断下载")
            break
        except Exception as e:
            fail_count += 1
            consecutive_failures += 1
            # 每100个错误输出一次，避免刷屏
            if fail_count % 100 == 0:
                tqdm.write(f"⚠️  已失败 {fail_count} 只股票，最后错误: {str(e)[:50]}")
    
    # 输出统计信息
    print("\n" + "=" * 70)
    print("下载完成统计")
    print("=" * 70)
    print(f"总股票数: {len(stock_codes)}")
    print(f"成功下载: {success_count} 只 ({success_count/len(stock_codes)*100:.1f}%)")
    print(f"下载失败: {fail_count} 只 ({fail_count/len(stock_codes)*100:.1f}%)")
    print(f"新增记录: {total_new_records} 条")
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if fail_count > 0:
        print("\n注意：部分股票下载失败，可能原因：")
        print("1. API服务器暂时不可用")
        print("2. 网络连接问题")
        print("3. 股票已退市或停牌")
        print("\n可以稍后重新运行此脚本，已成功下载的数据不会重复下载")
    
    print("=" * 70)

if __name__ == "__main__":
    main()

