#!/usr/bin/env python
"""
按股票存储的每日数据更新策略
演示如何安全地更新股票数据并融合历史数据
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import json

class StockDailyUpdater:
    """股票每日数据更新器"""

    def __init__(self, data_dir: str = "data/daily"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load_existing_data(self, stock_code: str) -> Optional[pd.DataFrame]:
        """加载现有股票数据"""
        filepath = os.path.join(self.data_dir, f"{stock_code}_daily.csv")

        if not os.path.exists(filepath):
            return None

        try:
            df = pd.read_csv(filepath)
            df['日期'] = pd.to_datetime(df['日期'])
            return df
        except Exception as e:
            print(f"加载股票 {stock_code} 数据失败: {e}")
            return None

    def merge_daily_data(self, stock_code: str, new_data: pd.DataFrame) -> bool:
        """
        合并每日数据到现有文件

        Args:
            stock_code: 股票代码
            new_data: 新获取的数据（通常是1天或几天的数据）

        Returns:
            是否成功
        """
        filepath = os.path.join(self.data_dir, f"{stock_code}_daily.csv")

        # 确保新数据格式正确
        if '日期' not in new_data.columns:
            print(f"错误: 新数据缺少日期列")
            return False

        new_data['日期'] = pd.to_datetime(new_data['日期'])
        new_data = new_data.sort_values('日期')

        # 加载现有数据
        existing_data = self.load_existing_data(stock_code)

        if existing_data is None:
            # 新股票，直接保存
            new_data['日期'] = new_data['日期'].dt.strftime('%Y-%m-%d')
            new_data.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"✓ 新建股票 {stock_code} 数据文件，{len(new_data)} 条记录")
            return True

        # 检查是否有重复数据
        existing_dates = set(existing_data['日期'].dt.strftime('%Y-%m-%d'))
        new_dates = set(new_data['日期'].dt.strftime('%Y-%m-%d'))

        overlapping_dates = existing_dates & new_dates

        if overlapping_dates:
            print(f"⚠️ 股票 {stock_code} 有 {len(overlapping_dates)} 天重复数据，将使用新数据覆盖")

        # 合并数据
        # 1. 保留现有数据（除了重复日期）
        old_data_filtered = existing_data[
            ~existing_data['日期'].dt.strftime('%Y-%m-%d').isin(overlapping_dates)
        ].copy()

        # 2. 添加新数据
        combined_data = pd.concat([old_data_filtered, new_data], ignore_index=True)

        # 3. 排序并去重（以防万一）
        combined_data = combined_data.drop_duplicates(subset=['日期'], keep='last')
        combined_data = combined_data.sort_values('日期')

        # 4. 保存
        combined_data['日期'] = combined_data['日期'].dt.strftime('%Y-%m-%d')
        combined_data.to_csv(filepath, index=False, encoding='utf-8-sig')

        added_records = len(new_data)
        total_records = len(combined_data)

        print(f"✓ 更新股票 {stock_code}: 新增 {added_records} 条，累计 {total_records} 条")
        return True

    def update_single_stock_daily(self, stock_code: str, target_date: Optional[str] = None) -> bool:
        """
        更新单只股票的每日数据

        Args:
            stock_code: 股票代码
            target_date: 目标日期，默认为昨天

        Returns:
            是否成功
        """
        if target_date is None:
            # 默认更新昨天的数据
            target_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

        print(f"正在更新股票 {stock_code} 在 {target_date} 的数据...")

        try:
            # 这里应该调用实际的数据获取函数
            # 由于网络限制，我们用模拟数据演示
            new_data = self._simulate_fetch_data(stock_code, target_date)

            if new_data is not None and not new_data.empty:
                return self.merge_daily_data(stock_code, new_data)
            else:
                print(f"✗ 股票 {stock_code} 在 {target_date} 无数据")
                return False

        except Exception as e:
            print(f"✗ 更新股票 {stock_code} 失败: {e}")
            return False

    def update_multiple_stocks_daily(self, stock_codes: List[str], target_date: Optional[str] = None) -> dict:
        """
        批量更新多只股票的每日数据

        Args:
            stock_codes: 股票代码列表
            target_date: 目标日期

        Returns:
            更新结果统计
        """
        if target_date is None:
            target_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

        print(f"开始批量更新 {len(stock_codes)} 只股票在 {target_date} 的数据...")

        results = {
            'total': len(stock_codes),
            'success': 0,
            'failed': 0,
            'skipped': 0
        }

        for stock_code in stock_codes:
            try:
                if self.update_single_stock_daily(stock_code, target_date):
                    results['success'] += 1
                else:
                    results['failed'] += 1
            except Exception as e:
                print(f"处理股票 {stock_code} 时出错: {e}")
                results['failed'] += 1

        print(f"\n批量更新完成:")
        print(f"  总计: {results['total']}")
        print(f"  成功: {results['success']}")
        print(f"  失败: {results['failed']}")
        print(f"  成功率: {results['success']/results['total']*100:.1f}%")

        return results

    def _simulate_fetch_data(self, stock_code: str, target_date: str) -> Optional[pd.DataFrame]:
        """模拟数据获取（实际使用时替换为真实API调用）"""
        # 这里是模拟数据，实际使用时应该调用真实的API
        try:
            # 模拟获取到的数据
            data = {
                '日期': [target_date],
                '股票代码': [stock_code],
                '开盘': [10.5],
                '收盘': [10.8],
                '最高': [11.0],
                '最低': [10.3],
                '成交量': [1000000],
                '成交额': [10800000],
                '振幅': [6.67],
                '涨跌幅': [2.86],
                '涨跌额': [0.3],
                '换手率': [0.45]
            }

            df = pd.DataFrame(data)
            df['日期'] = pd.to_datetime(df['日期'])
            return df

        except Exception as e:
            print(f"模拟获取数据失败: {e}")
            return None

    def validate_data_integrity(self, stock_code: str) -> dict:
        """验证数据完整性"""
        filepath = os.path.join(self.data_dir, f"{stock_code}_daily.csv")

        if not os.path.exists(filepath):
            return {'exists': False}

        try:
            df = pd.read_csv(filepath)
            df['日期'] = pd.to_datetime(df['日期'])

            # 基本统计
            stats = {
                'exists': True,
                'total_records': len(df),
                'date_range': {
                    'start': df['日期'].min().strftime('%Y-%m-%d'),
                    'end': df['日期'].max().strftime('%Y-%m-%d')
                },
                'missing_values': df.isnull().sum().sum(),
                'duplicate_dates': df['日期'].duplicated().sum()
            }

            # 检查数据连续性
            df_sorted = df.sort_values('日期')
            date_diff = df_sorted['日期'].diff().dt.days
            gaps = (date_diff > 1).sum() if len(date_diff) > 1 else 0
            stats['data_gaps'] = gaps

            return stats

        except Exception as e:
            return {'exists': True, 'error': str(e)}

    def cleanup_old_data(self, days_to_keep: int = 365*3) -> int:
        """清理旧数据，只保留最近N天的记录"""
        print(f"开始清理超过 {days_to_keep} 天的旧数据...")

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_count = 0

        for filename in os.listdir(self.data_dir):
            if filename.endswith('_daily.csv'):
                filepath = os.path.join(self.data_dir, filename)

                try:
                    df = pd.read_csv(filepath)
                    df['日期'] = pd.to_datetime(df['日期'])

                    # 保留最近的数据
                    recent_data = df[df['日期'] >= cutoff_date]

                    if len(recent_data) < len(df):
                        # 有数据被清理
                        recent_data['日期'] = recent_data['日期'].dt.strftime('%Y-%m-%d')
                        recent_data.to_csv(filepath, index=False, encoding='utf-8-sig')

                        cleaned_records = len(df) - len(recent_data)
                        print(f"✓ 清理 {filename}: 删除了 {cleaned_records} 条旧记录")
                        cleaned_count += cleaned_records

                except Exception as e:
                    print(f"清理 {filename} 时出错: {e}")

        print(f"数据清理完成，共删除 {cleaned_count} 条记录")
        return cleaned_count

def main():
    """演示每日更新流程"""
    updater = StockDailyUpdater()

    print("=" * 60)
    print("股票每日数据更新演示")
    print("=" * 60)

    # 演示更新几只股票
    test_stocks = ['000001', '000002', '000004']

    print("\n[1] 单个股票更新演示")
    for stock_code in test_stocks:
        updater.update_single_stock_daily(stock_code)

    print("\n[2] 批量更新演示")
    results = updater.update_multiple_stocks_daily(test_stocks)

    print("\n[3] 数据完整性检查")
    for stock_code in test_stocks:
        stats = updater.validate_data_integrity(stock_code)
        if stats.get('exists'):
            print(f"{stock_code}: {stats['total_records']} 条记录, "
                  f"{stats['date_range']['start']} 至 {stats['date_range']['end']}")

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)

if __name__ == "__main__":
    main()

