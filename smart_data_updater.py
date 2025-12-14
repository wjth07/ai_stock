#!/usr/bin/env python
"""
智能股票数据更新器
自动更新从2024-01-01到当天的每日数据和最近一周的分钟数据
支持重复运行，智能检测和合并历史数据
"""
import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# 禁用代理
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'

try:
    import akshare as ak
except ImportError:
    print("错误: 无法导入akshare，请检查安装")
    sys.exit(1)

class SmartStockDataUpdater:
    """智能股票数据更新器"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.daily_dir = os.path.join(data_dir, "daily")
        self.minute_dir = os.path.join(data_dir, "minute")

        # 创建目录
        os.makedirs(self.daily_dir, exist_ok=True)
        os.makedirs(self.minute_dir, exist_ok=True)

        # 设置时间范围
        self.start_date = "20240101"  # 从2024年开始
        self.today = datetime.now().strftime('%Y%m%d')

    def get_stock_list(self) -> List[str]:
        """获取A股股票列表"""
        try:
            print("正在获取A股股票列表...")
            stock_list = ak.stock_info_a_code_name()
            stock_codes = stock_list['code'].tolist()
            print(f"✓ 成功获取 {len(stock_codes)} 只A股股票")
            return stock_codes
        except Exception as e:
            print(f"✗ 获取股票列表失败: {e}")
            return []

    def check_data_status(self, stock_code: str) -> Dict:
        """
        检查股票数据的当前状态

        Returns:
            {
                'daily_exists': bool,
                'daily_latest_date': str or None,
                'daily_total_records': int,
                'minute_exists': bool,
                'minute_latest_date': str or None,
                'minute_total_records': int,
                'daily_needs_update': bool,
                'minute_needs_update': bool
            }
        """
        status = {
            'daily_exists': False,
            'daily_latest_date': None,
            'daily_total_records': 0,
            'minute_exists': False,
            'minute_latest_date': None,
            'minute_total_records': 0,
            'daily_needs_update': False,
            'minute_needs_update': False
        }

        # 检查每日数据
        daily_file = os.path.join(self.daily_dir, f"{stock_code}_daily.csv")
        if os.path.exists(daily_file):
            try:
                df = pd.read_csv(daily_file)
                df['日期'] = pd.to_datetime(df['日期'])
                status['daily_exists'] = True
                status['daily_total_records'] = len(df)
                status['daily_latest_date'] = df['日期'].max().strftime('%Y%m%d')

                # 检查是否需要更新（最新日期不是昨天或今天）
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
                if status['daily_latest_date'] < yesterday:
                    status['daily_needs_update'] = True
            except Exception as e:
                print(f"读取每日数据失败 {stock_code}: {e}")

        # 检查分钟数据
        minute_file = os.path.join(self.minute_dir, f"{stock_code}_minute.csv")
        if os.path.exists(minute_file):
            try:
                df = pd.read_csv(minute_file)
                if '时间' in df.columns:
                    df['时间'] = pd.to_datetime(df['时间'])
                    status['minute_exists'] = True
                    status['minute_total_records'] = len(df)
                    status['minute_latest_date'] = df['时间'].max().strftime('%Y%m%d')

                    # 分钟数据需要每周更新
                    status['minute_needs_update'] = True  # 总是更新最近一周
                else:
                    status['minute_needs_update'] = True
            except Exception as e:
                print(f"读取分钟数据失败 {stock_code}: {e}")

        # 新股票需要初始化数据
        if not status['daily_exists']:
            status['daily_needs_update'] = True
        if not status['minute_exists']:
            status['minute_needs_update'] = True

        return status

    def calculate_update_range(self, stock_code: str) -> Tuple[str, str]:
        """
        计算需要更新的日期范围

        Returns:
            (start_date, end_date) 格式: YYYYMMDD
        """
        status = self.check_data_status(stock_code)

        if not status['daily_exists']:
            # 新股票，从2024年开始
            return self.start_date, self.today

        latest_date = status['daily_latest_date']
        if latest_date >= self.today:
            # 数据已是最新
            return None, None

        # 从最新日期的下一天开始更新到今天
        next_date = (datetime.strptime(latest_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
        return next_date, self.today

    def update_daily_data(self, stock_code: str) -> bool:
        """更新单只股票的每日数据"""
        start_date, end_date = self.calculate_update_range(stock_code)

        if start_date is None:
            print(f"✓ {stock_code} 每日数据已是最新")
            return True

        print(f"更新 {stock_code} 每日数据: {start_date} 至 {end_date}")

        try:
            # 获取新数据
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period='daily',
                start_date=start_date,
                end_date=end_date,
                adjust='qfq'  # 前复权
            )

            if df is None or df.empty:
                print(f"✗ {stock_code} 无新数据")
                return False

            # 合并历史数据
            success = self._merge_daily_data(stock_code, df)
            if success:
                print(f"✓ {stock_code} 更新成功，新增 {len(df)} 条记录")
            return success

        except Exception as e:
            print(f"✗ 更新 {stock_code} 每日数据失败: {e}")
            return False

    def update_minute_data(self, stock_code: str) -> bool:
        """更新单只股票的分钟数据（最近一周）"""
        # 计算最近一周的日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')

        print(f"更新 {stock_code} 分钟数据: {start_str} 至 {end_str}")

        try:
            # 获取分钟数据
            df = ak.stock_zh_a_hist_min_em(
                symbol=stock_code,
                start_date=start_str,
                end_date=end_str,
                period='1'  # 1分钟K线
            )

            if df is None or df.empty:
                print(f"✗ {stock_code} 无分钟数据")
                return False

            # 合并历史数据
            success = self._merge_minute_data(stock_code, df)
            if success:
                print(f"✓ {stock_code} 分钟数据更新成功，新增 {len(df)} 条记录")
            return success

        except Exception as e:
            print(f"✗ 更新 {stock_code} 分钟数据失败: {e}")
            return False

    def _merge_daily_data(self, stock_code: str, new_df: pd.DataFrame) -> bool:
        """合并每日数据"""
        filepath = os.path.join(self.daily_dir, f"{stock_code}_daily.csv")

        try:
            # 确保新数据格式正确
            new_df['日期'] = pd.to_datetime(new_df['日期'])
            new_df['股票代码'] = stock_code

            # 加载现有数据
            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath)
                existing_df['日期'] = pd.to_datetime(existing_df['日期'])

                # 合并并去重
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=['日期'], keep='last')
                combined = combined.sort_values('日期')
            else:
                combined = new_df

            # 保存
            combined['日期'] = combined['日期'].dt.strftime('%Y-%m-%d')
            combined.to_csv(filepath, index=False, encoding='utf-8-sig')
            return True

        except Exception as e:
            print(f"合并每日数据失败 {stock_code}: {e}")
            return False

    def _merge_minute_data(self, stock_code: str, new_df: pd.DataFrame) -> bool:
        """合并分钟数据"""
        filepath = os.path.join(self.minute_dir, f"{stock_code}_minute.csv")

        try:
            # 确保新数据格式正确
            if '时间' in new_df.columns:
                new_df['时间'] = pd.to_datetime(new_df['时间'])

            # 加载现有数据
            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath)
                if '时间' in existing_df.columns:
                    existing_df['时间'] = pd.to_datetime(existing_df['时间'])

                    # 合并并去重（按时间去重）
                    combined = pd.concat([existing_df, new_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=['时间'], keep='last')
                    combined = combined.sort_values('时间')
                else:
                    combined = new_df
            else:
                combined = new_df

            # 保存
            if '时间' in combined.columns:
                combined['时间'] = combined['时间'].dt.strftime('%Y-%m-%d %H:%M:%S')
            combined.to_csv(filepath, index=False, encoding='utf-8-sig')
            return True

        except Exception as e:
            print(f"合并分钟数据失败 {stock_code}: {e}")
            return False

    def update_all_stocks(self, max_stocks: Optional[int] = None) -> Dict:
        """
        更新所有股票的数据

        Args:
            max_stocks: 最大更新股票数量，用于测试
        """
        print("=" * 70)
        print("智能股票数据更新器")
        print("=" * 70)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 获取股票列表
        stock_codes = self.get_stock_list()
        if not stock_codes:
            return {'error': '无法获取股票列表'}

        # 限制数量（用于测试）
        if max_stocks:
            stock_codes = stock_codes[:max_stocks]
            print(f"⚠️ 测试模式：只更新前 {max_stocks} 只股票")

        # 统计信息
        stats = {
            'total_stocks': len(stock_codes),
            'daily_success': 0,
            'daily_fail': 0,
            'minute_success': 0,
            'minute_fail': 0,
            'skipped_daily': 0,
            'skipped_minute': 0,
            'new_records_daily': 0,
            'new_records_minute': 0
        }

        print(f"\n开始更新 {len(stock_codes)} 只股票的数据...")
        print("=" * 70)

        # 更新每只股票
        for stock_code in tqdm(stock_codes, desc="更新进度", unit="只"):
            try:
                # 检查数据状态
                status = self.check_data_status(stock_code)

                # 更新每日数据
                if status['daily_needs_update']:
                    if self.update_daily_data(stock_code):
                        stats['daily_success'] += 1
                    else:
                        stats['daily_fail'] += 1
                else:
                    stats['skipped_daily'] += 1

                # 更新分钟数据
                if status['minute_needs_update']:
                    if self.update_minute_data(stock_code):
                        stats['minute_success'] += 1
                    else:
                        stats['minute_fail'] += 1
                else:
                    stats['skipped_minute'] += 1

            except Exception as e:
                print(f"处理股票 {stock_code} 时出错: {e}")
                stats['daily_fail'] += 1
                stats['minute_fail'] += 1

            # 请求间隔
            time.sleep(10)

        # 输出统计结果
        self._print_final_report(stats)
        return stats

    def _print_final_report(self, stats: Dict):
        """打印最终报告"""
        print("\n" + "=" * 70)
        print("更新完成报告")
        print("=" * 70)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总股票数: {stats['total_stocks']}")

        print("\n每日数据:")
        print(f"  成功更新: {stats['daily_success']}")
        print(f"  更新失败: {stats['daily_fail']}")
        print(f"  已最新: {stats['skipped_daily']}")
        daily_rate = (stats['daily_success'] / (stats['daily_success'] + stats['daily_fail'])) * 100 if (stats['daily_success'] + stats['daily_fail']) > 0 else 100
        print(f"  成功率: {daily_rate:.1f}%")

        print("\n分钟数据:")
        print(f"  成功更新: {stats['minute_success']}")
        print(f"  更新失败: {stats['minute_fail']}")
        print(f"  已最新: {stats['skipped_minute']}")
        minute_rate = (stats['minute_success'] / (stats['minute_success'] + stats['minute_fail'])) * 100 if (stats['minute_success'] + stats['minute_fail']) > 0 else 100
        print(f"  成功率: {minute_rate:.1f}%")

        # 数据质量检查
        self._check_data_quality()

        print("=" * 70)

    def _check_data_quality(self):
        """检查数据质量"""
        print("\n数据质量检查:")

        # 检查每日数据文件数量
        daily_files = len([f for f in os.listdir(self.daily_dir) if f.endswith('_daily.csv')])
        minute_files = len([f for f in os.listdir(self.minute_dir) if f.endswith('_minute.csv')])

        print(f"  每日数据文件: {daily_files}")
        print(f"  分钟数据文件: {minute_files}")

        # 检查存储空间
        try:
            daily_size = sum(os.path.getsize(os.path.join(self.daily_dir, f))
                           for f in os.listdir(self.daily_dir) if f.endswith('_daily.csv'))
            minute_size = sum(os.path.getsize(os.path.join(self.minute_dir, f))
                            for f in os.listdir(self.minute_dir) if f.endswith('_minute.csv'))

            print(f"  每日数据大小: {daily_size / 1024 / 1024:.1f} MB")
            print(f"  分钟数据大小: {minute_size / 1024 / 1024:.1f} MB")
        except:
            pass

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='智能股票数据更新器')
    parser.add_argument('--max-stocks', type=int, help='最大更新股票数量（测试用）')
    parser.add_argument('--test', action='store_true', help='测试模式（只更新前10只股票）')

    args = parser.parse_args()

    # 创建更新器
    updater = SmartStockDataUpdater()

    # 设置测试模式
    max_stocks = args.max_stocks
    if args.test and not max_stocks:
        max_stocks = 10

    # 执行更新
    updater.update_all_stocks(max_stocks=max_stocks)

if __name__ == "__main__":
    main()
