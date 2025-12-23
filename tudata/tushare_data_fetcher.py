#!/usr/bin/env python
"""
Tushare数据获取器
使用Tushare接口获取2024年以来的A股日K线数据
"""
import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import inspect

# 添加上级目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import tushare as ts
except ImportError:
    print("错误: 无法导入tushare，请先安装: pip install tushare")
    sys.exit(1)

try:
    import akshare as ak
except ImportError:
    print("警告: 无法导入akshare，将无法使用备用方案")
    ak = None


def debug_print(message: str, show_line_number: bool = False):
    """
    调试打印函数，可选择显示当前行号

    Args:
        message: 要打印的消息
        show_line_number: 是否显示行号
    """
    if show_line_number:
        frame = inspect.currentframe().f_back
        filename = os.path.basename(frame.f_code.co_filename)
        line_number = frame.f_lineno
        print(f"[{filename}:{line_number}] {message}")
    else:
        print(message)


class TokenWorker:
    """用于并行处理的Token工作器（轻量化）"""

    def __init__(self, token: str, data_dir: str, adjust: str, start_date: str, today: str, verbose: bool = False):
        """
        初始化Token工作器

        Args:
            token: Tushare API token
            data_dir: 数据目录
            adjust: 复权方式
            start_date: 开始日期
            today: 今天日期
            verbose: 是否显示详细处理信息（并行模式下默认False）
        """
        self.token = token
        self.data_dir = data_dir
        self.daily_dir = os.path.join(data_dir, "daily")
        self.adjust = adjust
        self.start_date = start_date
        self.today = today
        self.verbose = verbose

        # 初始化Tushare API
        ts.set_token(token)
        self.pro = ts.pro_api()

    def check_data_status(self, stock_code: str, end_date: str = None) -> Dict:
        """检查股票数据的当前状态

        Args:
            stock_code: 股票代码
            end_date: 用户指定的结束日期，如果提供，将与现有数据比较
        """
        status = {
            'exists': False,
            'latest_date': None,
            'total_records': 0,
            'needs_update': False
        }

        filename = f"{stock_code}_daily.csv"
        filepath = os.path.join(self.daily_dir, filename)

        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                status['exists'] = True
                status['total_records'] = len(df)
                status['latest_date'] = df['trade_date'].max().strftime('%Y%m%d')

                # 检查是否需要更新
                if end_date:
                    # 如果用户指定了结束日期，检查现有数据是否已经覆盖到该日期
                    if status['latest_date'] >= end_date:
                        # 现有数据已经包含或超过用户指定的结束日期，无需更新
                        status['needs_update'] = False
                        if self.verbose:
                            debug_print(f"✓ {stock_code} 数据已覆盖至 {status['latest_date']}，无需更新到 {end_date}", show_line_number=True)
                    else:
                        # 现有数据没有覆盖到用户指定的结束日期，需要更新
                        status['needs_update'] = True
                        if self.verbose:
                            debug_print(f"📅 {stock_code} 数据只到 {status['latest_date']}，需要更新到 {end_date}", show_line_number=True)
                else:
                    # 默认逻辑：检查是否需要更新到昨天
                    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
                    if status['latest_date'] < yesterday:
                        status['needs_update'] = True
            except Exception as e:
                pass

        if not status['exists']:
            status['needs_update'] = True

        return status

    def calculate_update_range(self, stock_code: str) -> tuple[str, str]:
        """计算需要更新的日期范围"""
        status = self.check_data_status(stock_code)

        if not status['exists']:
            return self.start_date, self.today

        latest_date = status['latest_date']
        if latest_date >= self.today:
            return None, None

        next_date = (datetime.strptime(latest_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
        return next_date, self.today

    def fetch_daily_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取单只股票的日K线数据"""
        try:
            df = self.pro.daily(
                ts_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                adj=self.adjust,
                fields='ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount'
            )

            if df is None or df.empty:
                return df

            df = df.sort_values('trade_date')

            if 'pct_chg' not in df.columns or df['pct_chg'].isnull().all():
                df['pct_chg'] = ((df['close'] - df['pre_close']) / df['pre_close'] * 100).round(2)

            df['change'] = (df['close'] - df['pre_close']).round(2)
            df['vol'] = (df['vol'] / 100).round(0)
            df['amount'] = (df['amount'] / 10000).round(2)

            return df

        except Exception as e:
            return None

    def merge_daily_data(self, stock_code: str, new_df: pd.DataFrame) -> bool:
        """合并每日数据"""
        filename = f"{stock_code}_daily.csv"
        filepath = os.path.join(self.daily_dir, filename)

        try:
            new_df['trade_date'] = pd.to_datetime(new_df['trade_date'], format='%Y%m%d')

            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath)
                existing_df['trade_date'] = pd.to_datetime(existing_df['trade_date'], format='%Y%m%d')

                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=['trade_date'], keep='last')
                combined = combined.sort_values('trade_date')
            else:
                combined = new_df

            combined['trade_date'] = combined['trade_date'].dt.strftime('%Y%m%d')
            combined.to_csv(filepath, index=False, encoding='utf-8-sig')
            return True

        except Exception as e:
            return False

    def process_single_stock(self, stock_code: str, end_date: str = None) -> Tuple[str, bool, int]:
        """处理单个股票的完整更新流程

        Args:
            stock_code: 股票代码
            end_date: 用户指定的结束日期
        """
        try:
            status = self.check_data_status(stock_code, end_date)

            if status['needs_update']:
                if end_date:
                    # 用户指定了结束日期，使用用户的结束日期
                    start_date, calculated_end_date = self.calculate_update_range(stock_code)
                    # 如果计算出的结束日期超过了用户指定的结束日期，使用用户指定的
                    actual_end_date = min(calculated_end_date, end_date) if calculated_end_date > end_date else calculated_end_date
                else:
                    # 默认逻辑
                    start_date, actual_end_date = self.calculate_update_range(stock_code)

                if start_date is None:
                    return stock_code, True, 0
                df = self.fetch_daily_data(stock_code, start_date, actual_end_date)
                if df is None:
                    if self.verbose:
                        debug_print(f"✗ {stock_code} 数据获取失败，跳过处理", show_line_number=True)
                    return stock_code, False, 0
                elif df.empty:
                    if self.verbose:
                        debug_print(f"✗ {stock_code} 没有新数据，跳过处理", show_line_number=True)
                    return stock_code, True, 0

                success = self.merge_daily_data(stock_code, df)
                records_added = len(df) if success else 0
                if self.verbose:
                    if success:
                        debug_print(f"✓ {stock_code} 更新成功，新增 {records_added} 条记录", show_line_number=True)
                    else:
                        debug_print(f"✗ {stock_code} 数据合并失败", show_line_number=True)
                return stock_code, success, records_added
            else:
                return stock_code, True, 0

        except Exception as e:
            if self.verbose:
                import traceback
                tb = traceback.extract_tb(e.__traceback__)
                if tb:
                    filename, line_number, func_name, text = tb[-1]
                    print(f"✗ 处理股票 {stock_code} 时出错 [{filename}:{line_number}]: {e}")
                else:
                    print(f"✗ 处理股票 {stock_code} 时出错: {e}")
            return stock_code, False, 0

class TushareDataFetcher:
    """Tushare数据获取器"""

    def __init__(self, token: str, data_dir: str = ".", adjust: str = "qfq", tokens: List[str] = None, verbose: bool = True):
        """
        初始化Tushare数据获取器

        Args:
            token: Tushare API token（主要token）
            data_dir: 数据存储目录
            adjust: 复权方式，'qfq'-前复权，'hfq'-后复权，''-不复权
            tokens: 多个token列表，用于并行处理避免限流
            verbose: 是否显示详细的处理信息
        """
        self.token = token
        self.tokens = tokens or [token]  # 如果没有提供多个token，使用单个token
        self.data_dir = data_dir
        self.daily_dir = os.path.join(data_dir, "daily")
        self.adjust = adjust  # 复权方式
        self.verbose = verbose  # 是否显示详细处理信息

        # 创建目录
        os.makedirs(self.daily_dir, exist_ok=True)

        # 初始化多个Tushare API实例
        self.pro_instances = []
        for i, tk in enumerate(self.tokens):
            ts.set_token(tk)
            pro = ts.pro_api()
            self.pro_instances.append(pro)
            if verbose:
                print(f"✓ 初始化Token {i+1}/{len(self.tokens)}: {tk[:10]}...")

        # 使用第一个token作为默认实例
        self.pro = self.pro_instances[0]

        # 设置时间范围
        self.start_date = "20240101"  # 从2024年开始
        self.today = datetime.now().strftime('%Y%m%d')

        adjust_desc = {"qfq": "前复权", "hfq": "后复权", "": "不复权"}.get(adjust, "未知")
        if verbose:
            debug_print(f"✓ Tushare API初始化完成，{len(self.tokens)}个token，数据将保存到: {self.data_dir}，复权方式: {adjust_desc}", show_line_number=True)

    def get_stock_list(self, force_refresh: bool = False) -> List[str]:
        """获取A股股票列表（支持本地缓存）

        Args:
            force_refresh: 是否强制刷新缓存，忽略本地文件
        """
        # 定义缓存文件路径
        cache_file = os.path.join(self.data_dir, "stock_list_cache.txt")

        # 检查缓存文件是否存在且不强制刷新
        if not force_refresh and os.path.exists(cache_file):
            try:
                print("正在从本地缓存读取股票列表...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    stock_codes = [line.strip() for line in f if line.strip()]

                if stock_codes:
                    debug_print(f"✓ 从缓存读取成功，共 {len(stock_codes)} 只A股股票", show_line_number=True)
                    return stock_codes
                else:
                    debug_print("⚠️ 缓存文件为空，将重新获取", show_line_number=True)

            except Exception as e:
                print(f"⚠️ 读取缓存文件失败: {e}，将重新获取")

        # 从API获取股票列表
        try:
            print("正在获取A股股票列表（Tushare）...")

            # 获取沪深A股基本信息
            df = self.pro.stock_basic(
                exchange='',
                list_status='L',  # L-上市，D-退市，P-暂停上市
                fields='ts_code,symbol,name,area,industry,list_date'
            )

            stock_codes = df['ts_code'].tolist()
            debug_print(f"✓ Tushare获取成功，共 {len(stock_codes)} 只A股股票", show_line_number=True)

            # 保存到缓存文件
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    for code in stock_codes:
                        f.write(f"{code}\n")
                debug_print(f"✓ 股票列表已保存到缓存文件: {cache_file}", show_line_number=True)
            except Exception as e:
                debug_print(f"⚠️ 保存缓存文件失败: {e}（不影响程序运行）", show_line_number=True)

            return stock_codes

        except Exception as e:
            print(f"✗ Tushare获取股票列表失败: {e}")

            # 尝试使用akshare作为备用方案
            if ak is not None:
                try:
                    print("正在尝试使用akshare获取股票列表...")
                    stock_list = ak.stock_info_a_code_name()
                    stock_codes = stock_list['code'].tolist()

                    # 转换为tushare格式（添加.SH/.SZ/.BJ后缀）
                    converted_codes = []
                    for code in stock_codes:
                        code_int = int(code)
                        if (code.startswith('0') and 1 <= code_int <= 4999) or \
                           (code.startswith('3') and 300000 <= code_int <= 399999):
                            converted_codes.append(f"{code}.SZ")  # 深圳交易所
                        elif (code.startswith('6') and 600000 <= code_int <= 699999):
                            converted_codes.append(f"{code}.SH")  # 上海交易所
                        elif (code.startswith('8') and 830000 <= code_int <= 879999) or \
                             (code.startswith('4') and 430000 <= code_int <= 479999) or \
                             (code.startswith('9') and 920000 <= code_int <= 999999):
                            converted_codes.append(f"{code}.BJ")  # 北京交易所
                        elif code.startswith('9') and 900000 <= code_int <= 919999:
                            converted_codes.append(f"{code}.SH")  # 上海B股
                        else:
                            converted_codes.append(f"{code}.SH")  # 默认SH

                    print(f"✓ akshare备用获取成功，共 {len(converted_codes)} 只A股股票")

                    # 保存到缓存文件
                    try:
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            for code in converted_codes:
                                f.write(f"{code}\n")
                        print(f"✓ 股票列表已保存到缓存文件: {cache_file}")
                    except Exception as e:
                        print(f"⚠️ 保存缓存文件失败: {e}（不影响程序运行）")

                    return converted_codes

                except Exception as e2:
                    print(f"✗ akshare备用方案也失败: {e2}")
            else:
                print("✗ akshare不可用，无备用方案")

            print("✗ 所有获取股票列表的方法都失败，程序终止")
            sys.exit(1)

    def check_data_status(self, stock_code: str, end_date: str = None) -> Dict:
        """
        检查股票数据的当前状态

        Args:
            stock_code: 股票代码，格式为XXXXXX.SH或XXXXXX.SZ
            end_date: 用户指定的结束日期，如果提供，将与现有数据比较

        Returns:
            状态信息字典
        """
        status = {
            'exists': False,
            'latest_date': None,
            'total_records': 0,
            'needs_update': False
        }

        # 检查数据文件
        filename = f"{stock_code}_daily.csv"
        filepath = os.path.join(self.daily_dir, filename)

        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                status['exists'] = True
                status['total_records'] = len(df)
                status['latest_date'] = df['trade_date'].max().strftime('%Y%m%d')

                # 检查是否需要更新
                if end_date:
                    # 如果用户指定了结束日期，检查现有数据是否已经覆盖到该日期
                    if status['latest_date'] >= end_date:
                        # 现有数据已经包含或超过用户指定的结束日期，无需更新
                        status['needs_update'] = False
                        if self.verbose:
                            debug_print(f"✓ {stock_code} 数据已覆盖至 {status['latest_date']}，无需更新到 {end_date}", show_line_number=True)
                    else:
                        # 现有数据没有覆盖到用户指定的结束日期，需要更新
                        status['needs_update'] = True
                        if self.verbose:
                            debug_print(f"📅 {stock_code} 数据只到 {status['latest_date']}，需要更新到 {end_date}", show_line_number=True)
                else:
                    # 默认逻辑：检查是否需要更新到昨天
                    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
                    if status['latest_date'] < yesterday:
                        status['needs_update'] = True
            except Exception as e:
                print(f"读取数据失败 {stock_code}: {e}")

        # 新股票需要初始化数据
        if not status['exists']:
            status['needs_update'] = True

        return status

    def calculate_update_range(self, stock_code: str) -> tuple[str, str]:
        """
        计算需要更新的日期范围

        Args:
            stock_code: 股票代码

        Returns:
            (start_date, end_date) 格式: YYYYMMDD
        """
        status = self.check_data_status(stock_code)

        if not status['exists']:
            # 新股票，从2024年开始
            return self.start_date, self.today

        latest_date = status['latest_date']
        if latest_date >= self.today:
            # 数据已是最新
            return None, None

        # 从最新日期的下一天开始更新到今天
        next_date = (datetime.strptime(latest_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
        return next_date, self.today

    def fetch_daily_data(self, stock_code: str, start_date: str, end_date: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        获取单只股票的日K线数据（带重试机制）

        Args:
            stock_code: 股票代码，格式为XXXXXX.SH或XXXXXX.SZ
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            max_retries: 最大重试次数

        Returns:
            日K线数据DataFrame
        """
        import time

        for attempt in range(max_retries):
            try:
                # 调用Tushare日线行情接口
                df = self.pro.daily(
                    ts_code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    adj=self.adjust,  # 复权方式
                    fields='ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount'
                )

                if df is None or df.empty:
                    if self.verbose and attempt == max_retries - 1:
                        print(f"⚠️ {stock_code} API返回空数据 ({start_date} 至 {end_date})")
                    return None

                # 数据预处理
                df = df.sort_values('trade_date')  # 按日期排序

                # 计算涨跌幅（如果API没有提供）
                if 'pct_chg' not in df.columns or df['pct_chg'].isnull().all():
                    df['pct_chg'] = ((df['close'] - df['pre_close']) / df['pre_close'] * 100).round(2)

                # 计算涨跌额
                df['change'] = (df['close'] - df['pre_close']).round(2)

                # 成交量转换为手（API返回的是股）
                df['vol'] = (df['vol'] / 100).round(0)  # 转换为手

                # 成交额转换为万元
                df['amount'] = (df['amount'] / 10000).round(2)  # 转换为万元

                return df

            except Exception as e:
                if attempt < max_retries - 1:
                    # 不是最后一次尝试，等待后重试
                    wait_time = (attempt + 1) * 2  # 指数退避: 2s, 4s, 6s...
                    if self.verbose:
                        print(f"⚠️ {stock_code} 第{attempt+1}次尝试失败，{wait_time}秒后重试: {str(e)[:50]}...")
                    time.sleep(wait_time)
                else:
                    # 最后一次尝试失败
                    if self.verbose:
                        print(f"✗ {stock_code} 获取失败，已重试{max_retries}次: {str(e)[:50]}...")
                    return None

    def update_daily_data(self, stock_code: str) -> bool:
        """更新单只股票的每日数据"""
        start_date, end_date = self.calculate_update_range(stock_code)

        if start_date is None:
            if self.verbose:
                debug_print(f"✓ {stock_code} 每日数据已是最新", show_line_number=True)
            return True

        if self.verbose:
            debug_print(f"更新 {stock_code} 每日数据: {start_date} 至 {end_date}", show_line_number=True)

        try:
            # 获取新数据
            df = self.fetch_daily_data(stock_code, start_date, end_date)

            if df is None or df.empty:
                if self.verbose:
                    debug_print(f"✗ {stock_code} 无新数据", show_line_number=True)
                return False

            # 合并历史数据
            success = self._merge_daily_data(stock_code, df)
            if success:
                debug_print(f"✓ {stock_code} 更新成功，新增 {len(df)} 条记录", show_line_number=self.verbose)
            return success

        except Exception as e:
            if self.verbose:
                print(f"✗ 更新 {stock_code} 每日数据失败: {e}")
            return False

    def process_single_stock(self, stock_code: str) -> Tuple[str, bool, int]:
        """
        处理单个股票的完整更新流程

        Returns:
            (stock_code, success, records_added)
        """
        try:
            # 检查数据状态
            status = self.check_data_status(stock_code)

            if status['needs_update']:
                success = self.update_daily_data(stock_code)
                records_added = len(self._get_new_records_count(stock_code)) if success else 0
                return stock_code, success, records_added
            else:
                return stock_code, True, 0  # 跳过但算成功

        except Exception as e:
            if self.verbose:
                import traceback
                tb = traceback.extract_tb(e.__traceback__)
                if tb:
                    filename, line_number, func_name, text = tb[-1]
                    print(f"✗ 处理股票 {stock_code} 时出错 [{filename}:{line_number}]: {e}")
                else:
                    print(f"✗ 处理股票 {stock_code} 时出错: {e}")
            return stock_code, False, 0

    def _get_new_records_count(self, stock_code: str) -> pd.DataFrame:
        """获取最新获取的数据量（用于统计）"""
        try:
            # 这里可以记录最新获取的数据条数
            # 由于合并逻辑复杂，这里简化返回空DataFrame
            return pd.DataFrame()
        except:
            return pd.DataFrame()

    def update_stocks_parallel(self, stock_codes: List[str], max_workers: int = None, force_verbose: bool = False, end_date: str = None) -> Dict:
        """
        并发更新股票数据（使用多个token避免API限流）

        Args:
            stock_codes: 股票代码列表
            max_workers: 最大并发数，默认使用CPU核心数的一半

        Returns:
            更新统计结果
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if max_workers is None:
            # 降低并发数，避免API限流。Tushare免费账户限制严格
            max_workers = min(2, len(self.tokens))  # 最多使用2个线程，或者token数量

        print(f"🎯 使用 {max_workers} 个线程，{len(self.tokens)} 个token并发处理")

        stats = {
            'total_stocks': len(stock_codes),
            'success': 0,
            'fail': 0,
            'skipped': 0,
            'new_records': 0
        }

        # 为每个token创建fetcher实例（简化版，只包含必要方法）
        # 如果强制要求verbose，即使在并行模式下也显示详细信息
        worker_verbose = force_verbose if force_verbose else False
        token_fetchers = []
        for i, token in enumerate(self.tokens):
            fetcher = TokenWorker(token, self.data_dir, self.adjust, self.start_date, self.today, verbose=worker_verbose)
            token_fetchers.append(fetcher)

        # 使用线程池进行并发处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 预先检查哪些股票需要更新，避免创建不必要的线程
            stocks_to_update = []
            skipped_stocks = []

            print("🔍 正在检查数据更新状态...")
            for stock_code in stock_codes:
                # 创建一个临时worker来检查状态（使用第一个token）
                temp_worker = token_fetchers[0]
                status = temp_worker.check_data_status(stock_code, end_date)
                if status['needs_update']:
                    stocks_to_update.append(stock_code)
                else:
                    skipped_stocks.append(stock_code)
                    stats['skipped'] += 1
                    if worker_verbose:
                        if end_date:
                            debug_print(f"⏭️ {stock_code} 数据已覆盖至 {status['latest_date']}，无需更新到 {end_date}", show_line_number=True)
                        else:
                            debug_print(f"⏭️ {stock_code} 数据已是最新，跳过更新", show_line_number=True)

            print(f"📊 检查完成: {len(stocks_to_update)} 只股票需要更新，{len(skipped_stocks)} 只已跳过")

            # 只为需要更新的股票提交任务
            future_to_stock = {}
            for i, stock_code in enumerate(stocks_to_update):
                # 轮流使用不同的token
                token_index = i % len(token_fetchers)
                fetcher = token_fetchers[token_index]

                future = executor.submit(fetcher.process_single_stock, stock_code, end_date)
                future_to_stock[future] = stock_code

                # 在提交任务之间添加小延迟，避免API限流
                if (i + 1) % max_workers == 0:
                    time.sleep(1)  # 每批任务后暂停1秒

            # 使用tqdm显示进度（只显示需要更新的股票）
            total_to_process = len(stocks_to_update)
            if total_to_process > 0:
                with tqdm(total=total_to_process, desc="更新进度", unit="只") as pbar:
                    for future in as_completed(future_to_stock):
                        stock_code = future_to_stock[future]
                        try:
                            result_stock, success, records = future.result()
                            if success:
                                if records > 0:
                                    stats['success'] += 1
                                else:
                                    stats['skipped'] += 1
                            else:
                                stats['fail'] += 1
                            stats['new_records'] += records

                        except Exception as e:
                            import traceback
                            tb = traceback.extract_tb(e.__traceback__)
                            if tb:
                                filename, line_number, func_name, text = tb[-1]
                                print(f"处理股票 {stock_code} 结果获取失败 [{filename}:{line_number}]: {e}")
                            else:
                                print(f"处理股票 {stock_code} 结果获取失败: {e}")
                            stats['fail'] += 1

                        pbar.update(1)

        return stats

    def _merge_daily_data(self, stock_code: str, new_df: pd.DataFrame) -> bool:
        """合并每日数据"""
        filename = f"{stock_code}_daily.csv"
        filepath = os.path.join(self.daily_dir, filename)

        try:
            # 确保新数据格式正确
            new_df['trade_date'] = pd.to_datetime(new_df['trade_date'], format='%Y%m%d')

            # 加载现有数据
            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath)
                existing_df['trade_date'] = pd.to_datetime(existing_df['trade_date'], format='%Y%m%d')

                # 合并并去重
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=['trade_date'], keep='last')
                combined = combined.sort_values('trade_date')
            else:
                combined = new_df

            # 保存文件
            combined['trade_date'] = combined['trade_date'].dt.strftime('%Y%m%d')
            combined.to_csv(filepath, index=False, encoding='utf-8-sig')
            return True

        except Exception as e:
            if self.verbose:
                import traceback
                tb = traceback.extract_tb(e.__traceback__)
                if tb:
                    filename, line_number, func_name, text = tb[-1]
                    print(f"合并每日数据失败 {stock_code} [{filename}:{line_number}]: {e}")
                else:
                    print(f"合并每日数据失败 {stock_code}: {e}")
            return False

    def update_all_stocks(self, max_stocks: Optional[int] = None, test_mode: bool = False,
                         parallel: bool = True, max_workers: int = None, force_verbose: bool = False,
                         force_refresh: bool = False, end_date: str = None) -> Dict:
        """
        更新所有股票的数据

        Args:
            max_stocks: 最大更新股票数量，用于测试
            test_mode: 测试模式，使用更小的API请求间隔
            parallel: 是否使用并行处理（默认True）
            max_workers: 最大进程数（并行模式下，默认CPU核心数一半）
            force_verbose: 强制显示详细信息
            force_refresh: 强制刷新股票列表缓存
            end_date: 指定结束日期，如果现有数据已覆盖此日期则跳过更新
        """
        print("=" * 70)
        print("Tushare股票数据更新器")
        print("=" * 70)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 获取股票列表
        stock_codes = self.get_stock_list(force_refresh=force_refresh)
        if not stock_codes:
            return {'error': '无法获取股票列表'}

        # 限制数量（用于测试）
        if max_stocks:
            stock_codes = stock_codes[:max_stocks]
            print(f"⚠️ 测试模式：只更新前 {max_stocks} 只股票")

        # 统计信息
        stats = {
            'total_stocks': len(stock_codes),
            'success': 0,
            'fail': 0,
            'skipped': 0,
            'new_records': 0
        }

        print(f"\n开始更新 {len(stock_codes)} 只股票的数据...")

        if parallel and len(stock_codes) > 1:
            # 并行处理
            print("🚀 使用并行处理模式")
            stats = self.update_stocks_parallel(stock_codes, max_workers=max_workers, force_verbose=force_verbose, end_date=end_date)
        else:
            # 串行处理（兼容模式）
            print("🔄 使用串行处理模式")
            stats = self.update_stocks_serial(stock_codes, test_mode)

        # 输出统计结果
        self._print_final_report(stats)
        return stats

    def update_stocks_serial(self, stock_codes: List[str], test_mode: bool = False) -> Dict:
        """串行更新股票数据（原始方法）"""
        stats = {
            'total_stocks': len(stock_codes),
            'success': 0,
            'fail': 0,
            'skipped': 0,
            'new_records': 0
        }

        print("=" * 70)

        # 更新每只股票
        for stock_code in tqdm(stock_codes, desc="更新进度", unit="只"):
            try:
                # 检查数据状态
                status = self.check_data_status(stock_code)

                # 更新数据
                if status['needs_update']:
                    if self.update_daily_data(stock_code):
                        stats['success'] += 1
                    else:
                        print(f"✗ {stock_code} 数据获取失败，跳过处理")
                        stats['fail'] += 1
                else:
                    stats['skipped'] += 1
                    continue

            except Exception as e:
                if self.verbose:
                    import traceback
                    tb = traceback.extract_tb(e.__traceback__)
                    if tb:
                        filename, line_number, func_name, text = tb[-1]
                        print(f"处理股票 {stock_code} 时出错 [{filename}:{line_number}]: {e}")
                    else:
                        print(f"处理股票 {stock_code} 时出错: {e}")
                stats['fail'] += 1

            # API请求间隔（Tushare有限流，建议间隔长一些）
            if test_mode:
                time.sleep(0.5)  # 测试模式0.5秒
            else:
                time.sleep(0.5)  # 正常模式1秒

        return stats

    def _print_final_report(self, stats: Dict):
        """打印最终报告"""
        print("\n" + "=" * 70)
        print("更新完成报告")
        print("=" * 70)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总股票数: {stats['total_stocks']}")
        print(f"成功更新: {stats['success']}")
        print(f"更新失败: {stats['fail']}")
        print(f"已最新: {stats['skipped']}")

        success_rate = (stats['success'] / (stats['success'] + stats['fail'])) * 100 if (stats['success'] + stats['fail']) > 0 else 100
        print(f"  成功率: {success_rate:.1f}%")
        # 数据质量检查
        self._check_data_quality()

        print("=" * 70)

    def _check_data_quality(self):
        """检查数据质量"""
        print("\n数据质量检查:")

        # 检查数据文件数量
        daily_files = len([f for f in os.listdir(self.daily_dir) if f.endswith('_daily.csv')])

        print(f"  数据文件数量: {daily_files}")

        # 检查存储空间
        try:
            total_size = sum(os.path.getsize(os.path.join(self.daily_dir, f))
                           for f in os.listdir(self.daily_dir) if f.endswith('_daily.csv'))

            print(f"  数据总大小: {total_size / 1024 / 1024:.1f} MB")
        except:
            pass

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Tushare股票数据获取器')
    parser.add_argument('--max-stocks', type=int, help='最大更新股票数量（测试用）')
    parser.add_argument('--test', action='store_true', help='测试模式（只更新前10只股票，较短请求间隔）')
    parser.add_argument('--token', type=str, help='Tushare API token（主要token）')
    parser.add_argument('--tokens', type=str, nargs='+', help='多个Tushare API token列表（用于并行处理避免限流）')
    parser.add_argument('--adjust', type=str, choices=['qfq', 'hfq', ''], default='qfq',
                       help='复权方式：qfq-前复权，hfq-后复权，空字符串-不复权（默认：qfq）')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='启用并行处理（默认启用）')
    parser.add_argument('--no-parallel', action='store_true',
                       help='禁用并行处理，使用串行模式')
    parser.add_argument('--max-workers', type=int, help='最大进程数（并行模式，默认CPU核心数一半）')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='显示详细的处理信息（并行模式默认关闭，串行模式默认开启）')
    parser.add_argument('--quiet', action='store_true', default=False,
                       help='静默模式，不显示详细处理信息')
    parser.add_argument('--force-refresh', action='store_true', default=False,
                       help='强制刷新股票列表缓存，重新从API获取')
    parser.add_argument('--end-date', type=str, help='指定结束日期(YYYYMMDD)，如果现有数据已覆盖此日期则跳过更新')

    args = parser.parse_args()

    # 设置tokens
    default_tokens = [
        '2d884a7e7c0468f3af578b61146ddb764c2e12a0ccfaf8fbb6d63528',  # 原有token
        'bd3a3e286bafb8c1cf602a5eca0e4cf7c2bbeaa28b45e0ab47f260a7'   # 新增token
    ]

    if args.tokens:
        tokens = args.tokens
    elif args.token:
        tokens = [args.token]
    else:
        tokens = default_tokens

    print(f"使用 {len(tokens)} 个token: {[t[:10] + '...' for t in tokens]}")

    # 设置verbose模式
    # 并行模式默认不显示详细信息（避免进度条干扰），串行模式默认显示
    is_parallel = args.parallel and not args.no_parallel
    if args.quiet:
        verbose = False
    elif args.verbose:
        verbose = True
    else:
        # 默认：串行模式显示详细信息，并行模式不显示
        verbose = not is_parallel

    # 创建获取器
    fetcher = TushareDataFetcher(token=tokens[0], tokens=tokens, adjust=args.adjust, verbose=verbose)

    # 如果强制要求verbose，在并行模式下也显示详细信息
    force_verbose = args.verbose

    # 设置测试模式
    max_stocks = args.max_stocks
    if args.test and not max_stocks:
        max_stocks = 10

    # 执行更新
    parallel = args.parallel and not args.no_parallel
    fetcher.update_all_stocks(
        max_stocks=max_stocks,
        test_mode=args.test,
        parallel=parallel,
        max_workers=args.max_workers,
        force_verbose=force_verbose,
        force_refresh=args.force_refresh,
        end_date=args.end_date
    )

if __name__ == "__main__":
    main()
