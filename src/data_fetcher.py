"""
A股股票数据获取模块
使用akshare和东方财富API获取股票数据
"""
import os
import time
import akshare as ak
import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import requests

from .utils import ensure_dir, format_date, get_date_range, get_recent_days_date_range

# 配置代理设置（如果需要）
def setup_proxy(use_proxy: bool = True):
    """
    配置代理设置
    
    Args:
        use_proxy: 是否使用代理，如果为False则禁用代理
    """
    if not use_proxy:
        # 禁用代理
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        # 设置requests不使用代理
        import requests
        requests.Session().proxies = {}
    else:
        # 尝试从系统获取代理设置
        try:
            import subprocess
            result = subprocess.run(['scutil', '--proxy'], capture_output=True, text=True)
            if 'HTTPProxy' in result.stdout:
                # 解析代理设置（简化版本）
                # 实际应该解析scutil输出，这里使用环境变量
                pass
        except:
            pass


class StockDataFetcher:
    """股票数据获取类"""
    
    def __init__(self, data_root: str = "data", use_proxy: bool = False):
        """
        初始化数据获取器
        
        Args:
            data_root: 数据根目录，默认为"data"
            use_proxy: 是否使用代理，默认False（不使用代理，因为VPN代理可能导致连接问题）
        """
        self.data_root = data_root
        self.daily_dir = os.path.join(data_root, "daily")
        self.minute_dir = os.path.join(data_root, "minute")
        
        # 确保目录存在
        ensure_dir(self.daily_dir)
        ensure_dir(self.minute_dir)
        
        # 配置代理
        setup_proxy(use_proxy=use_proxy)
        
        # API请求延时（秒），避免限流
        self.request_delay = 10
    
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取全部A股股票列表
        
        Returns:
            包含股票代码和名称的DataFrame
        """
        try:
            print("正在获取A股股票列表...")
            stock_list = ak.stock_info_a_code_name()
            print(f"成功获取 {len(stock_list)} 只股票")
            return stock_list
        except Exception as e:
            print(f"获取股票列表失败: {e}")
            raise
    
    def fetch_daily_data(
        self, 
        stock_code: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        period: str = "daily",
        adjust: str = ""
    ) -> Optional[pd.DataFrame]:
        """
        获取单只股票的每日数据
        
        Args:
            stock_code: 股票代码（如：000001）
            start_date: 开始日期，格式YYYYMMDD，如果为None则使用默认值（最近1年）
            end_date: 结束日期，格式YYYYMMDD，如果为None则使用今天
            period: 周期，默认为"daily"（日K线）
            adjust: 复权类型，""不复权，"qfq"前复权，"hfq"后复权
            
        Returns:
            包含股票数据的DataFrame，失败返回None
        """
        if start_date is None or end_date is None:
            start_date, end_date = get_date_range(days=365)
        
        try:
            # 使用akshare获取日K线数据
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period=period,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust=adjust
            )
            
            if df is not None and not df.empty:
                # 确保日期列为datetime类型
                if '日期' in df.columns:
                    df['日期'] = pd.to_datetime(df['日期'])
                return df
            else:
                print(f"股票 {stock_code} 无数据")
                return None
                
        except Exception as e:
            print(f"获取股票 {stock_code} 每日数据失败: {e}")
            return None
    
    def save_daily_data(self, stock_code: str, df: pd.DataFrame) -> str:
        """
        保存每日数据到CSV文件
        
        Args:
            stock_code: 股票代码
            df: 数据DataFrame
            
        Returns:
            保存的文件路径
        """
        filename = f"{stock_code}_daily.csv"
        filepath = os.path.join(self.daily_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return filepath
    
    def fetch_all_daily_data(
        self, 
        stock_codes: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> None:
        """
        批量获取全部A股的每日数据
        
        Args:
            stock_codes: 股票代码列表，如果为None则获取全部A股
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
        """
        if stock_codes is None:
            stock_list = self.get_stock_list()
            stock_codes = stock_list['code'].tolist()
        
        if start_date is None or end_date is None:
            start_date, end_date = get_date_range(days=365)
        
        print(f"\n开始批量获取每日数据...")
        print(f"股票数量: {len(stock_codes)}")
        print(f"日期范围: {start_date} 至 {end_date}")
        
        success_count = 0
        fail_count = 0
        
        for stock_code in tqdm(stock_codes, desc="下载每日数据"):
            df = self.fetch_daily_data(stock_code, start_date, end_date)
            
            if df is not None and not df.empty:
                self.save_daily_data(stock_code, df)
                success_count += 1
            else:
                fail_count += 1
            
            # 添加延时，避免API限流
            time.sleep(self.request_delay)
        
        print(f"\n每日数据下载完成！")
        print(f"成功: {success_count} 只")
        print(f"失败: {fail_count} 只")
    
    def fetch_minute_data(
        self, 
        stock_code: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1",
        adjust: str = ""
    ) -> Optional[pd.DataFrame]:
        """
        获取单只股票的分钟级数据
        
        Args:
            stock_code: 股票代码（如：000001）
            start_date: 开始日期，格式YYYYMMDD，如果为None则使用最近7天
            end_date: 结束日期，格式YYYYMMDD，如果为None则使用今天
            period: 分钟周期，"1"=1分钟，"5"=5分钟，"15"=15分钟，"30"=30分钟，"60"=60分钟
            adjust: 复权类型，""不复权，"qfq"前复权，"hfq"后复权
            
        Returns:
            包含分钟级数据的DataFrame，失败返回None
        """
        if start_date is None or end_date is None:
            start_date, end_date = get_recent_days_date_range(days=7)
        
        try:
            # 使用akshare获取分钟K线数据（东方财富接口）
            df = ak.stock_zh_a_hist_min_em(
                symbol=stock_code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                period=period,
                adjust=adjust
            )
            
            if df is not None and not df.empty:
                # 确保时间列为datetime类型
                if '时间' in df.columns:
                    df['时间'] = pd.to_datetime(df['时间'])
                return df
            else:
                print(f"股票 {stock_code} 无分钟级数据")
                return None
                
        except Exception as e:
            print(f"获取股票 {stock_code} 分钟级数据失败: {e}")
            return None
    
    def save_minute_data(self, stock_code: str, df: pd.DataFrame) -> str:
        """
        保存分钟级数据到CSV文件
        
        Args:
            stock_code: 股票代码
            df: 数据DataFrame
            
        Returns:
            保存的文件路径
        """
        filename = f"{stock_code}_minute.csv"
        filepath = os.path.join(self.minute_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return filepath
    
    def fetch_all_minute_data(
        self,
        stock_codes: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1"
    ) -> None:
        """
        批量获取全部A股的分钟级数据
        
        Args:
            stock_codes: 股票代码列表，如果为None则获取全部A股
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            period: 分钟周期，默认"1"（1分钟K线）
        """
        if stock_codes is None:
            stock_list = self.get_stock_list()
            stock_codes = stock_list['code'].tolist()
        
        if start_date is None or end_date is None:
            start_date, end_date = get_recent_days_date_range(days=7)
        
        print(f"\n开始批量获取分钟级数据...")
        print(f"股票数量: {len(stock_codes)}")
        print(f"日期范围: {start_date} 至 {end_date}")
        print(f"周期: {period}分钟")
        
        success_count = 0
        fail_count = 0
        
        for stock_code in tqdm(stock_codes, desc="下载分钟级数据"):
            df = self.fetch_minute_data(stock_code, start_date, end_date, period)
            
            if df is not None and not df.empty:
                self.save_minute_data(stock_code, df)
                success_count += 1
            else:
                fail_count += 1
            
            # 添加延时，避免API限流
            time.sleep(self.request_delay)
        
        print(f"\n分钟级数据下载完成！")
        print(f"成功: {success_count} 只")
        print(f"失败: {fail_count} 只")


def main():
    """主函数，用于测试和直接运行"""
    fetcher = StockDataFetcher()
    
    print("=" * 50)
    print("A股股票数据获取系统")
    print("=" * 50)
    
    # 获取股票列表
    stock_list = fetcher.get_stock_list()
    print(f"\n获取到 {len(stock_list)} 只A股股票")
    print("\n前10只股票示例:")
    print(stock_list.head(10))
    
    # 示例：获取单只股票的每日数据
    print("\n" + "=" * 50)
    print("示例：获取单只股票每日数据")
    print("=" * 50)
    test_code = "000001"  # 平安银行
    daily_df = fetcher.fetch_daily_data(test_code)
    if daily_df is not None:
        print(f"\n股票 {test_code} 每日数据:")
        print(daily_df.head())
        fetcher.save_daily_data(test_code, daily_df)
        print(f"\n数据已保存到: {fetcher.daily_dir}")
    
    # 示例：获取单只股票的分钟级数据
    print("\n" + "=" * 50)
    print("示例：获取单只股票分钟级数据")
    print("=" * 50)
    minute_df = fetcher.fetch_minute_data(test_code)
    if minute_df is not None:
        print(f"\n股票 {test_code} 分钟级数据:")
        print(minute_df.head())
        fetcher.save_minute_data(test_code, minute_df)
        print(f"\n数据已保存到: {fetcher.minute_dir}")


if __name__ == "__main__":
    main()

