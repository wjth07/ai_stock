"""
A股股票数据获取系统 - 主程序入口
"""
import argparse
from src.data_fetcher import StockDataFetcher
from src.utils import get_date_range, get_recent_days_date_range


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='A股股票数据获取系统')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['daily', 'minute', 'both', 'test'],
        default='test',
        help='运行模式: daily(仅每日数据), minute(仅分钟级数据), both(两者), test(测试模式)'
    )
    parser.add_argument(
        '--stocks',
        type=str,
        nargs='+',
        help='指定股票代码列表，如: --stocks 000001 000002'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='开始日期，格式: YYYYMMDD'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='结束日期，格式: YYYYMMDD'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='获取最近N天的每日数据，默认365天'
    )
    parser.add_argument(
        '--minute-days',
        type=int,
        default=7,
        help='获取最近N天的分钟级数据，默认7天'
    )
    parser.add_argument(
        '--period',
        type=str,
        default='1',
        choices=['1', '5', '15', '30', '60'],
        help='分钟级数据周期，默认1分钟'
    )
    
    args = parser.parse_args()
    
    # 初始化数据获取器
    fetcher = StockDataFetcher()
    
    print("=" * 60)
    print("A股股票数据获取系统")
    print("=" * 60)
    
    # 获取股票列表
    if args.stocks:
        stock_codes = args.stocks
        print(f"\n使用指定的股票代码: {stock_codes}")
    else:
        stock_list = fetcher.get_stock_list()
        stock_codes = stock_list['code'].tolist()
        print(f"\n获取到 {len(stock_codes)} 只A股股票")
    
    # 确定日期范围
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        if args.mode == 'minute':
            start_date, end_date = get_recent_days_date_range(days=args.minute_days)
        else:
            start_date, end_date = get_date_range(days=args.days)
    
    print(f"日期范围: {start_date} 至 {end_date}")
    
    # 根据模式执行
    if args.mode == 'test':
        # 测试模式：只获取前3只股票的数据作为示例
        print("\n" + "=" * 60)
        print("测试模式：获取前3只股票的数据")
        print("=" * 60)
        test_codes = stock_codes[:3]
        
        # 每日数据
        print("\n[1/2] 获取每日数据...")
        for code in test_codes:
            df = fetcher.fetch_daily_data(code, start_date, end_date)
            if df is not None:
                fetcher.save_daily_data(code, df)
                print(f"  ✓ {code}: {len(df)} 条记录")
        
        # 分钟级数据
        minute_start, minute_end = get_recent_days_date_range(days=args.minute_days)
        print(f"\n[2/2] 获取分钟级数据（{minute_start} 至 {minute_end}）...")
        for code in test_codes:
            df = fetcher.fetch_minute_data(code, minute_start, minute_end, args.period)
            if df is not None:
                fetcher.save_minute_data(code, df)
                print(f"  ✓ {code}: {len(df)} 条记录")
        
        print("\n测试完成！数据已保存到 data/ 目录")
        
    elif args.mode == 'daily':
        # 仅获取每日数据
        print("\n开始批量获取每日数据...")
        fetcher.fetch_all_daily_data(stock_codes, start_date, end_date)
        
    elif args.mode == 'minute':
        # 仅获取分钟级数据
        minute_start, minute_end = get_recent_days_date_range(days=args.minute_days)
        print("\n开始批量获取分钟级数据...")
        fetcher.fetch_all_minute_data(stock_codes, minute_start, minute_end, args.period)
        
    elif args.mode == 'both':
        # 获取两种数据
        print("\n[1/2] 开始批量获取每日数据...")
        fetcher.fetch_all_daily_data(stock_codes, start_date, end_date)
        
        minute_start, minute_end = get_recent_days_date_range(days=args.minute_days)
        print("\n[2/2] 开始批量获取分钟级数据...")
        fetcher.fetch_all_minute_data(stock_codes, minute_start, minute_end, args.period)
    
    print("\n" + "=" * 60)
    print("数据获取完成！")
    print("=" * 60)
    print(f"每日数据保存位置: {fetcher.daily_dir}")
    print(f"分钟级数据保存位置: {fetcher.minute_dir}")


if __name__ == "__main__":
    main()

