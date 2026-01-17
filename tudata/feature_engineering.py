#!/usr/bin/env python
"""
股票预测特征工程模块
为短期收益率预测构建技术指标和统计特征
"""
import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
import talib
import warnings
warnings.filterwarnings('ignore')

class StockFeatureEngineer:
    """股票特征工程器"""

    def __init__(self):
        self.technical_indicators = []
        self.statistical_features = []
        self.price_volume_features = []
        
        # 初始化特征列表（通过模拟运行）
        self._init_feature_columns()

    def _init_feature_columns(self):
        """通过模拟数据初始化特征列表，确保 get_feature_columns 在处理真实数据前就能返回完整列表"""
        try:
            # 构造足够的模拟数据 (需要满足最大 rolling window, 例如 MA30, rolling_20)
            dates = pd.date_range(start='2020-01-01', periods=50)
            data = {
                'ts_code': ['000001.SZ'] * 50,
                'trade_date': dates,
                'open': np.random.rand(50) * 10 + 10,
                'high': np.random.rand(50) * 10 + 20,
                'low': np.random.rand(50) * 10 + 5,
                'close': np.random.rand(50) * 10 + 15,
                'pre_close': np.random.rand(50) * 10 + 14,
                'change': np.random.randn(50),
                'pct_chg': np.random.randn(50),
                'vol': np.random.rand(50) * 1000,
                'amount': np.random.rand(50) * 10000
            }
            df = pd.DataFrame(data)
            
            # 计算收益率 (load_stock_data 中的逻辑)
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # 运行各特征生成函数
            # 这里的目的仅仅是填充 self.xxx_features 列表
            self.add_price_volume_features(df)
            self.add_technical_indicators(df)
            self.add_statistical_features(df)
            
            # 这里的 df 只是临时变量，用完即丢
        except Exception as e:
            print(f"初始化特征列表失败: {e}")

    def load_stock_data(self, file_path: str) -> pd.DataFrame:
        """加载股票数据"""
        try:
            df = pd.read_csv(file_path)
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.sort_values('trade_date').reset_index(drop=True)

            # 计算收益率
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            return df
        except Exception as e:
            print(f"加载数据失败 {file_path}: {e}")
            return pd.DataFrame()

    def add_price_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加基础量价特征 (保留原始结构信息)"""
        if df.empty:
            return df
            
        try:
            # 1. 基础价格比率 (归一化到当日基准)
            # 使用 pre_close 作为基准，保留了 Open/High/Low 相对于昨日收盘的位置信息
            # 这比纯粹的 Close/MA 更能反映当日的K线结构 (如跳空高开、长上影线等)
            df['OPEN_RATIO'] = df['open'] / df['pre_close']
            df['HIGH_RATIO'] = df['high'] / df['pre_close']
            df['LOW_RATIO'] = df['low'] / df['pre_close']
            df['CLOSE_RATIO'] = df['close'] / df['pre_close']
            
            # 2. 振幅 (Intraday Volatility)
            # 反映当日多空博弈的激烈程度
            df['AMPLITUDE'] = (df['high'] - df['low']) / df['pre_close']
            
            # 3. K线形态特征 (Shadows and Body)
            # 实体大小
            df['BODY_SIZE'] = abs(df['close'] - df['open']) / df['pre_close']
            # 上影线长度
            df['UPPER_SHADOW'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['pre_close']
            # 下影线长度
            df['LOWER_SHADOW'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['pre_close']
            
            # 4. 成交量变化 (Volume Dynamics)
            df['VOL_PCT_CHG'] = df['vol'].pct_change()
            df['AMOUNT_PCT_CHG'] = df['amount'].pct_change()
            
            # 5. 量价关系 (VWAP Relative Position)
            # 均价 = 金额 / 成交量。反映当日平均持仓成本。
            # 避免除零错误
            vwap = df['amount'] / (df['vol'] + 1e-5) 
            # 如果单位不一致，VWAP的绝对值没有意义，但其相对Close的变化有意义
            # 这里我们假设 amount 和 vol 的单位比率在时间上是恒定的
            # 我们关注 VWAP 相对于 Close 的位置
            # 由于不知道具体单位，我们使用 (Amount/Vol) / Close 的变化率是不安全的（如果单位换算导致数值差异极大）
            # 但 (Amount_t / Vol_t) / (Amount_{t-1} / Vol_{t-1}) 是安全的（单位抵消）
            # 或者简单的: Close / VWAP (需要校准单位，或者看趋势)
            
            # 另一种思路：量价相关性 (Rolling Correlation) 在 statistical features 里可能有
            
            # 这里先只加最稳健的量比变化
            # 以及换手率的代理变量 (Vol / Circulating_Shares)，但我们没有流通股本数据
            # 用 Vol / Vol_MA 已经在 technical indicators 里了 (VOL_RATIO)
            
            # 记录新特征
            self.price_volume_features = [
                'OPEN_RATIO', 'HIGH_RATIO', 'LOW_RATIO', 'CLOSE_RATIO', 
                'AMPLITUDE', 'BODY_SIZE', 'UPPER_SHADOW', 'LOWER_SHADOW',
                'VOL_PCT_CHG', 'AMOUNT_PCT_CHG'
            ]
                                        
        except Exception as e:
            print(f"计算基础量价特征失败: {e}")
            
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        if df.empty:
            return df

        try:
            # 基础价格数据
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['vol'].values

            # 均线系统
            df['MA5'] = talib.SMA(close, timeperiod=5)
            df['MA10'] = talib.SMA(close, timeperiod=10)
            df['MA20'] = talib.SMA(close, timeperiod=20)
            df['MA30'] = talib.SMA(close, timeperiod=30)

            # 指数移动平均
            df['EMA5'] = talib.EMA(close, timeperiod=5)
            df['EMA10'] = talib.EMA(close, timeperiod=10)
            df['EMA20'] = talib.EMA(close, timeperiod=20)

            # MACD指标
            macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df['MACD'] = macd
            df['MACD_SIGNAL'] = macdsignal
            df['MACD_HIST'] = macdhist

            # RSI相对强弱指数
            df['RSI'] = talib.RSI(close, timeperiod=14)

            # 威廉指标
            df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

            # 动量指标
            df['MOM'] = talib.MOM(close, timeperiod=10)
            df['ROC'] = talib.ROC(close, timeperiod=10)

            # KDJ随机指标
            k, d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            df['KDJ_K'] = k
            df['KDJ_D'] = d
            df['KDJ_J'] = 3 * k - 2 * d

            # ATR真实波动幅度
            df['ATR'] = talib.ATR(high, low, close, timeperiod=14)

            # 布林带
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['BB_UPPER'] = upper
            df['BB_MIDDLE'] = middle
            df['BB_LOWER'] = lower
            df['BB_WIDTH'] = (upper - lower) / middle  # 带宽

            # 成交量指标
            df['VOL_MA5'] = talib.SMA(volume, timeperiod=5)
            df['VOL_MA10'] = talib.SMA(volume, timeperiod=10)
            df['VOL_RATIO'] = volume / df['VOL_MA5']  # 量比

            # OBV能量潮
            df['OBV'] = talib.OBV(close, volume)

            # AD成交量指标
            df['AD'] = talib.AD(high, low, close, volume)

            # 记录技术指标
            self.technical_indicators = [col for col in df.columns if col not in
                                       ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                                        'pre_close', 'change', 'pct_chg', 'vol', 'amount',
                                        'returns', 'log_returns']]

        except Exception as e:
            print(f"计算技术指标失败: {e}")

        return df

    def add_statistical_features(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """添加统计特征"""
        if df.empty:
            return df

        try:
            # 滚动统计特征
            df[f'ROLLING_MEAN_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ROLLING_STD_{window}'] = df['close'].rolling(window=window).std()
            df[f'ROLLING_SKEW_{window}'] = df['close'].rolling(window=window).skew()
            df[f'ROLLING_KURT_{window}'] = df['close'].rolling(window=window).kurt()

            # 收益率统计
            df[f'RETURNS_MEAN_{window}'] = df['returns'].rolling(window=window).mean()
            df[f'RETURNS_STD_{window}'] = df['returns'].rolling(window=window).std()
            df[f'RETURNS_SKEW_{window}'] = df['returns'].rolling(window=window).skew()

            # 价格位置特征
            df[f'PRICE_PERCENTILE_{window}'] = df['close'].rolling(window=window).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
            )

            # 波动率特征
            df[f'VOLATILITY_{window}'] = df['returns'].rolling(window=window).std() * np.sqrt(252)  # 年化波动率

            # 自相关特征
            df[f'AUTOCORR_1_{window}'] = df['returns'].rolling(window=window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
            )

            # 成交量统计
            df[f'VOL_PERCENTILE_{window}'] = df['vol'].rolling(window=window).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
            )

            # 记录统计特征
            self.statistical_features = [col for col in df.columns if any(prefix in col for prefix in
                                                ['ROLLING_', 'RETURNS_', 'PRICE_PERCENTILE_', 'VOLATILITY_',
                                                 'AUTOCORR_', 'VOL_PERCENTILE_'])]

        except Exception as e:
            print(f"计算统计特征失败: {e}")

        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征"""
        if df.empty:
            return df

        try:
            # 日期特征
            df['DAY_OF_WEEK'] = df['trade_date'].dt.dayofweek
            df['DAY_OF_MONTH'] = df['trade_date'].dt.day
            df['MONTH'] = df['trade_date'].dt.month
            df['QUARTER'] = df['trade_date'].dt.quarter
            df['IS_MONTH_START'] = df['trade_date'].dt.is_month_start.astype(int)
            df['IS_MONTH_END'] = df['trade_date'].dt.is_month_end.astype(int)

            # 周期特征（正弦余弦编码）
            df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
            df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
            df['DAY_SIN'] = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)
            df['DAY_COS'] = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)

        except Exception as e:
            print(f"计算时间特征失败: {e}")

        return df

    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加趋势特征"""
        if df.empty:
            return df

        try:
            # 连续上涨/下跌天数
            df['CONSECUTIVE_UP'] = (df['close'] > df['close'].shift(1)).groupby(
                (df['close'] <= df['close'].shift(1)).cumsum()).cumcount()
            df['CONSECUTIVE_DOWN'] = (df['close'] < df['close'].shift(1)).groupby(
                (df['close'] >= df['close'].shift(1)).cumsum()).cumcount()

            # 新高新低
            df['NEW_HIGH_20'] = (df['high'] == df['high'].rolling(20).max()).astype(int)
            df['NEW_LOW_20'] = (df['low'] == df['low'].rolling(20).min()).astype(int)

            # 均线位置
            df['PRICE_ABOVE_MA5'] = (df['close'] > df['MA5']).astype(int)
            df['PRICE_ABOVE_MA10'] = (df['close'] > df['MA10']).astype(int)
            df['PRICE_ABOVE_MA20'] = (df['close'] > df['MA20']).astype(int)

            # 布林带位置
            df['PRICE_POSITION_BB'] = (df['close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])
            df['PRICE_ABOVE_BB_MIDDLE'] = (df['close'] > df['BB_MIDDLE']).astype(int)

        except Exception as e:
            print(f"计算趋势特征失败: {e}")

        return df

    def create_target_variables(self, df: pd.DataFrame, exclude_one_word_limit: bool = False) -> pd.DataFrame:
        """创建目标变量

        Args:
            df: 股票数据DataFrame
            exclude_one_word_limit: 是否排除一字板股票的涨幅标签
        """
        if df.empty:
            return df

        try:
            # T+1日收益率
            df['TARGET_RETURNS_T1'] = df['close'].shift(-1) / df['close'] - 1
            df['TARGET_RETURNS_T1_PCT'] = df['TARGET_RETURNS_T1'] * 100

            # T+3日收益率（3个交易日后的累积收益率）
            df['TARGET_RETURNS_T3'] = df['close'].shift(-3) / df['close'] - 1
            df['TARGET_RETURNS_T3_PCT'] = df['TARGET_RETURNS_T3'] * 100

            # 检测一字板：使用更灵活的检测逻辑
            # 主要特征：价格波动极小 + 大幅涨跌
            price_range = (df['high'] - df['low']) / df['close']  # 价格波动范围

            # 进一步放宽检测：降低价格波动要求，提高涨跌幅要求，避免过度排除
            is_large_change = abs(df['pct_chg']) >= 9.5  # 9.5%以上算大幅涨跌（接近涨跌停）
            is_one_word = (price_range <= 0.005) & is_large_change  # 价格波动<=0.5%且大幅涨跌

            # 打印检测到的市场信息（可选）
            if exclude_one_word_limit and len(df) > 0:
                stock_codes = df['ts_code'].unique()[:5]  # 展示前5个股票代码
                print(f"一字板检测: 分析了{len(df)}条记录，发现{is_one_word.sum()}个一字板 (示例股票: {list(stock_codes)})")

            # 创建基础目标变量
            df['TARGET_T1_OVER_5PCT'] = (df['TARGET_RETURNS_T1_PCT'] >= 5).astype(int)
            df['TARGET_T3_OVER_5PCT'] = (df['TARGET_RETURNS_T3_PCT'] >= 5).astype(int)

            # 如果排除一字板，将一字板当天的涨幅标签设为0
            if exclude_one_word_limit:
                df.loc[is_one_word, 'TARGET_T1_OVER_5PCT'] = 0
                df.loc[is_one_word, 'TARGET_T3_OVER_5PCT'] = 0
                print(f"已排除 {is_one_word.sum()} 个一字板样本的涨幅标签")

            # 其他收益率阈值
            for threshold in [1, 2, 3, 10]:
                df[f'TARGET_T1_OVER_{threshold}PCT'] = (df['TARGET_RETURNS_T1_PCT'] > threshold).astype(int)
                df[f'TARGET_T3_OVER_{threshold}PCT'] = (df['TARGET_RETURNS_T3_PCT'] > threshold).astype(int)

                # 同样排除一字板
                if exclude_one_word_limit:
                    df.loc[is_one_word, f'TARGET_T1_OVER_{threshold}PCT'] = 0
                    df.loc[is_one_word, f'TARGET_T3_OVER_{threshold}PCT'] = 0

        except Exception as e:
            print(f"创建目标变量失败: {e}")

        return df

    def process_single_stock(self, file_path: str, exclude_one_word_limit: bool = False) -> pd.DataFrame:
        """处理单个股票的完整特征工程

        Args:
            file_path: 股票数据文件路径
            exclude_one_word_limit: 是否排除一字板股票的涨幅标签
        """
        # 加载数据
        df = self.load_stock_data(file_path)
        if df.empty:
            return df

        # 添加各类特征
        df = self.add_price_volume_features(df)
        df = self.add_technical_indicators(df)
        df = self.add_statistical_features(df)
        df = self.add_temporal_features(df)
        df = self.add_trend_features(df)
        df = self.create_target_variables(df, exclude_one_word_limit=exclude_one_word_limit)

        # 去除NaN值（技术指标计算需要一定周期）
        df = df.dropna().reset_index(drop=True)

        return df

    def process_single_stock_for_prediction(self, file_path: str) -> pd.DataFrame:
        """处理单个股票的预测特征工程（不包含目标变量）"""
        # 加载数据
        df = self.load_stock_data(file_path)
        if df.empty:
            return df

        # 添加各类特征（不包含目标变量）
        df = self.add_price_volume_features(df)
        df = self.add_technical_indicators(df)
        df = self.add_statistical_features(df)
        df = self.add_temporal_features(df)
        df = self.add_trend_features(df)

        # 去除NaN值（技术指标计算需要一定周期）
        df = df.dropna().reset_index(drop=True)

        return df

    def get_feature_columns(self) -> List[str]:
        """获取所有特征列名"""
        all_features = (self.price_volume_features + self.technical_indicators + self.statistical_features +
                       ['DAY_OF_WEEK', 'DAY_OF_MONTH', 'MONTH', 'QUARTER',
                        'IS_MONTH_START', 'IS_MONTH_END', 'MONTH_SIN', 'MONTH_COS',
                        'DAY_SIN', 'DAY_COS', 'CONSECUTIVE_UP', 'CONSECUTIVE_DOWN',
                        'NEW_HIGH_20', 'NEW_LOW_20', 'PRICE_ABOVE_MA5', 'PRICE_ABOVE_MA10',
                        'PRICE_ABOVE_MA20', 'PRICE_POSITION_BB', 'PRICE_ABOVE_BB_MIDDLE'])
        return all_features

    def get_target_columns(self) -> List[str]:
        """获取所有目标列名"""
        targets = ['TARGET_RETURNS_T1', 'TARGET_RETURNS_T1_PCT', 'TARGET_RETURNS_T3',
                  'TARGET_RETURNS_T3_PCT', 'TARGET_T1_OVER_5PCT', 'TARGET_T3_OVER_5PCT']

        # 添加其他阈值目标
        for threshold in [1, 2, 3, 10]:
            targets.extend([f'TARGET_T1_OVER_{threshold}PCT', f'TARGET_T3_OVER_{threshold}PCT'])

        return targets

    def create_sequence_data(self, df: pd.DataFrame, window_size: int = 30, feature_cols: List[str] = None, target_col: str = 'TARGET_T1_OVER_5PCT') -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列样本数据 (Rolling Window)
        用于LSTM/GRU/Transformer等序列模型训练

        Args:
            df: 经过预处理的单个股票DataFrame
            window_size: 时间窗口大小 (序列长度)
            feature_cols: 特征列列表，如果为None则使用所有特征
            target_col: 目标变量列名

        Returns:
            X: 特征数据，形状 (N, window_size, feature_dim)
            y: 目标数据，形状 (N,)
        """
        if df.empty or len(df) < window_size:
            return np.array([]), np.array([])

        if feature_cols is None:
            feature_cols = self.get_feature_columns()

        # 确保所有特征列都存在
        valid_feature_cols = [col for col in feature_cols if col in df.columns]
        if not valid_feature_cols:
            return np.array([]), np.array([])

        # 提取特征和目标
        # data_x: (num_rows, num_features)
        data_x = df[valid_feature_cols].values
        # data_y: (num_rows, )
        if target_col in df.columns:
            data_y = df[target_col].values
        else:
            # 如果没有目标列（可能是预测模式），则y为空或全0
            data_y = np.zeros(len(df))

        X = []
        y = []

        # 遍历数据创建序列
        # 从 window_size-1 开始，因为需要足够的回溯窗口
        # 例如 window_size=3, i从2开始 (0,1,2)
        for i in range(window_size - 1, len(df)):
            # 截取窗口特征: 从 i-window_size+1 到 i+1 (不包含i+1)
            # 即 [i-window_size+1, ..., i] 共 window_size 行
            # 例如 i=2, window=3 -> 2-3+1=0 -> [0:3] -> 行 0, 1, 2
            window_features = data_x[i - window_size + 1 : i + 1]

            # 对应的目标是当前时间点 i 的目标 (T+1收益)
            target = data_y[i]

            X.append(window_features)
            y.append(target)

        return np.array(X), np.array(y)

def main():
    """演示特征工程"""
    engineer = StockFeatureEngineer()

    # 处理一个股票文件
    sample_file = "daily/000001.SZ_daily.csv"
    if os.path.exists(sample_file):
        df = engineer.process_single_stock(sample_file)

        print("特征工程演示:")
        print(f"数据形状: {df.shape}")
        print(f"特征数量: {len(engineer.get_feature_columns())}")
        print(f"目标变量: {len(engineer.get_target_columns())}")

        print("\n特征列示例:")
        features = engineer.get_feature_columns()[:10]
        for feature in features:
            if feature in df.columns:
                sample_value = df[feature].iloc[-1]
                print(f"  {feature}: {sample_value:.4f}")

        print("\n目标变量示例:")
        targets = ['TARGET_T1_OVER_5PCT', 'TARGET_T3_OVER_5PCT', 'TARGET_RETURNS_T1_PCT', 'TARGET_RETURNS_T3_PCT']
        for target in targets:
            if target in df.columns:
                sample_value = df[target].iloc[-5]  # 显示前5个（有目标值的）
                if 'PCT' in target:
                    print(f"  {target}: {sample_value:.2f}%")
                else:
                    print(f"  {target}: {sample_value}")

if __name__ == "__main__":
    main()
