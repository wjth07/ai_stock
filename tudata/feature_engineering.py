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

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建目标变量"""
        if df.empty:
            return df

        try:
            # T+1日收益率
            df['TARGET_RETURNS_T1'] = df['close'].shift(-1) / df['close'] - 1
            df['TARGET_RETURNS_T1_PCT'] = df['TARGET_RETURNS_T1'] * 100

            # T+3日收益率（3个交易日后的累积收益率）
            df['TARGET_RETURNS_T3'] = df['close'].shift(-3) / df['close'] - 1
            df['TARGET_RETURNS_T3_PCT'] = df['TARGET_RETURNS_T3'] * 100

            # 分类目标：是否超过5%
            df['TARGET_T1_OVER_5PCT'] = (df['TARGET_RETURNS_T1_PCT'] > 5).astype(int)
            df['TARGET_T3_OVER_5PCT'] = (df['TARGET_RETURNS_T3_PCT'] > 5).astype(int)

            # 其他收益率阈值
            for threshold in [1, 2, 3, 10]:
                df[f'TARGET_T1_OVER_{threshold}PCT'] = (df['TARGET_RETURNS_T1_PCT'] > threshold).astype(int)
                df[f'TARGET_T3_OVER_{threshold}PCT'] = (df['TARGET_RETURNS_T3_PCT'] > threshold).astype(int)

        except Exception as e:
            print(f"创建目标变量失败: {e}")

        return df

    def process_single_stock(self, file_path: str) -> pd.DataFrame:
        """处理单个股票的完整特征工程"""
        # 加载数据
        df = self.load_stock_data(file_path)
        if df.empty:
            return df

        # 添加各类特征
        df = self.add_technical_indicators(df)
        df = self.add_statistical_features(df)
        df = self.add_temporal_features(df)
        df = self.add_trend_features(df)
        df = self.create_target_variables(df)

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
        df = self.add_technical_indicators(df)
        df = self.add_statistical_features(df)
        df = self.add_temporal_features(df)
        df = self.add_trend_features(df)

        # 去除NaN值（技术指标计算需要一定周期）
        df = df.dropna().reset_index(drop=True)

        return df

    def get_feature_columns(self) -> List[str]:
        """获取所有特征列名"""
        all_features = (self.technical_indicators + self.statistical_features +
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
