#!/usr/bin/env python
"""
滚动股票预测器 - 连续30天滚动预测
使用截止到T-1的所有历史数据训练模型，预测第T天的结果
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
from feature_engineering import StockFeatureEngineer
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

class RollingStockPredictor:
    """滚动股票预测器"""

    def __init__(self, model_dir: str = "models", majority_undersampling_ratio: float = 0.1, max_stocks: int = None):
        self.model_dir = os.path.join(model_dir, "rolling_models")
        self.cache_dir = os.path.join(self.model_dir, "cache")
        self.feature_engineer = StockFeatureEngineer()
        self.majority_undersampling_ratio = majority_undersampling_ratio
        self.max_stocks = max_stocks

        # 创建目录
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def prepare_data_for_date(self, data_dir: str = "daily", cutoff_date: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备截止到指定日期之前的所有训练数据

        Args:
            data_dir: 数据目录
            cutoff_date: 截止日期 (字符串格式 'YYYY-MM-DD')

        Returns:
            (X, y_t1): 特征数据和目标变量
        """
        # 生成缓存文件名（包含股票数量限制）
        cutoff_str = cutoff_date.replace('-', '') if cutoff_date else 'all'
        stocks_str = f"_{self.max_stocks}" if self.max_stocks else "_all"
        cache_prefix = f"data_cutoff_{cutoff_str}{stocks_str}_full"
        X_cache_path = os.path.join(self.cache_dir, f"{cache_prefix}_X.pkl")
        y_cache_path = os.path.join(self.cache_dir, f"{cache_prefix}_y.pkl")

        # 检查全量数据缓存是否存在
        if os.path.exists(X_cache_path) and os.path.exists(y_cache_path):
            print(f"✓ 检测到缓存数据 ({cutoff_date})，正在加载...")
            X = joblib.load(X_cache_path)
            y_t1 = joblib.load(y_cache_path)
            print(f"✓ 全量缓存数据加载完成: {X.shape}")

            # 根据当前的下采样比例进行动态下采样
            if self.majority_undersampling_ratio < 1.0:
                print(f"✓ 应用多数类下采样 (ratio={self.majority_undersampling_ratio})...")
                X, y_t1 = self._apply_majority_undersampling(X, y_t1)
                print(f"✓ 下采样完成: {X.shape}")

            return X, y_t1

        print(f"准备截止到 {cutoff_date} 的训练数据...")
        all_data = []
        file_count = 0

        # 获取所有股票文件
        files = [f for f in os.listdir(data_dir) if f.endswith('_daily.csv')]

        # 限制股票数量用于调试
        if self.max_stocks and self.max_stocks < len(files):
            files = files[:self.max_stocks]
            print(f"调试模式：仅处理前 {self.max_stocks} 只股票")

        print(f"处理 {len(files)} 只股票...")

        for file in tqdm(files, desc=f"准备 {cutoff_date} 数据"):
            file_path = os.path.join(data_dir, file)
            stock_code = file.replace('_daily.csv', '')

            try:
                # 处理单个股票的特征工程
                df = self.feature_engineer.process_single_stock(file_path)

                if not df.empty and len(df) > 30:
                    # 过滤掉cutoff_date之后的数据
                    if cutoff_date:
                        df = df[df['trade_date'] <= cutoff_date]

                    if not df.empty:
                        df['stock_code'] = stock_code
                        all_data.append(df)
                        file_count += 1

            except Exception as e:
                print(f"处理股票 {stock_code} 失败: {e}")
                continue

        if not all_data:
            raise ValueError(f"截止到 {cutoff_date} 没有找到有效的训练数据")

        # 合并所有股票数据
        combined_data = pd.concat(all_data, ignore_index=True)

        # 确保数据按交易日期和股票代码全局排序
        combined_data = combined_data.sort_values(by=['trade_date', 'stock_code']).reset_index(drop=True)

        # 分离特征和目标
        feature_cols = self.feature_engineer.get_feature_columns()
        available_features = [col for col in feature_cols if col in combined_data.columns]

        X = combined_data[available_features].copy().reset_index(drop=True)
        y_t1 = combined_data['TARGET_T1_OVER_5PCT'].copy().reset_index(drop=True)

        # 处理缺失值
        X = X.fillna(method='ffill').fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        # 保存全量数据到缓存（不包含下采样）
        print(f"✓ 保存全量数据到缓存: {self.cache_dir}")
        joblib.dump(X, X_cache_path)
        joblib.dump(y_t1, y_cache_path)
        print("✓ 缓存保存完成")

        # 根据当前的下采样比例进行动态下采样
        if self.majority_undersampling_ratio < 1.0:
            print(f"✓ 应用多数类下采样 (ratio={self.majority_undersampling_ratio})...")
            X, y_t1 = self._apply_majority_undersampling(X, y_t1)
            print(f"✓ 下采样完成: {X.shape}")

        return X, y_t1

    def _apply_majority_undersampling(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        应用多数类下采样

        Args:
            X: 特征数据
            y: 目标变量

        Returns:
            下采样后的数据
        """
        class_counts = y.value_counts()
        majority_class = class_counts.idxmax()
        n_majority = class_counts[majority_class]

        # 计算目标多数类样本数量
        target_majority_count = int(n_majority * self.majority_undersampling_ratio)

        if target_majority_count < n_majority:
            print(f"多数类样本 ({n_majority}) 将按比例 {self.majority_undersampling_ratio} 欠采样至 {target_majority_count}...")

            undersampler = RandomUnderSampler(
                sampling_strategy={majority_class: target_majority_count},
                random_state=42
            )
            X_resampled, y_resampled = undersampler.fit_resample(X, y)

            print(f"欠采样后类别分布: {pd.Series(y_resampled).value_counts().to_dict()}")
            return X_resampled, y_resampled
        else:
            print(f"多数类样本 ({n_majority}) 无需欠采样")
            return X, y

    def predict_single_day(self, train_cutoff_date: str, predict_date: str) -> Dict:
        """
        使用截止到train_cutoff_date的数据训练模型，预测predict_date的结果

        Args:
            train_cutoff_date: 训练数据截止日期
            predict_date: 预测日期

        Returns:
            预测结果字典
        """
        print(f"\n=== 预测 {predict_date} ===")
        print(f"训练数据截止: {train_cutoff_date}")

        try:
            # 准备训练数据
            X_train, y_train = self.prepare_data_for_date(cutoff_date=train_cutoff_date)

            print(f"训练数据形状: {X_train.shape}")
            print(f"正样本比例: {y_train.mean():.3f}")

            # 训练模型
            model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                verbose=1  # 显示训练进度
            )

            model.fit(X_train, y_train)

            # 准备预测数据 - 只需要predict_date当天的数据
            predict_data = self.load_single_day_data(predict_date)
            print(f"predict_data: {predict_data}")
            if predict_data is None:
                return {"date": predict_date, "error": f"找不到 {predict_date} 的预测数据"}

            # 预测
            pred_proba = model.predict_proba(predict_data)[0]
            pred_class = model.predict(predict_data)[0]

            result = {
                "date": predict_date,
                "train_cutoff": train_cutoff_date,
                "train_samples": len(X_train),
                "positive_ratio": float(y_train.mean()),
                "prediction": int(pred_class),
                "probability_over_5pct": float(pred_proba[1]),
                "probability_under_5pct": float(pred_proba[0]),
                "confidence": "high" if abs(pred_proba[1] - pred_proba[0]) > 0.3 else "medium" if abs(pred_proba[1] - pred_proba[0]) > 0.1 else "low"
            }

            print(f"预测结果: {'上涨' if pred_class == 1 else '下跌'} (概率: {pred_proba[1]:.3f})")

            return result

        except Exception as e:
            print(f"预测 {predict_date} 失败: {e}")
            return {"date": predict_date, "error": str(e)}

    def load_single_day_data(self, predict_date: str) -> Optional[pd.DataFrame]:
        """
        加载指定日期的预测数据（单日）

        Args:
            predict_date: 预测日期

        Returns:
            预测特征数据
        """
        # 生成缓存文件名（包含股票数量限制）
        predict_date_str = predict_date.replace('-', '')
        stocks_str = f"_{self.max_stocks}" if self.max_stocks else "_all"
        cache_path = os.path.join(self.cache_dir, f"predict_data_{predict_date_str}{stocks_str}.pkl")

        # 检查缓存是否存在
        if os.path.exists(cache_path):
            print(f"✓ 检测到预测数据缓存 ({predict_date})，正在加载...")
            X_pred = joblib.load(cache_path)
            print(f"✓ 预测数据缓存加载完成: {X_pred.shape}")
            return X_pred

        print(f"生成预测数据缓存 ({predict_date})...")
        all_predict_data = []

        # 获取所有股票文件
        files = [f for f in os.listdir("daily") if f.endswith('_daily.csv')]

        # 限制股票数量用于调试
        if self.max_stocks and self.max_stocks < len(files):
            files = files[:self.max_stocks]
            print(f"调试模式：仅处理前 {self.max_stocks} 只股票")

        for file in tqdm(files, desc=f"加载预测数据 {predict_date}"):
            file_path = os.path.join("daily", file)
            stock_code = file.replace('_daily.csv', '')

            try:
                # 使用预测专用的特征工程方法（不包含目标变量，与训练保持一致）
                df = self.feature_engineer.process_single_stock_for_prediction(file_path)
                print(f'df: {df}, {max(df["trade_date"])}')

                if not df.empty:
                    # 找到指定日期的数据
                    # 将predict_date转换为datetime进行比较
                    predict_dt = pd.to_datetime(predict_date)
                    day_data = df[df['trade_date'] == predict_dt]

                    if not day_data.empty:
                        # 提取这一天的特征
                        feature_cols = self.feature_engineer.get_feature_columns()
                        available_features = [col for col in feature_cols if col in day_data.columns]

                        features = day_data[available_features].copy()
                        if not features.empty:
                            features['stock_code'] = stock_code
                            all_predict_data.append(features)

            except Exception as e:
                print(e)
                continue

        if not all_predict_data:
            return None

        # 合并所有股票的预测数据
        predict_df = pd.concat(all_predict_data, ignore_index=True)

        # 选择特征列（排除目标列）
        feature_cols = self.feature_engineer.get_feature_columns()
        available_features = [col for col in feature_cols if col in predict_df.columns and not col.startswith('TARGET_')]

        X_pred = predict_df[available_features].copy()

        # 处理缺失值
        X_pred = X_pred.fillna(method='ffill').fillna(0)
        X_pred = X_pred.replace([np.inf, -np.inf], 0)

        # 保存到缓存
        print(f"✓ 保存预测数据到缓存: {self.cache_dir}")
        joblib.dump(X_pred, cache_path)
        print("✓ 预测数据缓存保存完成")

        return X_pred

    def rolling_predict(self, start_date: str = None, end_date: str = None, n_days: int = 30) -> List[Dict]:
        """
        执行滚动预测

        Args:
            start_date: 开始预测的日期
            end_date: 结束预测的日期，配合n_days=1使用
            n_days: 预测天数

        Returns:
            预测结果列表
        """
        print("=" * 60)
        print("开始滚动股票预测")
        print("=" * 60)

        # 处理预测日期范围
        if end_date is not None:
            # 如果指定了end_date
            end_dt = pd.to_datetime(end_date)

            if n_days == 1:
                # predict-days=1时，只预测end_date当天
                predict_dates = [end_dt]
                print(f"预测日期: {end_date}")
            else:
                # predict-days>1时，预测end_date前面的n_days天
                # 获取数据中的交易日
                sample_files = [f for f in os.listdir("daily") if f.endswith('_daily.csv')]
                if sample_files:
                    sample_df = pd.read_csv(os.path.join("daily", sample_files[0]))
                    if 'trade_date' in sample_df.columns:
                        sample_df['trade_date'] = pd.to_datetime(sample_df['trade_date'], format='%Y%m%d')
                        all_dates = sorted(sample_df['trade_date'].unique())

                        # 找到end_dt在数据中的位置
                        end_idx = None
                        for i, date in enumerate(all_dates):
                            if date >= end_dt:
                                end_idx = i
                                break

                        if end_idx is not None and end_idx >= n_days:
                            # 取end_dt前面的n_days个交易日
                            predict_dates = all_dates[end_idx-n_days:end_idx]
                        else:
                            # 如果数据不够，尽量取可用的日期
                            predict_dates = all_dates[-n_days:] if len(all_dates) >= n_days else all_dates
                    else:
                        # 如果无法读取数据日期，使用简单的工作日计算
                        predict_dates = pd.date_range(end=end_dt, periods=n_days, freq='B')
                else:
                    # 如果无法读取数据，使用简单的工作日计算
                    predict_dates = pd.date_range(end=end_dt, periods=n_days, freq='B')

                print(f"预测日期范围: {predict_dates[0].strftime('%Y-%m-%d')} 到 {predict_dates[-1].strftime('%Y-%m-%d')}")

        elif start_date is None:
            # 没有指定start_date和end_date，使用原来的逻辑：预测最后n_days天
            sample_files = [f for f in os.listdir("daily") if f.endswith('_daily.csv')]
            if not sample_files:
                raise ValueError("没有找到股票数据文件")

            sample_df = pd.read_csv(os.path.join("daily", sample_files[0]))

            # 检查日期格式 - tushare数据是YYYYMMDD格式
            if 'trade_date' in sample_df.columns:
                # tushare数据是YYYYMMDD格式，如20240102
                sample_df['trade_date'] = pd.to_datetime(sample_df['trade_date'], format='%Y%m%d')

                # 获取最后n_days个交易日
                unique_dates = sorted(sample_df['trade_date'].unique())
                predict_dates = unique_dates[-n_days:]

                print(f"自动选择最后 {n_days} 个交易日进行预测")
                print(f"预测日期范围: {predict_dates[0].strftime('%Y-%m-%d')} 到 {predict_dates[-1].strftime('%Y-%m-%d')}")
            else:
                raise ValueError("数据文件中没有trade_date列")
        else:
            # 使用指定的start_date
            start_dt = pd.to_datetime(start_date)
            predict_dates = pd.date_range(start=start_dt, periods=n_days, freq='B')  # 工作日
            print(f"预测日期范围: {predict_dates[0].strftime('%Y-%m-%d')} 到 {predict_dates[-1].strftime('%Y-%m-%d')}")

        print(f"将预测以下 {len(predict_dates)} 个日期:")
        for i, date in enumerate(predict_dates):
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            print(f"  {i+1}. {date_str}")

        results = []

        for i, predict_date in enumerate(predict_dates):
            # 确保predict_date是datetime对象
            if not isinstance(predict_date, pd.Timestamp):
                predict_dt = pd.to_datetime(predict_date)
            else:
                predict_dt = predict_date

            # 计算训练数据截止日期（预测日期前一天）
            train_cutoff_dt = predict_dt - pd.Timedelta(days=1)

            # 执行预测
            result = self.predict_single_day(
                train_cutoff_date=train_cutoff_dt.strftime('%Y-%m-%d'),
                predict_date=predict_dt.strftime('%Y-%m-%d')
            )

            results.append(result)

            print(f"完成 {i+1}/{len(predict_dates)} 预测")

        return results

    def evaluate_predictions(self, predictions: List[Dict]) -> Dict:
        """
        评估预测结果

        Args:
            predictions: 预测结果列表

        Returns:
            评估指标字典
        """
        print("\n" + "=" * 60)
        print("预测结果评估")
        print("=" * 60)

        # 过滤掉有错误的结果
        valid_predictions = [p for p in predictions if 'error' not in p]

        if not valid_predictions:
            return {"error": "没有有效的预测结果"}

        # 提取预测和概率
        y_pred = [p['prediction'] for p in valid_predictions]
        y_proba = [p['probability_over_5pct'] for p in valid_predictions]

        print(f"总预测数: {len(predictions)}")
        print(f"有效预测数: {len(valid_predictions)}")

        # 计算基础指标
        accuracy = sum(y_pred) / len(y_pred) if y_pred else 0  # 正样本比例

        # Top-5 准确率（概率最高的5个预测中实际上涨的比例）
        if len(valid_predictions) >= 5:
            # 按概率排序，取前5个
            sorted_preds = sorted(valid_predictions, key=lambda x: x['probability_over_5pct'], reverse=True)
            top5_predictions = sorted_preds[:5]
            top5_accuracy = sum(p['prediction'] for p in top5_predictions) / 5
        else:
            top5_accuracy = None

        # 置信度分布
        confidence_dist = {}
        for pred in valid_predictions:
            conf = pred['confidence']
            confidence_dist[conf] = confidence_dist.get(conf, 0) + 1

        # 概率分布统计
        probas = [p['probability_over_5pct'] for p in valid_predictions]
        proba_stats = {
            'mean': np.mean(probas),
            'std': np.std(probas),
            'min': np.min(probas),
            'max': np.max(probas),
            'median': np.median(probas)
        }

        results = {
            'total_predictions': len(predictions),
            'valid_predictions': len(valid_predictions),
            'positive_predictions': sum(y_pred),
            'positive_ratio': sum(y_pred) / len(y_pred) if y_pred else 0,
            'top5_accuracy': top5_accuracy,
            'confidence_distribution': confidence_dist,
            'probability_stats': proba_stats,
            'predictions': valid_predictions
        }

        print(f"总预测数: {results['total_predictions']}")
        print(f"有效预测数: {results['valid_predictions']}")
        print(f"正样本预测数: {results['positive_predictions']}")
        print(f"正样本比例: {results['positive_ratio']:.3f}")
        if top5_accuracy is not None:
            print(f"Top-5准确率: {top5_accuracy:.3f}")
        print(f"置信度分布: {confidence_dist}")
        print(f"平均概率: {proba_stats['mean']:.3f}")
        print(f"概率标准差: {proba_stats['std']:.3f}")
        print(f"最小概率: {proba_stats['min']:.3f}")
        print(f"最大概率: {proba_stats['max']:.3f}")
        print(f"中位数概率: {proba_stats['median']:.3f}")

        return results

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='滚动股票预测器')
    parser.add_argument('--predict-days', type=int, default=30, help='预测天数')
    parser.add_argument('--start-date', type=str, help='开始预测日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='结束预测日期 (YYYY-MM-DD)，配合predict-days=1使用')
    parser.add_argument('--majority-undersampling-ratio', type=float, default=0.1,
                       help='多数类样本下采样比例 (0-1之间，默认0.1)')
    parser.add_argument('--max-stocks', type=int, help='最大股票数量，用于调试 (默认使用全部股票)')

    args = parser.parse_args()

    predictor = RollingStockPredictor(
        majority_undersampling_ratio=args.majority_undersampling_ratio,
        max_stocks=args.max_stocks
    )

    try:
        # 执行滚动预测
        results = predictor.rolling_predict(
            start_date=args.start_date,
            end_date=args.end_date,
            n_days=args.predict_days
        )

        # 评估结果
        evaluation = predictor.evaluate_predictions(results)

        print("\n预测完成！")

    except Exception as e:
        print(f"执行失败: {e}")

if __name__ == "__main__":
    main()
