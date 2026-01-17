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

    def __init__(self, model_dir: str = "models", majority_undersampling_ratio: float = 0.2, max_stocks: int = None, exclude_one_word_limit: bool = False):
        self.model_dir = os.path.join(model_dir, "rolling_models")
        self.cache_dir = os.path.join(self.model_dir, "cache")
        self.feature_engineer = StockFeatureEngineer()
        self.majority_undersampling_ratio = majority_undersampling_ratio
        self.max_stocks = max_stocks
        self.exclude_one_word_limit = exclude_one_word_limit

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
                df = self.feature_engineer.process_single_stock(file_path, exclude_one_word_limit=self.exclude_one_word_limit)

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

        # 分离特征和目标变量
        feature_cols = self.feature_engineer.get_feature_columns()
        target_cols = self.feature_engineer.get_target_columns()

        # 只选择有效的特征列
        available_features = [col for col in feature_cols if col in combined_data.columns]
        available_targets = [col for col in target_cols if col in combined_data.columns]

        if not available_targets:
            raise ValueError("没有找到有效的目标变量列")

        # 确保数据按日期和股票代码排序
        combined_data = combined_data.sort_values(by=['trade_date', 'stock_code']).reset_index(drop=True)

        X = combined_data[available_features].copy()
        y_t1 = combined_data['TARGET_T1_OVER_5PCT'].copy()

        # 移除包含NaN的行
        valid_rows = ~(X.isna().any(axis=1) | y_t1.isna())
        X = X[valid_rows].reset_index(drop=True)
        y_t1 = y_t1[valid_rows].reset_index(drop=True)

        print(f"数据预处理完成: {len(X)} 条有效样本")

        # 保存到缓存
        print(f"✓ 保存全量数据到缓存: {self.cache_dir}")
        os.makedirs(self.cache_dir, exist_ok=True)
        joblib.dump(X, X_cache_path)
        joblib.dump(y_t1, y_cache_path)
        print("✓ 全量数据缓存保存完成")

        # 根据当前的下采样比例进行动态下采样
        if self.majority_undersampling_ratio < 1.0:
            print(f"✓ 应用多数类下采样 (ratio={self.majority_undersampling_ratio})...")
            X, y_t1 = self._apply_majority_undersampling(X, y_t1)
            print(f"✓ 下采样完成: {X.shape}")

        return X, y_t1

    def _apply_majority_undersampling(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """应用多数类下采样"""
        if len(y.unique()) < 2:
            return X, y

        # 计算各类别样本数
        class_counts = y.value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()

        # 当ratio很小时，直接按ratio比例采样多数类
        if self.majority_undersampling_ratio < 0.1:
            target_majority_count = int(class_counts[majority_class] * self.majority_undersampling_ratio)
        else:
            target_majority_count = int(class_counts[minority_class] * (1 / self.majority_undersampling_ratio - 1))
        target_majority_count = min(target_majority_count, class_counts[majority_class])

        if target_majority_count < class_counts[majority_class]:
            # 使用随机下采样
            rus = RandomUnderSampler(sampling_strategy={majority_class: target_majority_count}, random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)

            print(f"下采样结果: {dict(y_resampled.value_counts())}")
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        else:
            return X, y

    def predict_single_day(self, train_cutoff_date: str, predict_date: str) -> Dict:
        """预测单日结果"""
        try:
            # 准备训练数据
            X_train, y_train = self.prepare_data_for_date(cutoff_date=train_cutoff_date)

            print(f"训练数据形状: {X_train.shape}")
            print(f"正样本比例: {y_train.mean():.3f}")

            # 训练模型
            model = GradientBoostingClassifier(
                n_estimators=200,        # 增加树的数量
                max_depth=5,            # 增加树深度，捕捉更复杂模式
                learning_rate=0.1,      # 降低学习率，提高稳定性
                subsample=0.8,          # 使用80%样本训练，防止过拟合
                min_samples_split=20,   # 内部节点最小样本数
                min_samples_leaf=10,    # 叶子节点最小样本数
                max_features='sqrt',    # 每次分裂考虑sqrt(n_features)个特征
                random_state=42,
                verbose=1  # 显示训练进度
            )

            model.fit(X_train, y_train)

            # 在训练集上进行评估
            y_train_pred = model.predict(X_train)
            y_train_proba = model.predict_proba(X_train)[:, 1]
            train_accuracy = accuracy_score(y_train, y_train_pred)
            try:
                train_auc = roc_auc_score(y_train, y_train_proba)
            except ValueError:
                train_auc = 0.5  # 如果只有一个类别，AUC无法计算
            
            print(f"训练集评估 - Accuracy: {train_accuracy:.3f}, AUC: {train_auc:.3f}")

            # 准备预测数据 - 只需要predict_date当天的数据
            predict_data = self.load_single_day_data(predict_date)
            print(f"predict_data: {predict_data}")
            if predict_data is None:
                return {"date": predict_date, "error": f"找不到 {predict_date} 的预测数据"}

            # 预测所有股票
            predict_features = predict_data.drop(columns=['stock_code'], errors='ignore')

            # 预测所有股票
            pred_proba_all = model.predict_proba(predict_features)
            pred_class_all = model.predict(predict_features)

            # 为每支股票创建预测结果
            # 从预测数据中获取真实的股票代码
            stock_predictions = []
            for i, (proba, pred_class) in enumerate(zip(pred_proba_all, pred_class_all)):
                # 从预测数据中获取股票代码
                stock_code = predict_data.iloc[i].get('stock_code', f"stock_{i}")
                stock_result = {
                    "stock_code": stock_code,
                    "prediction": int(pred_class),
                    "probability_over_5pct": float(proba[1]),
                    "probability_under_5pct": float(proba[0]),
                    "confidence": "high" if abs(proba[1] - proba[0]) > 0.3 else "medium" if abs(proba[1] - proba[0]) > 0.1 else "low"
                }
                stock_predictions.append(stock_result)

            result = {
                "date": predict_date,
                "train_cutoff": train_cutoff_date,
                "train_samples": len(X_train),
                "positive_ratio": float(y_train.mean()),
                "train_accuracy": float(train_accuracy),
                "train_auc": float(train_auc),
                "total_stocks": len(stock_predictions),
                "predictions": stock_predictions,
                # 为了向后兼容，保留单个股票的统计信息
                "positive_predictions": sum(p['prediction'] for p in stock_predictions),
                "avg_probability": sum(p['probability_over_5pct'] for p in stock_predictions) / len(stock_predictions)
            }

            # 打印上涨预测的股票（只显示Top10）
            positive_stocks = [(p['stock_code'], p['probability_over_5pct']) for p in stock_predictions if p['prediction'] == 1]
            if positive_stocks:
                sorted_positive = sorted(positive_stocks, key=lambda x: x[1], reverse=True)
                top_n = min(10, len(sorted_positive))
                print(f"预计上涨>5%的股票 (Top{top_n}/{len(positive_stocks)}只):")
                for code, prob in sorted_positive[:top_n]:
                    print(f"  {code}: {prob:.3f}")
                if len(positive_stocks) > 10:
                    print(f"  ... 还有 {len(positive_stocks) - 10} 只股票")
            else:
                print("没有股票预计上涨>5%")

            print(f"预测完成: {len(stock_predictions)} 只股票，平均上涨概率: {result['avg_probability']:.3f}")

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
                            # 重新添加stock_code列（因为特征工程可能重置了索引）
                            features = features.reset_index(drop=True)
                            features['stock_code'] = stock_code
                            all_predict_data.append(features)

            except Exception as e:
                print(e)
                continue

        if not all_predict_data:
            return None

        # 合并所有股票的预测数据
        predict_df = pd.concat(all_predict_data, ignore_index=True)

        # 选择特征列（排除目标列，但保留stock_code用于预测结果）
        feature_cols = self.feature_engineer.get_feature_columns()
        available_features = [col for col in feature_cols if col in predict_df.columns and not col.startswith('TARGET_')]

        # 保留stock_code列用于预测结果
        if 'stock_code' not in available_features:
            available_features.append('stock_code')

        X_pred = predict_df[available_features].copy()

        # 处理缺失值
        X_pred = X_pred.fillna(method='ffill').fillna(0)
        X_pred = X_pred.replace([np.inf, -np.inf], 0)

        # 保存到缓存（包含股票代码）
        print(f"✓ 保存预测数据到缓存: {self.cache_dir}")
        os.makedirs(self.cache_dir, exist_ok=True)
        joblib.dump(predict_df[available_features], cache_path)
        print("✓ 预测数据缓存保存完成")

        return X_pred

    def rolling_predict(self, start_date: str = None, n_days: int = 30, end_date: str = None) -> List[Dict]:
        """
        执行滚动预测

        Args:
            start_date: 开始预测日期 (YYYY-MM-DD)
            n_days: 预测天数
            end_date: 结束预测日期 (YYYY-MM-DD)，优先级高于n_days

        Returns:
            预测结果列表
        """
        # 确定预测日期范围
        if end_date and n_days > 1:
            # 同时指定了结束日期和天数：从结束日期向前n_days个工作日进行滚动预测
            end_dt = pd.to_datetime(end_date)
            # 生成从end_date向前n_days个工作日的日期序列
            predict_dates = []
            current_dt = end_dt
            count = 0
            while count < n_days:
                if current_dt.weekday() < 5:  # 工作日 (0-4: Mon-Fri)
                    predict_dates.append(current_dt)
                    count += 1
                current_dt = current_dt - pd.Timedelta(days=1)
            predict_dates.reverse()  # 确保按时间顺序
            print(f"滚动预测日期范围: {predict_dates[0].strftime('%Y-%m-%d')} 到 {predict_dates[-1].strftime('%Y-%m-%d')} ({len(predict_dates)}天)")
        elif end_date:
            # 只指定了结束日期：预测这一天
            end_dt = pd.to_datetime(end_date)
            predict_dates = [end_dt]
            print(f"预测指定日期: {end_date}")
        elif start_date:
            # 使用指定的开始日期和天数
            start_dt = pd.to_datetime(start_date)
            predict_dates = pd.date_range(start=start_dt, periods=n_days, freq='B')  # 工作日
            print(f"预测日期范围: {start_dt.strftime('%Y-%m-%d')} 到 {(start_dt + pd.Timedelta(days=n_days-1)).strftime('%Y-%m-%d')}")
        else:
            # 自动选择最近的交易日
            sample_files = [f for f in os.listdir("daily") if f.endswith('_daily.csv')]
            if not sample_files:
                raise ValueError("daily目录中没有找到CSV文件")

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

            # 增加实际收益率标签评估
            if result is not None and 'predictions' in result:
                eval_gt = self.evaluate_predictions_with_ground_truth(result['predictions'], predict_dt.strftime('%Y-%m-%d'), daily_dir="daily")
                if eval_gt:
                    result['real_metrics'] = eval_gt

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

        # 收集所有股票的预测结果
        all_stock_predictions = []
        for pred_result in valid_predictions:
            if 'predictions' in pred_result:
                # 新格式：包含多支股票的预测
                all_stock_predictions.extend(pred_result['predictions'])
            else:
                # 兼容旧格式：单个预测
                all_stock_predictions.append(pred_result)

        # 提取预测和概率
        y_pred = [p['prediction'] for p in all_stock_predictions]
        y_proba = [p['probability_over_5pct'] for p in all_stock_predictions]

        # 计算基础统计
        positive_predictions = sum(y_pred)

        print(f"总日期数: {len(predictions)}")
        print(f"有效日期数: {len(valid_predictions)}")
        print(f"总股票预测数: {len(all_stock_predictions)}")
        print(f"正样本预测数: {positive_predictions}")
        print(f"正样本比例: {positive_predictions / len(y_pred) if y_pred else 0:.3f}")

        # 打印上涨预测的股票
        positive_stocks = [p['stock_code'] for p in all_stock_predictions if p['prediction'] == 1]
        if positive_stocks:
            print(f"预计上涨>5%的股票 ({len(positive_stocks)}只): {', '.join(positive_stocks[:10])}")  # 只显示前10个
            if len(positive_stocks) > 10:
                print(f"  ... 还有 {len(positive_stocks) - 10} 只股票")
        else:
            print("没有股票预计上涨>5%")

        # 计算基础指标
        accuracy = sum(y_pred) / len(y_pred) if y_pred else 0  # 正样本比例

        # Top-5 准确率（概率最高的5个股票预测中实际上涨的比例）
        if len(all_stock_predictions) >= 5:
            # 按概率排序，取前5个
            sorted_preds = sorted(all_stock_predictions, key=lambda x: x['probability_over_5pct'], reverse=True)
            top5_predictions = sorted_preds[:5]
            top5_accuracy = sum(p['prediction'] for p in top5_predictions) / 5
        else:
            top5_accuracy = None

        # 置信度分布
        confidence_dist = {}
        for pred in all_stock_predictions:
            conf = pred['confidence']
            confidence_dist[conf] = confidence_dist.get(conf, 0) + 1

        # 概率分布统计
        probas = [p['probability_over_5pct'] for p in all_stock_predictions]
        proba_stats = {
            'mean': np.mean(probas),
            'std': np.std(probas),
            'min': np.min(probas),
            'max': np.max(probas),
            'median': np.median(probas)
        }

        eval_results = {
            'total_dates': len(predictions),
            'valid_dates': len(valid_predictions),
            'total_stocks': len(all_stock_predictions),
            'positive_predictions': sum(y_pred),
            'positive_ratio': sum(y_pred) / len(y_pred) if y_pred else 0,
            'top5_accuracy': top5_accuracy,
            'confidence_distribution': confidence_dist,
            'probability_stats': proba_stats,
            'predictions': valid_predictions,
            'all_stock_predictions': all_stock_predictions
        }

        print(f"总日期数: {eval_results['total_dates']}")
        print(f"有效日期数: {eval_results['valid_dates']}")
        print(f"总股票预测数: {eval_results['total_stocks']}")
        print(f"正样本预测数: {eval_results['positive_predictions']}")
        print(f"正样本比例: {eval_results['positive_ratio']:.3f}")

        if top5_accuracy is not None:
            print(f"Top-5准确率: {top5_accuracy:.3f}")
        print(f"置信度分布: {confidence_dist}")
        print(f"平均概率: {proba_stats['mean']:.3f}")
        print(f"概率标准差: {proba_stats['std']:.3f}")
        print(f"最小概率: {proba_stats['min']:.3f}")
        print(f"最大概率: {proba_stats['max']:.3f}")
        print(f"中位数概率: {proba_stats['median']:.3f}")

        return eval_results

    def evaluate_predictions_with_ground_truth(self, predictions: List[Dict], predict_date: str, daily_dir: str = "daily") -> Dict:
        """
        根据实际T+1收益标签，评估模型对T+1日的预测效果
        预测使用截至T日的数据，预测T+1日是否上涨>5%，评估使用T+1日的实际涨幅

        Args:
            predictions: List[Dict] 模型预测结果（对T+1日的预测）
            predict_date: 预测日期字符串（"YYYY-MM-DD"），如"2026-01-06"
            daily_dir: 行情csv目录
        Returns:
            dict, 包含准确率、精确率、召回率、F1、混淆矩阵等
        """
        import pandas as pd
        import os
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

        date_str = predict_date.replace('-', '')
        # 收集每个股票的预测和真实收益
        pred_labels, real_labels, codes, probs = [], [], [], []
        # 组织预测结果为dict
        pred_dict = {p['stock_code']: p for p in predictions if 'stock_code' in p}
        for file in os.listdir(daily_dir):
            if file.endswith('_daily.csv'):
                code = file.replace('_daily.csv', '')
                df = pd.read_csv(os.path.join(daily_dir, file))
                df['trade_date'] = df['trade_date'].astype(str)
                # 找到predict_date和predict_date+1行数据（用于验证对T+1日的预测）
                d_today = df[df['trade_date'] == date_str]
                d_next = df[df['trade_date'] == self.get_next_trade_date(df, date_str)]
                if not d_today.empty and not d_next.empty and code in pred_dict:
                    close_today = d_today.iloc[0]['close']
                    close_next = d_next.iloc[0]['close']
                    # 计算T+1日收益率（验证对明天的预测）
                    real_return = (close_next / close_today) - 1
                    real_label = 1 if real_return > 0.05 else 0
                    pred = pred_dict[code]['prediction']
                    prob = pred_dict[code]['probability_over_5pct']
                    pred_labels.append(pred)
                    real_labels.append(real_label)
                    codes.append(code)
                    probs.append(prob)

        metrics = {}
        if pred_labels and real_labels:
            metrics["准确率"] = accuracy_score(real_labels, pred_labels)
            metrics["精确率"] = precision_score(real_labels, pred_labels, zero_division=0)
            metrics["召回率"] = recall_score(real_labels, pred_labels, zero_division=0)
            metrics["F1"] = f1_score(real_labels, pred_labels, zero_division=0)
            
            if len(set(real_labels)) > 1:
                try:
                    metrics["AUC"] = roc_auc_score(real_labels, probs)
                except ValueError:
                    metrics["AUC"] = 0.5
            else:
                metrics["AUC"] = 0.5
                
            metrics["分类报告"] = classification_report(real_labels, pred_labels, zero_division=0, target_names=["<=5%", ">5%"], labels=[0, 1])
            metrics["混淆矩阵"] = confusion_matrix(real_labels, pred_labels, labels=[0, 1]).tolist()

            # 新增Top-N准确率计算（预测为正的分数从高到低排序，前N个预测正确的比例）
            positive_predictions = [(p['stock_code'], p['probability_over_5pct'], pred_dict.get(p['stock_code'], {}).get('prediction', 0)) for p in predictions if p['stock_code'] in pred_dict and pred_dict[p['stock_code']]['prediction'] == 1]
            if positive_predictions:
                # 按概率排序，取前20个（或全部如果小于20）
                top_n = min(20, len(positive_predictions))
                sorted_positive = sorted(positive_predictions, key=lambda x: x[1], reverse=True)[:top_n]

                correct_count = 0
                top_details = []
                for code, prob, pred in sorted_positive:
                    # 从real_labels中找到对应的真实标签和实际涨幅
                    real_label = 0
                    actual_return_pct = 0.0
                    if code in codes:
                        idx = codes.index(code)
                        real_label = real_labels[idx]

                        # 计算实际涨幅百分比：需要重新从数据中提取
                        # 找到对应的股票数据
                        stock_file = os.path.join(daily_dir, f"{code}_daily.csv")
                        if os.path.exists(stock_file):
                            stock_df = pd.read_csv(stock_file)
                            stock_df['trade_date'] = stock_df['trade_date'].astype(str)
                            d_today_data = stock_df[stock_df['trade_date'] == date_str]
                            d_next_data = stock_df[stock_df['trade_date'] == self.get_next_trade_date(stock_df, date_str)]

                            if not d_today_data.empty and not d_next_data.empty:
                                close_today = d_today_data.iloc[0]['close']
                                close_next = d_next_data.iloc[0]['close']
                                actual_return_pct = ((close_next / close_today) - 1) * 100

                    if real_label == 1:  # T+1日实际也上涨了>5%
                        correct_count += 1

                    top_details.append((code, prob, pred, real_label, actual_return_pct))

                top_n_accuracy = correct_count / top_n if top_n > 0 else 0
                metrics["Top-20准确率"] = top_n_accuracy
                metrics["Top-20详情"] = top_details

                print(f"Top-20准确率: {top_n_accuracy:.3f} (预测T+1日上涨概率最高的前{min(20, len(positive_predictions))}只股票中，T+1日实际涨幅>5%的比例)")
                print("\nTop-20预测详情:")
                print("排名 股票代码   预测概率  预测结果  实际涨幅>5%  实际涨幅(%)")
                print("-" * 70)
                for i, (code, prob, pred, real_label, actual_return) in enumerate(top_details, 1):
                    status = "✓" if real_label == 1 else "✗"
                    pred_result = "上涨>5%" if pred == 1 else "下跌<=5%"
                    print(f"{i:2d}  {code:<10} {prob:.3f}     {pred_result:<10} {status:<12} {actual_return:+.2f}%")
            else:
                metrics["Top-20准确率"] = 0
                print("Top-20准确率: N/A (没有预测为上涨的股票)")

            print("\n===== 真实T+1收益标签评估 =====")
            print(f"总样本数: {len(pred_labels)}")
            print(f"准确率: {metrics['准确率']:.3f}")
            print(f"精确率: {metrics['精确率']:.3f}")
            print(f"召回率: {metrics['召回率']:.3f}")
            print(f"F1分数: {metrics['F1']:.3f}")
            print(f"AUC: {metrics['AUC']:.3f}")
            print(f"混淆矩阵(n=0是<=5%，n=1是>5%): ")
            print(metrics['混淆矩阵'])

        return metrics

    @staticmethod
    def get_last_trade_date(df: pd.DataFrame, date_str: str) -> str:
        all_dates = sorted(df['trade_date'].unique())
        if date_str in all_dates:
            idx = all_dates.index(date_str)
            if idx >= 1:
                return all_dates[idx-1]
        return None

    @staticmethod
    def get_next_trade_date(df: pd.DataFrame, date_str: str) -> str:
        all_dates = sorted(df['trade_date'].unique())
        if date_str in all_dates:
            idx = all_dates.index(date_str)
            if idx < len(all_dates) - 1:
                return all_dates[idx+1]
        return None

    def _cleanup_old_prediction_files(self, results_dir: str, max_files: int = 20):
        """
        清理旧的预测结果文件，保持最多max_files个文件

        Args:
            results_dir: 结果文件目录
            max_files: 最大文件数量
        """
        import glob

        # 获取所有预测结果文件
        json_files = glob.glob(os.path.join(results_dir, "predictions_*.json"))
        csv_files = glob.glob(os.path.join(results_dir, "predictions_*.csv"))

        # 合并所有文件，按修改时间排序
        all_files = json_files + csv_files
        all_files.sort(key=lambda x: os.path.getmtime(x))

        # 如果文件数量超过限制，删除最旧的文件
        if len(all_files) > max_files:
            files_to_delete = all_files[:len(all_files) - max_files]  # 保留最新的max_files个，删除其余的
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"✓ 删除旧预测文件: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"删除文件失败 {file_path}: {e}")

    def save_predictions(self, results: List[Dict], evaluation: Dict):
        """
        保存预测结果到文件（最多保留20个文件）

        Args:
            results: 预测结果列表
            evaluation: 评估结果字典
        """
        import json
        from datetime import datetime

        # 创建结果目录
        results_dir = os.path.join(self.model_dir, "prediction_results")
        os.makedirs(results_dir, exist_ok=True)

        # 清理旧文件，保持最多20个文件
        self._cleanup_old_prediction_files(results_dir, max_files=20)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = os.path.join(results_dir, f"predictions_{timestamp}.json")
        csv_file = os.path.join(results_dir, f"predictions_{timestamp}.csv")

        # 准备数据用于保存
        all_predictions = []
        for date_result in results:
            if 'error' not in date_result and 'predictions' in date_result:
                for stock_pred in date_result['predictions']:
                    prediction_record = {
                        'date': date_result['date'],
                        'stock_code': stock_pred['stock_code'],
                        'prediction': stock_pred['prediction'],
                        'probability_over_5pct': stock_pred['probability_over_5pct'],
                        'probability_under_5pct': stock_pred['probability_under_5pct'],
                        'confidence': stock_pred['confidence'],
                        'train_cutoff': date_result.get('train_cutoff'),
                        'train_samples': date_result.get('train_samples'),
                        'positive_ratio': date_result.get('positive_ratio')
                    }
                    all_predictions.append(prediction_record)

        # 保存为JSON
        result_data = {
            'metadata': {
                'timestamp': timestamp,
                'total_dates': len(results),
                'total_stocks': len(all_predictions),
                'evaluation': evaluation
            },
            'predictions': all_predictions
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        # 保存为CSV
        if all_predictions:
            df = pd.DataFrame(all_predictions)
            df.to_csv(csv_file, index=False, encoding='utf-8')

        print(f"\n预测结果已保存:")
        print(f"  JSON: {json_file}")
        print(f"  CSV: {csv_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='A股股票预测器')
    parser.add_argument('--end-date', type=str, help='预测结束日期 (YYYY-MM-DD)')
    parser.add_argument('--predict-days', type=int, default=1, help='预测天数')
    parser.add_argument('--max-stocks', type=int, help='限制股票数量用于调试')
    parser.add_argument('--majority-undersampling-ratio', type=float, default=0.2, help='多数类下采样比例 (建议0.1-0.3)')
    parser.add_argument('--exclude-one-word-limit', action='store_true', default=False, help='排除一字板股票的涨幅标签')
    parser.add_argument('--include-one-word-limit', action='store_false', dest='exclude_one_word_limit', help='包含一字板股票的涨幅标签')

    args = parser.parse_args()

    # 创建预测器
    predictor = RollingStockPredictor(
        majority_undersampling_ratio=args.majority_undersampling_ratio,
        max_stocks=args.max_stocks,
        exclude_one_word_limit=args.exclude_one_word_limit
    )

    # 执行滚动预测
    results = predictor.rolling_predict(
        end_date=args.end_date,
        n_days=args.predict_days
    )

    # 评估结果
    evaluation = predictor.evaluate_predictions(results)

    # 保存预测结果
    predictor.save_predictions(results, evaluation)

    print("\n预测完成！")


if __name__ == "__main__":
    main()
