#!/usr/bin/env python
"""
股票短期收益率预测器
使用机器学习模型预测T+1和T+3日收益率是否超过5%
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler # 新增导入
from tqdm import tqdm
# import xgboost as xgb  # 暂时注释，需要安装OpenMP
# import lightgbm as lgb  # 暂时注释，需要OpenMP
from feature_engineering import StockFeatureEngineer
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """股票预测器"""

    def __init__(self, model_dir: str = "models", enable_feature_selection: bool = False):
        self.model_dir = os.path.join(model_dir, "tudata_models")
        self.feature_engineer = StockFeatureEngineer()
        self.scaler = StandardScaler()
        self.enable_feature_selection = enable_feature_selection
        
        if self.enable_feature_selection:
            self.feature_selector = None # 只有在启用特征选择时才初始化

        # 创建模型目录
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "t1"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "t3"), exist_ok=True)

    def prepare_training_data(self, data_dir: str = "daily", max_stocks: int = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        准备训练数据

        Args:
            data_dir: 数据目录
            max_stocks: 最大股票数量，用于测试

        Returns:
            (X_train, y_train) 元组
        """
        print("开始准备训练数据...")

        # 定义缓存文件路径
        cache_dir = os.path.join(self.model_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        X_cache_path = os.path.join(cache_dir, "processed_X.pkl")
        y_t1_cache_path = os.path.join(cache_dir, "processed_y_t1.pkl")
        y_t3_cache_path = os.path.join(cache_dir, "processed_y_t3.pkl")
        stock_codes_cache_path = os.path.join(cache_dir, "processed_stock_codes.pkl")
        trade_dates_cache_path = os.path.join(cache_dir, "processed_trade_dates.pkl")

        # 如果没有指定max_stocks，并且缓存文件存在，则直接加载
        if max_stocks is None and \
           os.path.exists(X_cache_path) and \
           os.path.exists(y_t1_cache_path) and \
           os.path.exists(y_t3_cache_path) and \
           os.path.exists(stock_codes_cache_path) and \
           os.path.exists(trade_dates_cache_path):
            print("✓ 检测到缓存数据，正在加载...")
            X = joblib.load(X_cache_path)
            y_t1 = joblib.load(y_t1_cache_path)
            y_t3 = joblib.load(y_t3_cache_path)
            stock_codes_df = joblib.load(stock_codes_cache_path)
            trade_dates = joblib.load(trade_dates_cache_path)
            print(f"✓ 缓存数据加载完成: {X.shape}")
            return X, y_t1, y_t3, stock_codes_df, trade_dates

        print("没有找到缓存数据或指定了max_stocks，开始完整数据准备...")

        all_data = []
        file_count = 0

        # 获取所有股票文件
        files = [f for f in os.listdir(data_dir) if f.endswith('_daily.csv')]
        if max_stocks:
            files = files[:max_stocks]

        print(f"处理 {len(files)} 只股票...")

        for file in tqdm(files, desc="准备训练数据"):
            file_path = os.path.join(data_dir, file)
            stock_code = file.replace('_daily.csv', '')

            try:
                # 处理单个股票的特征工程
                df = self.feature_engineer.process_single_stock(file_path)

                if not df.empty and len(df) > 30:  # 确保有足够的数据
                    df['stock_code'] = stock_code
                    all_data.append(df)
                    file_count += 1

                    # if file_count % 10 == 0:
                    #     print(f"已处理 {file_count} 只股票...")

            except Exception as e:
                print(f"处理股票 {stock_code} 失败: {e}")
                continue

        if not all_data:
            raise ValueError("没有找到有效的训练数据")

        # 合并所有股票数据
        combined_data = pd.concat(all_data, ignore_index=True)

        # 确保数据按交易日期和股票代码全局排序，以便时间序列拆分正确
        combined_data = combined_data.sort_values(by=['trade_date', 'stock_code']).reset_index(drop=True)

        print(f"训练数据准备完成: {combined_data.shape}")
        print(f"combined_data['trade_date'] is monotonic increasing: {combined_data['trade_date'].is_monotonic_increasing}")

        # 分离特征和目标
        feature_cols = self.feature_engineer.get_feature_columns()
        target_cols = ['TARGET_T1_OVER_5PCT', 'TARGET_T3_OVER_5PCT']

        # 确保所有特征列都存在
        available_features = [col for col in feature_cols if col in combined_data.columns]
        print(f"可用特征数量: {len(available_features)} / {len(feature_cols)}")

        # 创建训练数据
        X = combined_data[available_features].copy().reset_index(drop=True)
        y_t1 = combined_data['TARGET_T1_OVER_5PCT'].copy().reset_index(drop=True)
        y_t3 = combined_data['TARGET_T3_OVER_5PCT'].copy().reset_index(drop=True)
        stock_codes_series = combined_data['stock_code'].copy().reset_index(drop=True)
        trade_dates_series = combined_data['trade_date'].copy().reset_index(drop=True)

        # 处理缺失值
        X = X.fillna(method='ffill').fillna(0)

        # 移除inf值
        X = X.replace([np.inf, -np.inf], 0)

        print("数据清理完成:")
        print(f"  X形状: {X.shape}")
        print(f"  T+1目标分布: {y_t1.value_counts().to_dict()}")
        print(f"  T+3目标分布: {y_t3.value_counts().to_dict()}")

        # 如果没有指定max_stocks，则保存处理后的数据到缓存
        if max_stocks is None:
            print(f"✓ 正在保存处理后的训练数据到缓存: {cache_dir}")
            joblib.dump(X, X_cache_path)
            joblib.dump(y_t1, y_t1_cache_path)
            joblib.dump(y_t3, y_t3_cache_path)
            joblib.dump(stock_codes_series, stock_codes_cache_path)
            joblib.dump(trade_dates_series, trade_dates_cache_path) # 保存trade_date
            print("✓ 缓存保存完成")

        return X, y_t1, y_t3, stock_codes_series, trade_dates_series

    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series, desired_minority_to_majority_ratio: float = 1.0) -> Tuple[pd.DataFrame, pd.Series]:
        """使用SMOTE进行过采样，并可选地使用RandomUnderSampler进行欠采样，以达到指定的少数类与多数类比例"""
        print(f"原始类别分布: {y.value_counts().to_dict()}")

        # 找出多数类和少数类
        class_counts = y.value_counts()
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        n_minority = class_counts[minority_class]
        n_majority = class_counts[majority_class]

        # 检查是否有足够的样本进行SMOTE
        if n_minority < 6:  # SMOTE需要至少6个样本
            print("少数类样本数量不足，跳过SMOTE过采样")
            X_resampled, y_resampled = X, y
        else:
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, n_minority - 1))
                X_resampled, y_resampled = smote.fit_resample(X, y)
                print(f"SMOTE过采样后类别分布: {pd.Series(y_resampled).value_counts().to_dict()}")
            except Exception as e:
                print(f"SMOTE过采样失败: {e}，将使用原始数据")
                X_resampled, y_resampled = X, y
        
        # 根据目标比例进行欠采样
        current_class_counts = pd.Series(y_resampled).value_counts()
        current_minority_count = current_class_counts.get(minority_class, 0)
        current_majority_count = current_class_counts.get(majority_class, 0)

        if current_minority_count == 0:
            print("少数类样本为0，无法进行欠采样")
            return X_resampled, y_resampled

        # 计算目标多数类数量
        target_majority_count = int(current_minority_count / desired_minority_to_majority_ratio)
        
        if target_majority_count < current_majority_count:
            print(f"多数类样本数量 {current_majority_count} 超过目标 {target_majority_count}，进行欠采样...")
            # 创建一个只包含多数类的子集进行欠采样
            X_majority = X_resampled[y_resampled == majority_class]
            y_majority = y_resampled[y_resampled == majority_class]
            
            # 欠采样多数类
            undersampler = RandomUnderSampler(sampling_strategy={majority_class: target_majority_count}, random_state=42)
            X_undersampled_majority, y_undersampled_majority = undersampler.fit_resample(X_majority, y_majority)
            
            # 合并欠采样后的多数类和过采样后的少数类
            X_final = pd.concat([X_resampled[y_resampled == minority_class], X_undersampled_majority], axis=0)
            y_final = pd.concat([y_resampled[y_resampled == minority_class], y_undersampled_majority], axis=0)
            
            print(f"欠采样后类别分布: {y_final.value_counts().to_dict()}")
            return X_final, y_final
        else:
            print("无需进行欠采样")
            return X_resampled, y_resampled

    def select_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 30) -> Tuple[pd.DataFrame, List[str]]:
        """特征选择"""
        if not self.enable_feature_selection:
            print("特征选择已禁用，直接使用所有特征。")
            return X, X.columns.tolist()
        
        print(f"开始特征选择，目标特征数: {n_features}")

        try:
            # 使用随机森林进行特征选择
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42),
                max_features=n_features
            )
            selector.fit(X, y)

            # 获取选择的特征
            selected_features = X.columns[selector.get_support()].tolist()
            print(f"选择了 {len(selected_features)} 个特征")

            # 保存特征选择器
            self.feature_selector = selector

            return X[selected_features], selected_features

        except Exception as e:
            print(f"特征选择失败: {e}")
            return X, X.columns.tolist()

    def train_model(self, X: pd.DataFrame, y: pd.Series, trade_dates: pd.Series, target_name: str,
                   model_type: str = 'xgboost') -> Dict:
        """
        训练单个模型

        Args:
            X: 特征数据
            y: 目标变量
            target_name: 目标名称 ('t1' 或 't3')
            model_type: 模型类型

        Returns:
            训练结果字典
        """
        print(f"\n开始训练 {target_name} 模型 ({model_type})...")

        # 自定义时间序列交叉验证 - 确保训练集和验证集在时间上分离
        def custom_time_series_split(n_splits=3):
            """
            自定义时间序列拆分，确保训练集和验证集在时间上分离
            """
            unique_dates = trade_dates.sort_values().unique()
            n_dates = len(unique_dates)

            # 将日期分为 n_splits + 1 份，确保训练集逐步扩大，验证集在时间上靠后
            fold_size = n_dates // (n_splits + 1)

            for fold in range(n_splits):
                # 训练集：前 (fold + 1) * fold_size 个日期
                train_end_idx = (fold + 1) * fold_size
                train_dates_fold = unique_dates[:train_end_idx]

                # 验证集：接下来的 fold_size 个日期，确保在训练集时间之后
                val_start_idx = train_end_idx
                val_end_idx = min(val_start_idx + fold_size, n_dates)
                val_dates_fold = unique_dates[val_start_idx:val_end_idx]

                # 获取对应日期的行索引
                train_mask = trade_dates.isin(train_dates_fold)
                val_mask = trade_dates.isin(val_dates_fold)

                train_idx = trade_dates.index[train_mask]
                val_idx = trade_dates.index[val_mask]

                yield train_idx, val_idx

        # 使用自定义时间序列交叉验证
        tscv = custom_time_series_split

        # 模型选择
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            model = LogisticRegression(random_state=42)

        # 交叉验证
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(tqdm(tscv(), desc=f"交叉验证 {target_name} {model_type}", total=3)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_dates = trade_dates.iloc[train_idx]
            val_dates = trade_dates.iloc[val_idx]

            print(f"\nFold {fold + 1} 诊断信息:")
            print(f"  原始索引范围 - train_idx: [{train_idx.min()}, {train_idx.max()}], val_idx: [{val_idx.min()}, {val_idx.max()}]")
            print("  train_dates head:", train_dates[:5].dt.strftime('%Y-%m-%d').to_list())
            print("  train_dates tail:", train_dates[-5:].dt.strftime('%Y-%m-%d').to_list())
            print("  val_dates head:", val_dates[:5].dt.strftime('%Y-%m-%d').to_list())
            print("  val_dates tail:", val_dates[-5:].dt.strftime('%Y-%m-%d').to_list())
            print(f"  训练集时间范围: {train_dates.min().strftime('%Y-%m-%d')} - {train_dates.max().strftime('%Y-%m-%d')}")
            print(f"  验证集时间范围: {val_dates.min().strftime('%Y-%m-%d')} - {val_dates.max().strftime('%Y-%m-%d')}")
            print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"  y_train class distribution: {y_train.value_counts().to_dict()}")
            print(f"  X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            print(f"  y_val class distribution: {y_val.value_counts().to_dict()}")

            # --- 新增代码：打印单类别折叠的样本数据 ---
            if len(y_train.value_counts()) < 2 or len(y_val.value_counts()) < 2:
                print("  !!! 警告: 检测到单类别训练集或验证集，以下为样本数据 !!!")
                print("  y_train 样本:")
                print(y_train.head(5).to_string()) # 打印y_train的前5个样本，使用to_string()保持格式
                print("  y_val 样本:")
                print(y_val.head(5).to_string())   # 打印y_val的前5个样本
                print("  X_train (前5行前5列) 样本:")
                print(X_train.iloc[:5, :5].to_string()) # 打印X_train的前5行和前5列
                print("  X_val (前5行前5列) 样本:")
                print(X_val.iloc[:5, :5].to_string())   # 打印X_val的前5行和前5列
            # --- 新增代码结束 ---

            # 标准化
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            score = accuracy_score(y_val, y_pred)
            cv_scores.append(score)

        print(".4f")

        # 在全量数据上训练最终模型
        X_scaled = self.scaler.fit_transform(X)
        model.fit(X_scaled, y)

        # 保存模型
        model_path = os.path.join(self.model_dir, target_name, f"{model_type}_model.pkl")
        joblib.dump(model, model_path)
        joblib.dump(self.scaler, os.path.join(self.model_dir, target_name, "scaler.pkl"))

        if self.enable_feature_selection and hasattr(self, 'feature_selector') and self.feature_selector is not None: # 只有在启用特征选择且选择器存在时才保存
            joblib.dump(self.feature_selector, os.path.join(self.model_dir, target_name, "feature_selector.pkl"))

        # 存储模型信息
        self.models[target_name] = {
            'model': model,
            'model_type': model_type,
            'cv_score': np.mean(cv_scores),
            'features': X.columns.tolist(),
            'feature_selector': self.feature_selector # 保存特征选择器到模型信息中
        }

        return {
            'model_type': model_type,
            'cv_score': np.mean(cv_scores),
            'feature_count': len(X.columns),
            'model_path': model_path
        }

    def train_all_models(self, X: pd.DataFrame, y_t1: pd.Series, y_t3: pd.Series, trade_dates: pd.Series,
                        balance_classes: bool = False, desired_minority_to_majority_ratio: float = 1.0,
                        majority_undersampling_ratio: Optional[float] = None) -> Dict:
        """训练所有模型"""
        print("=" * 60)
        print("开始训练股票预测模型")
        print("=" * 60)

        results = {}

        for target_name, y in [('t1', y_t1), ('t3', y_t3)]:
            print(f"\n处理 {target_name} 目标...")

            X_current, y_current = X.copy(), y.copy()

            # 类别平衡 (SMOTE过采样和/或比例欠采样)
            if balance_classes:
                X_processed, y_processed = self.handle_class_imbalance(X_current, y_current, desired_minority_to_majority_ratio)
            else:
                X_processed, y_processed = X_current, y_current

            # 如果设置了majority_undersampling_ratio，则对多数类进行最终比例欠采样
            if majority_undersampling_ratio is not None and 0 < majority_undersampling_ratio <= 1.0:
                class_counts = y_processed.value_counts()
                majority_class = class_counts.idxmax()
                n_majority = class_counts[majority_class]

                # 计算目标多数类样本数量
                target_majority_count = int(n_majority * majority_undersampling_ratio)

                if target_majority_count < n_majority: # 只有当目标数量小于当前数量时才进行欠采样
                    print(f"最终多数类样本 ({n_majority}) 将按比例 {majority_undersampling_ratio} 欠采样至 {target_majority_count}...")
                    undersampler = RandomUnderSampler(sampling_strategy={majority_class: target_majority_count}, random_state=42)
                    X_final, y_final = undersampler.fit_resample(X_processed, y_processed)
                    print(f"最终欠采样后多数类分布: {pd.Series(y_final).value_counts().to_dict()}")
                    X_balanced, y_balanced = X_final, y_final
                else:
                    print(f"最终多数类样本 ({n_majority}) 未达到欠采样条件或比例不合理，无需额外欠采样")
                    X_balanced, y_balanced = X_processed, y_processed
            else:
                X_balanced, y_balanced = X_processed, y_processed

            # 特征选择
            if self.enable_feature_selection:
                X_selected, selected_features = self.select_features(X_balanced, y_balanced)
            else:
                X_selected, selected_features = X_balanced, X_balanced.columns.tolist()

            # 训练不同模型
            models_to_train = ['random_forest', 'gradient_boosting', 'logistic_regression']
            best_score = 0
            best_model = None

            for model_type in models_to_train:
                try:
                    result = self.train_model(X_selected, y_balanced, trade_dates.iloc[X_selected.index].reset_index(drop=True), target_name, model_type)
                    results[f"{target_name}_{model_type}"] = result

                    if result['cv_score'] > best_score:
                        best_score = result['cv_score']
                        best_model = model_type

                except Exception as e:
                    print(f"训练 {model_type} 模型失败: {e}")

            print(f"{target_name} 最佳模型: {best_model} (CV分数: {best_score:.4f})")

        return results

    def predict_single_stock(self, stock_code: str, target_days: int = 1) -> Dict:
        """
        预测单个股票

        Args:
            stock_code: 股票代码 (不含.SZ/.SH后缀)
            target_days: 预测天数 (1或3)

        Returns:
            预测结果字典
        """
        target_name = f"t{target_days}"

        # 检查模型是否存在
        model_path = os.path.join(self.model_dir, target_name, "gradient_boosting_model.pkl")
        scaler_path = os.path.join(self.model_dir, target_name, "scaler.pkl")
        selector_path = os.path.join(self.model_dir, target_name, "feature_selector.pkl")

        if not os.path.exists(model_path):
            return {"error": f"{target_name} 模型不存在，请先训练模型"}

        # 加载模型
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        selector = joblib.load(selector_path)

        # 查找股票数据文件
        data_file = None
        for ext in ['.SZ', '.SH']:
            candidate = f"{stock_code}{ext}_daily.csv"
            if os.path.exists(os.path.join("daily", candidate)):
                data_file = os.path.join("daily", candidate)
                break

        if not data_file:
            return {"error": f"找不到股票 {stock_code} 的数据文件"}

        # 处理特征工程
        df = self.feature_engineer.process_single_stock(data_file)

        if df.empty:
            return {"error": f"股票 {stock_code} 数据处理失败"}

        # 获取最新数据
        latest_data = df.iloc[-1:]

        # 准备特征
        # 预测时使用训练时选择的特征
        X_pred = latest_data[selector.get_feature_names_out()].fillna(method='ffill').fillna(0)
        X_pred = X_pred.replace([np.inf, -np.inf], 0)

        # 标准化
        X_pred_scaled = scaler.transform(X_pred)

        # 预测
        pred_proba = model.predict_proba(X_pred_scaled)[0]
        pred_class = model.predict(X_pred_scaled)[0]

        # 获取当前价格
        current_price = latest_data['close'].iloc[0]

        return {
            "stock_code": stock_code,
            "target_days": target_days,
            "current_price": current_price,
            "prediction": int(pred_class),
            "probability_over_5pct": float(pred_proba[1]),
            "probability_under_5pct": float(pred_proba[0]),
            "prediction_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "confidence": "high" if abs(pred_proba[1] - pred_proba[0]) > 0.3 else "medium" if abs(pred_proba[1] - pred_proba[0]) > 0.1 else "low"
        }

    def predict_multiple_stocks(self, stock_codes: List[str], target_days: int = 1) -> List[Dict]:
        """预测多个股票"""
        results = []

        print(f"开始预测 {len(stock_codes)} 只股票的 T+{target_days} 日收益率...")

        for stock_code in tqdm(stock_codes, desc=f"预测 T+{target_days} 日收益率"):
            try:
                result = self.predict_single_stock(stock_code, target_days)
                results.append(result)

                # if result.get('prediction') == 1:
                #     prob = result.get('probability_over_5pct', 0)
                #     print(f"{stock_code:<10} | 预测上涨 | 概率: {prob:.1%}")
            except Exception as e:
                results.append({
                    "stock_code": stock_code,
                    "error": str(e)
                })
        return results

    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict:
        """评估模型性能"""
        model_path = os.path.join(self.model_dir, target_name, "gradient_boosting_model.pkl")
        scaler_path = os.path.join(self.model_dir, target_name, "scaler.pkl")
        selector_path = os.path.join(self.model_dir, target_name, "feature_selector.pkl")

        if not os.path.exists(model_path):
            return {"error": "模型不存在"}

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        selector = joblib.load(selector_path)

        # 使用与训练时相同的特征选择
        X_selected = selector.transform(X)
        # 标准化
        X_scaled = scaler.transform(X_selected)

        # 预测
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]

        # 计算指标
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)

        # 分类报告
        report = classification_report(y, y_pred, output_dict=True)

        return {
            "accuracy": accuracy,
            "auc": auc,
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y, y_pred).tolist()
        }

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='股票短期收益率预测器')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--predict', nargs='*', help='预测股票列表')
    parser.add_argument('--target-days', type=int, choices=[1, 3], default=1,
                       help='预测天数 (1或3)')
    parser.add_argument('--max-stocks', type=int, help='训练时使用的最大股票数量')
    parser.add_argument('--evaluate', action='store_true', help='评估模型性能')
    parser.add_argument('--balance-ratio', type=float, default=1.0,
                       help='类别平衡时，少数类与多数类的比例 (例如：0.5 表示少数类是多数类的一半)')
    parser.add_argument('--no-balance', action='store_true', help='关闭类别平衡（即使设置了 --balance-ratio）')
    parser.add_argument('--majority-undersampling-ratio', type=float,
                       help='多数类样本的下采样比例（0-1之间，例如：0.1表示保留10%的多数类样本）')
    parser.add_argument('--enable-feature-selection', action='store_true', help='启用特征选择（默认关闭）')

    args = parser.parse_args()

    predictor = StockPredictor(enable_feature_selection=args.enable_feature_selection)

    if args.train:
        print("开始训练模型...")
        X, y_t1, y_t3, _, trade_dates = predictor.prepare_training_data(max_stocks=args.max_stocks)
        balance_classes_flag = not args.no_balance # 只有当用户没有指定--no-balance时才进行平衡
        results = predictor.train_all_models(X, y_t1, y_t3, trade_dates=trade_dates, balance_classes=balance_classes_flag, 
                                           desired_minority_to_majority_ratio=args.balance_ratio,
                                           majority_undersampling_ratio=args.majority_undersampling_ratio)
        print("\n训练完成!")
        for model_name, result in results.items():
            print(".4f")

    elif args.predict:
        if not args.predict:
            print("请指定要预测的股票代码")
            return

        results = predictor.predict_multiple_stocks(args.predict, args.target_days)

        print(f"\n预测结果 (T+{args.target_days} 日收益率超过5%):")
        print("=" * 60)

        positive_predictions = [r for r in results if r.get('prediction') == 1]

        if positive_predictions:
            print(f"发现 {len(positive_predictions)} 只可能上涨超过5%的股票:")
            for result in positive_predictions:
                if 'error' not in result:
                    prob = result['probability_over_5pct'] * 100
                    print("30")
        else:
            print("没有发现符合条件的股票")

    elif args.evaluate:
        print("评估模型性能...")
        X, y_t1, y_t3, _, trade_dates = predictor.prepare_training_data(max_stocks=args.max_stocks)  # 使用args.max_stocks来控制评估数据量

        for target_name, y in [('t1', y_t1), ('t3', y_t3)]:
            eval_result = predictor.evaluate_model(X, y, target_name)
            if 'error' not in eval_result:
                print(f"\n{target_name} 模型评估结果:")
                print(".4f")
                print(".4f")
            else:
                print(f"{target_name} 评估失败: {eval_result['error']}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
