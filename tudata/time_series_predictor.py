
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_engineering import StockFeatureEngineer
import json
import argparse
import warnings
warnings.filterwarnings('ignore')
import time

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()

class StockDataset(Dataset):
    """股票时间序列数据集"""
    def __init__(self, X, y=None, stock_indices=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.stock_indices = torch.LongTensor(stock_indices) if stock_indices is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            if self.stock_indices is not None:
                return self.X[idx], self.y[idx], self.stock_indices[idx]
            return self.X[idx], self.y[idx]
        
        if self.stock_indices is not None:
            return self.X[idx], self.stock_indices[idx]
        return self.X[idx]

class LSTMModel(nn.Module):
    """LSTM股票预测模型"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, num_stocks=None, embedding_dim=16):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Stock Embedding
        self.num_stocks = num_stocks
        self.embedding_dim = embedding_dim if num_stocks else 0
        if self.num_stocks:
            self.stock_embedding = nn.Embedding(num_stocks, self.embedding_dim)
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        # FC input dim = hidden_dim + embedding_dim
        fc_input_dim = hidden_dim + self.embedding_dim
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, stock_idx=None):
        # x shape: (batch_size, seq_len, input_dim)
        # h0, c0 shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # out shape: (batch_size, seq_len, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        # out[:, -1, :] shape: (batch_size, hidden_dim)
        out = out[:, -1, :]
        
        if self.num_stocks and stock_idx is not None:
            # stock_idx shape: (batch_size,)
            emb = self.stock_embedding(stock_idx) # (batch_size, embedding_dim)
            out = torch.cat([out, emb], dim=1)
            
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    """Transformer股票预测模型"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2, num_stocks=None, embedding_dim=16):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        # Stock Embedding
        self.num_stocks = num_stocks
        self.embedding_dim = embedding_dim if num_stocks else 0
        if self.num_stocks:
            self.stock_embedding = nn.Embedding(num_stocks, self.embedding_dim)
        
        # 输入线性变换
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # FC input dim = d_model + embedding_dim
        fc_input_dim = d_model + self.embedding_dim
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, stock_idx=None):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 添加位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码器
        # 注意：PyTorch的Transformer需要src_mask来处理padding，这里我们假设所有序列长度相同
        transformer_out = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # 全局平均池化
        pooled = torch.mean(transformer_out, dim=1)  # (batch_size, d_model)
        
        if self.num_stocks and stock_idx is not None:
            emb = self.stock_embedding(stock_idx)
            pooled = torch.cat([pooled, emb], dim=1)
            
        # 分类
        output = self.classifier(pooled)  # (batch_size, 1)
        return output

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class TimeSeriesStockPredictor:
    """基于时间序列的股票预测器"""

    def __init__(self, model_dir: str = "models", window_size: int = 30, 
                 hidden_dim: int = 64, num_layers: int = 2, 
                 batch_size: int = 64, epochs: int = 10, learning_rate: float = 0.001,
                 max_stocks: int = None, exclude_one_word_limit: bool = False,
                 model_type: str = "LSTM", d_model: int = 64, nhead: int = 4,
                 neg_ratio: int = 8, rank_loss_weight: float = 0.2):
        self.model_dir = os.path.join(model_dir, "ts_models")
        self.cache_dir = os.path.join(self.model_dir, "cache")
        self.scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        self.stock_map_path = os.path.join(self.model_dir, "stock2idx.json")
        self.feature_engineer = StockFeatureEngineer()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_stocks = max_stocks
        self.exclude_one_word_limit = exclude_one_word_limit
        self.model_type = model_type.upper()
        self.d_model = d_model
        self.nhead = nhead
        self.neg_ratio = neg_ratio
        self.rank_loss_weight = rank_loss_weight
        
        # 股票代码映射
        self.stock2idx = {'<UNK>': 0}
        
        # 创建目录
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def prepare_data_for_date(self, data_dir: str = "daily", cutoff_date: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        准备截止到指定日期之前的所有训练数据 (序列格式)
        Returns:
            X: 特征数据
            y: 标签数据
            S: 股票ID数据
        """
        # 生成缓存文件名
        cutoff_str = cutoff_date.replace('-', '') if cutoff_date else 'all'
        stocks_str = f"_{self.max_stocks}" if self.max_stocks else "_all"
        cache_prefix = f"ts_data_{cutoff_str}{stocks_str}_w{self.window_size}"
        X_cache_path = os.path.join(self.cache_dir, f"{cache_prefix}_X.pkl")
        y_cache_path = os.path.join(self.cache_dir, f"{cache_prefix}_y.pkl")
        S_cache_path = os.path.join(self.cache_dir, f"{cache_prefix}_S.pkl")

        # 检查缓存
        cache_valid = False
        if os.path.exists(X_cache_path) and os.path.exists(y_cache_path) and os.path.exists(S_cache_path) and os.path.exists(self.stock_map_path):
            print(f"✓ 检测到序列缓存数据 ({cutoff_date})，正在加载...")
            try:
                X = joblib.load(X_cache_path)
                
                # 检查特征维度
                current_features = self.feature_engineer.get_feature_columns()
                if X.shape[2] != len(current_features):
                    print(f"⚠ 缓存特征维度 ({X.shape[2]}) 与当前配置 ({len(current_features)}) 不匹配，将重新生成...")
                else:
                    y = joblib.load(y_cache_path)
                    S = joblib.load(S_cache_path)
                    
                    # 加载股票映射
                    with open(self.stock_map_path, 'r') as f:
                        self.stock2idx = json.load(f)
                        
                    print(f"✓ 序列缓存加载完成: X={X.shape}, y={y.shape}, S={S.shape}")
                    print(f"✓ 股票映射加载完成: {len(self.stock2idx)} 只股票")
                    cache_valid = True
            except Exception as e:
                print(f"⚠ 加载缓存失败: {e}")

        if cache_valid:
            # 强制转换为 float32 以节省一半内存
            X = X.astype(np.float32)
            
            # 标准化处理 (使用采样优化)
            if not os.path.exists(self.scaler_path):
                print("正在进行数据标准化处理 (Cache/Sampling)...")
                N, T, F = X.shape
                
                import time
                t0 = time.time()
                
                # 1. 采样计算均值和方差
                sample_size = min(100000, N)
                indices = np.random.choice(N, sample_size, replace=False)
                X_sample = X[indices].reshape(-1, F)
                
                scaler = StandardScaler(copy=False)
                scaler.fit(X_sample)
                joblib.dump(scaler, self.scaler_path)
                print(f"  统计特征耗时: {time.time()-t0:.2f}s")
            else:
                scaler = joblib.load(self.scaler_path)
            
            # 应用变换
            print("正在应用标准化变换...")
            mean = scaler.mean_.astype(np.float32)
            scale = scaler.scale_.astype(np.float32)
            X = (X - mean) / scale
            
            return X, y, S

        print(f"准备截止到 {cutoff_date} 的序列训练数据...")
        all_X = []
        all_y = []
        all_S = [] # Stock Indices
        
        # 重置 stock2idx
        self.stock2idx = {'<UNK>': 0}
        
        data_dir_resolved = data_dir
        if not os.path.isabs(data_dir_resolved):
            if not os.path.isdir(data_dir_resolved):
                script_dir = os.path.dirname(__file__)
                candidates = [
                    os.path.join(script_dir, data_dir_resolved),
                    os.path.join(os.path.dirname(script_dir), "data", data_dir_resolved),
                ]
                for cand in candidates:
                    if os.path.isdir(cand):
                        data_dir_resolved = cand
                        break
        if not os.path.isdir(data_dir_resolved):
            raise FileNotFoundError(f"数据目录不存在: {data_dir_resolved}")
        files = [f for f in os.listdir(data_dir_resolved) if f.endswith('_daily.csv')]
        
        # 限制股票数量
        if self.max_stocks and self.max_stocks < len(files):
            files = files[:self.max_stocks]
            print(f"调试模式：仅处理前 {self.max_stocks} 只股票")
            
        # 内存优化：分块保存机制
        # 创建临时目录
        temp_chunk_dir = os.path.join(self.cache_dir, "temp_chunks")
        if os.path.exists(temp_chunk_dir):
            import shutil
            shutil.rmtree(temp_chunk_dir)
        os.makedirs(temp_chunk_dir, exist_ok=True)
        
        current_chunk_X = []
        current_chunk_y = []
        current_chunk_S = []
        chunk_files = []
        total_samples = 0
        chunk_size = 100 # 每100只股票保存一次
        
        processed_count = 0
            
        for file in tqdm(files, desc=f"构建序列数据"):
            file_path = os.path.join(data_dir_resolved, file)
            # 解析股票代码
            stock_code = file.replace('_daily.csv', '')
            
            # 分配 ID
            if stock_code not in self.stock2idx:
                self.stock2idx[stock_code] = len(self.stock2idx)
            stock_idx = self.stock2idx[stock_code]
            
            try:
                # 处理单个股票
                df = self.feature_engineer.process_single_stock(file_path, exclude_one_word_limit=self.exclude_one_word_limit)
                
                if not df.empty and len(df) > self.window_size:
                    # 过滤日期
                    if cutoff_date:
                        df = df[df['trade_date'] <= cutoff_date]
                    
                    if len(df) > self.window_size:
                        # 创建序列数据
                        X_seq, y_seq = self.feature_engineer.create_sequence_data(
                            df, 
                            window_size=self.window_size,
                            target_col='TARGET_T1_OVER_5PCT'
                        )
                        
                        if len(X_seq) > 0:
                            # 立即转换为 float32 节省内存
                            X_seq = X_seq.astype(np.float32)
                            y_seq = y_seq.astype(np.float32)
                            S_seq = np.full(len(X_seq), stock_idx, dtype=np.int64)
                            
                            current_chunk_X.append(X_seq)
                            current_chunk_y.append(y_seq)
                            current_chunk_S.append(S_seq)
                            
                            total_samples += len(X_seq)
                            processed_count += 1
                            
                            # 达到 Chunk 大小，保存并释放内存
                            if processed_count % chunk_size == 0:
                                chunk_path = os.path.join(temp_chunk_dir, f"chunk_{len(chunk_files)}.pkl")
                                chunk_data = {
                                    'X': np.concatenate(current_chunk_X, axis=0),
                                    'y': np.concatenate(current_chunk_y, axis=0),
                                    'S': np.concatenate(current_chunk_S, axis=0)
                                }
                                joblib.dump(chunk_data, chunk_path)
                                chunk_files.append(chunk_path)
                                
                                # 清空当前 chunk 列表并强制回收
                                current_chunk_X = []
                                current_chunk_y = []
                                current_chunk_S = []
                                import gc
                                gc.collect()
                            
            except Exception as e:
                # print(f"处理失败: {e}")
                continue
                
        # 保存最后一个 chunk
        if current_chunk_X:
            chunk_path = os.path.join(temp_chunk_dir, f"chunk_{len(chunk_files)}.pkl")
            chunk_data = {
                'X': np.concatenate(current_chunk_X, axis=0),
                'y': np.concatenate(current_chunk_y, axis=0),
                'S': np.concatenate(current_chunk_S, axis=0)
            }
            joblib.dump(chunk_data, chunk_path)
            chunk_files.append(chunk_path)
            current_chunk_X = []
            current_chunk_y = []
            current_chunk_S = []
        
        if total_samples == 0:
            raise ValueError(f"截止到 {cutoff_date} 没有找到有效的训练数据")
            
        print(f"数据构建完成，共 {total_samples} 个样本，正在合并 {len(chunk_files)} 个分块...")
        
        # 预分配最终的大数组 (使用 float32)
        # 获取特征维度
        sample_chunk = joblib.load(chunk_files[0])
        _, window_size, feature_dim = sample_chunk['X'].shape
        
        X = np.zeros((total_samples, window_size, feature_dim), dtype=np.float32)
        y = np.zeros((total_samples,), dtype=np.float32)
        S = np.zeros((total_samples,), dtype=np.int64)
        
        # 填充大数组
        start_idx = 0
        for chunk_path in tqdm(chunk_files, desc="合并分块"):
            chunk_data = joblib.load(chunk_path)
            chunk_len = len(chunk_data['X'])
            end_idx = start_idx + chunk_len
            
            X[start_idx:end_idx] = chunk_data['X']
            y[start_idx:end_idx] = chunk_data['y']
            S[start_idx:end_idx] = chunk_data['S']
            
            start_idx = end_idx
            
            # 删除处理完的 chunk 文件以释放磁盘空间 (可选)
            # os.remove(chunk_path)
            
        # 清理临时目录
        import shutil
        shutil.rmtree(temp_chunk_dir)
        
        # 保存缓存
        print(f"✓ 保存序列数据到缓存: {self.cache_dir}")
        joblib.dump(X, X_cache_path)
        joblib.dump(y, y_cache_path)
        joblib.dump(S, S_cache_path)
        
        # 保存股票映射
        with open(self.stock_map_path, 'w') as f:
            json.dump(self.stock2idx, f)
        
        # 标准化处理 (使用采样优化)
        print("正在进行数据标准化处理 (Sampling)...")
        N, T, F = X.shape
        print(f"  数据规模: {N} 样本 x {T} 时间步 x {F} 特征 = {N*T*F} 元素")
        
        import time
        t0 = time.time()
        
        # 1. 采样计算均值和方差
        sample_size = min(100000, N)
        indices = np.random.choice(N, sample_size, replace=False)
        X_sample = X[indices].reshape(-1, F)
        
        scaler = StandardScaler(copy=False)
        scaler.fit(X_sample)
        print(f"  统计特征耗时 (基于 {sample_size} 样本): {time.time()-t0:.2f}s")
        
        # 2. 应用变换
        t1 = time.time()
        mean = scaler.mean_.astype(np.float32)
        scale = scaler.scale_.astype(np.float32)
        
        # 利用广播机制直接在 (N, T, F) 上操作，避免 reshape 内存开销
        X = (X - mean) / scale
        print(f"  全量变换耗时: {time.time()-t1:.2f}s")
        
        joblib.dump(scaler, self.scaler_path)
        
        print(f"数据准备完成: X={X.shape}, y={y.shape}, S={S.shape}")
        return X, y, S

    def train_model(self, X_train, y_train, S_train=None, predict_date=None):
        """训练模型（支持LSTM和Transformer），全量数据训练，无验证集"""
        # 全量训练，不划分验证集
        X_train_sub = X_train
        y_train_sub = y_train
        S_train_sub = S_train
        
        input_dim = X_train.shape[2]
        
        # 确定股票数量 (用于 Embedding)
        num_stocks = len(self.stock2idx) if self.stock2idx else None
        if S_train is not None and num_stocks is None:
             num_stocks = S_train.max() + 1
        
        # 打印训练数据详细信息
        print(f"\n{'='*50}")
        print(f"数据详情 (全量训练):")
        print(f"  训练集样本: {X_train_sub.shape[0]}")
        print(f"  特征维度: {input_dim}")
        print(f"  正样本比例: {np.mean(y_train_sub):.3f}")
        if S_train_sub is not None:
            print(f"  股票Embedding: True (Num Stocks: {num_stocks})")
        print(f"{'='*50}")
        
        # 小样本模式下禁用 Dropout
        dropout_rate = 0.0 if (self.max_stocks is not None and self.max_stocks <= 50) else 0.2
        if dropout_rate == 0.0:
            print("小样本模式：禁用 Dropout")

        if self.model_type == "LSTM":
            model = LSTMModel(input_dim, self.hidden_dim, self.num_layers, dropout=dropout_rate, num_stocks=num_stocks).to(device)
            print(f"\n开始训练 LSTM 模型 (Device: {device})...")
        elif self.model_type == "TRANSFORMER":
            model = TransformerModel(input_dim, self.d_model, self.nhead, self.num_layers, dropout=dropout_rate, num_stocks=num_stocks).to(device)
            print(f"\n开始训练 Transformer 模型 (Device: {device})...")
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 计算正样本权重 (用于处理类别不平衡)
        pos_count = np.sum(y_train_sub)
        neg_count = len(y_train_sub) - pos_count
        pos_weight_val = neg_count / (pos_count + 1e-5) # 简单的反比权重
        
        # 使用 Focal Loss 应对类别不平衡
        # alpha=0.25 意味着降低负样本（多数类）的权重
        # gamma=2 意味着更关注难以分类的样本
        # criterion = FocalLoss(alpha=0.25, gamma=2)
        
        # 使用带权重的 BCELoss
        # 注意：BCELoss 的 weight 参数是 batch 级的，这里我们使用 reduction='none' 手动加权
        # 或者更简单：如果下采样已经平衡了数据，weight 可以设为 1
        # 如果未下采样（小样本模式），则使用计算出的 pos_weight
        
        current_criterion = nn.BCELoss(reduction='none')
        rank_criterion = nn.MarginRankingLoss(margin=0.2) # Ranking Loss
        
        # criterion = nn.BCELoss() # 回退到 BCELoss 以测试过拟合能力
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # 创建DataLoader
        train_dataset = StockDataset(X_train_sub, y_train_sub, stock_indices=S_train_sub)
        # val_dataset = StockDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 准备测试集数据 (如果提供了预测日期)
        test_loader = None
        test_data_info = None
        if predict_date:
            print(f"\n正在加载测试集 ({predict_date}) 用于Epoch评估...")
            # 兼容性处理：尝试获取 S_test
            pred_data = self._prepare_prediction_data(predict_date)
            if len(pred_data) == 5:
                X_test, y_test, S_test, _, _ = pred_data
            else:
                X_test, y_test, _, _ = pred_data
                S_test = None
                
            if len(X_test) > 0:
                test_dataset = StockDataset(X_test, y_test, stock_indices=S_test)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                test_data_info = (X_test, y_test)
                print(f"测试集加载完成: {len(X_test)} 样本")
        
        # 学习率调度器
        # 全量模式下监控 Train Loss (min)
        # 如果有测试集，且用户希望用 Test AUC (max)
        if test_loader is not None:
             scheduler_mode = 'max'
             print("Scheduler Mode: Max (Monitoring Test AUC)")
        else:
             scheduler_mode = 'min'
             print("Scheduler Mode: Min (Monitoring Train Loss)")
             
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.5, patience=3)
        
        # 早停机制 (基于 Train Loss)
        best_train_loss = float('inf')
        best_model_state = None
        best_test_model_state = None
        patience_counter = 0
        patience = 20
        
        # 记录最佳测试集表现
        best_test_auc = 0
        best_test_epoch = 0
        best_test_metrics = {}
        
        for epoch in range(self.epochs):
            # === 训练阶段 ===
            model.train()
            total_loss = 0
            train_preds = []
            train_targets = []
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for batch in progress_bar:
                if len(batch) == 3:
                    X_batch, y_batch, S_batch = batch
                    S_batch = S_batch.to(device)
                else:
                    X_batch, y_batch = batch
                    S_batch = None
                
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(X_batch, stock_idx=S_batch)
                # 计算损失
                # loss = criterion(outputs, y_batch)
                loss_per_sample = current_criterion(outputs, y_batch)
                
                # 如果没有下采样且正负样本极度不平衡，应用正样本权重
                weights = torch.ones_like(y_batch)
                # 如果是小样本未下采样模式，应用权重
                if dropout_rate == 0.0: # 借用 dropout_rate==0.0 作为小样本未下采样模式的标记
                     if (self.max_stocks is not None and self.max_stocks <= 50):
                         weights[y_batch == 1] = pos_weight_val
                
                loss_bce = (loss_per_sample * weights).mean()
                
                # === Ranking Loss ===
                # 在 batch 内构建正负样本对
                # y_batch shape: (batch_size, 1) -> flatten -> (batch_size,)
                y_flat = y_batch.view(-1)
                pos_indices = (y_flat == 1).nonzero(as_tuple=True)[0]
                neg_indices = (y_flat == 0).nonzero(as_tuple=True)[0]
                
                loss_rank = torch.tensor(0.0).to(device)
                
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    # 简单策略：随机采样构建 Pair
                    # 为了不过度增加计算量，限制 Pair 的数量为 min(pos, neg) * 2
                    num_pairs = min(len(pos_indices), len(neg_indices)) * 2
                    num_pairs = min(num_pairs, 1024) # 上限
                    
                    # 随机选择索引（允许重复以增加 Pair 多样性）
                    pos_idx_sel = pos_indices[torch.randint(0, len(pos_indices), (num_pairs,))]
                    neg_idx_sel = neg_indices[torch.randint(0, len(neg_indices), (num_pairs,))]
                    
                    pos_scores = outputs[pos_idx_sel]
                    neg_scores = outputs[neg_idx_sel]
                    
                    # Target=1 表示 x1 > x2
                    target_rank = torch.ones(num_pairs, 1).to(device)
                    loss_rank = rank_criterion(pos_scores, neg_scores, target_rank)
                
                # 总 Loss = BCE + 0.2 * Ranking
                loss = loss_bce + self.rank_loss_weight * loss_rank
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                train_preds.extend(outputs.detach().cpu().numpy())
                train_targets.extend(y_batch.detach().cpu().numpy())
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_loss / len(train_loader)
            train_auc = roc_auc_score(train_targets, train_preds)
            
            # === 测试集评估 ===
            test_info = ""
            if test_loader:
                model.eval()
                test_preds = []
                test_targets = []
                with torch.no_grad():
                    for batch in test_loader:
                        if len(batch) == 3:
                            X_batch, y_batch, S_batch = batch
                            S_batch = S_batch.to(device)
                        else:
                            X_batch, y_batch = batch
                            S_batch = None
                        
                        X_batch = X_batch.to(device)
                        outputs = model(X_batch, stock_idx=S_batch)
                        test_preds.extend(outputs.cpu().numpy())
                        test_targets.extend(y_batch.numpy())
                
                test_auc = roc_auc_score(test_targets, test_preds)
                test_info = f" | Test AUC: {test_auc:.4f}"
                
                # 记录最佳测试集AUC
                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    best_test_epoch = epoch + 1
                    best_test_metrics = {
                        'auc': test_auc,
                        'epoch': epoch + 1,
                        'train_loss': avg_train_loss,
                        'train_auc': train_auc
                    }
                    # 保存最佳测试集模型状态
                    best_test_model_state = model.state_dict().copy()
                    test_info += " (New Best!)"
            
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train AUC: {train_auc:.4f}{test_info}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 学习率调度
            if test_loader:
                scheduler.step(test_auc)
            else:
                scheduler.step(avg_train_loss)
            
            # 保存最佳模型 (基于 Train Loss)
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                # print(f"  ✓ 新的最佳模型 (Train Loss: {best_train_loss:.4f})")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"早停触发 (Epoch {epoch+1})")
                break
        
        # 打印最佳测试集结果
        if best_test_metrics:
            print(f"\n{'='*50}")
            print(f"最佳测试集表现 (Epoch {best_test_metrics['epoch']}):")
            print(f"  Test AUC: {best_test_metrics['auc']:.4f}")
            print(f"  Train Loss: {best_test_metrics['train_loss']:.4f}")
            print(f"  Train AUC: {best_test_metrics['train_auc']:.4f}")
            print(f"{'='*50}\n")
            
            # 如果存在最佳测试集模型，优先使用它
            if best_test_model_state is not None:
                print(f"正在加载最佳测试集模型 (Epoch {best_test_metrics['epoch']})...")
                model.load_state_dict(best_test_model_state)
            elif best_model_state is not None:
                print("未找到最佳测试集模型，加载最小训练损失模型...")
                model.load_state_dict(best_model_state)
        elif best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        # 寻找最佳阈值 (基于全量训练数据)
        model.eval()
        train_preds_final = []
        # 使用不打乱的 DataLoader 进行预测
        train_eval_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in train_eval_loader:
                if len(batch) == 3:
                    X_batch, _, S_batch = batch
                    S_batch = S_batch.to(device)
                else:
                    X_batch, _ = batch
                    S_batch = None
                
                X_batch = X_batch.to(device)
                outputs = model(X_batch, stock_idx=S_batch)
                train_preds_final.extend(outputs.cpu().numpy())
        
        train_preds_final = np.array(train_preds_final)
        
        best_threshold = 0.5
        best_f1 = 0
        thresholds = np.arange(0.3, 0.9, 0.05)
        
        for thresh in thresholds:
            y_pred_bin = (train_preds_final > thresh).astype(int)
            f1 = f1_score(y_train_sub, y_pred_bin)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        self.best_threshold = best_threshold
        print(f"最佳阈值 (基于训练集): {self.best_threshold:.3f} (Train F1: {best_f1:.4f})")
            
        # 训练结束后，在全量训练集上进行评估
        print("\n模型训练完成，开始在全量训练集上评估...")
        self._evaluate_model(model, X_train, y_train, "全量训练集", S_data=S_train)
            
        return model

    def _evaluate_model(self, model, X_data, y_data, dataset_name: str, S_data=None):
        """评估模型并打印指标"""
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        model.eval()
        
        dataset = StockDataset(X_data, y_data, stock_indices=S_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    X_batch, _, S_batch = batch
                    S_batch = S_batch.to(device)
                else:
                    X_batch, _ = batch
                    S_batch = None
                
                X_batch = X_batch.to(device)
                outputs = model(X_batch, stock_idx=S_batch)
                all_preds.extend(outputs.cpu().numpy())
        
        y_pred_proba = np.array(all_preds).flatten()
        
        # 使用动态阈值 (如果存在)
        threshold = getattr(self, 'best_threshold', 0.5)
        y_pred_class = (y_pred_proba > threshold).astype(int)
        
        print(f"评估使用阈值: {threshold:.3f}")
        
        # 计算指标
        acc = accuracy_score(y_data, y_pred_class)
        try:
            auc = roc_auc_score(y_data, y_pred_proba)
        except ValueError:
            auc = 0.5  # 如果标签只有一类
        
        print(f"\n========== {dataset_name} 评估结果 ==========")
        print(f"整体准确率 (Accuracy): {acc:.4f}")
        print(f"AUC: {auc:.4f}")
        
        print("\n分类报告:")
        print(classification_report(y_data, y_pred_class, labels=[0, 1], target_names=['涨幅<=5%', '涨幅>5%']))
        
        print("混淆矩阵:")
        print(confusion_matrix(y_data, y_pred_class, labels=[0, 1]))
        print("========================================\n")
        model.train() # 恢复训练模式

    def _prepare_prediction_data(self, predict_date: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """为预测日准备数据，包括特征、目标、StockID、收益率和原始DataFrame"""
        # 统一日期格式为 YYYY-MM-DD
        predict_date = pd.to_datetime(predict_date).strftime('%Y-%m-%d')
        
        # === 缓存检查 ===
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        stocks_str = f"_{self.max_stocks}" if self.max_stocks else "_all"
        cache_file = os.path.join(cache_dir, f"pred_data_{predict_date}_w{self.window_size}{stocks_str}.pkl")
        
        if os.path.exists(cache_file):
            print(f"\n[Cache] 发现预测日 ({predict_date}) 的缓存数据，正在加载...")
            try:
                data = joblib.load(cache_file)
                # 兼容旧缓存
                if len(data) == 5:
                     return data
                else:
                     print("[Cache] 缓存格式过时，重新处理数据...")
            except Exception as e:
                print(f"[Cache] 加载缓存失败 ({e})，将重新处理数据...")
        
        print(f"\n准备预测日 ({predict_date}) 的评估数据...")
        
        # 我们需要 predict_date 前 window_size 天的数据来构建当天的序列
        start_date = (pd.to_datetime(predict_date) - pd.Timedelta(days=self.window_size * 2)).strftime('%Y-%m-%d')
        
        all_X, all_y, all_S, all_returns, all_codes = [], [], [], [], []
        
        # 加载股票映射 (如果尚未加载)
        if not self.stock2idx or len(self.stock2idx) <= 1:
            if os.path.exists(self.stock_map_path):
                with open(self.stock_map_path, 'r') as f:
                    self.stock2idx = json.load(f)
        
        files = sorted([f for f in os.listdir("daily") if f.endswith('_daily.csv')])
        if self.max_stocks:
            files = files[:self.max_stocks]

        for file in tqdm(files, desc=f"构建预测日序列"):
            file_path = os.path.join("daily", file)
            # 解析股票代码
            stock_code = file.replace('_daily.csv', '')
            stock_idx = self.stock2idx.get(stock_code, 0) # 0 is <UNK>
            
            try:
                # 使用 process_single_stock_for_prediction 以避免因 T+3 等长周期目标变量导致的行删除
                # 这对于保留最近的数据（预测日）至关重要
                df = self.feature_engineer.process_single_stock_for_prediction(file_path)
                
                # 手动计算 T+1 目标变量（用于评估）
                if not df.empty:
                    df['TARGET_RETURNS_T1'] = df['close'].shift(-1) / df['close'] - 1
                    df['TARGET_RETURNS_T1_PCT'] = df['TARGET_RETURNS_T1'] * 100
                    df['TARGET_T1_OVER_5PCT'] = (df['TARGET_RETURNS_T1_PCT'] >= 5).astype(int)
                
                # 筛选出包含预测日及其回溯窗口的数据
                df_slice = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= predict_date)]
                
                if not df_slice.empty and predict_date in df_slice['trade_date'].dt.strftime('%Y-%m-%d').values:
                    # 获取预测日当天的索引
                    predict_idx = df_slice.index[df_slice['trade_date'] == predict_date][0]
                    
                    # 确保有足够的回溯数据
                    if predict_idx - df_slice.index[0] >= self.window_size - 1:
                        
                        # 获取真实标签和收益率
                        y_true = df.loc[predict_idx, 'TARGET_T1_OVER_5PCT']
                        y_return = df.loc[predict_idx, 'TARGET_RETURNS_T1_PCT']
                        
                        # 创建序列
                        feature_cols = self.feature_engineer.get_feature_columns()
                        valid_cols = [c for c in feature_cols if c in df.columns]
                        
                        # 定位序列的起始和结束索引
                        seq_start_idx_in_slice = df_slice.index.get_loc(predict_idx) - self.window_size + 1
                        seq_end_idx_in_slice = df_slice.index.get_loc(predict_idx) + 1
                        
                        window_features = df_slice.iloc[seq_start_idx_in_slice:seq_end_idx_in_slice][valid_cols].values
                        
                        if window_features.shape[0] == self.window_size:
                            all_X.append(window_features)
                            all_y.append(y_true)
                            all_S.append(stock_idx)
                            all_returns.append(y_return)
                            all_codes.append(df.loc[predict_idx, 'ts_code'])

            except Exception as e:
                continue
        
        if not all_X:
            return np.array([]), np.array([]), np.array([]), np.array([]), pd.DataFrame()
            
        X_pred = np.array(all_X)
        y_true = np.array(all_y)
        S_pred = np.array(all_S)
        y_returns = np.array(all_returns)
        
        # 标准化处理
        if os.path.exists(self.scaler_path):
            scaler = joblib.load(self.scaler_path)
            N, T, F = X_pred.shape
            X_pred = scaler.transform(X_pred.reshape(N * T, F)).reshape(N, T, F)
        else:
            print("警告: 未找到Scaler文件，预测数据未标准化！")
        
        # 创建一个包含代码、真实标签和收益率的DataFrame，用于后续分析
        results_df = pd.DataFrame({
            'ts_code': all_codes,
            'y_true': y_true,
            'actual_return': y_returns
        })
        
        print(f"✓ 预测日数据准备完成: {X_pred.shape[0]} 个可评估样本")
        
        # === 保存缓存 ===
        result = (X_pred, y_true, S_pred, y_returns, results_df)
        try:
            joblib.dump(result, cache_file)
            print(f"[Cache] 已保存预测数据缓存至: {cache_file}")
        except Exception as e:
            print(f"[Cache] 保存缓存失败: {e}")
            
        return X_pred, y_true, S_pred, y_returns, results_df

    def predict_single_day(self, train_cutoff_date: str, predict_date: str, extra_predict_dates: Optional[List[str]] = None) -> Dict:
        """预测单日结果，并进行详细评估"""
        # 1. 准备训练数据
        X_train, y_train, S_train = self.prepare_data_for_date(cutoff_date=train_cutoff_date)
        
        # 下采样平衡类别 - 打印原始数据分布
        pos_indices = np.where(y_train == 1)[0]
        neg_indices = np.where(y_train == 0)[0]
        
        print(f"训练数据分布 - 正样本: {len(pos_indices)}, 负样本: {len(neg_indices)}, 比例: 1:{len(neg_indices)/len(pos_indices):.1f}")
        
        # 小样本测试模式（max_stocks <= 50）禁用下采样，以强制过拟合
        is_small_sample_test = self.max_stocks is not None and self.max_stocks <= 50
        if is_small_sample_test:
            print(f"小样本模式 (max_stocks={self.max_stocks})：禁用下采样，使用全量数据进行训练")
        
        neg_ratio = self.neg_ratio
        if not is_small_sample_test and len(pos_indices) > 0 and len(neg_indices) > len(pos_indices) * neg_ratio:
            # 修正：保持 1:neg_ratio 的比例
            n_neg = len(pos_indices) * neg_ratio
            selected_neg = np.random.choice(neg_indices, n_neg, replace=False)
            selected_indices = np.concatenate([pos_indices, selected_neg])
            np.random.shuffle(selected_indices)
            
            X_train = X_train[selected_indices]
            y_train = y_train[selected_indices]
            S_train = S_train[selected_indices]
            print(f"下采样后数据分布 - 正样本: {len(pos_indices)}, 负样本: {n_neg}, 比例: 1:{neg_ratio}")
            print(f"下采样后训练数据: {X_train.shape}")
            
        model = self.train_model(X_train, y_train, S_train=S_train, predict_date=predict_date)
        model_path = os.path.join(self.model_dir, f"best_model_{self.model_type.lower()}_{train_cutoff_date.replace('-', '')}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存: {model_path}")
        
        # 3. 准备预测日（测试集）的评估数据
        X_pred, y_true, S_pred, y_returns, results_df = self._prepare_prediction_data(predict_date)
        
        if X_pred.shape[0] == 0:
            print(f"在 {predict_date} 没有找到可供预测的股票数据。")
            return {}

        # 4. 在测试集上进行预测和评估
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        model.eval()
        
        pred_dataset = StockDataset(X_pred, y_true, stock_indices=S_pred)
        pred_dataloader = DataLoader(pred_dataset, batch_size=self.batch_size, shuffle=False)
        
        all_preds_proba = []
        with torch.no_grad():
            for batch in pred_dataloader:
                if len(batch) == 3:
                    X_batch, _, S_batch = batch
                    S_batch = S_batch.to(device)
                else:
                    X_batch, _ = batch
                    S_batch = None
                
                X_batch = X_batch.to(device)
                outputs = model(X_batch, stock_idx=S_batch)
                all_preds_proba.extend(outputs.cpu().numpy())
        
        y_pred_proba = np.array(all_preds_proba).flatten()
        results_df['pred_proba'] = y_pred_proba
        
        self._evaluate_model(model, X_pred, y_true, f"测试集 ({predict_date})", S_data=S_pred)
        
        top_20_df = results_df.sort_values('pred_proba', ascending=False).head(20)
        
        print("\n========== Top-20 预测结果 ==========")
        print(top_20_df[['ts_code', 'pred_proba', 'y_true', 'actual_return']].to_string(index=False))
        
        # 计算Top-20准确率
        top_20_correct = top_20_df[top_20_df['y_true'] == 1]
        top_20_accuracy = len(top_20_correct) / len(top_20_df) if len(top_20_df) > 0 else 0
        
        print(f"\nTop-20 预测准确率 (预测为1且真实为1): {top_20_accuracy:.2%}")
        print(f"Top-20 平均实际收益率: {top_20_df['actual_return'].mean():.2f}%")
        print("====================================\n")

        result = {
            "model_type": self.model_type,
            "predict_date": predict_date,
            "top_20_predictions": top_20_df.to_dict('records')
        }

        if extra_predict_dates:
            extra_results = {}
            for extra_date in extra_predict_dates:
                X_extra, y_extra, S_extra, y_returns_extra, results_df_extra = self._prepare_prediction_data(extra_date)
                if X_extra.shape[0] == 0:
                    continue
                device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
                model.eval()
                pred_dataset_extra = StockDataset(X_extra, y_extra, stock_indices=S_extra)
                pred_dataloader_extra = DataLoader(pred_dataset_extra, batch_size=self.batch_size, shuffle=False)
                all_preds_proba_extra = []
                with torch.no_grad():
                    for batch in pred_dataloader_extra:
                        if len(batch) == 3:
                            X_batch, _, S_batch = batch
                            S_batch = S_batch.to(device)
                        else:
                            X_batch, _ = batch
                            S_batch = None
                        X_batch = X_batch.to(device)
                        outputs = model(X_batch, stock_idx=S_batch)
                        all_preds_proba_extra.extend(outputs.cpu().numpy())
                y_pred_proba_extra = np.array(all_preds_proba_extra).flatten()
                results_df_extra['pred_proba'] = y_pred_proba_extra
                self._evaluate_model(model, X_extra, y_extra, f"测试集 ({extra_date})", S_data=S_extra)
                top_20_extra = results_df_extra.sort_values('pred_proba', ascending=False).head(20)
                print("\n========== Top-20 预测结果 ==========")
                print(top_20_extra[['ts_code', 'pred_proba', 'y_true', 'actual_return']].to_string(index=False))
                top_20_correct_extra = top_20_extra[top_20_extra['y_true'] == 1]
                top_20_accuracy_extra = len(top_20_correct_extra) / len(top_20_extra) if len(top_20_extra) > 0 else 0
                print(f"\nTop-20 预测准确率 (预测为1且真实为1): {top_20_accuracy_extra:.2%}")
                print(f"Top-20 平均实际收益率: {top_20_extra['actual_return'].mean():.2f}%")
                print("====================================\n")
                extra_results[extra_date] = {
                    "top_20_predictions": top_20_extra.to_dict('records')
                }
            if extra_results:
                result["extra_predict_results"] = extra_results

        return result

def main():
    import argparse
    parser = argparse.ArgumentParser(description='时间序列股票预测器 (LSTM/Transformer)')
    parser.add_argument('--predict-date', '--end-date', dest='end_date', type=str, help='预测日期 (YYYY-MM-DD)')
    parser.add_argument('--max-stocks', type=int, help='限制股票数量')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--window-size', type=int, default=30, help='时间窗口大小')
    parser.add_argument('--exclude-one-word-limit', action='store_true', default=False, help='排除一字板')
    parser.add_argument('--model-type', type=str.upper, default='LSTM', choices=['LSTM', 'TRANSFORMER'], help='模型类型')
    parser.add_argument('--d-model', type=int, default=128, help='Transformer模型维度')
    parser.add_argument('--hidden-dim', type=int, default=128, help='LSTM隐藏层维度')
    parser.add_argument('--nhead', type=int, default=8, help='Transformer注意力头数')
    parser.add_argument('--num-layers', type=int, default=3, help='Transformer/LSTM层数')
    parser.add_argument('--learning-rate', '--lr', dest='learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--neg-ratio', type=int, default=8, help='负样本下采样比例')
    parser.add_argument('--rank-loss-weight', type=float, default=0.2, help='Ranking Loss权重')
    parser.add_argument('--extra-predict-dates', type=str, help='使用当前训练模型额外评估的日期，逗号分隔')
    parser.add_argument('--eval-only', action='store_true', help='仅评估指定模型，不重新训练')
    parser.add_argument('--model-path', type=str, help='模型权重文件路径，仅eval-only模式使用')
    parser.add_argument('--train-cutoff', type=str, help='训练截止日期 (YYYY-MM-DD)，用于推断默认模型路径')
    
    args = parser.parse_args()
    extra_dates = None
    if args.extra_predict_dates:
        extra_dates = [d.strip() for d in args.extra_predict_dates.split(',') if d.strip()]
    
    predictor = TimeSeriesStockPredictor(
        max_stocks=args.max_stocks,
        window_size=args.window_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        exclude_one_word_limit=args.exclude_one_word_limit,
        model_type=args.model_type,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        neg_ratio=args.neg_ratio,
        rank_loss_weight=args.rank_loss_weight
    )
    
    predict_date = args.end_date if args.end_date else datetime.now().strftime('%Y-%m-%d')
    
    if args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        primary_date = predict_date
        dates_to_eval = [primary_date] + (extra_dates or [])
        if args.model_path:
            model_path = args.model_path
        else:
            if not args.train_cutoff:
                raise ValueError("eval-only 模式下未提供 train-cutoff 或 model-path")
            cutoff_str = args.train_cutoff.replace('-', '')
            model_suffix = args.model_type.lower()
            model_path = os.path.join(predictor.model_dir, f"best_model_{model_suffix}_{cutoff_str}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        X_sample, y_sample, S_sample, _, _ = predictor._prepare_prediction_data(primary_date)
        if X_sample.shape[0] == 0:
            raise ValueError(f"在 {primary_date} 没有可评估样本")
        input_dim = X_sample.shape[2]
        num_stocks = len(predictor.stock2idx) if predictor.stock2idx else None
        dropout_rate = 0.0 if (predictor.max_stocks is not None and predictor.max_stocks <= 50) else 0.2
        if predictor.model_type == "LSTM":
            model = LSTMModel(input_dim, predictor.hidden_dim, predictor.num_layers, dropout=dropout_rate, num_stocks=num_stocks).to(device)
        elif predictor.model_type == "TRANSFORMER":
            model = TransformerModel(input_dim, predictor.d_model, predictor.nhead, predictor.num_layers, dropout=dropout_rate, num_stocks=num_stocks).to(device)
        else:
            raise ValueError(f"不支持的模型类型: {predictor.model_type}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        eval_results = {}
        for d in dates_to_eval:
            X_eval, y_eval, S_eval, y_ret_eval, df_eval = predictor._prepare_prediction_data(d)
            if X_eval.shape[0] == 0:
                continue
            dataset = StockDataset(X_eval, y_eval, stock_indices=S_eval)
            dataloader = DataLoader(dataset, batch_size=predictor.batch_size, shuffle=False)
            all_proba = []
            with torch.no_grad():
                for batch in dataloader:
                    if len(batch) == 3:
                        X_batch, _, S_batch = batch
                        S_batch = S_batch.to(device)
                    else:
                        X_batch, _ = batch
                        S_batch = None
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch, stock_idx=S_batch)
                    all_proba.extend(outputs.cpu().numpy())
            y_pred_proba = np.array(all_proba).flatten()
            df_eval['pred_proba'] = y_pred_proba
            predictor._evaluate_model(model, X_eval, y_eval, f"测试集 ({d})", S_data=S_eval)
            top_20 = df_eval.sort_values('pred_proba', ascending=False).head(20)
            print("\n========== Top-20 预测结果 ==========")
            print(top_20[['ts_code', 'pred_proba', 'y_true', 'actual_return']].to_string(index=False))
            top_20_correct = top_20[top_20['y_true'] == 1]
            top_20_accuracy = len(top_20_correct) / len(top_20) if len(top_20) > 0 else 0
            print(f"\nTop-20 预测准确率 (预测为1且真实为1): {top_20_accuracy:.2%}")
            print(f"Top-20 平均实际收益率: {top_20['actual_return'].mean():.2f}%")
            print("====================================\n")
            eval_results[d] = {
                "top_20_predictions": top_20.to_dict('records')
            }
        result = {
            "model_type": args.model_type,
            "predict_dates": dates_to_eval,
            "eval_results": eval_results
        }
    else:
        train_cutoff = (pd.to_datetime(predict_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        result = predictor.predict_single_day(train_cutoff, predict_date, extra_predict_dates=extra_dates)
    
    # 保存结果
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = args.model_type.lower()
    with open(f"models/ts_models/prediction_{model_suffix}_{timestamp}.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"结果已保存: models/ts_models/prediction_{model_suffix}_{timestamp}.json")

if __name__ == "__main__":
    main()
