"""
模型训练脚本
用于训练股票预测模型
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from tqdm import tqdm
import json
from datetime import datetime

from .model_definitions import create_model, LSTMStockPredictor, GRUStockPredictor, TransformerStockPredictor


class StockDataset(Dataset):
    """
    股票数据集类
    用于加载和预处理股票数据
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 60,
        target_column: str = "收盘",
        feature_columns: Optional[list] = None
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径（CSV格式）
            sequence_length: 序列长度（用于时间序列预测）
            target_column: 目标列名（要预测的列）
            feature_columns: 特征列名列表，如果为None则使用默认列
        """
        self.sequence_length = sequence_length
        self.target_column = target_column
        
        # 加载数据
        df = pd.read_csv(data_path)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)
        
        # 默认特征列
        if feature_columns is None:
            feature_columns = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率']
        
        # 检查列是否存在
        available_columns = [col for col in feature_columns if col in df.columns]
        if len(available_columns) == 0:
            raise ValueError(f"没有找到可用的特征列: {feature_columns}")
        
        self.feature_columns = available_columns
        
        # 提取特征和目标
        self.features = df[available_columns].values.astype(np.float32)
        self.targets = df[target_column].values.astype(np.float32)
        
        # 数据标准化
        self.feature_mean = np.mean(self.features, axis=0)
        self.feature_std = np.std(self.features, axis=0) + 1e-8
        self.features = (self.features - self.feature_mean) / self.feature_std
        
        self.target_mean = np.mean(self.targets)
        self.target_std = np.std(self.targets) + 1e-8
        self.targets = (self.targets - self.target_mean) / self.target_std
        
        print(f"数据集加载完成: {len(df)} 条记录, {len(available_columns)} 个特征")
    
    def __len__(self) -> int:
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (序列特征, 目标值)
        """
        sequence = self.features[idx:idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length]
        
        return torch.tensor(sequence), torch.tensor(target)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
    save_dir: str = "models/checkpoints",
    model_name: str = "stock_predictor"
) -> Dict:
    """
    训练模型
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备（CPU或GPU）
        save_dir: 模型保存目录
        model_name: 模型名称
        
    Returns:
        训练历史字典
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    history = {
        'train_loss': [],
        'val_loss': [] if val_loader else None
    }
    
    best_val_loss = float('inf')
    
    print(f"开始训练，设备: {device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for sequences, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            sequences = sequences.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)
        
        # 验证阶段
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(device)
                    targets = targets.to(device).unsqueeze(1)
                    
                    outputs = model(sequences)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            history['val_loss'].append(avg_val_loss)
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(save_dir, f"{model_name}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, checkpoint_path)
                print(f"  ✓ 保存最佳模型 (val_loss: {avg_val_loss:.6f})")
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
    
    # 保存最终模型
    final_path = os.path.join(save_dir, f"{model_name}_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    
    print(f"\n训练完成！模型已保存到: {save_dir}")
    
    return history


def load_model(
    model_path: str,
    model_type: str = "lstm",
    input_size: int = 7,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        model_type: 模型类型
        input_size: 输入特征维度
        device: 设备
        
    Returns:
        加载的模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(model_type=model_type, input_size=input_size)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    使用模型进行预测
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 设备
        
    Returns:
        预测结果数组
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for sequences, _ in data_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            predictions.append(outputs.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)


def main():
    """主函数，用于测试训练流程"""
    print("=" * 60)
    print("股票预测模型训练")
    print("=" * 60)
    
    # 检查是否有数据文件
    data_dir = "data/daily"
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        print("请先运行数据获取脚本下载股票数据")
        return
    
    # 查找第一个数据文件作为示例
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not data_files:
        print(f"错误: 数据目录中没有CSV文件: {data_dir}")
        print("请先运行数据获取脚本下载股票数据")
        return
    
    # 使用第一个文件作为示例
    data_file = os.path.join(data_dir, data_files[0])
    print(f"\n使用数据文件: {data_file}")
    
    # 创建数据集
    try:
        dataset = StockDataset(data_file, sequence_length=60)
    except Exception as e:
        print(f"创建数据集失败: {e}")
        return
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    input_size = len(dataset.feature_columns)
    model = create_model(model_type="lstm", input_size=input_size, hidden_size=64, num_layers=2)
    
    print(f"\n模型类型: LSTM")
    print(f"输入特征数: {input_size}")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    
    # 训练模型
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,  # 示例使用较少轮数
        learning_rate=0.001,
        model_name="stock_lstm"
    )
    
    print("\n训练历史:")
    print(f"最终训练损失: {history['train_loss'][-1]:.6f}")
    if history['val_loss']:
        print(f"最终验证损失: {history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()

