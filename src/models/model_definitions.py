"""
AI模型架构定义模块
用于定义股票预测的机器学习模型
"""
import torch
import torch.nn as nn
from typing import Optional


class LSTMStockPredictor(nn.Module):
    """
    LSTM股票价格预测模型
    使用LSTM网络进行时间序列预测
    """
    
    def __init__(
        self,
        input_size: int = 7,  # 输入特征数（开盘、收盘、最高、最低、成交量、成交额、换手率等）
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1  # 预测未来1天的收盘价
    ):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比率
            output_size: 输出维度
        """
        super(LSTMStockPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_length, input_size)
            
        Returns:
            预测结果，形状为 (batch_size, output_size)
        """
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 只取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        output = self.fc(last_output)
        
        return output


class GRUStockPredictor(nn.Module):
    """
    GRU股票价格预测模型
    使用GRU网络进行时间序列预测（比LSTM更轻量）
    """
    
    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        初始化GRU模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: GRU隐藏层大小
            num_layers: GRU层数
            dropout: Dropout比率
            output_size: 输出维度
        """
        super(GRUStockPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_length, input_size)
            
        Returns:
            预测结果，形状为 (batch_size, output_size)
        """
        # GRU前向传播
        gru_out, _ = self.gru(x)
        
        # 只取最后一个时间步的输出
        last_output = gru_out[:, -1, :]
        
        # 全连接层
        output = self.fc(last_output)
        
        return output


class TransformerStockPredictor(nn.Module):
    """
    Transformer股票价格预测模型
    使用Transformer架构进行时间序列预测
    """
    
    def __init__(
        self,
        input_size: int = 7,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_size: int = 1
    ):
        """
        初始化Transformer模型
        
        Args:
            input_size: 输入特征维度
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer编码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            output_size: 输出维度
        """
        super(TransformerStockPredictor, self).__init__()
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码（可学习）
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出层
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_length, input_size)
            
        Returns:
            预测结果，形状为 (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer编码
        encoded = self.transformer_encoder(x)
        
        # 取最后一个时间步
        last_output = encoded[:, -1, :]
        
        # 输出层
        output = self.fc(last_output)
        
        return output


def create_model(
    model_type: str = "lstm",
    input_size: int = 7,
    **kwargs
) -> nn.Module:
    """
    创建模型工厂函数
    
    Args:
        model_type: 模型类型，可选: "lstm", "gru", "transformer"
        input_size: 输入特征维度
        **kwargs: 其他模型参数
        
    Returns:
        模型实例
    """
    if model_type.lower() == "lstm":
        return LSTMStockPredictor(input_size=input_size, **kwargs)
    elif model_type.lower() == "gru":
        return GRUStockPredictor(input_size=input_size, **kwargs)
    elif model_type.lower() == "transformer":
        return TransformerStockPredictor(input_size=input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: lstm, gru, transformer")

