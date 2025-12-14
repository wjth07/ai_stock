#!/bin/bash
# Conda镜像源配置脚本

# 查找conda命令
CONDA_CMD=""
if command -v conda &> /dev/null; then
    CONDA_CMD="conda"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/miniconda3/bin/conda"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/anaconda3/bin/conda"
elif [ -f "/opt/miniconda3/bin/conda" ]; then
    CONDA_CMD="/opt/miniconda3/bin/conda"
elif [ -f "/opt/anaconda3/bin/conda" ]; then
    CONDA_CMD="/opt/anaconda3/bin/conda"
fi

# 检查conda是否可用
if [ -z "$CONDA_CMD" ]; then
    echo "错误: 未找到conda命令"
    exit 1
fi

echo "=========================================="
echo "Conda 镜像源配置"
echo "=========================================="
echo ""

# 显示当前配置
echo "当前conda channels配置:"
$CONDA_CMD config --show channels
echo ""

# 询问操作
echo "请选择操作:"
echo "1) 添加清华源（推荐，国内加速）"
echo "2) 恢复默认源（官方源）"
echo "3) 查看当前配置"
echo "4) 退出"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "正在添加清华源..."
        
        # 添加清华源
        $CONDA_CMD config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
        $CONDA_CMD config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
        $CONDA_CMD config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
        
        # 设置显示channel URL
        $CONDA_CMD config --set show_channel_urls yes
        
        # 设置channel优先级
        $CONDA_CMD config --set channel_priority flexible
        
        echo ""
        echo "✓ 清华源配置完成！"
        echo ""
        echo "已添加的channels:"
        $CONDA_CMD config --show channels
        ;;
    2)
        echo ""
        echo "正在恢复默认源..."
        
        # 移除所有自定义channels
        $CONDA_CMD config --remove-key channels 2>/dev/null
        
        # 添加默认channels
        $CONDA_CMD config --add channels defaults
        
        echo ""
        echo "✓ 已恢复默认源"
        ;;
    3)
        echo ""
        echo "当前conda配置:"
        echo "----------------------------------------"
        $CONDA_CMD config --show
        ;;
    4)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "配置完成！"

