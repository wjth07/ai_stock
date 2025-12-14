#!/bin/bash
# Conda虚拟环境创建脚本

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
    echo ""
    echo "请先安装Miniconda（无需Homebrew）:"
    echo ""
    echo "方法1: 直接下载安装包"
    echo "  1. 访问: https://docs.conda.io/en/latest/miniconda.html"
    echo "  2. 下载 macOS 版本的 .pkg 文件"
    echo "  3. 双击安装"
    echo ""
    echo "方法2: 使用命令行自动下载安装"
    echo "  # Intel芯片:"
    echo "  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    echo "  bash Miniconda3-latest-MacOSX-x86_64.sh"
    echo ""
    echo "  # Apple Silicon (M1/M2/M3):"
    echo "  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    echo "  bash Miniconda3-latest-MacOSX-arm64.sh"
    echo ""
    echo "安装后运行: ~/miniconda3/bin/conda init zsh"
    echo "然后: source ~/.zshrc"
    echo ""
    echo "详细说明请查看: INSTALL_CONDA.md"
    exit 1
fi

# 创建env文件夹（如果不存在）
mkdir -p env

# 接受conda服务条款（如果需要）
echo "检查并接受conda服务条款..."
$CONDA_CMD tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null
$CONDA_CMD tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null

# 配置清华源（加速下载）
echo "配置清华源镜像..."
$CONDA_CMD config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main 2>/dev/null
$CONDA_CMD config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free 2>/dev/null
$CONDA_CMD config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 2>/dev/null
$CONDA_CMD config --set show_channel_urls yes 2>/dev/null

# 设置channel优先级（优先使用清华源）
$CONDA_CMD config --set channel_priority flexible 2>/dev/null

echo "✓ 清华源配置完成"

# 创建conda虚拟环境（带重试机制）
echo "正在创建conda虚拟环境: env/stock"
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if [ $RETRY_COUNT -gt 0 ]; then
        echo "重试中... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 5
    fi
    
    $CONDA_CMD create -p env/stock python=3.10 -y
    
    if [ $? -eq 0 ]; then
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "创建失败，将在5秒后重试..."
    fi
done

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 虚拟环境创建成功！"
    echo ""
    echo "激活环境的方法:"
    if [ "$CONDA_CMD" = "conda" ]; then
        echo "  conda activate env/stock"
    else
        echo "  $CONDA_CMD activate env/stock"
        echo ""
        echo "或者先初始化conda，然后使用:"
        echo "  $CONDA_CMD init zsh"
        echo "  source ~/.zshrc"
        echo "  conda activate env/stock"
    fi
    echo ""
    echo "安装依赖包:"
    echo "  conda activate env/stock"
    echo "  pip install -r requirements.txt"
else
    echo "错误: 虚拟环境创建失败"
    exit 1
fi

