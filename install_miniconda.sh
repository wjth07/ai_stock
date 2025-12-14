#!/bin/bash
# Miniconda自动安装脚本（无需Homebrew）

echo "=========================================="
echo "Miniconda 自动安装脚本"
echo "=========================================="
echo ""

# 检测系统架构
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "检测到: Apple Silicon (M1/M2/M3)"
    INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    INSTALLER_NAME="Miniconda3-latest-MacOSX-arm64.sh"
elif [ "$ARCH" = "x86_64" ]; then
    echo "检测到: Intel 芯片"
    INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    INSTALLER_NAME="Miniconda3-latest-MacOSX-x86_64.sh"
else
    echo "错误: 无法识别的系统架构: $ARCH"
    exit 1
fi

echo "下载地址: $INSTALLER_URL"
echo ""

# 检查是否已安装
if [ -f "$HOME/miniconda3/bin/conda" ] || [ -f "$HOME/anaconda3/bin/conda" ]; then
    echo "检测到已安装的conda:"
    if [ -f "$HOME/miniconda3/bin/conda" ]; then
        echo "  $HOME/miniconda3/bin/conda"
    fi
    if [ -f "$HOME/anaconda3/bin/conda" ]; then
        echo "  $HOME/anaconda3/bin/conda"
    fi
    echo ""
    read -p "是否继续安装？(y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消安装"
        exit 0
    fi
fi

# 下载安装包
echo "正在下载 Miniconda 安装包..."
cd /tmp
curl -L -O "$INSTALLER_URL"

if [ $? -ne 0 ]; then
    echo "错误: 下载失败"
    exit 1
fi

echo ""
echo "下载完成！"
echo ""

# 运行安装脚本
echo "开始安装 Miniconda..."
echo "安装过程中："
echo "  - 按 Enter 继续"
echo "  - 输入 'yes' 同意许可协议"
echo "  - 选择安装路径（默认: $HOME/miniconda3）"
echo "  - 输入 'yes' 初始化conda（推荐）"
echo ""

bash "$INSTALLER_NAME"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Miniconda 安装完成！"
    echo "=========================================="
    echo ""
    echo "下一步操作:"
    echo ""
    echo "1. 如果安装时选择了初始化，运行:"
    echo "   source ~/.zshrc"
    echo ""
    echo "2. 如果安装时没有初始化，运行:"
    echo "   ~/miniconda3/bin/conda init zsh"
    echo "   source ~/.zshrc"
    echo ""
    echo "3. 验证安装:"
    echo "   conda --version"
    echo ""
    echo "4. 创建项目虚拟环境:"
    echo "   cd /Users/mac/Documents/workspace/ai_stock"
    echo "   ./setup_conda_env.sh"
    echo ""
else
    echo ""
    echo "错误: 安装失败"
    exit 1
fi

# 清理下载的安装包
read -p "是否删除下载的安装包？(y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f "/tmp/$INSTALLER_NAME"
    echo "已删除安装包"
fi

