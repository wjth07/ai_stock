# 安装Miniconda指南（不使用Homebrew）

## 方法1：直接下载安装包（推荐）

### 步骤：

1. **下载Miniconda安装包**
   - 访问：https://docs.conda.io/en/latest/miniconda.html
   - 选择 macOS 版本（推荐选择 Python 3.10 或 3.11）
   - 下载 `.pkg` 文件（图形界面安装）或 `.sh` 文件（命令行安装）

2. **安装方式A：使用图形界面（.pkg文件）**
   - 双击下载的 `.pkg` 文件
   - 按照安装向导完成安装
   - 默认安装路径：`~/miniconda3` 或 `~/anaconda3`

3. **安装方式B：使用命令行（.sh文件）**
   ```bash
   # 下载后，在终端运行：
   bash ~/Downloads/Miniconda3-latest-MacOSX-x86_64.sh
   # 或者如果是Apple Silicon (M1/M2)：
   bash ~/Downloads/Miniconda3-latest-MacOSX-arm64.sh
   ```

4. **初始化conda**
   ```bash
   # 重新打开终端，或运行：
   ~/miniconda3/bin/conda init zsh
   source ~/.zshrc
   ```

5. **验证安装**
   ```bash
   conda --version
   ```

6. **创建虚拟环境**
   ```bash
   cd /Users/mac/Documents/workspace/ai_stock
   ./setup_conda_env.sh
   ```

## 方法2：使用官方安装脚本（自动下载）

```bash
# 下载并安装Miniconda（自动选择适合的版本）
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
# 如果是Apple Silicon (M1/M2/M3):
# curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# 运行安装脚本
bash Miniconda3-latest-MacOSX-x86_64.sh

# 按照提示完成安装，然后：
~/miniconda3/bin/conda init zsh
source ~/.zshrc
```

## 方法3：如果已经安装了Anaconda

如果您已经安装了Anaconda，可以直接使用：

```bash
# 检查conda是否可用
which conda

# 如果找到了，直接创建环境：
cd /Users/mac/Documents/workspace/ai_stock
conda create -p env/stock python=3.10 -y
```

## 安装后验证

安装完成后，运行以下命令验证：

```bash
conda --version
conda info
```

然后就可以使用 `setup_conda_env.sh` 脚本创建虚拟环境了。

