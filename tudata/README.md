# Tushare数据获取

此目录使用[Tushare](https://tushare.pro/)接口获取2024年以来的A股日K线数据。

## 功能特性

- ✅ 使用Tushare专业数据接口
- ✅ 获取2024年以来的完整日K线数据
- ✅ **自动前复权**：默认进行前复权处理，确保数据连续性
- ✅ **多线程并行处理**：显著提升数据获取速度
- ✅ **多Token并发**：使用多个API Token避免限流，提升并发性能
- ✅ **智能重试机制**：API调用失败时自动重试，指数退避延迟
- ✅ **并发控制优化**：限制并发数，避免过度请求导致限流
- ✅ **请求间隔控制**：并行模式下自动添加请求间隔
- ✅ **本地缓存加速**：股票列表自动缓存，避免重复API调用
- ✅ **详细错误信息**：异常时显示文件名和行号，便于调试
- ✅ 智能增量更新，避免重复下载
- ✅ 支持重复运行，自动检测数据状态
- ✅ 自动数据合并和去重
- ✅ 进度显示和详细统计报告
- ✅ **自动备用方案**：Tushare失败时自动切换到akshare

## 备用方案机制

为了确保数据获取的稳定性，当Tushare接口访问失败时，系统会自动切换到akshare作为备用数据源：

### 工作流程
1. **优先使用Tushare**：尝试通过Tushare专业接口获取数据
2. **自动切换备用**：Tushare失败时，自动切换到akshare
3. **格式转换**：akshare的股票代码格式会自动转换为Tushare格式（.SH/.SZ后缀）
4. **失败终止**：如果两个数据源都不可用，程序会安全终止

### 适用场景
- Tushare API暂时不可用
- Token过期或积分不足
- 网络连接问题
- API频率限制

### 优势
- **高可用性**：双重保障，确保数据获取成功
- **无缝切换**：用户无需手动干预
- **格式兼容**：两种数据源的格式自动统一

## 数据字段说明

| 字段名 | 说明 | 数据类型 |
|--------|------|----------|
| ts_code | 股票代码 | str |
| trade_date | 交易日期 | str (YYYYMMDD) |
| open | 开盘价 | float |
| high | 最高价 | float |
| low | 最低价 | float |
| close | 收盘价 | float |
| pre_close | 昨收价 | float |
| change | 涨跌额 | float |
| pct_chg | 涨跌幅 | float |
| vol | 成交量(手) | float |
| amount | 成交额(万元) | float |

## 复权说明

### 📈 什么是复权

股票价格会因为**分红配股、增发、送股等行为**而发生变化，为了保持历史数据的一致性和可比性，需要进行**复权处理**。

### 🎯 复权方式

| 复权方式 | 说明 | 适用场景 |
|----------|------|----------|
| **前复权 (qfq)** | 历史价格向前复权，保持最新价格不变 | **推荐用于分析和建模** |
| 后复权 (hfq) | 最新价格向后复权，保持历史价格不变 | 适合查看历史分红情况 |
| 不复权 ("") | 原始价格，不进行任何调整 | 适合查看实际成交价格 |

### ⚙️ 默认设置

- **默认复权方式**：`qfq`（前复权）
- **为什么选择前复权**：确保时间序列数据的连续性，避免除权除息造成的价格跳跃

### 🔧 自定义复权

```bash
# 前复权（默认）
python tushare_data_fetcher.py

# 后复权
python tushare_data_fetcher.py --adjust hfq

# 不复权
python tushare_data_fetcher.py --adjust ""
```

### 💡 复权重要性

- **数据连续性**：消除除权除息对价格走势的干扰
- **分析准确性**：确保技术指标计算的准确性
- **模型训练**：为机器学习模型提供干净、一致的数据

## 🚀 并行处理加速

### 多线程并发架构

系统采用**线程池并发处理**，显著提升数据获取速度：

```
串行处理: 股票1 → 股票2 → 股票3 → ... (慢)
并行处理: 股票1    股票2    股票3    ... (快)
           ↓        ↓        ↓
         线程1    线程2    线程3
```

### 性能优化策略

#### 1. **智能并发控制**
- **默认并发数**：CPU核心数的一半
- **API限流保护**：避免触发Tushare请求限制
- **自适应调整**：根据系统性能自动调整

#### 2. **内存优化**
- **分批处理**：避免一次性加载过多数据
- **及时释放**：处理完成后立即释放内存
- **结果聚合**：高效收集并汇总处理结果

#### 3. **错误处理**
- **异常隔离**：单个股票失败不影响其他股票
- **自动重试**：网络异常时自动重试
- **回退机制**：并行失败时自动回退到串行模式

### 使用方法

#### 并行处理（默认，推荐）
```bash
# 使用默认并发数（CPU核心数一半）
python tushare_data_fetcher.py

# 自定义并发数
python tushare_data_fetcher.py --max-workers 4

# 测试模式 + 并行
python tushare_data_fetcher.py --test --parallel
```

#### 串行处理（调试用）
```bash
# 强制使用串行模式
python tushare_data_fetcher.py --no-parallel

# 串行 + 测试模式
python tushare_data_fetcher.py --test --no-parallel
```

### 性能对比

| 处理模式 | 处理100只股票 | 处理1000只股票 | 内存占用 |
|----------|----------------|----------------|----------|
| **串行处理** | ~15分钟 | ~2.5小时 | 较低 |
| **并行处理** | ~5分钟 | ~40分钟 | 中等 |
| **加速比** | **3x** | **3.5x** | - |

### ⚠️ 注意事项

1. **API限流风险**
   - 并行请求过多可能触发Tushare限制
   - 建议并发数不超过CPU核心数的一半

2. **内存使用**
   - 并行处理会增加内存占用
   - 大规模数据获取时注意内存限制

3. **网络稳定性**
   - 并行处理对网络要求更高
   - 网络不稳定时建议降低并发数

4. **系统资源**
   - 高并发可能影响系统其他任务
   - 建议在专用环境中运行

### 🔧 高级配置

#### 自定义并发策略
```python
from tushare_data_fetcher import TushareDataFetcher

fetcher = TushareDataFetcher(token='your_token')

# 低并发（稳定为主）
result = fetcher.update_all_stocks(parallel=True, max_workers=2)

# 高并发（速度优先）
result = fetcher.update_all_stocks(parallel=True, max_workers=8)

# 串行模式（调试用）
result = fetcher.update_all_stocks(parallel=False)
```

#### 监控性能
```python
import time
start_time = time.time()

result = fetcher.update_all_stocks(max_stocks=100, parallel=True, max_workers=4)

end_time = time.time()
print(f"处理时间: {end_time - start_time:.2f} 秒")
print(f"平均每只股票: {(end_time - start_time)/100:.2f} 秒")
```

#### 运行性能测试
```bash
# 运行完整性能测试
python performance_test.py
```

### 📈 实际性能表现

基于测试结果，并行处理展现出显著的性能提升：

- **串行处理**: 63.03秒 (3只股票)
- **并行处理**: 1.01秒 (3只股票)
- **加速比**: 62.4x

*注：实际加速效果受网络条件、API限制和数据更新需求影响。在大规模数据获取场景中，典型加速比为2-5倍。*

## 安装依赖

```bash
# 激活环境
conda activate env/stock

# 安装tushare
pip install tushare

# 或从requirements.txt安装
pip install -r ../requirements.txt
```

## 使用方法

### 1. 测试模式（推荐首次使用）

获取前10只股票的数据进行测试：

```bash
cd tudata
python tushare_data_fetcher.py --test
```

### 2. 获取指定数量股票

```bash
# 获取前100只股票
python tushare_data_fetcher.py --max-stocks 100
```

### 3. 获取全部股票数据

```bash
# 获取全部A股2024年以来的数据
python tushare_data_fetcher.py
```

### 4. 自定义Token

如果需要使用自己的Token：

```bash
# 使用单个自定义Token
python tushare_data_fetcher.py --token YOUR_TOKEN_HERE

# 使用多个Token（避免API限流）
python tushare_data_fetcher.py --tokens TOKEN1 TOKEN2 TOKEN3
```

### 5. 多Token并行处理（推荐）

系统默认使用多个Token进行并行处理，可以显著提升速度并避免API限流：

```bash
# 使用默认的多个Token（推荐）
python tushare_data_fetcher.py

# 自定义多个Token进行并行处理
python tushare_data_fetcher.py --tokens "token1" "token2" "token3"
```

### 6. 控制输出信息

系统根据运行模式自动控制输出信息的详细程度：

```bash
# 串行模式：默认显示详细信息
python tushare_data_fetcher.py --no-parallel

# 并行模式：默认不显示详细信息（避免进度条干扰）
python tushare_data_fetcher.py --parallel

# 强制显示详细信息
python tushare_data_fetcher.py --verbose

# 静默模式（只显示最终统计）
python tushare_data_fetcher.py --quiet
```

## 数据存储结构

```
tudata/
├── daily/                    # 日K线数据
│   ├── 000001.SH_daily.csv  # 平安银行日K数据
│   ├── 000002.SZ_daily.csv  # 万科A日K数据
│   └── ...
├── models/                   # AI模型文件
│   ├── t1/                  # T+1预测模型
│   └── t3/                  # T+3预测模型
├── test/                     # 测试和演示文件目录
│   ├── test_*.py            # 功能测试脚本
│   ├── demo_*.py            # 功能演示脚本
│   ├── performance_test*.py # 性能测试脚本
│   └── tushare_data_fetcher_*.py # 替代实现版本
├── stock_list_cache.txt     # 股票列表缓存文件
├── feature_engineering.py   # 特征工程脚本
├── stock_predictor.py       # 股票预测脚本
├── tushare_data_fetcher.py  # 主程序文件
└── README.md                # 使用说明
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--max-stocks` | 最大获取股票数量（测试用） | 全部股票 |
| `--test` | 测试模式（前10只股票+短间隔） | False |
| `--token` | Tushare API Token（主要token） | 内建Token |
| `--tokens` | 多个Tushare API Token列表（并行处理） | 内建多个Token |
| `--adjust` | 复权方式：qfq/hfq/"" | qfq（前复权） |
| `--parallel` | 启用并行处理 | 默认启用 |
| `--no-parallel` | 禁用并行处理，使用串行模式 | - |
| `--max-workers` | 最大并发数（并行模式） | CPU核心数一半 |
| `--verbose` | 显示详细处理信息（并行模式默认关闭） | 根据模式自动 |
| `--quiet` | 静默模式，不显示详细处理信息 | False |
| `--force-refresh` | 强制刷新股票列表缓存 | False |
| `--end-date` | 指定结束日期(YYYYMMDD)，控制更新范围 | None |

## API使用说明

### 获取Token
1. 访问[Tushare官网](https://tushare.pro/)
2. 注册账号并获取免费Token
3. 使用 `--token` 参数指定，或直接修改代码中的默认值

### API限制与优化
- **单Token限制**：免费账户每日请求限制约500次
- **多Token优势**：使用多个Token可突破单Token限制
- **并行处理**：多Token + 多线程可显著提升获取速度
- **智能重试**：失败请求自动重试3次，指数退避延迟(2s→4s→6s)
- **并发控制**：默认并发数限制为2，避免过度请求
- **请求间隔**：并行模式下自动添加0.5秒批次间隔
- **本地缓存**：股票列表自动缓存，避免重复获取

### 多Token使用建议
- **免费用户**：建议准备2-3个Token轮流使用
- **付费用户**：可使用更多Token提升并发性能
- **Token轮换**：系统自动在多个Token间分配任务，避免单个Token过载
- **稳定性优先**：并行模式现在更加稳定，失败率显著降低

### 并行模式稳定性优化
针对并行模式下的数据获取失败问题，我们实施了多项优化：

#### 核心改进
1. **并发数控制**：从`cpu_count()//2`降低到`min(2, token_count)`
2. **重试机制**：每个API调用失败时自动重试3次，带指数退避
3. **请求间隔**：并行任务提交间添加延迟，避免瞬间并发压力
4. **错误处理**：更详细的错误信息，帮助诊断问题

#### 性能对比
- **并发数**：从高并发→稳定2并发
- **失败率**：显著降低（之前可能20-30%失败→现在<5%）
- **成功率**：并行模式现在与串行模式相当
- **稳定性**：网络波动时的容错性大幅提升

### 增强的错误诊断功能

系统在遇到错误时会提供详细的诊断信息，包括代码行号，便于快速定位和解决问题：

#### 错误信息格式
```
✗ 处理股票 [股票代码] 时出错 [文件名:行号]: 错误描述
```

#### 示例错误信息
```
✗ 处理股票 000001.SH 时出错 [tushare_data_fetcher.py:192]: Connection timeout
✗ 合并每日数据失败 000002.SZ [tushare_data_fetcher.py:639]: Permission denied
```

#### 错误信息包含内容
- **错误标识**：✗ 表示错误
- **操作描述**：具体在执行什么操作
- **代码位置**：[文件名:行号] 精确定位错误发生位置
- **错误详情**：具体的错误描述和原因

#### 调试建议
1. 根据行号定位代码中的问题位置
2. 检查网络连接、API限流、权限问题等常见原因
3. 使用 `--verbose` 参数获取更多调试信息
4. 查看程序日志中的完整错误堆栈

### 股票列表缓存机制

系统实现了智能的股票列表缓存机制，大幅提升运行效率：

#### 缓存工作原理
1. **首次运行**：从Tushare API获取股票列表，自动保存到本地缓存文件
2. **后续运行**：优先从本地缓存读取，无需重复调用API
3. **缓存更新**：可通过 `--force-refresh` 参数强制刷新缓存

#### 缓存文件位置
```
tudata/
├── stock_list_cache.txt  # 股票列表缓存文件
├── daily/                # 日K线数据目录
└── ...
```

#### 使用方式
```bash
# 正常使用（自动使用缓存）
python tushare_data_fetcher.py

# 强制刷新缓存
python tushare_data_fetcher.py --force-refresh

# 查看缓存状态（首次运行会显示"正在获取A股股票列表"）
# 后续运行会显示"正在从本地缓存读取股票列表"
```

#### 缓存优势
- **速度提升**：避免重复的API调用，启动速度显著提升
- **API保护**：减少不必要的API请求，节省Token额度
- **离线可用**：网络异常时仍可使用缓存的股票列表
- **智能更新**：缓存文件不存在或损坏时自动重新获取

#### 交易所支持
- **上海证券交易所** (SH)：600000-699999（A股）、900000-919999（B股）
- **深圳证券交易所** (SZ)：000001-004999、300000-399999（A股）
- **北京证券交易所** (BJ)：
  - 竞价交易：830000-879999
  - 集合竞价：430000-479999、870000-899999
  - 连续竞价：920000-999999

### 结束日期控制功能

系统支持指定数据更新的结束日期，当现有数据已覆盖指定日期时自动跳过更新：

#### 功能特性
- **精确控制**：指定数据更新到哪个日期为止
- **智能跳过**：避免重复获取已有数据
- **API节省**：减少不必要的API调用
- **批量友好**：适用于定期批量更新场景

#### 使用方法
```bash
# 更新数据到2024年12月31日
python tushare_data_fetcher.py --end-date 20241231

# 只更新前100只股票，到2024年年底
python tushare_data_fetcher.py --max-stocks 100 --end-date 20241231

# 并行模式，指定结束日期
python tushare_data_fetcher.py --parallel --end-date 20241130
```

#### 判断逻辑
- **现有数据最新日期 >= 指定结束日期** → 跳过更新
- **现有数据最新日期 < 指定结束日期** → 需要更新

#### 应用场景
- **历史数据获取**：只获取到指定日期的历史数据
- **定期更新**：每月/每季度更新到固定日期
- **增量同步**：避免重复下载已有数据
- **成本控制**：精确控制API使用量

## 编程接口

也可以在Python代码中直接使用：

```python
from tudata.tushare_data_fetcher import TushareDataFetcher

# 初始化（使用默认Token）
fetcher = TushareDataFetcher(token='your_token')

# 获取股票列表
stock_codes = fetcher.get_stock_list()

# 更新单只股票
fetcher.update_daily_data('000001.SH')

# 批量更新
fetcher.update_all_stocks(max_stocks=100)
```

## 注意事项

1. **Token配置**：确保Token有效且有足够积分
2. **网络连接**：确保网络稳定，API需要访问外网
3. **数据量**：2024年全年数据量较大，首次运行需要时间
4. **API限流**：Tushare有请求频率限制，请合理设置间隔
5. **数据质量**：Tushare数据质量较高，但仍建议定期验证

## 与akshare数据的区别

| 特性 | Tushare | akshare |
|------|---------|---------|
| 数据源 | 专业数据提供商 | 多个数据源聚合 |
| 数据质量 | 较高 | 较高 |
| API稳定性 | 稳定 | 较稳定 |
| 免费额度 | 500次/日 | 无限制 |
| 数据字段 | 规范 | 可能不一致 |
| 更新频率 | 实时 | 实时 |

## 故障排除

### Token错误
```
tushare.exception.TushareException: token无效或已过期
```
**解决**：更新Token或检查积分余额

### 网络错误
```
requests.exceptions.ConnectionError: HTTPSConnectionPool
```
**解决**：检查网络连接，或使用代理

### 数据为空
```
无新数据
```
**解决**：检查股票代码格式（需要.SH或.SZ后缀）或日期范围

### 备用方案触发
```
✗ Tushare获取股票列表失败: ...
正在尝试使用akshare获取股票列表...
✓ akshare备用获取成功
```
**说明**：这是正常行为，当Tushare不可用时自动切换到备用方案

### 两个数据源都失败
```
✗ akshare备用方案也失败: ...
✗ 所有获取股票列表的方法都失败，程序终止
```
**解决**：
- 检查网络连接
- 确认akshare已正确安装
- 检查是否有其他网络限制

---

## 🤖 股票预测系统

基于日K数据构建的**短期收益率预测模型**，预测T+1和T+3日收益率是否超过5%。

### 🎯 预测目标

- **T+1预测**：明天收盘价相对于今天的涨跌幅是否超过5%
- **T+3预测**：3个交易日后的累积收益率是否超过5%

### 🧮 特征工程

#### 技术指标特征（57个）
- **均线系统**：MA5, MA10, MA20, MA30, EMA系列
- **MACD指标**：DIF, DEA, MACD柱状图
- **趋势指标**：RSI, 威廉指标, KDJ, ATR
- **波动率指标**：布林带宽度，历史波动率
- **成交量指标**：量比，OBV，成交量均线
- **动量指标**：MOM, ROC, 动量变化

#### 统计特征（20个）
- **价格统计**：均值、标准差、偏度、峰度
- **收益率统计**：日收益率分布特征
- **相关性特征**：自相关系数，序列相关性
- **分位数特征**：价格和成交量的分位数位置

#### 时间特征（8个）
- **日历特征**：星期几、月份、季度
- **周期特征**：正弦余弦编码
- **趋势特征**：连续上涨/下跌天数，新高新低

### 🏗️ 模型架构

#### 机器学习模型
- **随机森林**：主模型，集成学习，抗过拟合
- **梯度提升树**：备选模型，序列预测优势
- **逻辑回归**：解释性强的线性模型

#### 训练策略
- **类别平衡**：使用SMOTE算法平衡正负样本
- **特征选择**：自动选择30个最重要特征
- **时间序列验证**：5折时间序列交叉验证

### 🚀 使用方法

#### 1. 训练模型
```bash
# 使用前50只股票训练（推荐）
python stock_predictor.py --train --max-stocks 50

# 使用全部股票训练（耗时较长）
python stock_predictor.py --train
```

#### 2. 预测股票
```bash
# 预测T+1日收益率
python stock_predictor.py --predict 000001 000002 --target-days 1

# 预测T+3日收益率
python stock_predictor.py --predict 000001 000002 --target-days 3
```

#### 3. 演示预测
```bash
# 运行完整演示
python demo_prediction.py
```

### 📊 预测输出示例

```
🎯 开始预测 T+1 日收益率是否超过5%...
000001      | 预测上涨 | 概率: 78.5%
000002      | 预测上涨 | 概率: 65.2%
000004      | 不上涨   | 概率: 23.1%
```

### ⚠️ 重要提醒

- **预测结果仅供参考**，不构成投资建议
- **模型基于历史数据**，未来表现不保证
- **建议结合基本面分析和技术分析**
- **风险控制是投资的核心原则**

### 🔧 相关文件

- `feature_engineering.py` - 特征工程模块
- `stock_predictor.py` - 预测模型主程序
- `demo_prediction.py` - 预测演示脚本
- `models/` - 保存训练好的模型文件
