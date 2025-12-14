# 每日股票数据组织方案

## 概述

当每天获取所有A股（约5457只）股票数据时，需要合理组织数据以便存储、查询和分析。本文档介绍几种数据组织方案及其优缺点。

## 当前数据结构

```
data/
├── daily/                    # 原始数据（每个股票一个文件）
│   ├── 000001_daily.csv     # 平安银行
│   ├── 000002_daily.csv     # 万科A
│   └── ...
├── organized/               # 组织化数据
│   ├── by_date/            # 按日期组织
│   ├── by_stock/           # 按股票组织（优化版）
│   └── summary/            # 汇总信息
```

## 数据组织方案对比

### 方案1：按股票存储（当前方案）
**文件结构：**
```
data/daily/
├── 000001_daily.csv    # 平安银行全部历史数据
├── 000002_daily.csv    # 万科A全部历史数据
└── ...
```

**优点：**
- ✅ 查询单只股票历史简单快速
- ✅ 数据更新方便（追加到对应文件）
- ✅ 文件损坏影响范围小
- ✅ 符合用户使用习惯

**缺点：**
- ❌ 文件数量过多（5457个文件）
- ❌ 难以查询某一天所有股票数据
- ❌ 统计分析需要读取多个文件

**适用场景：**
- 单股票分析
- 历史回溯分析
- 数据更新频繁

### 方案2：按日期存储
**文件结构：**
```
data/organized/by_date/
├── stocks_2024-12-13.csv    # 2024-12-13所有股票数据
├── stocks_2024-12-14.csv    # 2024-12-14所有股票数据
└── ...
```

**优点：**
- ✅ 查询某天所有股票数据高效
- ✅ 文件数量少（每天1个文件）
- ✅ 便于横向比较分析

**缺点：**
- ❌ 查询单股票历史需要读取多个文件
- ❌ 数据更新复杂（需要修改对应日期文件）
- ❌ 单文件过大（5457行×12列≈65KB/天）

**适用场景：**
- 日级别市场分析
- 横向比较研究
- 快照分析

### 方案3：数据库存储
**推荐方案：SQLite**
```python
import sqlite3
import pandas as pd

# 创建数据库
conn = sqlite3.connect('stocks_daily.db')

# 创建表
conn.execute('''
CREATE TABLE IF NOT EXISTS stock_daily (
    date TEXT,
    stock_code TEXT,
    open REAL,
    close REAL,
    high REAL,
    low REAL,
    volume REAL,
    amount REAL,
    amplitude REAL,
    change_pct REAL,
    change_amt REAL,
    turnover_rate REAL,
    PRIMARY KEY (date, stock_code)
)
''')

# 插入数据
df.to_sql('stock_daily', conn, if_exists='append', index=False)
```

**优点：**
- ✅ 查询灵活（单股票、日期范围、条件筛选）
- ✅ 数据完整性保证
- ✅ 支持复杂分析查询
- ✅ 存储效率高

**缺点：**
- ❌ 需要数据库知识
- ❌ 部署稍复杂
- ❌ 文件大小相对较大

**适用场景：**
- 复杂数据分析
- 大规模数据处理
- 多维度查询

### 方案4：混合存储（推荐）
**结合方案1和方案2的优点：**

```
data/
├── daily/                    # 主要存储：按股票
├── organized/
│   ├── by_date/             # 每日快照（可选）
│   ├── by_stock/            # 优化版股票文件
│   └── summary/             # 统计信息
└── stocks.db                # 数据库（可选）
```

**实现策略：**
1. **主要存储**：按股票存储（方案1）
2. **每日快照**：生成当天所有股票数据文件（方案2）
3. **统计索引**：日期索引和数据质量报告
4. **数据库**：可选，用于复杂查询

## 推荐的数据更新流程

### 1. 每日数据获取
```python
# 获取昨天的数据
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
fetcher = StockDataFetcher()
fetcher.fetch_all_daily_data(
    start_date=yesterday,
    end_date=yesterday
)
```

### 2. 数据组织
```python
organizer = DailyStockDataOrganizer()
organizer.organize_all()
```

### 3. 数据验证
```python
# 检查数据质量
quality_report = organizer.create_data_quality_report()
print("数据质量检查完成")
```

## 文件大小估算

**A股每日数据量估算：**
- 股票数量：~5457只
- 每日记录：1条/股票
- 字段数量：12个
- 单条记录大小：~200字节
- 每日总大小：~1MB

**存储空间需求：**
- 1年数据：~365MB
- 3年数据：~1GB
- 10年数据：~3.5GB

## 性能优化建议

### 1. 文件格式选择
- **CSV**: 简单易读，人 readable
- **Parquet**: 压缩更好，查询更快（推荐用于大数据）
- **SQLite**: 支持复杂查询，最灵活

### 2. 压缩策略
```python
# 使用gzip压缩CSV
df.to_csv('data.csv.gz', compression='gzip')

# 使用Parquet
df.to_parquet('data.parquet', compression='snappy')
```

### 3. 分区存储
```
data/
├── daily/
│   ├── 2024/
│   │   ├── 000001_daily.csv
│   │   └── ...
│   └── 2025/
└── ...
```

### 4. 索引优化
- 为常用查询字段建立索引
- 使用数据分区减少查询范围

## 总结

**推荐方案：按股票存储 + 混合组织**

1. **主要存储**：按股票存储（方案1），便于单股票查询和更新
2. **补充组织**：按日期生成快照，便于横向分析
3. **质量监控**：定期生成数据质量报告
4. **可选数据库**：复杂分析时使用SQLite

这种方案平衡了存储效率、查询性能和使用便利性，适合大多数股票数据分析场景。

