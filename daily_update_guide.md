# 按股票存储的每日数据更新策略

## 概述

当按股票存储数据时（每个股票一个CSV文件），需要设计合理的更新策略来处理每日新增数据，同时保证历史数据的完整性和准确性。

## 数据存储结构

```
data/daily/
├── 000001_daily.csv    # 平安银行完整历史数据
├── 000002_daily.csv    # 万科A完整历史数据
└── ...
```

## 每日更新流程

### 1. 数据获取策略

#### 单只股票更新
```python
def update_single_stock_daily(stock_code: str, target_date: str = None):
    """
    更新单只股票的每日数据

    Args:
        stock_code: 股票代码
        target_date: 目标日期，默认为昨天
    """
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

    # 1. 获取指定日期的数据
    new_data = fetcher.fetch_daily_data(stock_code, target_date, target_date)

    # 2. 合并到现有数据
    if new_data is not None:
        merge_daily_data(stock_code, new_data)
```

#### 批量更新策略
```python
def update_all_stocks_daily():
    """批量更新所有股票的每日数据"""
    # 1. 获取股票列表（可缓存）
    stock_list = get_stock_list()
    stock_codes = stock_list['code'].tolist()

    # 2. 并行或顺序更新
    for stock_code in tqdm(stock_codes):
        update_single_stock_daily(stock_code)

    # 3. 生成更新报告
    generate_update_report()
```

### 2. 历史数据融合策略

#### 数据合并逻辑
```python
def merge_daily_data(stock_code: str, new_data: pd.DataFrame):
    """
    安全合并每日数据

    策略：
    1. 加载现有数据
    2. 检查重复日期
    3. 合并并去重（新数据优先）
    4. 排序保存
    """
    filepath = f"data/daily/{stock_code}_daily.csv"

    # 1. 加载现有数据
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        existing_df['日期'] = pd.to_datetime(existing_df['日期'])
    else:
        existing_df = pd.DataFrame()

    # 2. 准备新数据
    new_data['日期'] = pd.to_datetime(new_data['日期'])

    # 3. 检查重复
    existing_dates = set(existing_df['日期']) if not existing_df.empty else set()
    new_dates = set(new_data['日期'])

    overlapping = existing_dates & new_dates
    if overlapping:
        print(f"发现 {len(overlapping)} 天重复数据，将使用新数据覆盖")

    # 4. 合并数据
    if existing_df.empty:
        combined = new_data
    else:
        # 保留非重复的旧数据 + 新数据
        old_filtered = existing_df[~existing_df['日期'].isin(overlapping)]
        combined = pd.concat([old_filtered, new_data], ignore_index=True)

    # 5. 去重和排序
    combined = combined.drop_duplicates(subset=['日期'], keep='last')
    combined = combined.sort_values('日期')

    # 6. 保存
    combined['日期'] = combined['日期'].dt.strftime('%Y-%m-%d')
    combined.to_csv(filepath, index=False, encoding='utf-8-sig')
```

## 重复数据处理策略

### 1. 日期重复处理
- **策略**: 新数据优先覆盖旧数据
- **原因**: API数据可能有修正，最新数据更准确

### 2. 字段重复处理
- **策略**: 完整记录替换
- **原因**: 避免部分字段更新导致的数据不一致

### 3. 数据验证
```python
def validate_merged_data(df: pd.DataFrame) -> bool:
    """验证合并后的数据"""
    # 1. 检查日期唯一性
    if df['日期'].duplicated().any():
        return False

    # 2. 检查日期连续性（可选）
    df_sorted = df.sort_values('日期')
    gaps = (df_sorted['日期'].diff().dt.days > 1).sum()
    if gaps > 0:
        print(f"警告: 发现 {gaps} 个数据缺口")

    # 3. 检查数据完整性
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"警告: 发现 {missing_values} 个缺失值")

    return True
```

## 性能优化策略

### 1. 增量更新优化
```python
# 只更新需要的数据
def smart_update_strategy():
    """智能更新策略"""
    # 1. 检查哪些股票需要更新
    need_update = check_update_needed()

    # 2. 优先更新活跃股票
    active_stocks = get_active_stocks()

    # 3. 分批更新，避免内存溢出
    batch_size = 100
    for i in range(0, len(need_update), batch_size):
        batch = need_update[i:i+batch_size]
        update_batch(batch)
```

### 2. 存储优化
```python
def optimize_storage():
    """存储优化"""
    # 1. 定期压缩旧数据
    compress_old_data()

    # 2. 分区存储
    organize_by_year()

    # 3. 清理异常数据
    cleanup_invalid_data()
```

### 3. 并发更新
```python
import concurrent.futures

def parallel_update(stocks: List[str], max_workers: int = 4):
    """并行更新"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(update_single_stock, stock) for stock in stocks]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results
```

## 错误处理和重试机制

### 1. API失败处理
```python
def update_with_retry(stock_code: str, max_retries: int = 3):
    """带重试的更新"""
    for attempt in range(max_retries):
        try:
            return update_single_stock_daily(stock_code)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 60  # 递增等待
                print(f"重试 {stock_code} ({attempt+1}/{max_retries}) 等待 {wait_time}s")
                time.sleep(wait_time)
            else:
                log_error(f"更新 {stock_code} 最终失败: {e}")
                return False
```

### 2. 数据完整性检查
```python
def post_update_validation():
    """更新后验证"""
    # 1. 检查文件完整性
    # 2. 验证数据一致性
    # 3. 生成更新报告
    # 4. 备份重要数据
    pass
```

## 监控和告警

### 1. 更新状态监控
```python
class UpdateMonitor:
    def __init__(self):
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_stocks': 0,
            'success_count': 0,
            'fail_count': 0,
            'error_details': []
        }

    def log_success(self, stock_code):
        self.success_count += 1

    def log_error(self, stock_code, error):
        self.fail_count += 1
        self.error_details.append({'stock': stock_code, 'error': error})

    def generate_report(self):
        """生成更新报告"""
        success_rate = self.success_count / self.total_stocks * 100

        report = f"""
        更新报告
        ========
        开始时间: {self.start_time}
        结束时间: {self.end_time}
        总股票数: {self.total_stocks}
        成功数量: {self.success_count}
        失败数量: {self.fail_count}
        成功率: {success_rate:.1f}%

        失败详情:
        {self.error_details[:10]}  # 显示前10个错误
        """

        return report
```

### 2. 告警机制
```python
def send_alert(message: str):
    """发送告警"""
    # 可以通过邮件、微信、钉钉等发送
    if success_rate < 90:  # 成功率低于90%告警
        alert(f"股票数据更新异常: {message}")
```

## 备份和恢复策略

### 1. 自动备份
```python
def backup_before_update():
    """更新前备份"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f"backup/daily_{timestamp}"

    # 复制整个data/daily目录
    shutil.copytree("data/daily", backup_dir)
    print(f"备份完成: {backup_dir}")
```

### 2. 增量备份
```python
def incremental_backup(changed_files: List[str]):
    """增量备份，只备份修改的文件"""
    for filepath in changed_files:
        backup_path = filepath.replace("data/", "backup/")
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(filepath, backup_path)
```

## 最佳实践总结

### 1. 更新频率
- **每日更新**: 收盘后更新前一天数据
- **定时任务**: 使用cron或调度器自动执行
- **手动更新**: 特殊情况下的手动干预

### 2. 数据一致性
- **原子性**: 每个股票的更新是原子操作
- **事务性**: 更新失败时回滚到备份状态
- **验证**: 更新后验证数据完整性

### 3. 性能优化
- **批量处理**: 避免逐个更新
- **并发控制**: 合理使用多线程
- **资源管理**: 控制内存和网络使用

### 4. 监控告警
- **成功率监控**: 低于阈值时告警
- **错误日志**: 详细记录失败原因
- **性能监控**: 监控更新耗时

这种按股票存储的更新策略既保证了数据的完整性和准确性，又提供了良好的性能和可维护性。
