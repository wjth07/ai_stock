#!/usr/bin/env python
"""
测试代理连接
"""
import os
import sys

print("=" * 60)
print("测试代理连接")
print("=" * 60)

# 测试1: 不使用代理
print("\n[测试1] 不使用代理...")
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'

try:
    import akshare as ak
    import pandas as pd
    df = ak.stock_zh_a_hist(symbol='000001', period='daily', start_date='20241213', end_date='20241213', adjust='')
    if df is not None and not df.empty:
        print("✓ 不使用代理：成功获取数据")
        print(f"  数据: {len(df)} 条记录")
    else:
        print("✗ 不使用代理：无数据返回")
except Exception as e:
    print(f"✗ 不使用代理：失败 - {str(e)[:100]}")

# 测试2: 使用系统代理
print("\n[测试2] 使用系统代理 (127.0.0.1:7897)...")
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'
os.environ.pop('NO_PROXY', None)
os.environ.pop('no_proxy', None)

try:
    # 重新导入以应用新的环境变量
    import importlib
    import akshare
    importlib.reload(akshare)
    import pandas as pd
    df = ak.stock_zh_a_hist(symbol='000001', period='daily', start_date='20241213', end_date='20241213', adjust='')
    if df is not None and not df.empty:
        print("✓ 使用代理：成功获取数据")
        print(f"  数据: {len(df)} 条记录")
    else:
        print("✗ 使用代理：无数据返回")
except Exception as e:
    print(f"✗ 使用代理：失败 - {str(e)[:100]}")

print("\n" + "=" * 60)
print("建议：")
print("1. 如果测试1成功，说明不使用代理可以正常工作")
print("2. 如果测试2成功，说明VPN代理可以正常工作")
print("3. 如果都失败，可能是网络或API服务器问题")
print("=" * 60)

