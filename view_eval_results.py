#!/usr/bin/env python3
import pickle
import gzip
import pandas as pd
import numpy as np
from pprint import pprint

def load_pickle_gz(filepath):
    """加载压缩的pickle文件"""
    try:
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def explore_data(data, max_items=10):
    """探索数据结构"""
    print(f"数据类型: {type(data)}")
    print(f"数据长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")
    print("-" * 50)
    
    if isinstance(data, dict):
        print("字典键值:")
        for key, value in list(data.items())[:max_items]:
            print(f"  {key}: {type(value)}")
            if hasattr(value, '__len__') and not isinstance(value, str):
                print(f"    长度: {len(value)}")
        
        # 如果有太多键，显示一个总结
        if len(data) > max_items:
            print(f"  ... 还有 {len(data) - max_items} 个键")
    
    elif isinstance(data, list):
        print("列表内容:")
        for i, item in enumerate(data[:max_items]):
            print(f"  [{i}]: {type(item)}")
            if isinstance(item, dict):
                print(f"    键: {list(item.keys())[:5]}")
            elif hasattr(item, '__len__') and not isinstance(item, str):
                print(f"    长度: {len(item)}")
        
        if len(data) > max_items:
            print(f"  ... 还有 {len(data) - max_items} 个项目")
    
    elif isinstance(data, pd.DataFrame):
        print("DataFrame信息:")
        print(f"  形状: {data.shape}")
        print(f"  列名: {list(data.columns)}")
        print("  前几行:")
        print(data.head())
    
    else:
        print(f"数据内容: {data}")

def main():
    filepath = "eval_files/noma-10k_collected.pkl.gz/noma-10k.pkl.gz"
    
    print(f"正在加载文件: {filepath}")
    data = load_pickle_gz(filepath)
    
    if data is not None:
        print("文件加载成功!")
        explore_data(data)
        
        # 如果是字典，尝试查看一些关键字段
        if isinstance(data, dict):
            print("\n" + "="*50)
            print("详细信息:")
            
            # 常见的评估指标字段
            metrics_keys = ['accuracy', 'loss', 'mse', 'mae', 'r2_score', 'rwp', 'rmsd', 'success_rate', 'generation_time']
            
            for key in metrics_keys:
                if key in data:
                    value = data[key]
                    print(f"{key}: {value}")
                    if isinstance(value, (list, np.ndarray)):
                        if len(value) > 0:
                            print(f"  均值: {np.mean(value):.4f}")
                            print(f"  标准差: {np.std(value):.4f}")
                            print(f"  最小值: {np.min(value):.4f}")
                            print(f"  最大值: {np.max(value):.4f}")
            
            # 如果有results字段，尝试查看
            if 'results' in data:
                print("\n结果详情:")
                results = data['results']
                explore_data(results, max_items=3)
    
    else:
        print("无法加载文件")

if __name__ == "__main__":
    main() 