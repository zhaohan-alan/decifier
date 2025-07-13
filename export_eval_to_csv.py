#!/usr/bin/env python3
import pickle
import gzip
import pandas as pd
import numpy as np

def load_pickle_gz(filepath):
    """加载压缩的pickle文件"""
    try:
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def export_to_csv():
    filepath = "eval_files/noma-10k_collected.pkl.gz/noma-10k.pkl.gz"
    
    print(f"正在加载文件: {filepath}")
    data = load_pickle_gz(filepath)
    
    if data is not None:
        print("文件加载成功!")
        
        if isinstance(data, pd.DataFrame):
            print(f"数据形状: {data.shape}")
            print(f"列名: {list(data.columns)}")
            
            # 导出为CSV
            csv_filename = "eval_results_noma-10k.csv"
            data.to_csv(csv_filename, index=False, encoding='utf-8')
            print(f"数据已导出为CSV文件: {csv_filename}")
            
            # 显示基本统计信息
            print("\n基本统计信息:")
            print("-" * 50)
            
            # 数值列的统计
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                print("数值列统计:")
                print(data[numeric_columns].describe())
            
            # 布尔列的统计
            bool_columns = data.select_dtypes(include=[bool]).columns
            if len(bool_columns) > 0:
                print("\n布尔列统计:")
                for col in bool_columns:
                    true_count = data[col].sum()
                    total_count = len(data[col])
                    percentage = (true_count / total_count * 100) if total_count > 0 else 0
                    print(f"{col}: {true_count}/{total_count} ({percentage:.2f}%)")
            
            # 显示前几行
            print("\n前5行数据:")
            print("-" * 50)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', 50)
            print(data.head())
            
        else:
            print("数据不是DataFrame格式")
    else:
        print("无法加载文件")

if __name__ == "__main__":
    export_to_csv() 