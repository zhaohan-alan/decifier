#!/usr/bin/env python3

import os
import sys
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from decifer.decifer_dataset import DeciferDataset
from decifer.tokenizer import Tokenizer

# 初始化tokenizer以获取padding ID
TOKENIZER = Tokenizer()
PADDING_ID = TOKENIZER.padding_id
START_ID = TOKENIZER.token_to_id["data_"]
NEWLINE_ID = TOKENIZER.token_to_id["\n"]

def debug_data_loading():
    """Debug data loading process"""
    print("=== 开始调试数据加载 ===")
    
    dataset_path = "data/noma/noma-1k"
    
    # 1. 检查文件是否存在
    train_file = os.path.join(dataset_path, "serialized/train.h5")
    print(f"检查训练文件: {train_file}")
    if not os.path.exists(train_file):
        print("❌ 训练文件不存在!")
        return
    print("✅ 训练文件存在")
    
    # 2. 测试数据集初始化
    print("初始化数据集...")
    try:
        dataset_fields = ["cif_tokens", "xrd.q", "xrd.iq"]
        train_dataset = DeciferDataset(train_file, dataset_fields)
        print(f"✅ 数据集初始化成功，样本数: {len(train_dataset)}")
    except Exception as e:
        print(f"❌ 数据集初始化失败: {e}")
        return
    
    # 3. 测试单个样本加载
    print("测试单个样本加载...")
    try:
        start_time = time.time()
        sample = train_dataset[0]
        load_time = time.time() - start_time
        print(f"✅ 样本加载成功，耗时: {load_time:.2f}秒")
        print(f"   CIF tokens长度: {len(sample['cif_tokens'])}")
        print(f"   XRD Q值长度: {len(sample['xrd.q'])}")
        print(f"   XRD IQ值长度: {len(sample['xrd.iq'])}")
    except Exception as e:
        print(f"❌ 样本加载失败: {e}")
        return
    
    # 4. 定义collate_fn来处理不同长度的序列
    def collate_fn(batch):
        # batch is a list of dictionaries
        batch_data = {}
        for key in batch[0].keys():
            field_data = [item[key] for item in batch]
            # Pad the sequences to the maximum length in the batch
            if "xrd" in key:
                padded_seqs = pad_sequence(field_data, batch_first=True, padding_value=0.0)
                batch_data[key] = padded_seqs
            elif "cif" in key:
                padded_seqs = pad_sequence(field_data, batch_first=True, padding_value=PADDING_ID)
                batch_data[key] = padded_seqs
            else:
                batch_data[key] = field_data  # Leave as is

        return batch_data
    
    # 5. 测试DataLoader（带collate_fn）
    print("测试DataLoader...")
    try:
        batch_size = 2
        sampler = SubsetRandomSampler(range(min(4, len(train_dataset))))
        dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
        
        start_time = time.time()
        batch = next(iter(dataloader))
        load_time = time.time() - start_time
        print(f"✅ 批次加载成功，耗时: {load_time:.2f}秒")
        print(f"   批次大小: {len(batch['cif_tokens'])}")
        print(f"   CIF tokens形状: {batch['cif_tokens'].shape}")
        print(f"   XRD Q值形状: {batch['xrd.q'].shape}")
        print(f"   XRD IQ值形状: {batch['xrd.iq'].shape}")
    except Exception as e:
        print(f"❌ 批次加载失败: {e}")
        return
    
    # 6. 测试XRD数据增强
    print("测试XRD数据增强...")
    try:
        from decifer.utility import discrete_to_continuous_xrd
        augmentation_kwargs = {
            'qmin': 0.0,
            'qmax': 10.0,
            'qstep': 0.01,
            'fwhm_range': (0.001, 0.1),
            'eta_range': (0.5, 0.5),
            'noise_range': (0.001, 0.05),
            'intensity_scale_range': (1.0, 1.0),
            'mask_prob': 0.0,
        }
        
        start_time = time.time()
        result = discrete_to_continuous_xrd(batch['xrd.q'], batch['xrd.iq'], **augmentation_kwargs)
        aug_time = time.time() - start_time
        print(f"✅ XRD数据增强成功，耗时: {aug_time:.2f}秒")
        print(f"   增强后数据长度: {len(result['iq'])}")
        print(f"   增强后数据形状: {result['iq'][0].shape}")
    except Exception as e:
        print(f"❌ XRD数据增强失败: {e}")
        return
    
    # 7. 测试完整的get_batch逻辑
    print("测试完整的get_batch逻辑...")
    try:
        # 模拟get_batch的复杂逻辑
        block_size = 3076  # 来自配置文件
        batch_size = 32    # 来自配置文件
        
        # 重新创建dataloader来模拟真实场景
        class RandomBatchSampler:
            def __init__(self, sampler, batch_size, drop_last=False):
                self.sampler = sampler
                self.batch_size = batch_size
                self.drop_last = drop_last
                
            def __iter__(self):
                import random
                batch_indices = list(self.sampler)
                random.shuffle(batch_indices)
                for i in range(0, len(batch_indices), self.batch_size):
                    yield batch_indices[i:i + self.batch_size]
        
        train_sampler = SubsetRandomSampler(range(len(train_dataset)))
        train_batch_sampler = RandomBatchSampler(train_sampler, batch_size=batch_size, drop_last=False)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=0, collate_fn=collate_fn)
        
        data_iter = iter(train_dataloader)
        
        # 模拟get_batch的逻辑
        start_time = time.time()
        
        # 初始化列表
        cond_list = []
        total_sequences = []
        
        # 收集序列直到达到batch_size
        while len(total_sequences) < batch_size:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)
            
            # 处理CIF tokens
            sequences = batch['cif_tokens']
            sequences = [torch.cat([seq[seq != PADDING_ID], torch.tensor([NEWLINE_ID, NEWLINE_ID], dtype=torch.long)]) for seq in sequences]
            total_sequences.extend(sequences)
            
            # 处理XRD数据
            cond_list.extend(discrete_to_continuous_xrd(batch['xrd.q'], batch['xrd.iq'], **augmentation_kwargs)['iq'])
        
        # 打包序列
        all_tokens = torch.cat(total_sequences)
        seq_lengths = torch.tensor([len(seq) for seq in total_sequences])
        seq_cum_lengths = torch.cumsum(seq_lengths, dim=0)
        
        # 计算批次
        num_full_blocks = all_tokens.size(0) // block_size
        num_batches = min(batch_size, num_full_blocks)
        
        if num_batches == 0:
            print("⚠️  警告：无法创建完整的块，数据可能太短")
            return
        
        # 截断和重塑
        total_tokens = all_tokens[:num_batches * block_size]
        total_tokens = total_tokens.view(num_batches, block_size)
        
        # 创建输入输出
        X_batch = total_tokens[:, :-1]
        Y_batch = total_tokens[:, 1:]
        
        # 找到开始索引
        start_token_mask = X_batch == START_ID
        start_indices = start_token_mask.nonzero(as_tuple=False)
        
        # 处理条件数据
        index = torch.searchsorted(seq_cum_lengths, num_batches * block_size) + 1
        cond_list = cond_list[:index]
        cond_batch = torch.stack(cond_list)
        
        get_batch_time = time.time() - start_time
        
        print(f"✅ get_batch逻辑成功，耗时: {get_batch_time:.2f}秒")
        print(f"   X_batch形状: {X_batch.shape}")
        print(f"   Y_batch形状: {Y_batch.shape}")
        print(f"   cond_batch形状: {cond_batch.shape}")
        print(f"   开始索引数量: {len(start_indices)}")
        
    except Exception as e:
        print(f"❌ get_batch逻辑失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("=== 调试完成 ===")

if __name__ == "__main__":
    debug_data_loading() 