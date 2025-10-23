import os
import sys
import torch
import pandas as pd
import time

# 添加当前目录到Python路径
sys.path.append('/root/Map-Mix')

# 导入必要的模块
from config import LIDConfig
from Datasets.datasetLID import LIDDataset, collate_fn

def debug_csv_files():
    """调试CSV文件是否存在且可读取"""
    print("=== CSV文件调试 ===")
    csv_paths = {
        'train': LIDConfig.train_path,
        'test': LIDConfig.test_path,
        'val': LIDConfig.val_path
    }
    
    for name, path in csv_paths.items():
        print(f"\n检查{name} CSV文件: {path}")
        if os.path.exists(path):
            print(f"✓ 文件存在，大小: {os.path.getsize(path)/1024:.2f} KB")
            try:
                # 读取少量行来测试
                df = pd.read_csv(path, nrows=5)
                print(f"✓ 成功读取，前5行数据:")
                print(df.head())
                print(f"✓ 总共有 {len(pd.read_csv(path))} 条数据")
            except Exception as e:
                print(f"✗ 读取文件出错: {e}")
        else:
            print(f"✗ 文件不存在")
    print("=== CSV文件调试完成 ===")

def debug_single_sample():
    """调试单个样本的加载"""
    print("\n=== 单个样本加载调试 ===")
    
    # 创建简单的参数字典
    hparams = {
        'batch_size': 1,
        'n_workers': 0
    }
    
    try:
        # 只加载前5个样本进行测试
        dataset = LIDDataset(
            CSVPath=LIDConfig.train_path,
            hparams=hparams,
            is_train=True
        )
        
        print(f"✓ 数据集初始化成功，总样本数: {len(dataset)}")
        print(f"✓ 类别映射: {dataset.classes}")
        
        # 测试加载前3个样本
        for i in range(min(3, len(dataset))):
            print(f"\n加载样本 {i}:")
            start_time = time.time()
            try:
                wav, wav_duration, label = dataset[i]
                load_time = time.time() - start_time
                print(f"  ✓ 加载成功 (耗时: {load_time:.4f}s)")
                print(f"  ✓ 音频形状: {wav.shape}")
                print(f"  ✓ 音频时长: {wav_duration.item()}")
                print(f"  ✓ 标签形状: {label.shape}")
                print(f"  ✓ 标签类型: {label.dtype}")
            except Exception as e:
                print(f"  ✗ 加载失败: {e}")
                # 打印样本信息以便调试
                try:
                    sample_info = dataset.data[i]
                    print(f"  样本信息: {sample_info}")
                except:
                    pass
                    
    except Exception as e:
        print(f"✗ 数据集初始化失败: {e}")
    
    print("=== 单个样本加载调试完成 ===")

def debug_dataloader():
    """调试DataLoader"""
    print("\n=== DataLoader调试 ===")
    
    hparams = {
        'batch_size': 2,  # 小批量以便调试
        'n_workers': 0   # 使用0个worker避免多进程问题
    }
    
    try:
        dataset = LIDDataset(
            CSVPath=LIDConfig.train_path,
            hparams=hparams,
            is_train=True
        )
        
        # 创建DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=hparams['batch_size'],
            shuffle=False,  # 不打乱以便调试
            num_workers=hparams['n_workers'],
            collate_fn=collate_fn
        )
        
        print(f"✓ DataLoader创建成功，批次总数: {len(dataloader)}")
        
        # 测试加载第一个批次
        print("\n加载第一个批次:")
        start_time = time.time()
        for i, batch in enumerate(dataloader):
            if i == 0:  # 只测试第一个批次
                x, x_len, y_l = batch
                load_time = time.time() - start_time
                print(f"  ✓ 批次加载成功 (耗时: {load_time:.4f}s)")
                print(f"  ✓ 音频数据形状: {x.shape}")
                print(f"  ✓ 音频长度: {x_len}")
                print(f"  ✓ 标签数量: {len(y_l)}")
                print(f"  ✓ 单个标签形状: {y_l[0].shape}")
                break
        
    except Exception as e:
        print(f"✗ DataLoader测试失败: {e}")
    
    print("=== DataLoader调试完成 ===")

def debug_torchaudio():
    """调试torchaudio加载"""
    print("\n=== torchaudio加载调试 ===")
    
    try:
        import torchaudio
        print(f"✓ torchaudio版本: {torchaudio.__version__}")
        
        # 读取CSV中的第一个音频文件路径
        df = pd.read_csv(LIDConfig.train_path, nrows=1)
        audio_path = df.iloc[0, 0]
        
        print(f"\n测试音频文件: {audio_path}")
        if os.path.exists(audio_path):
            print(f"✓ 文件存在")
            try:
                # 直接使用torchaudio加载
                start_time = time.time()
                waveform, sample_rate = torchaudio.load(audio_path)
                load_time = time.time() - start_time
                print(f"✓ 成功加载音频")
                print(f"  - 波形形状: {waveform.shape}")
                print(f"  - 采样率: {sample_rate}")
                print(f"  - 音频时长: {waveform.shape[1]/sample_rate:.4f}秒")
                print(f"  - 加载耗时: {load_time:.4f}秒")
            except Exception as e:
                print(f"✗ 加载音频出错: {e}")
                # 尝试使用其他方法
                try:
                    import librosa
                    print("\n尝试使用librosa加载:")
                    y, sr = librosa.load(audio_path, sr=None)
                    print(f"✓ librosa加载成功")
                    print(f"  - 音频形状: {y.shape}")
                    print(f"  - 采样率: {sr}")
                except Exception as e2:
                    print(f"✗ librosa加载也失败: {e2}")
        else:
            print(f"✗ 文件不存在")
    
    except ImportError:
        print("✗ 未安装torchaudio")
    except Exception as e:
        print(f"✗ torchaudio测试出错: {e}")
    
    print("=== torchaudio加载调试完成 ===")

if __name__ == "__main__":
    print("开始调试数据集加载...")
    
    # 1. 测试CSV文件
    debug_csv_files()
    
    # 2. 测试单个样本加载
    debug_single_sample()
    
    # 3. 测试DataLoader
    debug_dataloader()
    
    # 4. 测试torchaudio加载
    debug_torchaudio()
    
    print("\n数据集加载调试完成！")