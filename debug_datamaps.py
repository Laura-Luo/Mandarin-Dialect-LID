import os
import torch
import torch.utils.data as data
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from config import LIDConfig
from Datasets.datasetLID import LIDDataset, collate_fn
from Models.lightning import LightningModel
from Models.datamaps import DataMapsCallback, embed_datamaps_into_dataframe, generate_maps_plots_from_dataframe

# 设置随机种子确保可复现性
SEED = 100
pl.seed_everything(SEED)
torch.manual_seed(SEED)

class SmallBatchCallback(pl.callbacks.Callback):
    """
    自定义回调，只运行少量批次后停止
    """
    def __init__(self, max_batches=3):
        self.max_batches = max_batches
        self.batch_count = 0
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.batch_count += 1
        if self.batch_count >= self.max_batches:
            print(f"\n已运行{self.max_batches}个批次，正在停止训练...")
            trainer.should_stop = True

def debug_datamaps():
    """
    调试datamaps生成过程的函数
    """
    # 使用少量数据配置
    print("开始调试datamaps生成过程...")
    
    # 创建普通字典而不是mappingproxy
    HPARAMS = {
        'batch_size': 8,
        'n_workers': 2,
        'gpu': 1,
        'dev': False,
        'model_checkpoint': LIDConfig.model_checkpoint,
        'model_type': LIDConfig.model_type,
        'upstream_model': LIDConfig.upstream_model,
        'mixup_type': LIDConfig.mixup_type,
        'cluster': LIDConfig.cluster,
        'unfreeze_last_conv_layers': LIDConfig.unfreeze_last_conv_layers,
        'noise_dataset_path': None,
        'train_path': LIDConfig.train_path,
        'val_path': LIDConfig.val_path,
        'test_path': LIDConfig.test_path,
        'feature_dim': LIDConfig.feature_dim,
        'lr': LIDConfig.lr,
        'epochs': LIDConfig.epochs
    }
    
    # 为数据集创建一个简单的hparams对象
    class SimpleHParams:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    dataset_hparams = SimpleHParams(**HPARAMS)
    
    # 加载数据集（只使用少量数据）
    print("加载数据集（只使用少量数据）...")
    train_set = LIDDataset(
        CSVPath = LIDConfig.train_path,
        hparams = dataset_hparams,
        is_train=True,
    )
    
    # 只使用前几个样本进行调试
    debug_size = min(24, len(train_set))  # 使用24个样本（3个批次）
    train_subset = data.Subset(train_set, list(range(debug_size)))
    
    trainloader = data.DataLoader(
        train_subset,
        batch_size=HPARAMS['batch_size'],
        shuffle=False,  # 不打乱以便于调试
        num_workers=HPARAMS['n_workers'],
        collate_fn=collate_fn,
    )
    
    # 创建模型和回调
    print("初始化模型和回调...")
    model = LightningModel(HPARAMS)
    
    # 创建DataMapsCallback和停止回调
    datamaps_cb = DataMapsCallback(log_dir='./debug_plots', dataset_length=len(train_subset))
    small_batch_callback = SmallBatchCallback(max_batches=3)
    
    # 创建训练器
    trainer = pl.Trainer(
        fast_dev_run=False,
        devices=HPARAMS['gpu'],
        accelerator='gpu' if HPARAMS['gpu'] > 0 else 'cpu',
        max_epochs=1,
        callbacks=[datamaps_cb, small_batch_callback],
        logger=False,  # 禁用日志器以加快速度
    )
    
    # 运行训练
    print("开始运行少量批次的训练...")
    trainer.fit(model, trainloader)
    
    # 分析收集的metrics
    print(f"\n分析收集的metrics: {len(datamaps_cb.metrics)}个样本")
    
    # 检查metrics数据结构
    if len(datamaps_cb.metrics) > 0:
        sample_key = list(datamaps_cb.metrics.keys())[0]
        sample_metric = datamaps_cb.metrics[sample_key]
        print(f"\n样本metrics结构示例 ({sample_key}):")
        print(f"  - 'probs'类型: {type(sample_metric['probs'])}")
        if sample_metric['probs']:
            print(f"    - 第一个probs元素类型: {type(sample_metric['probs'][0])}")
            print(f"    - probs长度: {len(sample_metric['probs'])}")
        print(f"  - 'corrects'类型: {type(sample_metric['corrects'])}")
        if sample_metric['corrects']:
            print(f"    - corrects长度: {len(sample_metric['corrects'])}")
    
    # 创建简化的DataFrame进行测试
    print("\n创建简化的DataFrame进行测试...")
    # 只使用我们实际训练的样本对应的CSV行
    full_df = pd.read_csv(train_set.CSVPath).reset_index(drop=True)
    df = full_df.iloc[:debug_size].copy()
    
    # 打印DataFrame信息
    print(f"测试DataFrame形状: {df.shape}")
    print(f"DataFrame列名: {df.columns.tolist()}")
    
    # 尝试执行embed_datamaps_into_dataframe函数
    print("\n测试embed_datamaps_into_dataframe函数...")
    try:
        # 修复代码中的错误 - 确保probs已经是列表格式
        # 预处理metrics，确保probs是列表而不是需要调用tolist()的张量
        for key, metric in datamaps_cb.metrics.items():
            # 检查probs中的元素类型，如果是tensor或numpy数组则转换为列表
            if metric['probs'] and isinstance(metric['probs'][0], (torch.Tensor, np.ndarray)):
                metric['probs'] = [p.tolist() if isinstance(p, (torch.Tensor, np.ndarray)) else p for p in metric['probs']]
            # 确保confidence等计算正确
            metric['confidence'] = np.mean(metric['probs']) if metric['probs'] else 0.0
            metric['variability'] = np.std(metric['probs']) if metric['probs'] else 0.0
            metric['correctness'] = np.mean(metric['corrects']) if metric['corrects'] else 0.0
        
        # 调用embed函数
        df_result = embed_datamaps_into_dataframe(df, datamaps_cb.metrics)
        
        # 检查结果
        print(f"\nembed_datamaps_into_dataframe执行成功!")
        print(f"结果DataFrame形状: {df_result.shape}")
        
        # 打印部分结果进行验证
        if 'probs' in df_result.columns and not df_result.empty:
            for i in range(min(3, len(df_result))):
                if df_result.loc[i, 'probs'] != "":
                    print(f"\n样本{i}:")
                    print(f"  confidence: {df_result.loc[i, 'confidence']}")
                    print(f"  variability: {df_result.loc[i, 'variability']}")
                    print(f"  correctness: {df_result.loc[i, 'correctness']}")
                    print(f"  probs类型: {type(df_result.loc[i, 'probs'])}")
        
        # 尝试保存结果
        print("\n保存结果到debug_plots目录...")
        if not os.path.exists('./debug_plots'):
            os.makedirs('./debug_plots')
        df_result.to_csv('./debug_plots/debug_datamaps_result.csv', index=False)
        print("结果CSV文件已保存到 ./debug_plots/debug_datamaps_result.csv")
        
        # 测试generate_maps_plots_from_dataframe函数
        print("\n测试generate_maps_plots_from_dataframe函数...")
        generate_maps_plots_from_dataframe(df_result, './debug_plots')
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 提供修复建议
    print("\n修复建议:")
    print("1. 在DataMapsCallback.on_train_batch_end中，确保存储的probs是numpy数组或tensor")
    print("2. 在embed_datamaps_into_dataframe函数中，修改这一行:")
    print("   df.loc[i, 'probs'] = metrics[filename]['probs'].tolist()")
    print("   改为检查probs类型后再决定是否调用tolist()")
    print("3. 确保文件名标准化处理在DataMapsCallback和embed_datamaps_into_dataframe中保持一致")

if __name__ == "__main__":
    debug_datamaps()