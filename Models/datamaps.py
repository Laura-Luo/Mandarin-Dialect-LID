from pyparsing import Dict
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 移除全局metrics变量，使用DataMapsCallback实例的metrics

class DataMapsCallback(pl.callbacks.Callback):
    """
    A pytorch lightning callback, implementing `Data Maps` described in the publication (https://arxiv.org/abs/2009.10795).
    """

    def __init__(self, log_dir, dataset_length):
        self.metrics = {}
        self.log_dir = log_dir
        self.dataset_length = dataset_length
        print(f"DataMapsCallback initialized with log_dir: {log_dir}, dataset_length: {dataset_length}")


    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # 获取预测概率、标签和文件名
        if not isinstance(outputs, dict):
            print(f"Warning: outputs is not a dict at batch {batch_idx}. Type: {type(outputs)}")
            return
            
        # 检查必要的键
        required_keys = ['probs', 'labels', 'filenames']
        missing_keys = [key for key in required_keys if key not in outputs]
        if missing_keys:
            print(f"Warning: Missing required keys in outputs at batch {batch_idx}: {missing_keys}. Available keys: {list(outputs.keys())}")
            return
            
        batch_probs = outputs['probs']
        batch_labels = outputs['labels']
        filenames = outputs['filenames']
        
        # 确保索引不会越界
        min_length = min(len(batch_probs), len(batch_labels), len(filenames))
        
        for ix in range(min_length):
            try:
                # 确保filename是字符串类型并标准化
                filename = str(filenames[ix])
                # 标准化文件名，去除路径，只保留文件名部分
                if '/' in filename or '\\' in filename:
                    filename = os.path.basename(filename)
                    
                prob = batch_probs[ix]
                label = batch_labels[ix]

                if filename not in self.metrics:
                    self.metrics[filename] = {'probs': [], 'corrects': []}

                # 计算正确预测
                predicted_class = np.argmax(prob)
                correct = 1 if predicted_class == label else 0

                self.metrics[filename]['probs'].append(prob)
                self.metrics[filename]['corrects'].append(correct)
            except Exception as e:
                print(f"Error processing batch item {ix} at batch_idx {batch_idx}: {str(e)}")
                continue


    def on_train_epoch_start(self, trainer, pl_module):
        pass

    def on_train_end(self, trainer, pl_module):
        """
        Store the mean and std of probabilities across epochs for each instance after training ends.
        Also store correctness = mean of correct/total labels across epochs.
        """
        # 打印收集的metrics信息，帮助调试
        print(f"Training ended. Collected metrics for {len(self.metrics)} files")
        # 打印几个示例键，帮助识别文件格式
        if len(self.metrics) > 0:
            sample_keys = list(self.metrics.keys())[:5]
            print(f"Sample metric keys: {sample_keys}")
        
        # Store the training artefacts


def embed_datamaps_into_dataframe(df, metrics):
    """
    Store training artefacts in a dataframe
    """

    for key in metrics.keys():
            metrics[key]['confidence'] = np.mean(metrics[key]['probs'])
            metrics[key]['variability'] = np.std(metrics[key]['probs'])
            metrics[key]['correctness'] = np.mean(metrics[key]['corrects'])

    # 打印DataFrame的列名，帮助调试
    print("DataFrame columns:", df.columns.tolist())
    print("Metrics keys sample:", list(metrics.keys())[:5] if len(metrics) > 0 else "Empty metrics")

    # 动态查找文件路径列
    file_path_columns = ['audiopath', 'path', 'file_path', 'filename', 'filepath']
    filepath_col = None
    for col in file_path_columns:
        if col in df.columns:
            filepath_col = col
            break
    
    # 如果找不到预定义的文件路径列，尝试使用第一列
    if filepath_col is None and len(df.columns) > 0:
        filepath_col = df.columns[0]
        print(f"Warning: Using first column '{filepath_col}' as file path column")

    df['probs'] = ""
    df['probs'] = df['probs'].astype('object')
    
    # 确保添加必要的列
    for col in ['confidence', 'variability', 'correctness', 'label_score']:
        if col not in df.columns:
            df[col] = 0.0

    # 处理可能的metrics键不匹配问题
    processed_count = 0
    for i in range(len(df)):
        if filepath_col and filepath_col in df.columns:
            filename = df.loc[i, filepath_col]
            
            # 标准化文件名，与DataMapsCallback中的处理保持一致
            if isinstance(filename, str):
                # 只保留文件名部分
                filename = os.path.basename(filename)
            
            # 检查标准化后的filename是否在metrics中
            if filename in metrics:
                df.loc[i, "confidence"] = metrics[filename]['confidence']
                df.loc[i, "variability"] = metrics[filename]['variability']
                df.loc[i, "correctness"] = metrics[filename]['correctness']
                # 处理probs数据，确保它是一个可以保存到DataFrame的格式
                try:
                    probs_data = metrics[filename]['probs']
                    # 如果是tensor或numpy数组，转换为列表
                    if isinstance(probs_data, (torch.Tensor, np.ndarray)):
                        # 确保是一维的
                        if probs_data.ndim > 1:
                            probs_data = probs_data.flatten().tolist()
                        else:
                            probs_data = probs_data.tolist()
                    elif isinstance(probs_data, list):
                        # 如果是列表，检查是否需要进一步处理
                        if probs_data and isinstance(probs_data[0], (torch.Tensor, np.ndarray)):
                            # 转换内部元素
                            probs_data = [p.tolist() if isinstance(p, (torch.Tensor, np.ndarray)) else p for p in probs_data]
                        # 如果列表中的元素本身是列表或数组，将其展平
                        if probs_data and isinstance(probs_data[0], (list, np.ndarray)):
                            # 这里我们需要决定如何处理：可以取均值或展平
                            # 为了保持数据一致性，我们取每个预测的最大值
                            probs_data = [float(max(p) if isinstance(p, (list, np.ndarray)) else p) for p in probs_data]
                    # 确保最终结果是一个简单的一维列表或标量
                    if isinstance(probs_data, list):
                        # 如果列表太长，我们可以取均值或只保存部分
                        # 这里我们保存均值作为简化表示
                        probs_data = float(np.mean(probs_data))
                    df.loc[i, 'probs'] = probs_data
                except Exception as e:
                    print(f"Error processing probs for {filename}: {str(e)}")
                    df.loc[i, 'probs'] = 0.0  # 使用默认值
                df.loc[i, 'label_score'] = (df.loc[i, "variability"] ** 2 + df.loc[i, "confidence"] ** 2) ** 0.5
                processed_count += 1
            else:
                # 尝试模糊匹配，只打印前几个警告避免日志过多
                if processed_count < 5 or i % 100 == 0:
                    print(f"Warning: File {filename} not found in metrics")
        else:
            print(f"Error: Could not find filepath column in DataFrame")
    
    print(f"Processed {processed_count} out of {len(df)} rows successfully")
    
    return df

def generate_maps_plots_from_dataframe(df: pd.DataFrame, dir_: str):
    # 确保输出目录存在
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    
    # 首先保存CSV文件
    df.to_csv(os.path.join(dir_, "datamaps-metrics-3.csv"), index=False)
    
    # 检查是否有'class'列，如果没有则跳过绘图
    if 'class' not in df.columns:
        print("Warning: 'class' column not found, skipping plot generation")
        return
    
    # 继续原有逻辑生成图表
    try:
        unique_labels = df['class'].unique().tolist()
        fig, axs = plt.subplots(len(unique_labels), 1, figsize=(6, 6 * len(unique_labels)))
        
        # 确保axs是列表
        if len(unique_labels) == 1:
            axs = [axs]
        
        box_style = {"boxstyle": "round", "facecolor": "white", "ec": "black"}
        for ix, label in enumerate(unique_labels):
            # 过滤并绘制数据
            label_data = df[df['class'] == label]
            if not label_data.empty:
                axs[ix].scatter(
                    x=label_data["variability"].to_numpy(),
                    y=label_data["confidence"].to_numpy(),
                    marker="x",
                    s=30,
                    c=label_data["correctness"].to_numpy(),
                )
                axs[ix].set_title(f"datamap for label = {label}")
                axs[ix].set_xlabel("variability")
                axs[ix].set_ylabel("confidence")
                axs[ix].text(0.14, 0.84, "easy-to-learn", transform=axs[ix].transAxes, verticalalignment="top", bbox=box_style)
                axs[ix].text(0.75, 0.5, "ambiguous", transform=axs[ix].transAxes, verticalalignment="top", bbox=box_style)
                axs[ix].text(0.14, 0.14, "hard-to-learn", transform=axs[ix].transAxes, verticalalignment="top", bbox=box_style)
        
        plt.tight_layout()
        plt.savefig(os.path.join(dir_, "datamaps-3.png"), format="png")
        plt.close()
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        # 即使绘图失败，CSV文件已经保存成功