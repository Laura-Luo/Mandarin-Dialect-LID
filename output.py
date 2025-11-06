import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)

import os
import json
import torch
import torchaudio
import torch.nn.functional as F
from argparse import ArgumentParser

# 导入配置和模型
from config import LIDConfig
from Models.lightning import LightningModel

def predict_language(audio_path, model_checkpoint=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    预测音频文件的语言标签
    
    Args:
        audio_path: 音频文件路径
        model_checkpoint: 模型检查点路径，如果为None则使用配置文件中的路径
        device: 运行设备
    
    Returns:
        prediction: 预测的语言标签（'zho-cmn' 或 'zho-dia'）
        probability: 预测概率
    """
    # 加载配置
    if model_checkpoint is None:
        model_checkpoint = LIDConfig.model_checkpoint
    
    # 准备模型参数
    hparams = {
        'model_type': LIDConfig.model_type,
        'upstream_model': LIDConfig.upstream_model,
        'feature_dim': LIDConfig.feature_dim,
        'lr': LIDConfig.lr,
        'unfreeze_last_conv_layers': False
    }
    
    # 加载模型
    model = LightningModel.load_from_checkpoint(model_checkpoint, HPARAMS=hparams)
    model.to(device)
    model.eval()
    
    # 标签映射
    num2labels = {0: 'zho-cmn', 1: 'zho-dia'}
    
    # 加载并处理音频
    try:
        wav, _ = torchaudio.load(audio_path)
        wav = wav.view(-1)
        
        # 对长音频进行分块处理
        if wav.shape[-1] > 16000 * 6:  # 超过6秒的音频
            wav_tensor = wav.unfold(0, 16000 * 6, 16000 * 3)
        else:
            wav_tensor = wav.view(1, -1)
        
        wav_tensor = wav_tensor.to(device)
        x_lens = [wav_tensor.shape[-1]] * wav_tensor.shape[0]
        
        # 模型预测
        with torch.no_grad():
            y_hat_l = model(wav_tensor, x_lens)
            probs = F.softmax(y_hat_l, dim=1).mean(0).view(1, 2)
            pred_label_idx = probs.argmax(dim=1).item()
            
        # 转换为标签
        prediction = num2labels[pred_label_idx]
        probability = probs[0, pred_label_idx].item()
        
        return prediction, probability
        
    except Exception as e:
        print(f"处理音频文件时出错: {e}")
        return None, None

def main():
    # 解析命令行参数
    parser = ArgumentParser(description='音频语言识别预测脚本')
    parser.add_argument('--audio_path', type=str, required=True, help='要预测的音频文件路径')
    parser.add_argument('--model_checkpoint', type=str, default=None, help='模型检查点路径（可选）')
    args = parser.parse_args()
    
    # 验证文件是否存在
    if not os.path.exists(args.audio_path):
        print(f"错误：音频文件不存在: {args.audio_path}")
        return
    
    # 进行预测
    prediction, probability = predict_language(args.audio_path, args.model_checkpoint)
    
    if prediction is not None:
        print(f"预测结果:")
        print(f"语言标签: {prediction}")
        print(f"置信度: {probability:.4f}")
        
        # 根据标签输出对应的语言名称
        if prediction == 'zho-cmn':
            print("识别结果: 普通话")
        else:
            print("识别结果: 方言")
    else:
        print("预测失败")

if __name__ == '__main__':
    main()