# 语言识别模型接口文档

## 项目简介

本项目提供了一个用于区分普通话(zho-cmn)和方言(zho-dia)的音频语言识别模型。该模型基于预训练的音频特征提取器和Transformer架构，能够高效地对音频文件进行语言类型预测。

## 模型概述

- **模型类型**: UpstreamTransformerXLSR
- **上游预训练模型**: xlsr_300m
- **支持的语言标签**: 
  - `zho-cmn`: 普通话
  - `zho-dia`: 方言
- **最佳检查点**: `baseline-xlsr-3-epoch=13.ckpt`

## 快速开始

### 1. 环境要求

- Python 3.8+
- PyTorch
- torchaudio
- pytorch-lightning
- pandas
- argparse

### 2. 安装依赖

使用提供的 requirements.txt 文件安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

如果需要指定特定的 PyTorch 版本以匹配您的 CUDA 环境，请参考 [PyTorch 官方文档](https://pytorch.org/get-started/locally/) 进行安装。

## 接口说明

### 主要接口脚本: `output.py`

`output.py` 是提供给外部使用的主要接口脚本，它接收一个音频文件路径作为输入，并输出预测的语言标签和置信度。

#### 输入参数

- `--audio_path` (必需): 要预测的音频文件路径 (WAV格式)
- `--model_checkpoint` (可选): 模型检查点路径，如果不提供则使用配置文件中的默认路径

#### 输出格式

脚本将在控制台输出以下信息：

```
预测结果:
语言标签: zho-cmn/zho-dia
置信度: 0.9999
识别结果: 普通话/方言
```

## 使用示例

### 命令行使用

```bash
# 使用默认检查点
python output.py --audio_path /path/to/your/audio.wav

# 使用指定检查点
python output.py --audio_path /path/to/your/audio.wav --model_checkpoint /path/to/checkpoint.ckpt
```

### 作为模块导入使用

```python
from output import predict_language

# 进行预测
prediction, probability = predict_language(
    audio_path='/path/to/your/audio.wav',
    model_checkpoint='/path/to/checkpoint.ckpt'
)

print(f"预测标签: {prediction}, 置信度: {probability:.4f}")
```

## 配置说明

模型的默认配置存储在 `config.json` 文件中，主要配置项包括：

- **模型参数**:
  - `model_type`: 模型类型 (UpstreamTransformerXLSR)
  - `upstream_model`: 上游预训练模型 (xlsr_300m)
  - `feature_dim`: 特征维度 (1024)

- **路径配置**:
  - `model_checkpoint`: 默认模型检查点路径

## 音频处理说明

- 模型支持处理任意长度的WAV音频文件
- 对于超过6秒的长音频，会自动分块处理并综合各块的预测结果
- 音频采样率会在加载时自动处理，无需手动转换

## 需要打包的文件列表

将以下文件打包提供给外部用户：

1. **核心接口脚本**:
   - `output.py` - 主要使用接口

2. **模型相关文件**:
   - `Models/` 目录 - 包含模型定义
   - `config.py` - 配置加载模块
   - `utils.py` - 工具函数

3. **配置文件**:
   - `config.json` - 模型配置

4. **模型检查点**:
   - `baseline-xlsr-3-epoch=13.ckpt` (从 `/root/autodl-tmp/checkpoints/` 复制)

5. **文档**:
   - `README.md` - 本接口文档

## 注意事项

1. 确保提供的音频文件格式正确（WAV格式）
2. 对于最佳性能，建议使用与训练数据类似质量的音频
3. 模型主要针对普通话和方言的区分，不适用于其他语言的识别
4. 在GPU环境下运行可以获得更快的推理速度

## 故障排除

- 如果出现CUDA内存不足错误，可以尝试在CPU上运行：修改`output.py`中的`device`参数为`'cpu'`
- 如果音频文件无法识别，请检查文件格式是否正确
- 对于较长的音频文件，处理时间会相应增加