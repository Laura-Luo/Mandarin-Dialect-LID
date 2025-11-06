# 模型打包指南

本指南提供了将语言识别模型打包提供给外部用户的详细步骤。

## 需要打包的文件清单

### 1. 核心接口脚本
- `output.py` - 主要使用接口脚本
- `requirements.txt` - 依赖列表文件

### 2. 模型相关文件
- `Models/lightning.py` - Lightning模型封装
- `Models/model.py` - 核心模型定义
- `config.py` - 配置加载模块
- `utils.py` - 工具函数

### 3. 配置文件
- `config.json` - 模型配置

### 4. 模型检查点
- 从 `/root/autodl-tmp/checkpoints/baseline-xlsr-3-epoch=13.ckpt` 复制到打包目录

### 5. 文档
- `README.md` - 接口文档
- `PACKAGING_GUIDE.md` - 本打包指南

## 打包步骤

1. **创建打包目录**
```bash
mkdir -p /tmp/lid_model_package
```

2. **复制核心文件**
```bash
# 复制接口脚本和配置文件
cp /root/Map-Mix/output.py /tmp/lid_model_package/
cp /root/Map-Mix/config.py /tmp/lid_model_package/
cp /root/Map-Mix/utils.py /tmp/lid_model_package/
cp /root/Map-Mix/config.json /tmp/lid_model_package/
cp /root/Map-Mix/requirements.txt /tmp/lid_model_package/
cp /root/Map-Mix/README.md /tmp/lid_model_package/
cp /root/Map-Mix/PACKAGING_GUIDE.md /tmp/lid_model_package/

# 复制模型目录
mkdir -p /tmp/lid_model_package/Models
cp /root/Map-Mix/Models/lightning.py /tmp/lid_model_package/Models/
cp /root/Map-Mix/Models/model.py /tmp/lid_model_package/Models/

# 复制模型检查点
cp /root/autodl-tmp/checkpoints/baseline-xlsr-3-epoch=13.ckpt /tmp/lid_model_package/
```

3. **更新配置文件路径**
编辑 `/tmp/lid_model_package/config.json`，将模型检查点路径更新为相对路径：
```json
{
    "dataDir": {
        "model_checkpoint": "./baseline-xlsr-3-epoch=13.ckpt"
    },
    // 其他配置保持不变
}
```

4. **创建压缩包**
```bash
cd /tmp
zip -r lid_model_package.zip lid_model_package/
```

## 外部用户使用说明

1. 解压压缩包
```bash
unzip lid_model_package.zip
cd lid_model_package
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 使用模型
```bash
# 命令行使用
python output.py --audio_path /path/to/audio.wav

# 或者作为模块使用
python -c "from output import predict_language; pred, prob = predict_language('./test_audio.wav'); print(f'预测结果: {pred}, 置信度: {prob}')"
```

## 注意事项

- 确保所有文件路径在打包后保持一致
- 模型检查点文件较大，请确保有足够的存储空间
- 外部用户需要根据自己的环境安装相应的依赖库
- 如需在CPU环境下运行，可以修改output.py中的device参数