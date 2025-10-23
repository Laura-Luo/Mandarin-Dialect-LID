import os
import sys
import torch
import traceback
import faulthandler

faulthandler.enable()

# 添加当前目录到Python路径
sys.path.append('/root/Map-Mix')

def debug_torch_cuda():
    """调试PyTorch和CUDA"""
    print("=== PyTorch和CUDA调试 ===")
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    
    # 测试CUDA操作
    try:
        x = torch.randn(100, 100)
        if torch.cuda.is_available():
            x = x.cuda()
        y = torch.matmul(x, x)
        print("✓ 矩阵乘法测试成功")
    except Exception as e:
        print(f"✗ CUDA操作测试失败: {e}")
        traceback.print_exc()
    
    print("=== PyTorch和CUDA调试完成 ===")

def debug_upstream_transformer_xlsr():
    """直接调试UpstreamTransformerXLSR类"""
    print("\n=== UpstreamTransformerXLSR直接调试 ===")
    
    # 打印当前工作目录和环境信息
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path}")
    
    # 尝试导入UpstreamTransformerXLSR类
    try:
        print("\n1. 导入UpstreamTransformerXLSR类...")
        from Models.model import UpstreamTransformerXLSR
        print("✓ 成功导入UpstreamTransformerXLSR类")
        
        # 准备初始化参数
        upstream_model = 'xlsr_300m'  # 从config.json中获取的参数
        feature_dim = 1024  # 从config.json中获取的参数
        
        print(f"\n2. 准备初始化参数:")
        print(f"   - upstream_model: {upstream_model}")
        print(f"   - feature_dim: {feature_dim}")
        
        # 尝试初始化模型
        print("\n3. 初始化UpstreamTransformerXLSR模型...")
        model = UpstreamTransformerXLSR(
            upstream_model=upstream_model,
            feature_dim=feature_dim,
            unfreeze_last_conv_layers=False
        )
        print("✓ 模型初始化成功")
        
        # 将模型移至适当的设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f"✓ 模型已移至设备: {device}")
        
        # 测试简单的前向传播（使用小批量数据）
        print("\n4. 测试前向传播...")
        # 创建一个小批量的随机音频数据
        batch_size = 1  # 先用小批量测试
        sample_rate = 16000
        duration = 1  # 1秒
        test_input = torch.randn(batch_size, 1, sample_rate * duration).to(device)
        test_lengths = [sample_rate * duration]
        
        print(f"   - 输入形状: {test_input.shape}")
        print(f"   - 输入长度: {test_lengths}")
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            output = model(test_input, test_lengths)
        
        print(f"✓ 前向传播成功")
        print(f"   - 输出形状: {output.shape}")
        
        return model
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"✗ 模型初始化或前向传播失败: {e}")
        traceback.print_exc()
    
    print("\n=== UpstreamTransformerXLSR调试完成 ===")
    return None

if __name__ == "__main__":
    print("开始调试UpstreamTransformerXLSR模型...")
    
    # 1. 先调试PyTorch和CUDA环境
    debug_torch_cuda()
    
    # 2. 直接调试UpstreamTransformerXLSR类
    model = debug_upstream_transformer_xlsr()
    
    if model is not None:
        print("\n🎉 模型加载和前向传播测试成功！")
    else:
        print("\n❌ 模型调试失败，请检查错误信息。")
    
    print("\n调试完成！")