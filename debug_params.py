import os
import sys
from argparse import ArgumentParser

# 添加当前目录到Python路径
sys.path.append('/root/Map-Mix')

# 导入配置模块
from config import LIDConfig

# 打印配置信息
def debug_config():
    print("=== 配置参数调试 ===")
    print(f"数据目录: {LIDConfig.dir}")
    print(f"训练集路径: {LIDConfig.train_path}")
    print(f"测试集路径: {LIDConfig.test_path}")
    print(f"验证集路径: {LIDConfig.val_path}")
    print(f"批次大小: {LIDConfig.batch_size}")
    print(f"训练轮数: {LIDConfig.epochs}")
    print(f"模型类型: {LIDConfig.model_type}")
    print(f"上游模型: {LIDConfig.upstream_model}")
    print(f"特征维度: {LIDConfig.feature_dim}")
    print(f"GPU数量: {LIDConfig.gpu}")
    print(f"工作线程数: {LIDConfig.n_workers}")
    print(f"学习率: {LIDConfig.lr}")
    print(f"运行名称: {LIDConfig.run_name}")
    print(f"模型检查点: {LIDConfig.model_checkpoint}")
    print(f"结果路径: {LIDConfig.results_path}")
    print("=== 配置参数调试完成 ===")

# 测试命令行参数解析
def debug_argparse():
    print("\n=== 命令行参数解析调试 ===")
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--train_path', type=str, default=LIDConfig.train_path)
    parser.add_argument('--val_path', type=str, default=LIDConfig.val_path)
    parser.add_argument('--test_path', type=str, default=LIDConfig.test_path)
    parser.add_argument('--batch_size', type=int, default=LIDConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=LIDConfig.epochs)
    parser.add_argument('--feature_dim', type=int, default=LIDConfig.feature_dim)
    parser.add_argument('--lr', type=float, default=LIDConfig.lr)
    parser.add_argument('--gpu', type=int, default=LIDConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=LIDConfig.n_workers)
    parser.add_argument('--dev', type=bool, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=LIDConfig.model_checkpoint)
    parser.add_argument('--model_type', type=str, default=LIDConfig.model_type)
    parser.add_argument('--upstream_model', type=str, default=LIDConfig.upstream_model)
    parser.add_argument('--unfreeze_last_conv_layers', action='store_true')
    
    # 解析参数
    hparams = parser.parse_args([])  # 传入空列表避免解析实际命令行参数
    hparams_dict = vars(hparams)
    
    print("解析后的参数:")
    for key, value in hparams_dict.items():
        print(f"  {key}: {value}")
    
    print("=== 命令行参数解析调试完成 ===")
    return hparams_dict

# 测试配置文件读取
def debug_config_file():
    print("\n=== 配置文件读取调试 ===")
    try:
        import json
        with open("config.json", "r") as jsonfile:
            raw_config = json.load(jsonfile)
        
        print("原始配置文件内容:")
        print(json.dumps(raw_config, indent=2, ensure_ascii=False))
        
        # 检查路径替换是否正确
        if 'dataDir' in raw_config:
            dir_val = raw_config['dataDir']['dir']
            train_path = raw_config['dataDir']['train_path']
            replaced_path = train_path.replace('$dir', dir_val)
            print(f"\n路径替换测试:")
            print(f"原始路径模板: {train_path}")
            print(f"替换后路径: {replaced_path}")
            print(f"LIDConfig中路径: {LIDConfig.train_path}")
            print(f"替换是否一致: {replaced_path == LIDConfig.train_path}")
    
        print("=== 配置文件读取调试完成 ===")
    except Exception as e:
        print(f"读取配置文件出错: {e}")

if __name__ == "__main__":
    print("开始调试参数加载...")
    
    # 1. 测试配置模块
    debug_config()
    
    # 2. 测试命令行参数解析
    hparams = debug_argparse()
    
    # 3. 测试配置文件读取
    debug_config_file()
    
    print("\n调试完成！")