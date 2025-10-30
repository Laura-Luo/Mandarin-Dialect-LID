import pandas as pd
import sys
import os

# 添加Map-Mix目录到Python路径
sys.path.append('/root/Map-Mix')

# 导入绘图函数
try:
    from Models.datamaps import generate_maps_plots_from_dataframe
    print("成功导入generate_maps_plots_from_dataframe函数")
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

# 文件路径
csv_path = '/root/Langid/results/plots/datamaps-metrics-3-modified.csv'
output_dir = '/root/Langid/results/plots/'

# 检查CSV文件是否存在
if not os.path.exists(csv_path):
    print(f"错误: CSV文件不存在 - {csv_path}")
    sys.exit(1)

# 读取CSV文件
print(f"正在读取CSV文件: {csv_path}")
df = pd.read_csv(csv_path)

print(f"读取成功，数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")

# 处理重复的'class'列
if df.columns.duplicated().any():
    print("警告: 发现重复列名，保留第一个'class'列")
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    print(f"处理后的数据形状: {df.shape}")
    print(f"处理后的列名: {list(df.columns)}")

# 确保输出目录存在
if not os.path.exists(output_dir):
    print(f"创建输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

# 生成图表
print("开始生成datamaps图表...")
try:
    generate_maps_plots_from_dataframe(df, output_dir)
    print(f"图表生成成功，输出目录: {output_dir}")
    
    # 验证生成的文件
    csv_output = os.path.join(output_dir, "datamaps-metrics-3.csv")
    png_output = os.path.join(output_dir, "datamaps-3.png")
    
    if os.path.exists(csv_output):
        print(f"生成的CSV文件: {csv_output}, 大小: {os.path.getsize(csv_output) / 1024:.2f} KB")
    
    if os.path.exists(png_output):
        print(f"生成的图表文件: {png_output}, 大小: {os.path.getsize(png_output) / 1024:.2f} KB")
    else:
        print("警告: 图表文件未生成，请检查是否存在'class'列")
        
except Exception as e:
    print(f"生成图表时出错: {e}")
    import traceback
    traceback.print_exc()