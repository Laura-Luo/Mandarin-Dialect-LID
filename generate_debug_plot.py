import os
import pandas as pd
import sys

# 添加Map-Mix目录到Python路径
sys.path.append('/root/Map-Mix')

# 从Models.datamaps模块导入generate_maps_plots_from_dataframe函数
from Models.datamaps import generate_maps_plots_from_dataframe

def main():
    # 定义文件路径
    input_csv_path = '/root/Map-Mix/debug_plots/datamaps-metrics-3.csv'
    output_plot_path = '/root/Map-Mix/debug_plots/datamaps-3.png'
    output_csv_path = '/root/Map-Mix/debug_plots/datamaps-metrics-3-processed.csv'
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_plot_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_csv_path):
        print(f"错误: 输入文件不存在 - {input_csv_path}")
        return
    
    try:
        # 读取CSV文件
        print(f"正在读取CSV文件: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        
        # 打印数据信息
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print("前5行数据:")
        print(df.head())
        
        # 检查并处理列名中的空格
        df.columns = df.columns.str.strip()
        print(f"清理后的列名: {list(df.columns)}")
        
        # 检查是否有重复的'class'列
        if 'class' in df.columns:
            print("'class'列已存在，无需添加")
        else:
            print("错误: 缺少'class'列")
            return
        
        # 调用generate_maps_plots_from_dataframe函数生成图表
        print("正在生成datamaps图表...")
        # 函数签名是generate_maps_plots_from_dataframe(df: pd.DataFrame, dir_: str)
        generate_maps_plots_from_dataframe(
            df=df,
            dir_=output_dir
        )
        
        # 验证生成的文件
        if os.path.exists(output_plot_path):
            file_size = os.path.getsize(output_plot_path) / 1024  # KB
            print(f"图表生成成功: {output_plot_path} ({file_size:.2f} KB)")
        
        if os.path.exists(output_csv_path):
            file_size = os.path.getsize(output_csv_path) / 1024  # KB
            print(f"处理后的CSV生成成功: {output_csv_path} ({file_size:.2f} KB)")
            
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()