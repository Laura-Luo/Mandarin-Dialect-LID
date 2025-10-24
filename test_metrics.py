# 先执行test.py，生成results.csv
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import ast
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from netcal.metrics import ECE

def EER(y, y_softmax_scores):
    # 二分类EER计算
    y = label_binarize(y, classes=[0, 1])
    y_softmax_scores = np.array(y_softmax_scores)[:, 1]  # 使用正类的概率
    
    fpr, tpr, _ = roc_curve(y, y_softmax_scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer
    
def ECEMetric(y, y_softmax_scores):
    y_softmax_scores = np.stack(y_softmax_scores, axis=0)
    ece = ECE(10)
    return ece.measure(y_softmax_scores, y)

# 加载测试结果文件（假设结果保存在config中配置的路径）
from config import LIDConfig
results_df = pd.read_csv(LIDConfig.results_path)

# 标签映射
def label2num(label):
    # 处理字符串形式的列表，如"['zho-dia']"
    if isinstance(label, str) and label.startswith('[') and label.endswith(']'):
        import ast
        label_list = ast.literal_eval(label)
        label = label_list[0] if isinstance(label_list, list) and len(label_list) > 0 else label
    
    if label == 'zho-cmn':
        return 0
    elif label == 'zho-dia':
        return 1
    else:
        raise ValueError(f"Unknown label: {label}")


def get_ece(df):
    probs = []
    y_true = []
    for i in range(len(df)):
        # 处理嵌套列表格式的概率
        prob_data = ast.literal_eval(df.loc[i, 'probability'])
        if isinstance(prob_data[0], list):
            # 如果是嵌套列表，取第一个元素
            probs.append(np.array(prob_data[0]))
        else:
            probs.append(np.array(prob_data))
        # 处理class列
        class_value = df.loc[i, 'class']
        y_true.append(label2num(class_value))
    y_true = np.array(y_true)
    return ECEMetric(y_true, probs)




def metrics(df):
    # 按不同时长分组的评估
    duration_groups = {'short': 3, 'medium': 10, 'long': 30}
    duration_results = {}
    
    # 整体评估
    y_true = [label2num(cls) for cls in list(df['class'])]
    y_pred = [label2num(pred) for pred in list(df['prediction'])]
    
    # 获取概率值
    y_probs = []
    for i in range(len(df)):
        prob_data = ast.literal_eval(df.loc[i, 'probability'])
        if isinstance(prob_data[0], list):
            prob_array = np.array(prob_data[0])
        else:
            prob_array = np.array(prob_data)
        y_probs.append(prob_array)
    
    # 计算整体指标
    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    ece = get_ece(df)
    
    # 计算EER
    eer = EER(y_true, y_probs)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 按时长分组评估
    for group_name, duration in duration_groups.items():
        # 修正：使用近似匹配，因为duration是浮点数
        group_df = df[np.abs(df['duration'].astype(float) - duration) < 0.5]
        if len(group_df) > 0:
            group_true = [label2num(cls) for cls in list(group_df['class'])]
            group_pred = [label2num(pred) for pred in list(group_df['prediction'])]
            duration_results[group_name] = {
                'accuracy': accuracy_score(group_true, group_pred),
                'f1_weighted': f1_score(group_true, group_pred, average='weighted'),
                'count': len(group_df)
            }
    
    # 打印结果
    print("===== 二分类任务评估结果 =====")
    print(f"整体准确率: {acc:.4f}")
    print(f"F1分数 (weighted): {f1_weighted:.4f}")
    print(f"F1分数 (macro): {f1_macro:.4f}")
    print(f"F1分数 (micro): {f1_micro:.4f}")
    print(f"ECE (预期校准误差): {ece:.4f}")
    print(f"EER (等错误率): {eer:.4f}")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['zho-cmn', 'zho-dia']))
    print("\n混淆矩阵:")
    print(cm)
    
    print("\n按时长分组结果:")
    for group_name, results in duration_results.items():
        print(f"{group_name} (时长={duration_groups[group_name]}秒): 准确率={results['accuracy']:.4f}, F1={results['f1_weighted']:.4f}, 样本数={results['count']}")
    
    return acc, f1_weighted, ece, eer, duration_results

# 执行评估
if __name__ == "__main__":
    print("开始评估二分类模型性能...")
    acc, f1_weighted, ece, eer, duration_results = metrics(results_df)
    print("\n评估完成!")
    
    # 保存评估结果
    eval_results = {
        'accuracy': acc,
        'f1_weighted': f1_weighted,
        'ece': ece,
        'eer': eer,
        'duration_results': duration_results
    }
    
    # 可以将结果保存到文件
    import json
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)