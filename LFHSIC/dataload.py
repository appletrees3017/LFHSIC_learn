# dataload.py
import os
import time
import numpy as np
import pandas as pd
import torch
import h5py
import random  # 添加缺失的导入
from collections import defaultdict
import bisect
from sklearn.preprocessing import StandardScaler

def get_data_path():
    """自动检测数据目录位置"""
    # 在Colab环境下
    if os.path.exists('/content'):
        # 优先尝试符号链接路径
        if os.path.exists('/content/LFHSIC_learn/LFHSIC/data'):
            return '/content/LFHSIC_learn/LFHSIC/data'
        # 备选路径（如果符号链接未正确创建）
        return '/content/drive/MyDrive/Colab_experiments/nips_lfhsic_code_data/'
    
    # 本地环境（备用）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'data')

# 定义全局变量--数据根路径
DATA_ROOT = get_data_path()

# 3dshape数据导入
# 定义成全局变量--因子的解释
FACTORS_IN_ORDER = [
    'floor_hue',  # 地板色相
    'wall_hue',   # 墙壁色相
    'object_hue', # 物体色相
    'scale',      # 物体尺寸
    'shape',      # 物体形状
    'orientation' # 观察场景
]

NUM_VALUES_PER_FACTOR = {
    'floor_hue': 10,
    'wall_hue': 10,  
    'object_hue': 10,
    'scale': 8,
    'shape': 4,
    'orientation': 15
}

def get_index(factors):
    """将因子组合转换为数据集索引===代码修改===兼容Python 3
    参数:
        factors: numpy 数组形状 [6, batch_size]
                每个因子取值范围为 [0, num_values_per_factor-1]
    返回:
        indices: numpy 数组形状 [batch_size]
    """
    # 预先计算各维度的步长（每个因子变化的基值）
    strides = []
    product = 1
    for factor_name in reversed(_FACTORS_IN_ORDER):
        strides.insert(0, product)
        product *= _NUM_VALUES_PER_FACTOR[factor_name]
    strides = np.array(strides, dtype=np.int64)
    
    # 使用向量化操作替代循环加速计算
    indices = np.sum(factors * strides[:, np.newaxis], axis=0)
    return indices     
def load_3dshapes(batch_size, fixed_factor, fixed_factor_value):
    # 验证数据路径
    shape_path = os.path.join(DATA_ROOT, '3dshapes', '3dshapes.h5')
    if not os.path.exists(shape_path):
        raise FileNotFoundError(f"3Dshape数据未找到: {shape_path}. 当前工作目录: {os.getcwd()}")
    
    start_time = time.time()   
    # 加载数据准备
    dataset = h5py.File(shape_path, 'r')
    print(list(dataset.keys()))  # 在 Python 3 中需要显式转换为列表
    images = dataset['images']
    labels = dataset['labels']
    
    # 定义数据维度
    n_samples = labels.shape[0]
    image_shape = images.shape[1:]
    label_shape = labels.shape[1:]
    
    factors = np.zeros([len(FACTORS_IN_ORDER), batch_size], dtype=np.int32)
    
     # 为非固定因子生成随机值
    for factor_idx, name in enumerate(_FACTORS_IN_ORDER):
        if factor_idx != fixed_factor:
            factors[factor_idx] = np.random.choice(_NUM_VALUES_PER_FACTOR[name], batch_size)
    
    # 设置固定因子的值
    factors[fixed_factor] = fixed_factor_value
    
    # 计算索引并获取图像
    indices = get_index(factors)
    x_ims = np.array(images[indices])  # 转换为数组确保兼容性
    
    # 归一化并转换类型
    x_ims.astype(np.float32) / 255.0
    
    factors = factors.T
    orient_idx = FACTORS_IN_ORDER.index('orientation')
    y_orien = factors[:, orient_idx]
   
    elapsed = time.time() - start_time
    print(f"总耗时：{elapsed:.2f}秒")  # 修正f-string
    # return x_ims.reshape(batch_size, 64, 64, 3), y_orien.reshape(batch_size, 1) 重复reshape会导致顺序错乱
    return x_ims, y_orien

def calculate_samples_per_bin(bin_data, total_samples):
    bin_counts = {key: len(indices) for key, indices in bin_data.items()}
    total_count = sum(bin_counts.values())
    samples_per_bin = {}
    remaining_samples = total_samples  # 修正变量名
    
    for bin_key in bin_data.keys():
        bin_size = len(bin_data[bin_key])
        proportion = bin_size / total_count
        allocated = max(1, int(total_samples * proportion))
        allocated = min(allocated, bin_size)
        samples_per_bin[bin_key] = allocated
        remaining_samples -= allocated
    
    bin_keys = list(bin_data.keys())
    while remaining_samples > 0:
        bin_key = random.choice(bin_keys)
        if samples_per_bin[bin_key] < len(bin_data[bin_key]):
            samples_per_bin[bin_key] += 1
            remaining_samples -= 1
            
    return samples_per_bin
    
def load_ypmsd_stratified(bin_size=10, seed=42, train_samplesn=5000, test_samplesn=1000):
    msd_path = os.path.join(DATA_ROOT, 'msd', 'YearPredictionMSD.csv')
    if not os.path.exists(msd_path):
        raise FileNotFoundError(f"MSD数据集未找到: {msd_path}, 当前工作目录: {os.getcwd()}")
    
    start_time = time.time()
    # 官方数据建议划分
    OFFICIAL_TRAIN_SIZE = 463715
    OFFICIAL_TEST_SIZE = 51630
    TOTAL_SIZE = OFFICIAL_TRAIN_SIZE + OFFICIAL_TEST_SIZE
    
    # 随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 初始分箱数据
    bins = list(range(1920, 2020, bin_size))
    if 2020 not in bins:
        bins.append(2020)
        
    # 按分箱组织索引
    train_bins = defaultdict(list)
    test_bins = defaultdict(list)
    
    # 遍历整个文件，建立年份索引
    with open(msd_path, 'r') as f:
        for line_idx, line in enumerate(f):
            year = int(float(line.split(',')[0]))  # 修正缩进和括号
            bin_idx = bisect.bisect_left(bins, year) - 1
            if bin_idx < 0:
                bin_idx = 0
            elif bin_idx > len(bins) - 1:
                bin_idx = len(bins) - 2
                
            bin_key = f"{bins[bin_idx]}---{bins[bin_idx+1]}"
            
            # 确定样本所属数据集
            if line_idx < OFFICIAL_TRAIN_SIZE:  # 修正变量名
                train_bins[bin_key].append(line_idx)
            else:
                test_bins[bin_key].append(line_idx)
    
    print("\n训练集和测试集样本年份分布统计：")
    for bin_key in sorted(train_bins.keys()):  # 修正函数调用
        print(f"分箱 {bin_key}: {len(train_bins[bin_key])} 个样本")
    for bin_key in sorted(test_bins.keys()):  # 修正函数调用
        print(f"分箱 {bin_key}: {len(test_bins[bin_key])} 个样本")
    
    # 计算分箱抽样样本量
    train_samples_per_bin = calculate_samples_per_bin(train_bins, train_samplesn)
    test_samples_per_bin = calculate_samples_per_bin(test_bins, test_samplesn)
    
    # 分层抽样
    select_train_indices = []
    for bin_key, count in train_samples_per_bin.items():
        selected = random.sample(train_bins[bin_key], count)
        select_train_indices.extend(selected)
        
    select_test_indices = []
    for bin_key, count in test_samples_per_bin.items():
        selected = random.sample(test_bins[bin_key], count)
        select_test_indices.extend(selected)
        
    all_indices = select_train_indices + select_test_indices
    all_indices.sort()
    
    sampled_data = []
    index_map = []
    with open(msd_path, 'r') as f:
        for line_idx, line in enumerate(f):
            if line_idx > all_indices[-1]:
                break
                
            if line_idx in select_train_indices:
                data = list(map(float, line.strip().split(',')))  # 修正括号
                sampled_data.append(data)
                index_map.append(('train', line_idx))
                
            elif line_idx in select_test_indices:  # 修正变量名
                data = list(map(float, line.strip().split(',')))  # 修正括号
                sampled_data.append(data)
                index_map.append(('test', line_idx))
    
    # 转换成数据，分离训练集和测试集
    sampled_array = np.array(sampled_data, dtype=np.float32)
    train_data = []
    test_data = []
    
    for i, (label, line_idx) in enumerate(index_map):
        if label == 'train':
            train_data.append(sampled_array[i])
        else:
            test_data.append(sampled_array[i])
            
    train_array = np.array(train_data, dtype=np.float32)
    test_array = np.array(test_data, dtype=np.float32)
    
    # 分离特征和年份
    X_train = train_array[:, 1:]
    X_test = test_array[:, 1:]
    Y_train = train_array[:, 0].astype(np.int32)
    Y_test = test_array[:, 0].astype(np.int32)
    
    # 结束时间
    elapsed = time.time() - start_time
    print(f"总耗时：{elapsed:.2f}秒")  # 修正f-string
    return (Y_train, X_train), (Y_test, X_test)


def load_ypmsd(bin_size=10, seed=42, train_samplesn=5000, test_samplesn=1000):
    msd_path = os.path.join(DATA_ROOT, 'msd', 'YearPredictionMSD.csv')
    if not os.path.exists(msd_path):
        raise FileNotFoundError(f"MSD数据集未找到: {msd_path}, 当前工作目录: {os.getcwd()}")
    
    start_time = time.time()
    # 官方数据建议划分
    OFFICIAL_TRAIN_SIZE = 463715
    OFFICIAL_TEST_SIZE = 51630
    TOTAL_SIZE = OFFICIAL_TRAIN_SIZE + OFFICIAL_TEST_SIZE
    
    # 随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 初始分箱数据
    bins = list(range(1920, 2020, bin_size))
    if 2020 not in bins:
        bins.append(2020)
        
    # 按分箱组织索引
    samples_bins = defaultdict(list)
    
    # 遍历整个文件，建立年份索引
    with open(msd_path, 'r') as f:
        for line_idx, line in enumerate(f):
            year = int(float(line.split(',')[0]))
            bin_idx = bisect.bisect_left(bins, year) - 1
            if bin_idx < 0:
                bin_idx = 0
            elif bin_idx > len(bins) - 1:
                bin_idx = len(bins) - 2
                
            bin_key = f"{bins[bin_idx]}---{bins[bin_idx+1]}"
            samples_bins[bin_key].append(line_idx)
            
    print("\n样本年份分布统计：")
    for bin_key in sorted(samples_bins.keys()):  # 修复变量名
        print(f"分箱 {bin_key}: {len(samples_bins[bin_key])} 个样本")
    
    # 计算分箱抽样样本量
    samples_per_bin = calculate_samples_per_bin(samples_bins, train_samplesn+test_samplesn)
    
    # 分层抽样    
    all_indices = []
    for bin_key, count in samples_per_bin.items():
        selected = random.sample(samples_bins[bin_key], count)
        all_indices.extend(selected)  # 修复拼写
    
    all_indices.sort()
    max_index = all_indices[-1]
    
    # 高效读取选中行（使用集合提高查找效率）
    sampled_data = []
    selected_set = set(all_indices)
    with open(msd_path, 'r') as f:
        for line_idx, line in enumerate(f):
            if line_idx > max_index:
                break    
            if line_idx in selected_set:
                data = list(map(float, line.strip().split(',')))
                sampled_data.append(data)
    
    # 转换成数组并划分数据集
    for it in range(100):
        
        sampled_array = np.array(sampled_data, dtype=np.float32)
        rand_index = np.random.permutation(len(sampled_array))
        train_array = sampled_array[rand_index[:train_samplesn]]
        test_array = sampled_array[rand_index[train_samplesn:train_samplesn+test_samplesn]]
    
    # 分离特征和年份
    X_train = train_array[:, 1:]
    X_test = test_array[:, 1:]
    Y_train = train_array[:, 0].astype(np.int32)
    Y_test = test_array[:, 0].astype(np.int32)
    
    # 结束时间
    elapsed = time.time() - start_time
    print(f"总耗时：{elapsed:.2f}秒")
    return (Y_train, X_train), (Y_test, X_test)
    
        
