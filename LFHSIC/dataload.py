# dataload.py
import os
import time
import itertools
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
    """修正索引计算逻辑"""
    # 修正步长计算：每个因子的步长 = 后续所有因子取值的乘积
      strides = np.array([
          10 * 10 * 8 * 4 * 15,  # floor_hue: 10 * 10 * 8 * 4 * 15 = 48,000
          10 * 8 * 4 * 15,     # wall_hue: 10 * 8 * 4 * 15 = 4,800
          8 * 4 * 15,        # object_hue: 8 * 4 * 15 = 480
          4 * 15,          # scale: 4 * 15 = 60
          15,            # shape: 15
          1              # orientation: 1
      ], dtype=np.int64)
      
      # 验证维度匹配
      if factors.shape[0] != len(strides):
        raise ValueError(f"因子维度不匹配: 矩阵有{factors.shape[0]}行, " f"但步长数组有{len(strides)}个元素 ")
        # 计算索引: ∑(因子值 × 对应步长)
return np.sum(factors * strides.reshape(-1, 1), axis=0).astype(np.int64)


def get_indices_for_factors(fixed_factor, fixed_factor_value):
    """获取固定因子值对应的所有可能索引（修复版）"""
    # 1. 确保使用正确的全局常量
    factors_ranges = []
    for i, name in enumerate(FACTORS_IN_ORDER):  # 使用全局常量名
        if i == fixed_factor:
            factors_ranges.append([fixed_factor_value])
        else:
            factors_ranges.append(list(range(NUM_VALUES_PER_FACTOR[name])))  # 使用正确的常量名

    # 2. 生成所有组合
    all_combinations = list(itertools.product(*factors_ranges))

    if not all_combinations:
      raise ValueError("未生成任何因子组合")

    factors_matrix = np.array(all_combinations).T
    
    # 3. 计算所有索引
    indices = get_index(factors_matrix)
    return indices

def get_indices_for_factors(fixed_factor, fixed_factor_value):
    """获取固定因子值对应的所有可能索引（修复版）"""
    # 1. 确保使用正确的全局常量
    factors_ranges = []
    for i, name in enumerate(FACTORS_IN_ORDER):  # 使用全局常量名
        if i == fixed_factor:
            factors_ranges.append([fixed_factor_value])
        else:
            factors_ranges.append(list(range(NUM_VALUES_PER_FACTOR[name])))  # 使用正确的常量名
    
    # 2. 生成所有组合
    all_combinations = list(itertools.product(*factors_ranges))
    factors_matrix = np.array(all_combinations).T
    
    # 3. 计算所有索引
    indices = get_index(factors_matrix)
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
    
    # 3. 获取所有符合条件的索引
    all_indices = get_indices_for_factors(fixed_factor, fixed_factor_value)
        
    # 4. 随机抽样并确保索引唯一有序
    selected_indices = np.random.choice(all_indices, batch_size, replace=False)
    selected_indices.sort()  # 确保索引顺序递增
        
    # 5. 获取图像数据
    X_ims = np.array(images[selected_indices])  # 修正数组名拼写错误 (X_imgs -> X_ims)
        
    # 6. 获取标签数据
    all_labels = np.array(labels[selected_indices])   
    # 7. 提取方位标签
    orientation_index = FACTORS_IN_ORDER.index('orientation')
    y_orien = all_labels[:, orientation_index].astype(np.float32)
    
    # 提取所有因子标签（包括形状）
    factor_labels = {}
    for i, name in enumerate(FACTORS_IN_ORDER):
      factor_labels[name] = all_labels[:, i]

    # 验证固定因子是否一致
    fixed_name = FACTORS_IN_ORDER[fixed_factor]
    unique_vals = np.unique(factor_labels[fixed_name])
    if len(unique_vals) > 1 or unique_vals[0] != fixed_factor_value:
      raise RuntimeError(
          f"因子{fixed_name}未固定! 期望值: {fixed_factor_value}, "
          f"实际值: {unique_vals}"
          )
    else:
        print(f"✓ {fixed_name}因子成功固定为值: {fixed_factor_value}")
    
    # 8. 归一化图像数据
    X_ims = X_ims.astype(np.float32) / 255.0
    
    elapsed = time.time() - start_time
    print(f"数据加载耗时：{elapsed:.2f}秒")
    
    return X_ims, y_orien,all_labels

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
    
        
