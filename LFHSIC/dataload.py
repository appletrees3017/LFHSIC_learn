# dataload.py
import os
import time
import numpy as np
import pandas as pd
import torch
import h5py
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

#定义全局变量--数据根路径
DATA_ROOT = get_data_path()

#3dshape数据导入

#定义成全局变量--因子的解释
FACTORS_IN_ORDER=[
        'floor_hue', #地板色相
        'wall_hue',  #墙壁色相
        'object_hue',#物体色相
        'scale',#物体尺寸
        'shape',#物体形状
        'orientation'#观察场景
    ]
NUM_VALUES_PER_FACTOR={
         'floor_hue':10,
        'wall_hue':10,  
        'object_hue':10,
        'scale':8,
        'shape':4,
        'orientation':15
    #可以取值的范围
    }

#计算索引---3Dshape数据集处理工具
def get_index(factors):
    indices=np.zeros(factors.shape[1], dtype=np.int32) #数组操作--把indices定义成为数组
    base=1
    for factor,name in reversed(list(enumerate(FACTORS_IN_ORDER))):
        indices+=factors[factor]*base
        base*=NUM_VALUES_PER_FACTOR[name]
    return indices
        
def load_3dshapes(batch_size,fixed_factor,fixed_factor_value):
    
    # 验证数据路径
    shape_path = os.path.join(DATA_ROOT, '3dshapes', '3dshapes.h5')
    if not os.path.exists(shape_path):
        raise FileNotFoundError(f"3Dshape数据未找到: {shape_path}. 当前工作目录: {os.getcwd()}")
    start_time=time.time()   
    #加载数据准备 ---惰性加载
    dataset=h5py.File(shape_path,'r') #文件路径自适应
    print(dataset.keys())
    images=dataset['images']
    labels=dataset['labels']
    
    #定义数据维度
    n_samples=labels.shape[0]
    image_shape=images.shape[1:]
    label_shape=labels.shape[1:]

    
    factors=np.zeros([len(FACTORS_IN_ORDER),batch_size],dtype=np.int32)
    
    for factor,name in enumerate(FACTORS_IN_ORDER):
        num_choice=NUM_VALUES_PER_FACTOR[name]
        factors[factor] = np.random.choice(num_choice, batch_size)  
        
    factors[fixed_factor] = fixed_factor_value  # ‘’‘’

    indices=get_index(factors)
    
    factors=factors.T
    #y_orien=factors['orientation']
    orient_idx = FACTORS_IN_ORDER.index('orientation')  # 获取索引位置
    y_orien = factors[:, orient_idx]                   # 正确切片
    #y_orien=torch.tensor(y_orien)
    
    x_ims=[]
    
    for ind in indices:
        im=images[ind]
        im=np.asarray(im)
        x_ims.append(im)
        
    x_ims=np.stack(x_ims,axis=0) #数组堆叠--组成多维数组
    x_ims=x_ims/255.0 #标准化 转换到[0,1] (RGB最后通道255)
    x_ims=x_ims.astype(np.float32)
    #x_ims = torch.tensor(x_ims)
    
    #结束时间
    elapsed=time.time()-start_time
    print("总耗时：{elapsed:.2f}秒")
    return x_ims.reshape(batch_size,64,64,3),y_orien.reshape(batch_size,1)  #显示保证输出形状

def calculate_samples_per_bin(bin_data,total_samples):
    """根据比例分配每个箱的抽样数量
        bin_data 分箱数据--样本集经过分箱处理
        total_samples 总抽样数量"""
    bin_counts={key:len(indices) for key,indices in bin_data.items() }
    total_count=sum(bin_counts.values())

    samples_per_bin={}
    remeaning_samples=total_samples
    
    for bin_key in bin_data.keys():
        bin_size=len(bin_data[bin_key])
        proportion=bin_size/total_count
        allocated=max(1,int(total_samples*proportion))
        allocated=min(allocated,bin_size)
        samples_per_bin[bin_key]=allocated
        
    bin_keys=list(bin_data.keys())
    while remeaning_samples>0:
        bin_key=random.choice(bin_keys)
        if samples_per_bin[bin_key]<len(bin_data[bin_key]):
            samples_per_bin[bin_key]+=1
            remeaning_samples-=1
            
    return samples_per_bin
    
def load_ypmsd_stratified(bin_size=10,seed=42,train_samplesn=5000,test_samplesn=1000):
    msd_path=os.path.join(DATA_ROOT,'msd','YearPredictionMSD.csv')
    if not os.path.exists(msd_path):
        raise FileNotFoundError(f"MSD数据集未找到:{msd_path},当前工作目录:{os.getwd()}")
    start_time=time.time()

    #官方数据建议划分
    OFFICIAL_TRAIN_SIZE=463715
    OFFICIAL_TEST_SIZE=51630
    TOTAL_SIZE=OFFICIAL_TRAIN_SIZE+OFFICIAL_TEST_SIZE
    #随机种子
    random.seed(seed)
    np.random.seed(seed)

    #初始分箱数据
    bins=list(range(1920,2020,bin_size))
    if 2020 not in bins:
        bins.append(2020)
    #按分箱组织索引
    train_bins=defaultdic(list)
    test_bins=defaultdic(list)
    #遍历整个文件，建立年份索引
    with open(msd_path,'r') as f:
        for line_idx,line in enumerate(f):
            year=int(float(line.split(',')[0]) #取年份--第一列
            bin_idx=bisect.bisect_left(bins,year)-1
            if bin_idx<0:
                bin_idx=0
            elif bin_idx>len(bins)-1:
                bin_idx=len(bins)-2
            bin_key=f"{bins[bin_idx]}---{bins[bin_idx+1]}"
            #确定样本所属数据集--根据行号确定属于训练集或测试集
            if line_idx<OFFICIAL_TRAINSIZE:
                train_bins[bin_key].append(line_idx)
            else:
                test_bins[bin_key].append(line_idx)
    print("\n 训练集和测试集样本年份分布统计：")
    for bin_key in sort(train_bins.keys()):
        print(f"分箱{bin_key}:{len(train_bins[bin_key])}个样本")
    for bin_key in sort(test_bins.keys()):
        print(f"分箱{bin_key}:{len(test_bins[bin_key])}个样本")
    #按照样本分布情况，计算分箱抽样样本量
    train_samples_per_bin=calculate_samples_per_bin(train_bins,train_samplesn)
    test_samples_per_bin=calculate_samples_per_bin(test_bins,test_samplesn)
    #分层抽样--需要抽取的行号
    select_train_indices=[] #训练集抽样数据
    for bin_key,count in train_samples_per_bin.items():
        selected=random.sample(train_bins[bin_key],count)
        select_train_indices.extend(selected)
    select_test_indices=[] #测试集抽样数据
    for bin_key,count in test_samples_per_bin.items():
        selected=random.sample(test_bins[bin_key],count)
        select_test_indices.extend(selected)
    all_indices=select_train_indices+select_test_indices
    all_indices.sort()

    sampled_data=[]
    index_map=[]
    with open(msd_path,'r') as f:
        for lineidex,line in enumerate(f):
            if lineidex>all_indices[-1]:
                break
            if lineidex in select_train_indices:
                data=list(map(float,line.strip().split(',')))
                sampled_data.append(data)
                index_map.append(('train',lineidex))
            elif lineidex in select_tset_indices:
                data=list(map(float,line.strip().split(','))
                sampled_data.append(data)
                index_map.append(('test',lineidex))
    #转换成数据，分离训练集和测试集
    sampled_array=np.array(sampled_data,dtype=np.float32)
    train_data=[]
    test_data=[]
    for i,(label,lineidex) in enumerate(index_map):
        if label=='train':
            train_data.append(sampled_array[i])
        else:
            test_data.append(sampled_array[i])
    train_array=np.array(train_data,dtype=np.float32)
    test_array=np.array(test_data,dtype=np.float32)
    #分离特征和年份
    X_train=train_array[:,1:]
    X_test=test_array[:,1:]

    Y_train=train_array[:,0].astype(np.int32)
    Y_test=test_array[:,0].astype(np.int32)

    #结束时间
    elapsed=time.time()-start_time
    print("总耗时：{elapsed:.2f}秒")
    return (Y_train,X_train),(Y_test,X_test)
    
        
