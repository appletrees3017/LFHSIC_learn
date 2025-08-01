# dataload.py
import os
import numpy as np
import pandas as pd
import torch
import h5py
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
#3dshape数据导入
#定义成全局变量--因子的解释
DATA_ROOT = get_data_path()
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
    y_orien=torch.tensor(y_orien)
    
    x_ims=[]
    
    for ind in indices:
        im=images[ind]
        im=np.asarray(im)
        x_ims.append(im)
        
    x_ims=np.stack(x_ims,axis=0)
    x_ims=x_ims/255.0 #标准化 转换到[0,1] (RGB最后通道255)
    x_ims=x_ims.astype(np.float32)
    x_ims = torch.tensor(x_ims)
    
    return x_ims.reshape(batch_size,64,64,3),y_orien.reshape(batch_size,1)  #显示保证输出形状

#随机种子，保证实验的可重复性
RANDOM_SEED=42 

def load_yearprediction_msd(train_samplesn=5000,test_samplesn=1000,random_sample=True):
    
    #验证数据路径
    msd_path = os.path.join(DATA_ROOT, 'msd', 'YearPredictionMSD.csv')
    if not os.path.exists(msd_path):
        raise FileNotFoundError(f"MSD数据未找到: {msd_path}. 当前工作目录: {os.getcwd()}")

    #数据集官方建议划分方式
    OFFICIAL_TRAIN_SIZE=463715
    OFFICIAL_TEST_SIZE=51630
    total_size=OFFICAL_TRAIN_SIZE+OFFICAL_TEST_SIZE
    #采样数据大小规定
    train_samplesn=min(train_samplesn,OFFICIAL_TRAIN_SIZE)
    test_samplesn=min(test_samplesn,OFFICIAL_TEST_SIZE)
    #定义数据类型
    dtypes={0:np.int32}
    for i in range(1,91):
        dtypes[i]=np.float32
    
    #分批读取---整个文件读取
    chunk_size=10000
    chunks=[]
    for chunk in pd.read_csv(msd_path,header=None,dtype=dtypes,usecols=range(90),chunksize=chunk_size):
        chunks.append(chunk)
    data=pd.concat(chunks)
    #验证数据集大小
    if len(data)!=total_size: print(f"警告：数据集大小不匹配！预期：{total_size},实际:{len(data)}")
    
    #数据集分割
    train_sample=data.iloc[:OFFICIAL_TRAIN_SIZE]
    test_sample=data.iloc[OFFICIAL_TRAIN_SIZE:]

    del data
    
    #抽样
    if random_sample: #随机抽样
        if train_samplesn<OFFICIAL_TRAIN_SIZE:
            train_sample=train_sample.sample(n=train_samplesn,random_state=RANDOM_SEED)
        if test_samplesn<OFFICIAL_TEST_SIZE:
            test_sample=test_sample.sample(n=test_samplesn,random_state=RANDOM_SEED)
    else: #顺序抽样
            train_sample=train_sample.iloc[:train_samplesn]
            test_sample=test_sample.iloc[:test_samplesn]

    #分离特征和目标变量

    X_train = train_sample.iloc[:, 1:].values.astype(np.float32)  # 正确索引
    Y_train = train_sample.iloc[:, 0].values.astype(np.int32)    # 修正数据类型

    X_test = test_sample.iloc[:, 1:].values.astype(np.float32)
    Y_test = test_sample.iloc[:, 0].values.astype(np.int32)
    
    return (Y_train,X_train),(Y_test,X_test)
    
