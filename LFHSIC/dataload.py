# dataload.py
import os
import numpy as np
import panda as pd
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
        return '/content/drive/MyDrive/Colab_experiments/nips_lfhsic_code_data/data'
    
    # 本地环境（备用）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'data')

DATA_ROOT = get_data_path()

#计算索引---3Dshape数据集处理工具
def get_index(factors):
    indices=0
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
    dataset=h5py.File('3dshape.h5',r)
    print(dataset.keys())
    images=dataset['images']
    labels=dataset['labels']
    
    #定义数据维度
    n_samples=labels.shape[0]
    image_shape=images.shape[1:]
    label_shape=labels.shape[1:]
    
    FACTORS_IN_ORDER=[
        'floor_hue', #地板色相
        'wall_hue',  #墙壁色相
        'object_hue',#物体色相
        'scale',#物体尺寸
        'shape',#物体形状
        'orientation'#观察场景
    ]
    NUM_VALUES_PER_FACTOR{
         'floor_hue':10,
        'wall_hue':10,  
        'object_hue':10,
        'scale':8,
        'shape':4,
        'orientation':15
    #可以取值的范围
    }
    
    factors=np.zeros([len(FACTORS_IN_ORDER),batch_size],dtype=np.int32)
    
    for factor,name in enumerate(FACTORS_IN_ORDER):
        num_choice=NUM_VALUES_PER_FACTOR[name]
        fartors[factor]=np.ranom.choice(num_choice,batch_size)
    factors[fixed_fator]=fixed_factor
    
    indices=get_index(factors)
    
    factors=factors.T
    y_orien=factors['orientation']
    y-orien=torch.tensor(y_orien).
    
    x_ims=[]
    
    for ind in indices:
        im=images[ind]
        im=np.asarray(im)
        x_ims.append(im)
        
    x_ims=np.stack(ims,axis=0)
    x_ims=ims/255 #标准化 转换到[0,1] (RGB最后通道255)
    x_ims=ims.astype(np.float32)
    
    return ims.reshape(batch_size,64,64,3),y_orien.reshape(batch_size,6)  #显示保证输出形状
    
RANDOM_SEED  42 

def load_yearprediction_msd(train_samplesn=5000,test_samplesn=1000,random_sample=True):
    
    #验证数据路径
    msd_path = os.path.join(DATA_ROOT, 'msd', 'YearPredictionMSD.txt')
    if not os.path.exists(msd_path):
        raise FileNotFoundError(f"MSD数据未找到: {msd_path}. 当前工作目录: {os.getcwd()}")

    #数据集官方建议划分方式
    OFFICAL_TRAIN_SIZE=463715
    OFFICAL_TEST_SIZE=51630
    total_size=OFFICAL_TRAIN_SIZE+OFFICAL_TEST_SIZE
    #采样数据大小规定
    train_samplesn=min(train_samplesn,OFFICAL_TRAIN_SIZE)
    test_samplesn=min(test_samplesn,OFFICAL_TEST_SIZ)
    #定义数据类型
    dtypes={0:np.int32}
    for i in range(1,91):
        dtypes[i]=np.float32
    
    #分批读取---整个文件读取
    chunck_size=10000
    chuncks=[]
    for chunck in pd.read.csv(msd_path,header=None,dtype=dtypes,usecol=range(90),chuncksize=chunck_size):
        chuncks.append(chunck)
    data=pd.contat(chunck)
    #验证数据集大小
    if len(data)!=total_size: print(f"警告：数据集大小不匹配！预期：{total_size},实际:{len(data)}")
    
    #数据集分割
    train_sample=data.iloc[0:OFFICIAL_TRAIN_SIZE]
    test_sample=data.iloc[OFFICAL_TRAIN_SIZE:]

    del data
    
    #抽样
    if random_samples: #随机抽样
        if train_samplesn<OFFICIAL_TRAIN_SIZE:
            trainsample=train_sample.sample(n=train_samplesn,random_state=RANDOM_SEED)
        if test_samples<OFFICIAL_TEST_SIZE:
            testsample=test_sample.sample(n=test_samplesn,random_state=RANDOM_SEED)
    else: #顺序抽样
            trainsample=train_sample.iloc[:train_samplesn]
            testsample=test_sample.iloc[:test_samplesn]

    #分离特征和目标变量

    X_train=trainsample[:,1:].values.astype(np.float32)
    Y_train=trainsample[:,0].values.astype(np.flat32)

    X_test=testsample[:,1:].values.astype(np.float32)
    Y_test=testsample[:,0].values.astype(np.float32)
    
    return (Y_train,X_train),(Y_test,X_test)
    
