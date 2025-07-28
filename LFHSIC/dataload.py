# src/datasets.py
import numpy as np
import h5py
import os

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_3dshapes():
    with h5py.File(os.path.join(DATA_ROOT, '3dshapes/3dshapes.h5'), 'r') as f:
        # 仅使用前10000个样本（加速实验）
        return images[:10000], noisy_orient[:10000]

def load_msd():
    """加载MSD数据集"""
    msd_path = os.path.join(DATA_ROOT, 'MSD', 'YearPredictionMSD.txt')
    if not os.path.exists(msd_path):
        raise FileNotFoundError(f"MSD数据未找到: {msd_path}")
    
    data = np.loadtxt(msd_path, delimiter=',')
    years = data[:, 0]
    features = data[:, 1:]
    """
    # 添加强噪声（论文要求）
    noisy_features = features + np.random.normal(0, 1000, features.shape)
    """
    # 仅使用前5000个样本（加速实验）
    return noisy_features[:5000], years[:5000]        # 标准化特征
