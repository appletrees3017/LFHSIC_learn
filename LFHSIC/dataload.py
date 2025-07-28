import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, data_root='data/'):
        self.data_root = data_root
    
    def load_3dshapes(self):
        """加载3DShapes数据集"""
        with h5py.File(f"{self.data_root}/3dshapes/3dshapes.h5", 'r') as f:
            images = f['images'][:]
            labels = f['labels'][:]
        
        # 提取方向标签（第5列）
        orientations = labels[:, 4]
        
        """添加噪声（论文要求）
        noisy_orient = orientations + np.random.normal(0, 1, orientations.shape)"""
        
        return images, noisy_orient
    
    def load_msd(self):
        """加载MSD数据集"""
        data = np.loadtxt(f"{self.data_root}/msd/YearPredictionMSD.txt", delimiter=',')
        years = data[:, 0]
        features = data[:, 1:]
        
        """添加强噪声（论文要求）
        noisy_features = features + np.random.normal(0, 1000, features.shape)
        
        # 标准化特征
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(noisy_features)"""
        
        return scaled_features, years
