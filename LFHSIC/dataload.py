# dataload.py
import os
import numpy as np
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

def load_3dshapes(sample_size=None):
    """
    加载3DShapes数据集（来自DeepMind）
    sample_size: 可选，用于测试的小样本大小
    """
    # 验证数据路径
    shape_path = os.path.join(DATA_ROOT, '3dshapes', '3dshapes.h5')
    if not os.path.exists(shape_path):
        raise FileNotFoundError(f"3Dshape数据未找到: {shape_path}. 当前工作目录: {os.getcwd()}")
    
    with h5py.File(shape_path, 'r') as f:
        images = f['images'][:]
        labels = f['labels'][:]
    
    # 提取方向标签（第5列）并添加噪声
    orientations = labels[:, 4]
    noisy_orient = orientations + np.random.normal(0, 1, orientations.shape)
    
    # 限制样本大小用于快速测试
    if sample_size:
        return images[:sample_size], noisy_orient[:sample_size]
    return images, noisy_orient

def load_msd(sample_size=None):
    """
    加载百万歌曲数据集（MSD）来自UCI
    sample_size: 可选，用于测试的小样本大小
    """
    msd_path = os.path.join(DATA_ROOT, 'msd', 'YearPredictionMSD.txt')
    if not os.path.exists(msd_path):
        raise FileNotFoundError(f"MSD数据未找到: {msd_path}. 当前工作目录: {os.getcwd()}")
    
    # 逐块加载大文件
    data = np.loadtxt(msd_path, delimiter=',')
    years = data[:, 0]
    features = data[:, 1:]
    
    # 添加强噪声
    noisy_features = features + np.random.normal(0, 1000, features.shape)
    
    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(noisy_features)
    
    # 限制样本大小用于快速测试
    if sample_size:
        return scaled_features[:sample_size], years[:sample_size]
    return scaled_features, years

# 测试函数（可选）
if __name__ == "__main__":
    print(f"数据根目录: {DATA_ROOT}")
    try:
        img, orient = load_3dshapes(sample_size=10)
        feat, year = load_msd(sample_size=10)
        print("3DShapes测试通过:", img.shape, orient.shape)
        print("MSD测试通过:", feat.shape, year.shape)
    except Exception as e:
        print(f"数据加载错误: {str(e)}")
