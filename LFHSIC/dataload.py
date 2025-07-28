# src/datasets.py
import numpy as np
import h5py
import os

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_3dshapes():
    with h5py.File(os.path.join(DATA_ROOT, '3dshapes/3dshapes.h5'), 'r') as f:
        return f['images'][:], f['labels'][:]

def load_msd():
    return np.loadtxt(os.path.join(DATA_ROOT, 'msd/YearPredictionMSD.txt'), delimiter=',')