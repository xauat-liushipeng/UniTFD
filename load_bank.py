import os
import numpy as np

def load_feature_bank(feature_dir):
    """
    从feature_dir加载所有.npy特征文件，组成字典
    """
    feature_bank = {}
    for fname in os.listdir(feature_dir):
        if fname.endswith('.npy'):
            feat = np.load(os.path.join(feature_dir, fname))
            key = os.path.splitext(fname)[0]
            feature_bank[key] = feat
    return feature_bank

if __name__ == '__main__':
    import sys
    bank_dir = sys.argv[1]
    bank = load_feature_bank(bank_dir)
    print(f"Loaded {len(bank)} features from {bank_dir}.")
