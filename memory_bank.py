import os

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from data import get_transforms
from model import FeatureExtractor


def gen_memory_bank(root_dir, model_name='resnet50', save_path='memory_bank.npy', device='cuda'):
	"""
	构建缺陷样本的Memory Bank
	root_dir: 组织形式为 root/class_name/image.jpg
	"""

	model = FeatureExtractor(model_name, return_map=False).to(device)  # 确保模型是特征提取器
	model.eval()  # 设置为评估模式

	transforms = get_transforms((224, 224))  # 获取预处理函数

	memory_bank = {}  # {class_name: [feature1, feature2, ...]}

	with torch.no_grad():
		for class_name in os.listdir(root_dir):
			class_dir = os.path.join(root_dir, class_name)
			if not os.path.isdir(class_dir):
				continue
			features = []
			for img_name in os.listdir(class_dir):
				img_path = os.path.join(class_dir, img_name)
				try:
					image = Image.open(img_path).convert('RGB')  # 确保是RGB格式
					input_tensor = transforms(image).unsqueeze(0).to(device)
					feat = model(input_tensor)  # [1, C]
					feat = F.normalize(feat, p=2, dim=1)  # L2归一化便于匹配
					features.append(feat.squeeze(0).cpu().numpy())
				except Exception as e:
					print(f"Failed to process {img_path}: {e}")
			if features:
				memory_bank[class_name] = np.stack(features)

	# 保存为 numpy 字典
	np.save(save_path, memory_bank)
	print(f"Memory bank saved to {save_path}")


class MemoryBank:
	def __init__(self, path):
		"""
		加载保存好的 memory_bank.npy 文件
		"""
		self.path = path
		self.memory = np.load(path, allow_pickle=True).item()  # 载入为 dict
		self.classes = list(self.memory.keys())
		self.index_bank()

	def index_bank(self):
		"""
		将所有特征统一拼接为 (N, D)，并建立 index 到类别的映射
		"""
		self.all_features = []
		self.all_labels = []

		for cls in self.classes:
			feats = self.memory[cls]  # (N_cls, D)
			self.all_features.append(feats)
			self.all_labels.extend([cls] * len(feats))

		self.all_features = np.vstack(self.all_features)  # (N_total, D)

	def match(self, query_feat, topk=1, metric='cosine'):
		"""
		输入单个query特征，返回top-k最相似的类别标签和相似度
		query_feat: numpy array of shape (D,)
		"""
		if metric == 'cosine':
			sims = np.dot(self.all_features, query_feat)  # 特征已归一化，直接内积
		elif metric == 'l2':
			sims = -np.linalg.norm(self.all_features - query_feat, axis=1)
		else:
			raise ValueError("Unsupported metric")

		topk_idx = np.argsort(-sims)[:topk]
		topk_scores = sims[topk_idx]
		topk_labels = [self.all_labels[i] for i in topk_idx]

		return list(zip(topk_labels, topk_scores))


if __name__ == '__main__':
	gen_memory_bank(
		root_dir='examples/',
		model_name='moco',
		save_path='mem_banks/memory_bank_moco.npy',
		device='cpu'
	)

# mb = MemoryBank('memory_bank.npy')

# print()
