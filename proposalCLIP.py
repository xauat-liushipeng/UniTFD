import networkx as nx
import torch
import clip
import cv2
import numpy as np
import torch.nn.functional as F

from PIL import Image
from EdgeBoxes.edge_box import edgebox



# 计算两个候选框的 IoU（空间重叠）
def cal_iou(box1, box2):
	x1, y1, w1, h1 = box1
	x2, y2, w2, h2 = box2
	xi1 = max(x1, x2)
	yi1 = max(y1, y2)
	xi2 = min(x1 + w1, x2 + w2)
	yi2 = min(y1 + h1, y2 + h2)
	inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
	union = w1 * h1 + w2 * h2 - inter
	return inter / union if union > 0 else 0.0

# 初始化 CLIP 模型
def init_CLIP(model_path, classes_list, device="cuda"):
	model, preprocess = clip.load(model_path, device=device)
	prompts = []
	for class_name in classes_list:
		prompts.append(f"a photo of a {class_name}")
	text_tokens = clip.tokenize(prompts).to(device)
	return model, preprocess, text_tokens

# CLIP 相似度计算及目标得分估计（用于筛选低熵提案）
@torch.no_grad()
def clip_objectness(image, init_boxes, init_scores,
					model, preprocess, tokens,
					batch_size=32, high_entropy_ratio=0.4,
					lambda_sim=0.06, lambda_sl=1, device="cuda"):

	# 将候选框图像区域裁剪并预处理
	img_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	crops = []
	for (x, y, w, h) in init_boxes:
		crop = img_pil[y:y+h, x:x+w]
		pil = Image.fromarray(crop)
		crops.append(preprocess(pil))
	inputs = torch.stack(crops).to(device)

	# 批量提取图像区域的 CLIP 特征
	img_feats = []
	for i in range(0, len(inputs), batch_size):
		batch = inputs[i:i+batch_size]
		feats = model.encode_image(batch)
		feats /= feats.norm(dim=-1, keepdim=True)
		img_feats.append(feats)
	img_feats = torch.cat(img_feats, dim=0)

	# 提取文本特征（支持多类别）
	txt_feats = model.encode_text(tokens)
	txt_feats /= txt_feats.norm(dim=-1, keepdim=True)

	# 计算图像区域和文本之间的相似度矩阵
	sim_matrix = img_feats @ txt_feats.T
	sim_softmax = F.softmax(sim_matrix, dim=1)

	# 计算每个提案的相似度熵（衡量模糊性）
	entropy = -torch.sum(sim_softmax * torch.log(sim_softmax + 1e-6), dim=1)

	# 根据熵值筛除高熵提案
	N = sim_matrix.shape[0]
	T = int(N * (1 - high_entropy_ratio))
	sorted_entropy, indices = torch.sort(entropy)
	keep_idx = indices[:T]

	kept_entropy = sorted_entropy[:T]
	kept_sim = sim_matrix[keep_idx]
	kept_max_sim = torch.max(kept_sim, dim=1).values

	# 目标得分公式计算
	norm_entropy = kept_entropy / (torch.norm(kept_entropy) + 1e-6)
	C = txt_feats.shape[0]
	S_t = - (T / C) * norm_entropy + lambda_sim * kept_max_sim

	init_scores_tensor = torch.tensor(init_scores, device=device)
	SL_t = init_scores_tensor[keep_idx].squeeze()
	S_t += lambda_sl * SL_t

	return entropy, S_t, keep_idx, img_feats

# 基于图的提案合并（联通子图聚合）
def merge_boxes(boxes, scores, feats, entropy, thr_iou=0.5, thr_sim=0.9):
	T, D = feats.shape
	G = nx.Graph()

	# 构建图节点
	for i in range(T):
		G.add_node(i)

	# 根据 IoU 和语义相似度建图连边
	for i in range(T):
		for j in range(i + 1, T):
			iou = cal_iou(boxes[i], boxes[j])
			sim = F.cosine_similarity(feats[i].unsqueeze(0), feats[j].unsqueeze(0)).item()
			if iou > thr_iou and sim > thr_sim:
				G.add_edge(i, j)

	score_tensor = torch.tensor(scores, device=feats.device) if not isinstance(scores, torch.Tensor) else scores
	entropy_tensor = torch.tensor(entropy, device=feats.device) if not isinstance(entropy, torch.Tensor) else entropy

	merged_boxes, merged_feats, merged_scores, merged_entropy = [], [], [], []
	for comp in nx.connected_components(G):
		comp = list(comp)
		if len(comp) <= 1:
			continue

		# 合并框坐标为均值
		box_group = np.array([boxes[i] for i in comp])
		merged_box = np.round(box_group.mean(axis=0)).astype(int)

		# 合并语义特征、目标得分和熵
		merged_feat = F.normalize(torch.stack([feats[i] for i in comp]).mean(dim=0), dim=0)
		merged_score = torch.stack([score_tensor[i] for i in comp]).mean()
		merged_ent = torch.stack([entropy_tensor[i] for i in comp]).mean()

		merged_boxes.append(merged_box)
		merged_feats.append(merged_feat)
		merged_scores.append(merged_score)
		merged_entropy.append(merged_ent)

	if len(merged_boxes) == 0:
		return [], [], []

	merged_scores = torch.stack(merged_scores)
	merged_entropy = torch.stack(merged_entropy)
	max_E = entropy_tensor.max()
	keep_mask = merged_entropy <= max_E

	final_boxes = [merged_boxes[i] for i in range(len(merged_boxes)) if keep_mask[i]]
	final_scores = merged_scores[keep_mask]
	return final_boxes, final_scores

# 主流程入口函数
def main(img, clip_model, preprocess, tokens, device):
	"""
	主流程逻辑：
	1. 使用 EdgeBoxes 获取初始提案和打分（O_m, SL_m）
	2. 使用 CLIP 提取图像区域特征并计算文本相似度，获得每个提案的熵 E_m 和最终目标得分 S_t
	3. 保留低熵提案，得到保留索引 keep_id 和对应特征
	4. 构建图结构：边连接满足空间和语义相似度门限的提案节点
	5. 查找最大连通子图并进行特征/得分融合，去除熵过高的合并提案
	6. 返回最终筛选出的提案和得分
	"""
	# 获取初始 proposals 和低层次打分
	init_boxes, init_scores = edgebox("./EdgeBoxes/model/model.yml.gz", img, 300)

	# 使用 CLIP 特征评估各提案（计算相似度熵和目标得分）
	entropy, S_t, keep_id, img_feats = clip_objectness(
		img, init_boxes, init_scores,
		clip_model, preprocess, tokens,
		batch_size=32, high_entropy_ratio=0.4,
		lambda_sim=0.06, lambda_sl=1, device=device
	)

	# 提取保留的低熵提案信息
	kept_boxes = [init_boxes[i] for i in keep_id.cpu().numpy()]
	kept_scores = S_t
	kept_feats = img_feats[keep_id]
	kept_entropy = entropy[keep_id]

	# 基于图结构合并提案
	final_boxes, final_scores = merge_boxes(kept_boxes, kept_scores, kept_feats, kept_entropy, 0.5, 0.9)
	return final_boxes, final_scores

# 测试入口
if __name__ == "__main__":
	device = "cuda" if torch.cuda.is_available() else "cpu"
	classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

	clip_model, preprocess, tokens = init_CLIP("ViT-B/32", classes, device=device)
	img = cv2.imread("test/001192.jpg")

	boxes, scores = main(img, clip_model, preprocess, tokens, device)
	for x, y, w, h in boxes:
		cv2.rectangle(img, (x, y), (x + w, y + h), tuple(np.random.randint(0, 255, 3).tolist()), 1)
	cv2.imshow("ProposalCLIP Proposals", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
