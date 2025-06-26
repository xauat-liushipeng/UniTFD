import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.image_list import ImageList

from data import get_transforms

_rpn_model = None


def get_rpn_model(device='cpu'):
	global _rpn_model
	if _rpn_model is None:
		model = fasterrcnn_resnet50_fpn(pretrained=True)
		model.eval()
		_rpn_model = model.to(device)
	return _rpn_model


def extract_rpn_proposals(image, device='cpu', top_n=1000):
	"""
	generating proposals by RPN
	"""
	h, w = image.shape[:2]
	model = get_rpn_model(device)

	transforms = get_transforms((224, 224))  # 获取预处理函数

	# Step 1: 预处理图像
	pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	img_tensor = transforms(pil_image).to(device)

	# Step 2: 构造 ImageList
	img_list = ImageList(img_tensor.unsqueeze(0), [(h, w)])  # batch_size=1

	# Step 3: 提取特征 → RPN 推理
	with torch.no_grad():
		features = model.backbone(img_tensor.unsqueeze(0))
		proposals, _ = model.rpn(img_list, features)

	boxes = proposals[0].cpu().numpy()

	# Step 4: 裁剪合法坐标
	boxes_clipped = []
	for x1, y1, x2, y2 in boxes[:top_n]:
		x1 = max(0, min(w, x1))
		y1 = max(0, min(h, y1))
		x2 = max(0, min(w, x2))
		y2 = max(0, min(h, y2))
		if x2 - x1 >= 10 and y2 - y1 >= 10:
			boxes_clipped.append([int(x1), int(y1), int(x2), int(y2)])

	return boxes_clipped


def sliding_window(image, step_size, window_size):
	"""
	generating proposals by sliding window
	"""
	proposals = []
	for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
		for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
			proposals.append([x, y, x + window_size[0], y + window_size[1]])
	return proposals


def selective_search_proposals(image):
	"""
	generating proposals by selective search
	"""
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)
	ss.switchToSelectiveSearchFast()  # 或 switchToSelectiveSearchQuality()
	rects = ss.process()
	proposals = []
	for (x, y, w, h) in rects:
		proposals.append([x, y, x + w, y + h])
	return proposals


def compute_edge_map(image):
	"""
	计算整幅图像的边缘图（可选用 Canny 或 Sobel）
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edge = cv2.Canny(gray, 50, 150)  # 可调参数
	return edge


def proposals_filter(image, rects,
                     min_area=300,
                     max_area_ratio=0.4,
                     aspect_ratio_range=(0.3, 3.0),
                     edge_thresh=0.015):
	h_img, w_img = image.shape[:2]
	max_area = h_img * w_img * max_area_ratio
	edge_map = compute_edge_map(image)

	proposals = []
	for (x, y, w, h) in rects:
		x1, y1, x2, y2 = x, y, x + w, y + h
		area = w * h
		if area < min_area or area > max_area:
			continue

		aspect_ratio = w / h if h != 0 else 0
		if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
			continue

		patch_edge = edge_map[y1:y2, x1:x2]
		if patch_edge.size == 0:
			continue
		edge_density = np.count_nonzero(patch_edge) / (area + 1e-6)
		if edge_density < edge_thresh:
			continue

		proposals.append([x1, y1, x2, y2])

	return proposals


def selective_search_filtered(image,
                              mode='fast',
                              max_regions=1000,
                              min_area=300,
                              max_area_ratio=0.4,
                              aspect_ratio_range=(0.3, 3.0),
                              edge_thresh=0.015):
	"""
	结合边缘、面积和长宽比过滤的 Selective Search Proposals
	"""
	# 1. 初始化选择性搜索
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)
	if mode == 'fast':
		ss.switchToSelectiveSearchFast()
	else:
		ss.switchToSelectiveSearchQuality()
	rects = ss.process()[:max_regions]  # 限制最多返回 max_regions 个 proposals

	h_img, w_img = image.shape[:2]
	max_area = h_img * w_img * max_area_ratio
	edge_map = compute_edge_map(image)

	proposals = []
	for (x, y, w, h) in rects:
		x1, y1, x2, y2 = x, y, x + w, y + h
		area = w * h
		if area < min_area or area > max_area:
			continue

		aspect_ratio = w / h if h != 0 else 0
		if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
			continue

		patch_edge = edge_map[y1:y2, x1:x2]
		if patch_edge.size == 0:
			continue
		edge_density = np.count_nonzero(patch_edge) / (area + 1e-6)
		if edge_density < edge_thresh:
			continue

		proposals.append([x1, y1, x2, y2])

	return proposals


def merge_proposals_by_center(proposals, eps=30, min_samples=1):
	"""
	proposals: List of [x1, y1, x2, y2]
	返回：锚点列表 [(cx, cy)]
	"""
	centers = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in proposals])
	if len(centers) == 0:
		return []

	# 基于空间距离的聚类（或可选KMeans）
	clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
	labels = clustering.labels_
	unique_labels = set(labels)
	merged_points = []

	for lbl in unique_labels:
		if lbl == -1:
			continue  # -1 是离群点
		group = centers[labels == lbl]
		merged_center = group.mean(axis=0)
		merged_points.append(tuple(merged_center))
	return merged_points


def generate_fixed_box_from_points(points, box_size=(100, 100), image_shape=None):
	"""
	输入锚点列表，生成固定大小的proposal框
	"""
	w_box, h_box = box_size
	h_img, w_img = image_shape[:2] if image_shape is not None else (1e6, 1e6)

	boxes = []
	for cx, cy in points:
		x1 = int(cx - w_box / 2)
		y1 = int(cy - h_box / 2)
		x2 = x1 + w_box
		y2 = y1 + h_box

		# 裁剪到图像范围
		if x1 < 0 or y1 < 0 or x2 > w_img or y2 > h_img:
			continue
		boxes.append([x1, y1, x2, y2])
	return boxes


def smart_selective_search(image, mode='fast', eps=10, box_size=(100, 100)):
	proposals = selective_search_filtered(image, mode)
	anchors = merge_proposals_by_center(proposals, eps)
	proposals = generate_fixed_box_from_points(anchors, box_size, image_shape=image.shape)
	return proposals


if __name__ == '__main__':
	img_path = "data/image/image.jpg"
	image = cv2.imread(img_path)

	''' sliding window '''
	# proposals = sliding_window(image, step_size=50, window_size=(100, 100))

	''' selective search '''
	# proposals = selective_search_proposals(image)

	''' selective search + edges filter, merge anchor points, gen new proposals '''
	# proposals = selective_search_filtered(image, mode='fast')
	# anchors = merge_proposals_by_center(proposals, eps=10)
	# proposals = generate_fixed_box_from_points(anchors, box_size=(100, 100), image_shape=image.shape)

	''' RPN '''
	proposals = extract_rpn_proposals(image)
	proposals = proposals_filter(image, proposals)

	print(f"Generated {len(proposals)} proposals.")

	# visualize proposals
	vis = image.copy()
	for i, (x1, y1, x2, y2) in enumerate(proposals):
		color = tuple(np.random.randint(0, 255, 3).tolist())
		# color = (0, 255, 0)
		# cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
		cv2.circle(vis, (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2), 3, color, -1)

	cv2.imshow("Proposals", vis)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imwrite("proposals_output.jpg", vis)
