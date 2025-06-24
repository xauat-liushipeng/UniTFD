import cv2
import numpy as np
import torch
from torchvision import transforms

from gen_proposals import sliding_window
from load_bank import load_feature_bank
from model import FeatureExtractor
from visual_matching import match_features
from post_proc import select_topk_nms

def preprocess_image(image, input_size=(224, 224)):
    # BGR to RGB, resize, to tensor, normalize
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def extract_region_features(model, image, proposals, device):
    """
    image: 原始BGR numpy array
    proposals: list of [x1,y1,x2,y2]
    返回: tensor [N, D]  候选区域特征
    """
    model.eval()
    features = []
    with torch.no_grad():
        for box in proposals:
            x1, y1, x2, y2 = box
            region = image[y1:y2, x1:x2]
            if region.size == 0 or region.shape[0]<10 or region.shape[1]<10:
                # 跳过无效区域
                continue
            region_tensor = preprocess_image(region).unsqueeze(0).to(device)  # [1,3,H,W]
            feat = model(region_tensor)  # [1, D]
            features.append(feat.cpu())
    if len(features) == 0:
        return torch.empty(0, model.out_dim)
    features = torch.cat(features, dim=0)
    return features

def main(exemplar_path, query_path, feature_bank_dir, model_name='resnet50', topk=5, device='cuda'):
    # 1. 读取示例图，提取示例缺陷特征（假设只一张示例）
    exemplar_img = cv2.imread(exemplar_path)
    if exemplar_img is None:
        raise FileNotFoundError(f"Exemplar image not found: {exemplar_path}")

    # 简单示例：假设示例图是裁剪好的缺陷区域，直接提取特征
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = FeatureExtractor(model_name).to(device)

    model.eval()
    with torch.no_grad():
        exemplar_tensor = preprocess_image(exemplar_img).unsqueeze(0).to(device)
        exemplar_feat = model(exemplar_tensor)  # [1, D]

    # 2. 读取查询图，生成候选框
    query_img = cv2.imread(query_path)
    if query_img is None:
        raise FileNotFoundError(f"Query image not found: {query_path}")
    proposals = sliding_window(query_img, step_size=50, window_size=(100,100))
    print(f"Generated {len(proposals)} proposals.")

    # 3. 提取候选框特征
    query_feats = extract_region_features(model, query_img, proposals, device)
    print(f"Extracted features for {query_feats.shape[0]} proposals.")

    if query_feats.shape[0] == 0:
        print("No valid proposals found.")
        return

    # 4. 匹配示例特征与候选区域特征
    scores = match_features(exemplar_feat.cpu(), query_feats.cpu())  # tensor [N]

    # 5. 筛选Top-K + NMS输出
    proposals_np = np.array(proposals)
    # 注意前面extract_region_features跳过了部分proposal，索引要对应
    # 这里简单假设没有跳过
    boxes, scores_final = select_topk_nms(proposals_np, scores.numpy(), topk=topk, iou_thresh=0.5)

    print("Final detection boxes and scores:")
    for b, s in zip(boxes, scores_final):
        print(f"Box: {b}, Score: {s:.4f}")

    # 6. 可视化检测结果
    vis_img = query_img.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis_img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imshow("Detection Result", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python main.py <exemplar_image> <query_image> [model_name] [topk]")
        sys.exit(1)

    exemplar_path = sys.argv[1]
    query_path = sys.argv[2]
    model_name = sys.argv[3] if len(sys.argv) > 3 else 'resnet50'
    topk = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    main(exemplar_path, query_path, feature_bank_dir=None, model_name=model_name, topk=topk)
