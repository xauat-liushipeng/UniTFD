import time

import cv2
import torch
import numpy as np
from model import extract_features, FeatureExtractor
from memory_bank import MemoryBank
from gen_proposals import smart_selective_search, extract_rpn_proposals, proposals_filter
from post_proc import nms_boxes

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference(img_path,
              model_name='resnet50',
              bank_path='memory_bank.npy',
              score_thresh=0.3,
              nms_thresh=0.3,
              batch_size=32,
              input_size=(224, 224),
              save_path=None):
    """
    对单张图像进行缺陷检测推理，并可视化标注结果
    """
    # 加载图像
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Cannot read image: {img_path}")
    h, w = image.shape[:2]
    h_ratio = input_size[0] / h
    w_ratio = input_size[1] / w

    # 加载 memory bank
    bank = MemoryBank(bank_path)

    # 提取 proposals
    proposals = smart_selective_search(image)
    # proposals = extract_rpn_proposals(image)
    # proposals = proposals_filter(image, proposals)

    # visualize proposals
    vis = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(proposals):
        x1, y1, x2, y2 = int(x1 / w_ratio), int(y1 / h_ratio), int(x2 / w_ratio), int(y2 / h_ratio)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # color = (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        # cv2.circle(vis, (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2), 3, color, -1)

    cv2.imshow("Proposals", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 构建模型
    model = FeatureExtractor(model_name, return_map=True).to(device)

    # 提取特征
    features, coords = extract_features(image, proposals, model, device)

    # 匹配
    results = []
    for i, feat in enumerate(features):
        top_match = bank.match(feat, topk=1)
        label, score = top_match[0]
        if score >= score_thresh:
            results.append((coords[i], label, score))

    # NMS
    if not results:
        print("No matching results.")
        return

    boxes = np.array([r[0] for r in results])
    scores = np.array([r[2] for r in results])
    labels = [r[1] for r in results]

    keep = nms_boxes(boxes, scores, iou_thresh=nms_thresh)
    final_boxes = [boxes[i] for i in keep]
    final_labels = [labels[i] for i in keep]
    final_scores = [scores[i] for i in keep]

    # 可视化
    vis = image.copy()
    for (x1, y1, x2, y2), label, score in zip(final_boxes, final_labels, final_scores):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(vis, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 显示 or 保存
    if save_path:
        cv2.imwrite(save_path, vis)
        print(f"Saved result to {save_path}")
    else:
        cv2.imshow("Detection Result", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    st = time.time()
    inference(
        img_path="test/0.jpg",
        model_name="moco",
        bank_path="mem_banks/memory_bank_moco.npy",
        score_thresh=0.5,
        nms_thresh=0.3,
        batch_size=128,
        input_size=(224, 224),
        save_path="output/result.jpg"
    )
    et = time.time()
    print(f"Inference completed in {et - st:.2f} seconds")