import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

from proposalCLIP import init_CLIP, main  # 请修改为你代码文件的实际名称和函数

# VOC 类别
VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# VOC 标签读取函数
def load_voc_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        x1 = int(float(bbox.find("xmin").text))
        y1 = int(float(bbox.find("ymin").text))
        x2 = int(float(bbox.find("xmax").text))
        y2 = int(float(bbox.find("ymax").text))
        boxes.append([x1, y1, x2, y2])
    return boxes

# IoU计算函数
def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# xywh 转 xyxy
def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

# VOC验证集上测试 proposal recall
def evaluate_proposals_on_voc(voc_root, proposal_fn, device="cuda", top_k=300, iou_thresh=0.5):
    val_list = os.path.join(voc_root, "ImageSets/Main/val.txt")
    with open(val_list, 'r') as f:
        img_ids = [line.strip() for line in f.readlines()]

    total_gt = 0
    total_recalled = 0

    for img_id in tqdm(img_ids, desc=f"Evaluating Proposals@{top_k}"):
        img_path = os.path.join(voc_root, "JPEGImages", img_id + ".jpg")
        ann_path = os.path.join(voc_root, "Annotations", img_id + ".xml")

        img = cv2.imread(img_path)
        if img is None:
            continue
        gt_boxes = load_voc_annotation(ann_path)

        try:
            proposals, _ = proposal_fn(img)
        except Exception as e:
            print(f"Proposal error: {img_id}", e)
            continue

        proposals = [xywh_to_xyxy(p) for p in proposals[:top_k]]

        # 遍历 GT 框匹配
        matched = [False] * len(gt_boxes)
        for i, gt in enumerate(gt_boxes):
            for prop in proposals:
                iou = compute_iou(gt, prop)
                if iou >= iou_thresh:
                    matched[i] = True
                    break

        total_gt += len(gt_boxes)
        total_recalled += sum(matched)

    recall = total_recalled / total_gt
    print(f"\n[Recall@{top_k} | IoU>{iou_thresh}]  => {recall:.4f}")


if __name__ == "__main__":
    # VOC路径
    voc_root = "D:/PF/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/"  # 修改为你的VOC路径
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

    # 初始化模型
    clip_model, preprocess, tokens = init_CLIP("ViT-B/32", VOC_CLASSES, device=device)

    # 包装 proposalclip 接口
    def proposalclip_wrapper(img):
        return main(img, clip_model, preprocess, tokens, device)

    # 测试不同K值
    for k in [10, 30, 50, 100]:
        evaluate_proposals_on_voc(voc_root, proposalclip_wrapper, device=device, top_k=k, iou_thresh=0.5)
