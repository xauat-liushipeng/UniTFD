import time

import cv2
import torch
import numpy as np
from PIL import Image
from model import get_backbone
from memory_bank import MemoryBank
from gen_proposals import smart_selective_search
from torchvision import transforms
from post_proc import nms_boxes

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference_single_image(img_path,
                           model_name='resnet50',
                           bank_path='memory_bank.npy',
                           score_thresh=0.3,
                           nms_thresh=0.3,
                           batch_size=32,
                           box_resize=(224, 224),
                           save_path=None):
    """
    对单张图像进行缺陷检测推理，并可视化标注结果
    """
    # 加载图像
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Cannot read image: {img_path}")
    h_img, w_img = image.shape[:2]

    # 加载模型
    model = get_backbone(model_name).to(device)
    model.eval()

    # 加载 memory bank
    bank = MemoryBank(bank_path)

    # transforms
    preprocess = transforms.Compose([
        transforms.Resize(box_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # proposals
    proposals = smart_selective_search(image)
    crops = []
    coords = []
    for (x1, y1, x2, y2) in proposals:
        patch = image[y1:y2, x1:x2]
        if patch.shape[0] < 10 or patch.shape[1] < 10:
            continue
        pil_img = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(pil_img)
        crops.append(input_tensor)
        coords.append([x1, y1, x2, y2])

    if not crops:
        print("No valid proposals.")
        return

    # 特征提取
    features = []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch = torch.stack(crops[i:i+batch_size]).to(device)
            feat = model(batch)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            features.append(feat.cpu())
    features = torch.cat(features, dim=0).numpy()

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
    inference_single_image(
        img_path="test/0.jpg",
        model_name="resnet50",
        bank_path="memory_bank2.npy",
        score_thresh=0.6,
        nms_thresh=0.3,
        batch_size=128,
        box_resize=(224, 224),
        save_path="output/result.jpg"
    )
    et = time.time()
    print(f"Inference completed in {et - st:.2f} seconds")