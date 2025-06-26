import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from model import get_backbone
from memory_bank import MemoryBank
from gen_proposals import smart_selective_search
from post_proc import nms_boxes
from PIL import Image
from tqdm import tqdm

# --------- 参数 ---------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'resnet50'
test_dir = './test_images'
bank_path = 'memory_bank.npy'
batch_size = 32
score_thresh = 0.3
nms_thresh = 0.3
box_size = (100, 100)  # 用于新proposal生成的锚点尺度（如有需要）

# --------- 模型与预处理 ---------
model = get_backbone(model_name).to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------- 载入 Memory Bank ---------
bank = MemoryBank(bank_path)

# --------- 主推理流程 ---------
def run_inference():
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png'))]

    for fname in tqdm(image_files):
        img_path = os.path.join(test_dir, fname)
        image = cv2.imread(img_path)
        h_img, w_img = image.shape[:2]

        # Step 1: 生成 proposals
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
            print(f"No valid crops in {fname}")
            continue

        # Step 2: 批量提取特征
        features = []
        with torch.no_grad():
            for i in range(0, len(crops), batch_size):
                batch = torch.stack(crops[i:i+batch_size]).to(device)
                feat = model(batch)
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                features.append(feat.cpu())
        features = torch.cat(features, dim=0).numpy()

        # Step 3: 与 memory bank 匹配
        results = []
        for i, feat in enumerate(features):
            top_match = bank.match(feat, topk=1)
            label, score = top_match[0]
            if score >= score_thresh:
                results.append((coords[i], label, score))

        # Step 4: NMS 后处理
        if results:
            boxes = np.array([r[0] for r in results])
            scores = np.array([r[2] for r in results])
            labels = [r[1] for r in results]

            keep_idx = nms_boxes(boxes, scores, iou_thresh=nms_thresh)
            final_boxes = [boxes[i] for i in keep_idx]
            final_labels = [labels[i] for i in keep_idx]
            final_scores = [scores[i] for i in keep_idx]

            # 可视化或保存
            vis = image.copy()
            for (x1, y1, x2, y2), label, score in zip(final_boxes, final_labels, final_scores):
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f'{label} {score:.2f}', (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(f'results/{fname}', vis)

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    run_inference()
