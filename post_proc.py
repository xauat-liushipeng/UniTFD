import numpy as np
from torchvision.ops import nms
import torch

def nms_boxes(boxes, scores, iou_thresh=0.3):
    """
    boxes: ndarray Nx4, scores: ndarray N
    return: indices of kept boxes
    """
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    keep = nms(boxes_tensor, scores_tensor, iou_thresh)
    return keep.cpu().numpy()


def nms_(boxes, scores, iou_threshold=0.5):
    """
    简单NMS实现，boxes格式: [N,4], scores: [N]
    """
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def select_topk_nms(boxes, scores, topk=10, iou_thresh=0.5):
    scores = np.array(scores)
    boxes = np.array(boxes)
    if len(scores) == 0:
        return [], []

    topk_idx = scores.argsort()[::-1][:topk]
    boxes_topk = boxes[topk_idx]
    scores_topk = scores[topk_idx]

    keep = nms(boxes_topk, scores_topk, iou_thresh)
    boxes_final = boxes_topk[keep]
    scores_final = scores_topk[keep]
    return boxes_final, scores_final

if __name__ == '__main__':
    boxes = np.array([[10,10,50,50],[12,12,48,48],[100,100,150,150]])
    scores = [0.9, 0.8, 0.75]
    b,s = select_topk_nms(boxes, scores, topk=2)
    print(b, s)
