import cv2
import numpy as np
from skimage import segmentation

def sliding_window(image, step_size, window_size):
    """
    滑动窗口生成候选框
    """
    proposals = []
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            proposals.append([x, y, x + window_size[0], y + window_size[1]])
    return proposals

def selective_search_proposals(image):
    """
    使用OpenCV Selective Search生成候选区域
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()  # 或 switchToSelectiveSearchQuality()
    rects = ss.process()
    proposals = []
    for (x, y, w, h) in rects:
        proposals.append([x, y, x + w, y + h])
    return proposals

if __name__ == '__main__':
    import sys
    img_path = sys.argv[1]
    image = cv2.imread(img_path)
    # 这里用滑窗示例，步长50，窗口大小100x100
    proposals = sliding_window(image, step_size=50, window_size=(100, 100))
    print(f"Generated {len(proposals)} proposals.")
