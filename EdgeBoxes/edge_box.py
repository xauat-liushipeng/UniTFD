import cv2
import numpy as np

def edgebox(model_path: str, im, max_boxes: int = 100, show=False):
    """
    使用 OpenCV 的 ximgproc 接口调用 Edge Boxes，生成边缘候选框。
    :param model_path: 结构化边缘检测模型（.yml.gz）
    :param image_path: 输入图像路径
    :param max_boxes: 最多返回多少个候选框
    """

    '''
    1.初始提案
    
    输入一张图像$I$，用EdgeBoxes提取初始提案$\{O_m\}_{m=1}^M$，$O_m \in R^4$表示坐标，对应初始得分为$SL_m$
    '''

    # 创建结构化边缘检测器
    edge_detector = cv2.ximgproc.createStructuredEdgeDetection(model_path)

    # 计算边缘图与方向图
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    edges = edge_detector.detectEdges(np.float32(rgb) / 255.0)
    orient = edge_detector.computeOrientation(edges)
    edges = edge_detector.edgesNms(edges, orient)

    # 创建 Edge Boxes 生成器并设置参数
    eb = cv2.ximgproc.createEdgeBoxes()
    eb.setMaxBoxes(max_boxes)

    # 获取候选框列表 (x, y, w, h)
    boxes, scores = eb.getBoundingBoxes(edges, orient)

    if show:
        # 绘制候选框
        out = im.copy()
        for (x, y, w, h), s in zip(boxes, scores):
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # 显示图像
        cv2.imshow("EdgeBoxes Proposals", out)

    return boxes, scores

if __name__ == '__main__':
    img = cv2.imread("../test/0.jpg")
    edgebox("./model/model.yml.gz", img, 300)
