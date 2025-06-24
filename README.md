
# UniTFD: Unified Training-Free One-Shot Tunnel Defect Detection

本项目实现了一个**统一的、无需训练的 One-shot 隧道缺陷检测框架**，通过示例缺陷图像与查询图像的视觉特征匹配，完成缺陷的定位和检测。适用于缺陷样本稀缺、标注成本高的隧道检测场景。

---

## 项目结构

- `gen_proposals.py`  
  生成查询图像的候选区域（proposals），支持滑窗和可扩展到选择性搜索等方法。

- `load_bank.py`  
  加载预提取的示例缺陷特征库（Memory Bank），支持.npy格式文件读取。

- `model.py`  
  多种可选视觉特征提取模型封装（ResNet50、MobileNetV2、ViT等），用于提取图像区域特征。

- `post_proc.py`  
  对匹配分数进行后处理，支持Top-K筛选和非极大值抑制（NMS），输出最终缺陷边界框。

- `visual_matching.py`  
  实现示例与候选区域特征的匹配计算（余弦相似度和L2距离）。

- `main.py`  
  主程序示例，整合各模块，实现给定示例缺陷图与查询图的端到端缺陷检测流程。

---

## 环境依赖

- Python >=3.7  
- PyTorch >=1.7  
- torchvision  
- OpenCV (`opencv-python`, `opencv-contrib-python`)  
- numpy  
- scikit-image（可选，用于高级候选框生成）

安装示例：

```bash
pip install torch torchvision opencv-python opencv-contrib-python numpy scikit-image
````

---

## 快速运行示例

```bash
python main.py <exemplar_image> <query_image> [model_name] [topk]
```

示例：

```bash
python main.py data/exemplar.jpg data/query.jpg resnet50 5
```

* `<exemplar_image>`: 单张缺陷示例图（已裁剪缺陷区域）
* `<query_image>`: 待检测隧道图像
* `[model_name]`: 可选，特征提取模型，默认 `resnet50`，支持 `mobilenetv2`、`vit`
* `[topk]`: 可选，输出匹配最高的候选框数量，默认5

运行后将弹出带检测框的查询图窗口。

---

## 代码示例简要说明

* **特征提取**：使用预训练模型提取示例和候选区域的深度语义特征。
* **视觉匹配**：计算示例特征与所有候选区域特征的余弦相似度，作为匹配分数。
* **候选筛选**：选取得分最高的Top-K候选框，利用NMS去除冗余框。
* **输出结果**：打印和可视化最终检测的缺陷边界框。

---

## 扩展建议

* 替换更先进的候选区域生成方法（如RPN、Selective Search）。
* 支持多示例输入，构建更丰富的特征库。
* 集成更复杂的后处理策略，提升定位准确度。
* 增加批处理支持，提升推理效率。

---

## 联系方式

如有问题或建议，请联系：\[你的邮箱]

---
