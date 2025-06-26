import cv2
import torch.nn as nn
import torchvision.models as models

import torch
from PIL import Image
from torchvision.models import swin_t, vit_b_16
from torchvision.ops import roi_align

from data import get_transforms


def extract_features(image, proposals, model, device, output_size=(7, 7)):
    """
    image: 原图 (BGR, np.ndarray)
    proposals: [[x1, y1, x2, y2], ...]
    model: FeatureExtractor 实例
    return: NxC L2-normalized 特征，及对应坐标
    """
    h, w = image.shape[:2]
    if not proposals:
        print("No valid proposals.")
        return None, []

    # Step 1: 图像预处理
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    transforms = get_transforms((224, 224))
    img_tensor = transforms(pil_img).unsqueeze(0).to(device)

    # Step 2: 特征提取
    with torch.no_grad():
        if model.return_map:
            features = model(img_tensor)  # [1, C, H, W]
            feat_h, feat_w = features.shape[-2:]
            spatial_scale = feat_w / w
        else:
            # 模型不支持 ROIAlign（如CLIP或ViT），则不处理 proposals，直接整图特征
            vec = model(img_tensor)  # [1, C]
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            return vec.cpu().numpy(), [[0, 0, w, h]]  # 直接返回整图特征

    # Step 3: ROIAlign
    rois = []
    for (x1, y1, x2, y2) in proposals:
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        rois.append([0, x1, y1, x2, y2])  # batch_idx 为 0

    if not rois:
        print("No valid ROIs after filtering.")
        return None, []

    rois = torch.as_tensor(rois, dtype=torch.float32).to(device)
    pooled = roi_align(features, rois, output_size=output_size, spatial_scale=spatial_scale)

    pooled = torch.nn.functional.adaptive_avg_pool2d(pooled, 1).squeeze(-1).squeeze(-1)  # [N, C]
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

    return pooled.cpu().numpy(), rois[:, 1:].cpu().numpy().astype(int).tolist()



def get_backbone(name='resnet50'):
	model = getattr(models, name)(weights=True)
	model.fc = nn.Identity()  # 去掉分类层
	return model



class FeatureExtractor(nn.Module):
	def __init__(self, model_name='resnet50', pretrained=True, return_map=False):
		"""
		:param model_name: 模型名称 ['resnet50', 'mobilenetv2', 'vit', 'swin', 'clip', 'moco']
		:param return_map: 是否返回特征图，用于 ROIAlign
		"""
		super().__init__()
		self.model_name = model_name.lower()
		self.return_map = return_map

		if self.model_name == 'resnet50':
			backbone = models.resnet50(pretrained=pretrained)
			self.features = nn.Sequential(*list(backbone.children())[:-2])  # [B,2048,H,W]
			self.pool = nn.AdaptiveAvgPool2d(1)
			self.out_dim = 2048

		elif self.model_name == 'mobilenetv2':
			backbone = models.mobilenet_v2(pretrained=pretrained)
			self.features = backbone.features
			self.pool = nn.AdaptiveAvgPool2d(1)
			self.out_dim = 1280

		elif self.model_name == 'vit':
			model = vit_b_16(pretrained=pretrained)
			self.features = model._modules['encoder']
			self.patch_embed = model.conv_proj
			self.class_token = model.cls_token
			self.pos_embed = model.encoder.pos_embedding
			self.out_dim = model.heads.head.in_features

		elif self.model_name == 'swin':
			model = swin_t(pretrained=pretrained)
			self.backbone = nn.Sequential(*list(model.children())[:-1])  # 去掉分类头
			self.pool = nn.AdaptiveAvgPool2d(1)
			self.out_dim = model.head.in_features

		elif self.model_name == 'clip':
			import clip
			model, _ = clip.load("ViT-B/32", device='cpu')  # 或 "RN50", "ViT-L/14"
			self.features = model.visual
			self.out_dim = model.visual.output_dim
			self.is_clip = True
			self.return_map = False  # clip 不支持 return_map

		elif self.model_name == 'moco':
			# 加载MoCo v2 ResNet50 encoder_q
			state = torch.hub.load_state_dict_from_url(
				'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar',
				map_location='cpu'
			)
			backbone = models.resnet50()
			state_dict = {k.replace('module.encoder_q.', ''): v for k, v in state['state_dict'].items() if 'encoder_q' in k}
			backbone.load_state_dict(state_dict, strict=False)
			self.features = nn.Sequential(*list(backbone.children())[:-2])
			self.pool = nn.AdaptiveAvgPool2d(1)
			self.out_dim = 2048

		else:
			raise ValueError(f"Unsupported model: {model_name}")

	def forward(self, x):
		if self.model_name == 'clip':
			x = self.features(x.type(self.features.conv1.weight.dtype))
			return x

		elif self.model_name == 'vit':
			# 模拟ViT输入流程
			x = self.patch_embed(x)  # [B,768,H/16,W/16]
			B, C, H, W = x.shape
			x = x.flatten(2).transpose(1, 2)  # [B,N,C]
			cls_token = self.class_token.expand(B, -1, -1)
			x = torch.cat((cls_token, x), dim=1)
			x = x + self.pos_embed
			x = self.features(x)
			x = x[:, 0]  # cls token
			return x

		elif self.model_name == 'swin':
			x = self.backbone(x)
			if self.return_map:
				return x
			x = self.pool(x).flatten(1)
			return x

		else:
			x = self.features(x)  # [B, C, H, W]
			if self.return_map:
				return x
			x = self.pool(x).view(x.size(0), -1)
			return x


if __name__ == '__main__':
	# 测试
	model = FeatureExtractor('resnet50')
	dummy = torch.randn(2, 3, 224, 224)
	out = model(dummy)
	print(out.shape)  # 应该是 (2, 2048)
