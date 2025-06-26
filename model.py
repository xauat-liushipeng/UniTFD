import torch
import torch.nn as nn
import torchvision.models as models


def get_backbone(name='resnet50'):
    model = getattr(models, name)(weights=True)
    model.fc = nn.Identity()  # 去掉分类层
    return model


class FeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()
        self.model_name = model_name.lower()
        if self.model_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(*list(backbone.children())[:-1])  # 去掉fc层
            self.out_dim = 2048
        elif self.model_name == 'mobilenetv2':
            backbone = models.mobilenet_v2(pretrained=pretrained)
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.out_dim = 1280
        elif self.model_name == 'vit':
            from torchvision.models import vit_b_16
            backbone = vit_b_16(pretrained=pretrained)
            self.features = nn.Sequential(*list(backbone.children())[:-1])  # 去掉分类头
            self.out_dim = backbone.heads.head.in_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def forward(self, x):
        if self.model_name == 'mobilenetv2':
            x = self.features(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
        else:
            x = self.features(x)
            x = x.view(x.size(0), -1)
        return x

if __name__ == '__main__':
    # 测试
    model = FeatureExtractor('resnet50')
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(out.shape)  # 应该是 (2, 2048)
