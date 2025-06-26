from torchvision.transforms import transforms


def get_transforms(input_size=(224, 224)):
	"""
	获取图像预处理转换
	"""
	return transforms.Compose([
		transforms.Resize(input_size),  # 可调
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
		                     std=[0.229, 0.224, 0.225])
	])
