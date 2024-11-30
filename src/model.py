from torchvision import models
import torch.nn as nn

def create_model(model: str = 'resnet', dataset: str = 'cifar10'):
    if model == 'resnet':
        return _resnet18_pretrained()
    elif model == 'efficientnet':
        return _efficientb0_pretrained()
    elif model == 'mlp':
        return _create_mlp(dataset)

def _create_mlp(dataset):
    # 데이터셋에 따른 입력/출력 크기 설정
    if dataset == 'cifar10':
        input_size = 3 * 32 * 32  # 3072
        output_size = 10
    elif dataset == 'flowers':
        input_size = 3 * 224 * 224  # 150528 (새로운 이미지 크기에 맞춤)
        output_size = 5
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, output_size)
    )

def _resnet18_pretrained():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # 마지막 레이어 수정
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)  # CIFAR10용으로 수정
    return model

def _efficientb0_pretrained():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # 마지막 레이어 수정
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, 10)  # CIFAR10용으로 수정
    return model