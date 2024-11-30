import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os

# 이미지 크기 설정
IMAGE_SIZE = (128, 128)

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# 훈련 데이터 로드
train_dataset = ImageFolder(
    root='./dataset/flowers/train',
    transform=transform
)
image, label = train_dataset[0]
print(train_dataset)
print(f"이미지 크기: {image.shape}")

# 클래스 이름 가져오기
class_names = train_dataset.classes
print("클래스 목록:", class_names)
print("클래스 개수:", len(class_names))


# 훈련/검증 데이터 분할 (80:20)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# 데이터 크기 출력
print("훈련 데이터 크기:", len(train_dataset))
print("검증 데이터 크기:", len(val_dataset))

# 이미지 시각화
plt.figure(figsize=(10, 2))
for i in range(5):
    img, label = train_dataset[i]
    plt.subplot(1, 5, i + 1)
    # PyTorch tensor를 이미지로 변환 (CHW -> HWC)
    plt.imshow(img.permute(1, 2, 0))
    plt.title(class_names[label])
    plt.axis('off')
plt.show()

# 데이터로더 생성 (배치 처리를 위해)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


