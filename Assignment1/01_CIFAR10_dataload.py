import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 클래스 이름 정의
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# CIFAR10 데이터셋 로드
transform = transforms.ToTensor()
trainset = torchvision.datasets.CIFAR10(root='./dataset/cifar-10', train=True,
                                      download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./dataset/cifar-10', train=False,
                                     download=True, transform=transform)

# 데이터 형태 출력
print("Training Data:", len(trainset))
print("Test Data:", len(testset))

# 첫 번째 이미지와 라벨 가져오기
image, label = trainset[0]

# 이미지 크기 확인
print(f"이미지 크기: {image.shape}")  
# 출력: torch.Size([3, 32, 32]) -> (채널, 높이, 너비)

# 클래스 개수 확인
print(f"클래스 개수: {len(trainset.classes)}")  
# 출력: 10

# 클래스 이름 확인
print(f"클래스 목록: {trainset.classes}")  
# 출력: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 전체 데이터셋 크기
print(f"전체 훈련 데이터 개수: {len(trainset)}")  
# 출력: 50000

# 클래스별 개수 계산
class_counts = np.zeros(len(class_names), dtype=int)
for _, label in trainset:
    class_counts[label] += 1

# 클래스별 개수 출력
print("\n클래스별 데이터 개수:")
for i, (name, count) in enumerate(zip(class_names, class_counts)):
    print(f"{name}: {count}")

# 클래스별 분포 시각화
plt.figure(figsize=(12, 6))
plt.bar(class_names, class_counts)
plt.title('Class Distribution in Training Set')
plt.xticks(rotation=45)
plt.ylabel('Number of Images')
plt.tight_layout()
plt.show()

# 이미지 시각화
plt.figure(figsize=(10, 2))
for i in range(5):
    img, label = trainset[i]
    plt.subplot(1, 5, i + 1)
    # PyTorch tensor를 numpy 배열로 변환 (CHW -> HWC)
    plt.imshow(img.permute(1, 2, 0))
    plt.title(class_names[label])
    plt.axis('off')
plt.show()
