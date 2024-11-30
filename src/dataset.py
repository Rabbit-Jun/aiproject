import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import lightning as L

SEED = 36
L.seed_everything(SEED)

class CustomDataModule(L.LightningDataModule):
    def __init__(self, dataset_type: str = 'cifar10', data_path: str = './dataset', 
                 batch_size: int = 32):
        super().__init__()
        self.dataset_type = dataset_type
        self.data_path = data_path
        self.batch_size = batch_size
        self.batch_size_per_device = batch_size
        
        # 데이터셋별 이미지 크기 및 변환 설정
        if dataset_type == 'cifar10':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif dataset_type == 'flowers':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def setup(self, stage: str):
        if stage == 'fit' or stage == 'test':
            if self.dataset_type == 'cifar10':
                self.train_dataset = datasets.CIFAR10(
                    root=self.data_path,
                    train=True,
                    download=True,
                    transform=self.transform
                )
                self.test_dataset = datasets.CIFAR10(
                    root=self.data_path,
                    train=False,
                    download=True,
                    transform=self.transform
                )
                # 훈련 데이터의 20%를 검증 데이터로 사용
                train_size = int(0.8 * len(self.train_dataset))
                val_size = len(self.train_dataset) - train_size
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    self.train_dataset, [train_size, val_size]
                )
            
            elif self.dataset_type == 'flowers':
                train_dir = os.path.join(self.data_path, 'flowers', 'train')
                test_dir = os.path.join(self.data_path, 'flowers', 'test')
                
                # 훈련 데이터셋 로드
                self.train_dataset = datasets.ImageFolder(
                    root=train_dir,
                    transform=self.transform
                )
                
                # 테스트 데이터셋 로드
                self.test_dataset = datasets.ImageFolder(
                    root=test_dir,
                    transform=self.transform
                )
                
                # 훈련 데이터의 20%를 검증 데이터로 사용
                train_size = int(0.8 * len(self.train_dataset))
                val_size = len(self.train_dataset) - train_size
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    self.train_dataset, [train_size, val_size]
                )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                         batch_size=self.batch_size_per_device, 
                         shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                         batch_size=self.batch_size_per_device, 
                         shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                         batch_size=self.batch_size_per_device, 
                         shuffle=False) 