from torchvision import models
import torch.nn as nn
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            # Flatten
            nn.Flatten()
        )
        
    def forward(self, x):
        return self.features(x)

class HybridModel(nn.Module):
    def __init__(self, classifier_type='cnn', num_classes=5):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor()
        self.classifier_type = classifier_type
        self.num_classes = num_classes
        
        if classifier_type == 'cnn':
            # 입력 특징 크기
            input_size = 128 * 28 * 28
            
            # 더 깊은 분류기 네트워크
            self.classifier = nn.Sequential(
                # 첫 번째 블록
                nn.Linear(input_size, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                # 두 번째 블록
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                # 세 번째 블록
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                # 네 번째 블록
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                # 다섯 번째 블록
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                # 출력 레이어
                nn.Linear(128, num_classes)
            )
            
            # Skip Connection을 위한 추가 레이어들
            self.skip1 = nn.Linear(input_size, 1024)
            self.skip2 = nn.Linear(1024, 512)
            self.skip3 = nn.Linear(512, 256)
            self.skip4 = nn.Linear(256, 128)
            
        else:
            self.sklearn_classifier = self._create_classifier(classifier_type)
            self.is_fitted = False
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def _create_classifier(self, classifier_type):
        if classifier_type == 'knn':
            return KNeighborsClassifier(n_neighbors=5)
        elif classifier_type == 'svm':
            return SVC(kernel='rbf')
        elif classifier_type == 'dt':
            return DecisionTreeClassifier()
        elif classifier_type == 'mlp':
            return MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=1000)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        if self.classifier_type == 'cnn':
            # Skip Connection을 활용한 순전파
            x = features
            
            # Skip Connection 1
            identity1 = self.skip1(x)
            x = self.classifier[0:4](x)  # 첫 번째 블록
            x = x + identity1
            
            # Skip Connection 2
            identity2 = self.skip2(x)
            x = self.classifier[4:8](x)  # 두 번째 블록
            x = x + identity2
            
            # Skip Connection 3
            identity3 = self.skip3(x)
            x = self.classifier[8:12](x)  # 세 번째 블록
            x = x + identity3
            
            # Skip Connection 4
            identity4 = self.skip4(x)
            x = self.classifier[12:16](x)  # 네 번째 블록
            x = x + identity4
            
            # 나머지 레이어 통과
            x = self.classifier[16:](x)
            
            return x
        else:
            # sklearn 분류기 로직
            features = features.detach().cpu().numpy()
            
            if self.training:
                if not self.is_fitted:
                    self.sklearn_classifier.fit(features, self.current_targets)
                    self.is_fitted = True
                predictions = self.sklearn_classifier.predict(features)
            else:
                if self.is_fitted:
                    predictions = self.sklearn_classifier.predict(features)
                else:
                    predictions = np.zeros(features.shape[0])
            
            return torch.tensor(predictions, dtype=torch.long, device=x.device)
    
    def set_targets(self, targets):
        """sklearn 분류기 학습을 위한 타겟 설정"""
        self.current_targets = targets.detach().cpu().numpy()

def create_model(model: str = 'cnn', dataset: str = 'flowers'):
    if dataset == 'flowers':
        num_classes = 5
    else:  # cifar10
        num_classes = 10
        
    if model in ['cnn', 'knn', 'svm', 'dt', 'mlp']:
        return HybridModel(classifier_type=model, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model}")