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
            self.classifier = nn.Sequential(
                nn.Linear(128 * 28 * 28, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            # sklearn 분류기 초기화
            self.sklearn_classifier = self._create_classifier(classifier_type)
            self.is_fitted = False
            # 특징 추출기의 파라미터 고정
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
            return self.classifier(features)
        else:
            # sklearn 분류기를 위해 numpy 배열로 변환
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