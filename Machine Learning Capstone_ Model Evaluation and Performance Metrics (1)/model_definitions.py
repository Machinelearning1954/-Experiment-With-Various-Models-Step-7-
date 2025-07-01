"""
Model definitions for various architectures.
Includes both traditional ML and deep learning models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    """Simple CNN architecture for CIFAR-10."""
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 256 * 2 * 2)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class ImprovedCNN(nn.Module):
    """Improved CNN with residual connections."""
    
    def __init__(self, num_classes: int = 10):
        super(ImprovedCNN, self).__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.res_block1 = self._make_res_block(64, 64)
        self.res_block2 = self._make_res_block(64, 128, stride=2)
        self.res_block3 = self._make_res_block(128, 256, stride=2)
        self.res_block4 = self._make_res_block(256, 512, stride=2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def _make_res_block(self, in_channels, out_channels, stride=1):
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks with skip connections
        identity = x
        x = self.res_block1(x)
        x = F.relu(x + identity)
        
        x = self.res_block2(x)
        x = F.relu(x)
        
        x = self.res_block3(x)
        x = F.relu(x)
        
        x = self.res_block4(x)
        x = F.relu(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class ModelFactory:
    """Factory class for creating different types of models."""
    
    @staticmethod
    def create_pytorch_model(model_name: str, num_classes: int = 10, pretrained: bool = False) -> nn.Module:
        """
        Create a PyTorch model.
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            
        Returns:
            PyTorch model
        """
        if model_name.lower() == 'simplecnn':
            return SimpleCNN(num_classes)
        
        elif model_name.lower() == 'improvedcnn':
            return ImprovedCNN(num_classes)
        
        elif model_name.lower() == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        
        elif model_name.lower() == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        
        elif model_name.lower() == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        
        elif model_name.lower() == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            return model
        
        elif model_name.lower() == 'vgg19':
            model = models.vgg19(pretrained=pretrained)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            return model
        
        elif model_name.lower() == 'densenet121':
            model = models.densenet121(pretrained=pretrained)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            return model
        
        elif model_name.lower() == 'densenet169':
            model = models.densenet169(pretrained=pretrained)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            return model
        
        elif model_name.lower() == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            return model
        
        elif model_name.lower() == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            return model
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    @staticmethod
    def create_sklearn_model(model_name: str, **params) -> Any:
        """
        Create a scikit-learn model.
        
        Args:
            model_name: Name of the model
            **params: Model parameters
            
        Returns:
            Scikit-learn model
        """
        if model_name.lower() == 'randomforest':
            return RandomForestClassifier(**params)
        
        elif model_name.lower() == 'svm':
            return SVC(**params)
        
        elif model_name.lower() == 'logisticregression':
            return LogisticRegression(**params)
        
        else:
            raise ValueError(f"Unknown sklearn model: {model_name}")
    
    @staticmethod
    def create_xgboost_model(**params) -> xgb.XGBClassifier:
        """Create XGBoost model."""
        return xgb.XGBClassifier(**params)
    
    @staticmethod
    def create_lightgbm_model(**params) -> lgb.LGBMClassifier:
        """Create LightGBM model."""
        return lgb.LGBMClassifier(**params)
    
    @staticmethod
    def create_catboost_model(**params) -> cb.CatBoostClassifier:
        """Create CatBoost model."""
        return cb.CatBoostClassifier(**params)

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb

def initialize_weights(model: nn.Module):
    """Initialize model weights using Xavier initialization."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

