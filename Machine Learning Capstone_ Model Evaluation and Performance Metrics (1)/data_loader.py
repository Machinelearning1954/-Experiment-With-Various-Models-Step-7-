"""
Data loading and preprocessing module for CIFAR-10 dataset.
Handles data loading, preprocessing, and cross-validation splits.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
from typing import Tuple, List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CIFAR10DataLoader:
    """
    CIFAR-10 data loader with preprocessing and cross-validation support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = config['data']['data_dir']
        self.batch_size = config['data']['batch_size']
        self.num_workers = config['data']['num_workers']
        self.validation_split = config['data']['validation_split']
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load datasets
        self._load_datasets()
        
    def _load_datasets(self):
        """Load CIFAR-10 datasets."""
        logger.info("Loading CIFAR-10 datasets...")
        
        # Load training dataset
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # Load test dataset
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        # Get class names
        self.class_names = self.train_dataset.classes
        logger.info(f"Dataset loaded. Classes: {self.class_names}")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Test samples: {len(self.test_dataset)}")
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train, validation, and test data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Split training data into train and validation
        total_size = len(self.train_dataset)
        val_size = int(total_size * self.validation_split)
        train_size = total_size - val_size
        
        indices = list(range(total_size))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        
        # Create subsets
        train_subset = Subset(self.train_dataset, train_indices)
        val_subset = Subset(self.train_dataset, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_cross_validation_splits(self, n_splits: int = 5) -> List[Tuple[DataLoader, DataLoader]]:
        """
        Get cross-validation splits.
        
        Args:
            n_splits: Number of CV folds
            
        Returns:
            List of (train_loader, val_loader) tuples for each fold
        """
        # Get labels for stratification
        labels = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_splits = []
        for train_idx, val_idx in skf.split(range(len(self.train_dataset)), labels):
            # Create subsets
            train_subset = Subset(self.train_dataset, train_idx)
            val_subset = Subset(self.train_dataset, val_idx)
            
            # Create data loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            cv_splits.append((train_loader, val_loader))
        
        logger.info(f"Created {n_splits} cross-validation splits")
        return cv_splits
    
    def get_data_for_traditional_ml(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get flattened data for traditional ML models.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Convert datasets to numpy arrays
        X_train = []
        y_train = []
        
        # Use test transform for consistency
        train_dataset_no_aug = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=False,
            transform=self.test_transform
        )
        
        for i in range(len(train_dataset_no_aug)):
            image, label = train_dataset_no_aug[i]
            X_train.append(image.numpy().flatten())
            y_train.append(label)
        
        X_test = []
        y_test = []
        
        for i in range(len(self.test_dataset)):
            image, label = self.test_dataset[i]
            X_test.append(image.numpy().flatten())
            y_test.append(label)
        
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        logger.info(f"Traditional ML data shapes: X_train={X_train.shape}, X_test={X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'input_shape': self.config['data']['input_shape'],
            'train_size': len(self.train_dataset),
            'test_size': len(self.test_dataset)
        }

