"""
Training pipeline for automated model training and hyperparameter tuning.
Supports both traditional ML and deep learning models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import os
import json
import pickle
from typing import Dict, Any, List, Tuple, Optional
import logging
from tqdm import tqdm
import optuna
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from models.model_definitions import ModelFactory, count_parameters, get_model_size, initialize_weights

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop

class PyTorchTrainer:
    """Trainer for PyTorch models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mixed_precision = config['training'].get('mixed_precision', False)
        self.gradient_clipping = config['training'].get('gradient_clipping', 1.0)
        
        logger.info(f"Using device: {self.device}")
        
    def train_model(self, model: nn.Module, train_loader, val_loader, 
                   learning_rate: float = 0.001, epochs: int = 50,
                   experiment_name: str = "experiment") -> Dict[str, Any]:
        """
        Train a PyTorch model.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            epochs: Number of epochs
            experiment_name: Name for experiment tracking
            
        Returns:
            Training results dictionary
        """
        model = model.to(self.device)
        
        # Initialize weights
        initialize_weights(model)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        # Mixed precision scaler
        scaler = GradScaler() if self.use_mixed_precision else None
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config['training']['early_stopping']['patience'],
            min_delta=self.config['training']['early_stopping']['min_delta']
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        best_val_acc = 0.0
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                if self.use_mixed_precision and scaler:
                    with autocast():
                        output = model(data)
                        loss = criterion(output, target)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clipping)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clipping)
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                # Update progress bar
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for data, target in val_pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    if self.use_mixed_precision:
                        with autocast():
                            output = model(data)
                            loss = criterion(output, target)
                    else:
                        output = model(data)
                        loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            # Calculate metrics
            train_loss_avg = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss_avg = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss_avg)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss_avg)
            history['val_acc'].append(val_acc)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            # Early stopping check
            if early_stopping(val_acc):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            logger.info(f'Epoch {epoch+1}: Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, '
                       f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
        
        training_time = time.time() - start_time
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Model statistics
        num_params = count_parameters(model)
        model_size_mb = get_model_size(model)
        
        results = {
            'best_val_accuracy': best_val_acc,
            'training_time': training_time,
            'num_parameters': num_params,
            'model_size_mb': model_size_mb,
            'history': history,
            'final_lr': optimizer.param_groups[0]['lr']
        }
        
        return results

class TraditionalMLTrainer:
    """Trainer for traditional ML models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def train_model(self, model_type: str, X_train: np.ndarray, X_test: np.ndarray,
                   y_train: np.ndarray, y_test: np.ndarray, 
                   params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train a traditional ML model.
        
        Args:
            model_type: Type of model ('randomforest', 'xgboost', etc.)
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            params: Model parameters
            
        Returns:
            Training results dictionary
        """
        if params is None:
            params = {}
        
        start_time = time.time()
        
        # Create model
        if model_type.lower() == 'randomforest':
            model = ModelFactory.create_sklearn_model('randomforest', **params)
        elif model_type.lower() == 'xgboost':
            model = ModelFactory.create_xgboost_model(**params)
        elif model_type.lower() == 'lightgbm':
            model = ModelFactory.create_lightgbm_model(**params)
        elif model_type.lower() == 'catboost':
            model = ModelFactory.create_catboost_model(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        logger.info(f"Training {model_type} model...")
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results = {
            'train_accuracy': train_accuracy * 100,
            'test_accuracy': test_accuracy * 100,
            'cv_accuracy_mean': cv_scores.mean() * 100,
            'cv_accuracy_std': cv_scores.std() * 100,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time,
            'model': model
        }
        
        return results

class HyperparameterTuner:
    """Hyperparameter tuning using Optuna."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_trials = config['hyperparameter_tuning']['n_trials']
        self.timeout = config['hyperparameter_tuning']['timeout']
    
    def tune_pytorch_model(self, model_name: str, train_loader, val_loader) -> Dict[str, Any]:
        """
        Tune hyperparameters for PyTorch model.
        
        Args:
            model_name: Name of the model
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Best parameters and results
        """
        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
            
            # Create model
            model = ModelFactory.create_pytorch_model(model_name, num_classes=10)
            
            # Train model
            trainer = PyTorchTrainer(self.config)
            results = trainer.train_model(
                model, train_loader, val_loader,
                learning_rate=lr, epochs=20,  # Reduced epochs for tuning
                experiment_name=f"{model_name}_tuning"
            )
            
            return results['best_val_accuracy']
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
    
    def tune_traditional_ml(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                           param_grid: Dict[str, List]) -> Dict[str, Any]:
        """
        Tune hyperparameters for traditional ML model.
        
        Args:
            model_type: Type of model
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for tuning
            
        Returns:
            Best parameters and results
        """
        def objective(trial):
            # Suggest hyperparameters
            params = {}
            for param_name, param_values in param_grid.items():
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
            
            # Create and train model
            trainer = TraditionalMLTrainer(self.config)
            
            # Use a subset for faster tuning
            subset_size = min(5000, len(X_train))
            indices = np.random.choice(len(X_train), subset_size, replace=False)
            X_subset = X_train[indices]
            y_subset = y_train[indices]
            
            # Cross-validation score
            if model_type.lower() == 'randomforest':
                model = ModelFactory.create_sklearn_model('randomforest', **params)
            elif model_type.lower() == 'xgboost':
                model = ModelFactory.create_xgboost_model(**params)
            elif model_type.lower() == 'lightgbm':
                model = ModelFactory.create_lightgbm_model(**params)
            elif model_type.lower() == 'catboost':
                model = ModelFactory.create_catboost_model(**params)
            
            cv_scores = cross_val_score(model, X_subset, y_subset, cv=3, scoring='accuracy')
            return cv_scores.mean()
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }

