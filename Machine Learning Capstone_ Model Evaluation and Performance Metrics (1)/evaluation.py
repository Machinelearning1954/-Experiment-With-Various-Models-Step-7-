"""
Evaluation and metrics module for comprehensive model assessment.
Implements proper performance metrics and cross-validation strategies.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, matthews_corrcoef, cohen_kappa_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
from typing import Dict, Any, List, Tuple, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Comprehensive metrics calculation for classification tasks."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class and averaged metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Store per-class metrics
        metrics['per_class'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist()
        }
        
        # Averaged metrics
        for avg_type in ['macro', 'micro', 'weighted']:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average=avg_type, zero_division=0
            )
            metrics[f'precision_{avg_type}'] = prec
            metrics[f'recall_{avg_type}'] = rec
            metrics[f'f1_score_{avg_type}'] = f1
        
        # Additional metrics
        metrics['matthews_corr_coef'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Class-wise accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        metrics['class_accuracy'] = class_accuracy.tolist()
        
        # ROC AUC and PR AUC (if probabilities available)
        if y_prob is not None:
            try:
                # Multi-class ROC AUC
                if self.num_classes > 2:
                    metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                    metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo')
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                
                # Average Precision Score
                metrics['avg_precision_score'] = average_precision_score(
                    y_true, y_prob, multi_class='ovr'
                )
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        
        return metrics
    
    def calculate_pytorch_metrics(self, model: nn.Module, data_loader, device: torch.device) -> Dict[str, Any]:
        """
        Calculate metrics for PyTorch model.
        
        Args:
            model: PyTorch model
            data_loader: Data loader
            device: Device to run on
            
        Returns:
            Dictionary containing metrics
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Evaluating"):
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                probs = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        
        return self.calculate_metrics(y_true, y_pred, y_prob)

class CrossValidationEvaluator:
    """Cross-validation evaluation framework."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cv_config = config['cross_validation']
        self.n_splits = self.cv_config['n_splits']
        self.shuffle = self.cv_config['shuffle']
        self.random_state = self.cv_config['random_state']
    
    def evaluate_traditional_ml(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform cross-validation for traditional ML models.
        
        Args:
            model: Scikit-learn compatible model
            X: Features
            y: Labels
            
        Returns:
            Cross-validation results
        """
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro',
            'f1_weighted': 'f1_weighted'
        }
        
        # Stratified K-Fold
        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring,
            return_train_score=True, return_estimator=True
        )
        
        # Calculate statistics
        results = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results[f'{metric}_test_mean'] = np.mean(test_scores)
            results[f'{metric}_test_std'] = np.std(test_scores)
            results[f'{metric}_train_mean'] = np.mean(train_scores)
            results[f'{metric}_train_std'] = np.std(train_scores)
            results[f'{metric}_test_scores'] = test_scores.tolist()
            results[f'{metric}_train_scores'] = train_scores.tolist()
        
        # Calculate overfitting indicators
        for metric in scoring.keys():
            train_mean = results[f'{metric}_train_mean']
            test_mean = results[f'{metric}_test_mean']
            results[f'{metric}_overfitting_gap'] = train_mean - test_mean
        
        return results
    
    def evaluate_pytorch_model(self, model_class, model_params: Dict[str, Any],
                              cv_splits: List[Tuple], trainer) -> Dict[str, Any]:
        """
        Perform cross-validation for PyTorch models.
        
        Args:
            model_class: Model class
            model_params: Model parameters
            cv_splits: List of (train_loader, val_loader) tuples
            trainer: PyTorchTrainer instance
            
        Returns:
            Cross-validation results
        """
        fold_results = []
        
        for fold_idx, (train_loader, val_loader) in enumerate(cv_splits):
            logger.info(f"Training fold {fold_idx + 1}/{len(cv_splits)}")
            
            # Create fresh model for each fold
            model = model_class(**model_params)
            
            # Train model
            results = trainer.train_model(
                model, train_loader, val_loader,
                experiment_name=f"cv_fold_{fold_idx}"
            )
            
            fold_results.append(results)
        
        # Aggregate results
        metrics = ['best_val_accuracy', 'training_time', 'num_parameters', 'model_size_mb']
        aggregated_results = {}
        
        for metric in metrics:
            values = [result[metric] for result in fold_results]
            aggregated_results[f'{metric}_mean'] = np.mean(values)
            aggregated_results[f'{metric}_std'] = np.std(values)
            aggregated_results[f'{metric}_values'] = values
        
        # Calculate overfitting indicators
        train_accs = []
        val_accs = []
        
        for result in fold_results:
            history = result['history']
            train_accs.append(max(history['train_acc']))
            val_accs.append(max(history['val_acc']))
        
        aggregated_results['train_accuracy_mean'] = np.mean(train_accs)
        aggregated_results['val_accuracy_mean'] = np.mean(val_accs)
        aggregated_results['overfitting_gap'] = np.mean(train_accs) - np.mean(val_accs)
        
        return aggregated_results

class ModelComparator:
    """Compare multiple models systematically."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.results = {}
    
    def add_model_results(self, model_name: str, results: Dict[str, Any]):
        """Add results for a model."""
        self.results[model_name] = results
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table for all models."""
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {
                'Model': model_name,
                'Test Accuracy': results.get('test_accuracy', results.get('best_val_accuracy', 0)),
                'Training Time (s)': results.get('training_time', 0),
                'Parameters': results.get('num_parameters', 'N/A'),
                'Model Size (MB)': results.get('model_size_mb', 'N/A'),
                'F1 Score (Macro)': results.get('f1_score_macro', 'N/A'),
                'Precision (Macro)': results.get('precision_macro', 'N/A'),
                'Recall (Macro)': results.get('recall_macro', 'N/A')
            }
            
            # Add cross-validation results if available
            if 'accuracy_test_mean' in results:
                row['CV Accuracy (Mean)'] = results['accuracy_test_mean']
                row['CV Accuracy (Std)'] = results['accuracy_test_std']
                row['Overfitting Gap'] = results.get('accuracy_overfitting_gap', 'N/A')
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def rank_models(self, primary_metric: str = 'Test Accuracy') -> pd.DataFrame:
        """Rank models by primary metric."""
        df = self.generate_comparison_table()
        
        if primary_metric in df.columns:
            df_ranked = df.sort_values(primary_metric, ascending=False).reset_index(drop=True)
            df_ranked['Rank'] = range(1, len(df_ranked) + 1)
            return df_ranked
        else:
            logger.warning(f"Metric {primary_metric} not found in results")
            return df

class VisualizationGenerator:
    """Generate visualizations for model evaluation."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                             model_name: str, save: bool = True) -> str:
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save:
            filepath = os.path.join(self.output_dir, f'{model_name}_confusion_matrix.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        else:
            plt.show()
            return ""
    
    def plot_training_curves(self, history: Dict[str, List], model_name: str, 
                           save: bool = True) -> str:
        """Plot training curves."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        ax3.plot(epochs, history['learning_rates'], 'g-')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Overfitting analysis
        gap = np.array(history['train_acc']) - np.array(history['val_acc'])
        ax4.plot(epochs, gap, 'purple', label='Accuracy Gap')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Overfitting Analysis (Train - Val Accuracy)')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Accuracy Gap (%)')
        ax4.legend()
        ax4.grid(True)
        
        plt.suptitle(f'Training Analysis - {model_name}', fontsize=16)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'{model_name}_training_curves.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        else:
            plt.show()
            return ""
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, save: bool = True) -> str:
        """Plot model comparison chart."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Test Accuracy', 'Training Time', 'Model Size', 'F1 Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        models = comparison_df['Model'].tolist()
        
        # Test Accuracy
        fig.add_trace(
            go.Bar(x=models, y=comparison_df['Test Accuracy'], name='Test Accuracy'),
            row=1, col=1
        )
        
        # Training Time
        fig.add_trace(
            go.Bar(x=models, y=comparison_df['Training Time (s)'], name='Training Time'),
            row=1, col=2
        )
        
        # Model Size (handle 'N/A' values)
        model_sizes = []
        for size in comparison_df['Model Size (MB)']:
            if size == 'N/A':
                model_sizes.append(0)
            else:
                model_sizes.append(float(size))
        
        fig.add_trace(
            go.Bar(x=models, y=model_sizes, name='Model Size'),
            row=2, col=1
        )
        
        # F1 Score
        f1_scores = []
        for score in comparison_df['F1 Score (Macro)']:
            if score == 'N/A':
                f1_scores.append(0)
            else:
                f1_scores.append(float(score))
        
        fig.add_trace(
            go.Bar(x=models, y=f1_scores, name='F1 Score'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Model Comparison Dashboard",
            showlegend=False,
            height=800
        )
        
        if save:
            filepath = os.path.join(self.output_dir, 'model_comparison.html')
            fig.write_html(filepath)
            return filepath
        else:
            fig.show()
            return ""

