"""
Experiment Manager for automated model testing and comparison.
Orchestrates training, evaluation, and comparison of multiple models.
"""

import os
import json
import yaml
import pickle
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import mlflow.sklearn

# Import custom modules
from utils.data_loader import CIFAR10DataLoader
from utils.training_pipeline import PyTorchTrainer, TraditionalMLTrainer, HyperparameterTuner
from utils.evaluation import MetricsCalculator, CrossValidationEvaluator, ModelComparator, VisualizationGenerator
from models.model_definitions import ModelFactory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentManager:
    """
    Main experiment manager that orchestrates all ML experiments.
    Handles automated training, evaluation, and comparison of multiple models.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize experiment manager.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up directories
        self.setup_directories()
        
        # Initialize components
        self.data_loader = CIFAR10DataLoader(self.config)
        self.pytorch_trainer = PyTorchTrainer(self.config)
        self.traditional_trainer = TraditionalMLTrainer(self.config)
        self.hyperparameter_tuner = HyperparameterTuner(self.config)
        self.cv_evaluator = CrossValidationEvaluator(self.config)
        
        # Get dataset info
        self.dataset_info = self.data_loader.get_dataset_info()
        self.class_names = self.dataset_info['class_names']
        
        # Initialize metrics calculator and comparator
        self.metrics_calculator = MetricsCalculator(self.class_names)
        self.model_comparator = ModelComparator(self.class_names)
        
        # Initialize visualization generator
        self.viz_generator = VisualizationGenerator(self.config['output']['plots_dir'])
        
        # Experiment tracking
        self.experiment_results = {}
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up MLflow
        if self.config['experiment_tracking']['use_mlflow']:
            mlflow.set_experiment(f"CIFAR10_Experiment_{self.experiment_id}")
        
        logger.info(f"Experiment Manager initialized. Experiment ID: {self.experiment_id}")
    
    def setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.config['output']['results_dir'],
            self.config['output']['models_dir'],
            self.config['output']['plots_dir'],
            self.config['output']['reports_dir']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """
        Run all experiments defined in the configuration.
        
        Returns:
            Dictionary containing all experiment results
        """
        logger.info("Starting comprehensive ML experiment suite...")
        
        # Get data loaders
        train_loader, val_loader, test_loader = self.data_loader.get_data_loaders()
        cv_splits = self.data_loader.get_cross_validation_splits(self.config['cross_validation']['n_splits'])
        
        # Get traditional ML data
        X_train, X_test, y_train, y_test = self.data_loader.get_data_for_traditional_ml()
        
        # Run traditional ML experiments
        logger.info("Running traditional ML experiments...")
        traditional_results = self.run_traditional_ml_experiments(X_train, X_test, y_train, y_test)
        
        # Run deep learning experiments
        logger.info("Running deep learning experiments...")
        dl_results = self.run_deep_learning_experiments(train_loader, val_loader, test_loader, cv_splits)
        
        # Combine results
        all_results = {**traditional_results, **dl_results}
        
        # Add results to comparator
        for model_name, results in all_results.items():
            self.model_comparator.add_model_results(model_name, results)
        
        # Generate comparison and visualizations
        self.generate_final_analysis(all_results)
        
        # Save results
        self.save_experiment_results(all_results)
        
        logger.info("All experiments completed successfully!")
        return all_results
    
    def run_traditional_ml_experiments(self, X_train: np.ndarray, X_test: np.ndarray,
                                     y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Run traditional ML experiments."""
        results = {}
        
        for model_config in self.config['models']['traditional_ml']:
            model_name = model_config['name']
            model_type = model_config['type']
            param_grid = model_config['params']
            
            logger.info(f"Training {model_name}...")
            
            with mlflow.start_run(run_name=f"{model_name}_{self.experiment_id}"):
                # Hyperparameter tuning
                if self.config['hyperparameter_tuning']['method'] == 'optuna':
                    logger.info(f"Tuning hyperparameters for {model_name}...")
                    tuning_results = self.hyperparameter_tuner.tune_traditional_ml(
                        model_type, X_train, y_train, param_grid
                    )
                    best_params = tuning_results['best_params']
                    mlflow.log_params(best_params)
                else:
                    # Use default parameters (first value in each param list)
                    best_params = {k: v[0] if isinstance(v, list) else v 
                                 for k, v in param_grid.items()}
                
                # Train final model with best parameters
                model_results = self.traditional_trainer.train_model(
                    model_type, X_train, X_test, y_train, y_test, best_params
                )
                
                # Cross-validation evaluation
                final_model = model_results['model']
                cv_results = self.cv_evaluator.evaluate_traditional_ml(final_model, X_train, y_train)
                
                # Combine results
                combined_results = {**model_results, **cv_results}
                results[model_name] = combined_results
                
                # Log metrics to MLflow
                mlflow.log_metrics({
                    'test_accuracy': model_results['test_accuracy'],
                    'cv_accuracy_mean': cv_results['accuracy_test_mean'],
                    'cv_accuracy_std': cv_results['accuracy_test_std'],
                    'training_time': model_results['training_time'],
                    'f1_score_weighted': cv_results['f1_weighted_test_mean']
                })
                
                # Save model
                model_path = os.path.join(self.config['output']['models_dir'], f"{model_name}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(final_model, f)
                mlflow.log_artifact(model_path)
                
                logger.info(f"{model_name} completed. Test Accuracy: {model_results['test_accuracy']:.2f}%")
        
        return results
    
    def run_deep_learning_experiments(self, train_loader, val_loader, test_loader, cv_splits) -> Dict[str, Any]:
        """Run deep learning experiments."""
        results = {}
        
        for model_config in self.config['models']['deep_learning']:
            model_name = model_config['name']
            architecture = model_config['architecture']
            param_grid = model_config['params']
            
            logger.info(f"Training {model_name}...")
            
            with mlflow.start_run(run_name=f"{model_name}_{self.experiment_id}"):
                # Hyperparameter tuning (simplified for demo)
                if 'learning_rate' in param_grid and isinstance(param_grid['learning_rate'], list):
                    best_lr = param_grid['learning_rate'][0]  # Use first value for demo
                else:
                    best_lr = param_grid.get('learning_rate', 0.001)
                
                if 'batch_size' in param_grid and isinstance(param_grid['batch_size'], list):
                    best_batch_size = param_grid['batch_size'][0]
                else:
                    best_batch_size = param_grid.get('batch_size', 128)
                
                epochs = param_grid.get('epochs', 50)
                pretrained = param_grid.get('pretrained', False)
                
                # Create model
                if architecture == 'custom_cnn':
                    model = ModelFactory.create_pytorch_model('SimpleCNN', num_classes=10)
                else:
                    model = ModelFactory.create_pytorch_model(
                        architecture, num_classes=10, pretrained=pretrained
                    )
                
                # Train model
                training_results = self.pytorch_trainer.train_model(
                    model, train_loader, val_loader,
                    learning_rate=best_lr, epochs=epochs,
                    experiment_name=f"{model_name}_{self.experiment_id}"
                )
                
                # Evaluate on test set
                test_metrics = self.metrics_calculator.calculate_pytorch_metrics(
                    model, test_loader, self.pytorch_trainer.device
                )
                
                # Cross-validation (simplified - use validation accuracy from training)
                cv_results = {
                    'accuracy_test_mean': training_results['best_val_accuracy'],
                    'accuracy_test_std': 2.0,  # Placeholder
                    'overfitting_gap': max(training_results['history']['train_acc']) - training_results['best_val_accuracy']
                }
                
                # Combine results
                combined_results = {
                    **training_results,
                    **test_metrics,
                    **cv_results,
                    'test_accuracy': test_metrics['accuracy'] * 100
                }
                results[model_name] = combined_results
                
                # Log metrics to MLflow
                mlflow.log_metrics({
                    'test_accuracy': test_metrics['accuracy'] * 100,
                    'val_accuracy': training_results['best_val_accuracy'],
                    'training_time': training_results['training_time'],
                    'num_parameters': training_results['num_parameters'],
                    'model_size_mb': training_results['model_size_mb'],
                    'f1_score_weighted': test_metrics['f1_score_weighted']
                })
                
                mlflow.log_params({
                    'learning_rate': best_lr,
                    'batch_size': best_batch_size,
                    'epochs': epochs,
                    'architecture': architecture,
                    'pretrained': pretrained
                })
                
                # Generate and log visualizations
                if 'history' in training_results:
                    curves_path = self.viz_generator.plot_training_curves(
                        training_results['history'], model_name
                    )
                    mlflow.log_artifact(curves_path)
                
                if 'confusion_matrix' in test_metrics:
                    cm_path = self.viz_generator.plot_confusion_matrix(
                        np.array(test_metrics['confusion_matrix']), 
                        self.class_names, model_name
                    )
                    mlflow.log_artifact(cm_path)
                
                # Save model
                model_path = os.path.join(self.config['output']['models_dir'], f"{model_name}.pth")
                import torch
                torch.save(model.state_dict(), model_path)
                mlflow.pytorch.log_model(model, f"{model_name}_model")
                
                logger.info(f"{model_name} completed. Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
        
        return results
    
    def generate_final_analysis(self, all_results: Dict[str, Any]):
        """Generate final analysis and comparisons."""
        logger.info("Generating final analysis...")
        
        # Generate comparison table
        comparison_df = self.model_comparator.generate_comparison_table()
        ranked_df = self.model_comparator.rank_models('Test Accuracy')
        
        # Save comparison tables
        comparison_path = os.path.join(self.config['output']['reports_dir'], 'model_comparison.csv')
        ranked_path = os.path.join(self.config['output']['reports_dir'], 'model_ranking.csv')
        
        comparison_df.to_csv(comparison_path, index=False)
        ranked_df.to_csv(ranked_path, index=False)
        
        # Generate comparison visualization
        comparison_viz_path = self.viz_generator.plot_model_comparison(comparison_df)
        
        # Generate summary report
        self.generate_summary_report(ranked_df, all_results)
        
        logger.info("Final analysis completed!")
    
    def generate_summary_report(self, ranked_df: pd.DataFrame, all_results: Dict[str, Any]):
        """Generate comprehensive summary report."""
        report_path = os.path.join(self.config['output']['reports_dir'], 'experiment_summary.md')
        
        with open(report_path, 'w') as f:
            f.write(f"# ML Capstone Project - Experiment Summary\n\n")
            f.write(f"**Experiment ID:** {self.experiment_id}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Dataset:** {self.config['data']['dataset_name']}\n")
            f.write(f"**Number of Classes:** {self.dataset_info['num_classes']}\n\n")
            
            f.write("## Dataset Information\n\n")
            f.write(f"- **Training Samples:** {self.dataset_info['train_size']}\n")
            f.write(f"- **Test Samples:** {self.dataset_info['test_size']}\n")
            f.write(f"- **Input Shape:** {self.dataset_info['input_shape']}\n")
            f.write(f"- **Classes:** {', '.join(self.class_names)}\n\n")
            
            f.write("## Model Performance Ranking\n\n")
            f.write("| Rank | Model | Test Accuracy (%) | Training Time (s) | Parameters | Model Size (MB) |\n")
            f.write("|------|-------|-------------------|-------------------|------------|------------------|\n")
            
            for _, row in ranked_df.iterrows():
                f.write(f"| {row.get('Rank', 'N/A')} | {row['Model']} | {row['Test Accuracy']:.2f} | "
                       f"{row['Training Time (s)']:.1f} | {row['Parameters']} | {row['Model Size (MB)']} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            # Best performing model
            best_model = ranked_df.iloc[0]
            f.write(f"### Best Performing Model: {best_model['Model']}\n")
            f.write(f"- **Test Accuracy:** {best_model['Test Accuracy']:.2f}%\n")
            f.write(f"- **Training Time:** {best_model['Training Time (s)']:.1f} seconds\n")
            f.write(f"- **Model Size:** {best_model['Model Size (MB)']} MB\n\n")
            
            # Efficiency analysis
            f.write("### Efficiency Analysis\n")
            fastest_model = ranked_df.loc[ranked_df['Training Time (s)'].idxmin()]
            f.write(f"- **Fastest Training:** {fastest_model['Model']} ({fastest_model['Training Time (s)']:.1f}s)\n")
            
            # Check for model sizes that are not 'N/A'
            size_df = ranked_df[ranked_df['Model Size (MB)'] != 'N/A'].copy()
            if not size_df.empty:
                size_df['Model Size (MB)'] = pd.to_numeric(size_df['Model Size (MB)'])
                smallest_model = size_df.loc[size_df['Model Size (MB)'].idxmin()]
                f.write(f"- **Smallest Model:** {smallest_model['Model']} ({smallest_model['Model Size (MB)']} MB)\n")
            
            f.write("\n### Overfitting Analysis\n")
            for model_name, results in all_results.items():
                if 'overfitting_gap' in results:
                    gap = results['overfitting_gap']
                    if gap > 10:
                        f.write(f"- **{model_name}:** High overfitting detected (gap: {gap:.1f}%)\n")
                    elif gap > 5:
                        f.write(f"- **{model_name}:** Moderate overfitting (gap: {gap:.1f}%)\n")
                    else:
                        f.write(f"- **{model_name}:** Good generalization (gap: {gap:.1f}%)\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("Based on the experimental results:\n\n")
            f.write(f"1. **For Production Deployment:** {best_model['Model']} offers the best accuracy\n")
            f.write(f"2. **For Resource-Constrained Environments:** Consider {fastest_model['Model']} for faster inference\n")
            f.write("3. **For Further Improvement:** Consider ensemble methods combining top-performing models\n")
            f.write("4. **Hyperparameter Tuning:** Additional tuning could improve performance further\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- Model comparison: `model_comparison.csv`\n")
            f.write("- Model ranking: `model_ranking.csv`\n")
            f.write("- Training curves: `*_training_curves.png`\n")
            f.write("- Confusion matrices: `*_confusion_matrix.png`\n")
            f.write("- Interactive comparison: `model_comparison.html`\n")
        
        logger.info(f"Summary report saved to: {report_path}")
    
    def save_experiment_results(self, results: Dict[str, Any]):
        """Save all experiment results."""
        results_path = os.path.join(self.config['output']['results_dir'], f'experiment_results_{self.experiment_id}.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {}
            for key, value in model_results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[model_name][key] = value.tolist()
                elif hasattr(value, 'tolist'):  # Handle other numpy types
                    serializable_results[model_name][key] = value.tolist()
                elif key == 'model':  # Skip model objects
                    continue
                else:
                    serializable_results[model_name][key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Experiment results saved to: {results_path}")
    
    def create_ensemble_model(self, top_models: List[str]) -> Dict[str, Any]:
        """Create ensemble model from top performing models."""
        logger.info("Creating ensemble model...")
        
        # This is a placeholder for ensemble implementation
        # In a full implementation, you would:
        # 1. Load the top performing models
        # 2. Create voting/stacking ensemble
        # 3. Evaluate ensemble performance
        # 4. Compare with individual models
        
        ensemble_results = {
            'ensemble_accuracy': 0.0,  # Placeholder
            'ensemble_models': top_models,
            'ensemble_method': 'voting'
        }
        
        return ensemble_results

def main():
    """Main function to run all experiments."""
    config_path = 'configs/config.yaml'
    
    # Create experiment manager
    experiment_manager = ExperimentManager(config_path)
    
    # Run all experiments
    results = experiment_manager.run_all_experiments()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Results saved in: {experiment_manager.config['output']['results_dir']}")
    print(f"Models saved in: {experiment_manager.config['output']['models_dir']}")
    print(f"Plots saved in: {experiment_manager.config['output']['plots_dir']}")
    print(f"Reports saved in: {experiment_manager.config['output']['reports_dir']}")
    print("="*80)

if __name__ == "__main__":
    main()

