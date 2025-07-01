#!/usr/bin/env python3
"""
Main script to run ML capstone project experiments.
This is a demonstration version that runs a subset of models.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import CIFAR10DataLoader
from utils.training_pipeline import PyTorchTrainer, TraditionalMLTrainer
from utils.evaluation import MetricsCalculator, ModelComparator, VisualizationGenerator
from models.model_definitions import ModelFactory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_config():
    """Create a simplified configuration for demonstration."""
    return {
        'data': {
            'dataset_name': 'CIFAR-10',
            'num_classes': 10,
            'input_shape': [3, 32, 32],
            'data_dir': './data',
            'batch_size': 128,
            'num_workers': 2,
            'validation_split': 0.2
        },
        'training': {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'mixed_precision': False,
            'gradient_clipping': 1.0,
            'early_stopping': {
                'patience': 5,
                'min_delta': 0.001
            }
        },
        'cross_validation': {
            'strategy': 'stratified_kfold',
            'n_splits': 3,  # Reduced for demo
            'shuffle': True,
            'random_state': 42
        },
        'output': {
            'results_dir': './results',
            'models_dir': './models',
            'plots_dir': './results/plots',
            'reports_dir': './results/reports'
        }
    }

def setup_directories(config):
    """Create necessary directories."""
    directories = [
        config['output']['results_dir'],
        config['output']['models_dir'],
        config['output']['plots_dir'],
        config['output']['reports_dir']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_demo_experiments():
    """Run a subset of experiments for demonstration."""
    logger.info("Starting ML Capstone Project Demo...")
    
    # Create configuration
    config = create_demo_config()
    setup_directories(config)
    
    # Initialize data loader
    logger.info("Loading CIFAR-10 dataset...")
    data_loader = CIFAR10DataLoader(config)
    
    # Get data
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    X_train, X_test, y_train, y_test = data_loader.get_data_for_traditional_ml()
    
    # Initialize components
    pytorch_trainer = PyTorchTrainer(config)
    traditional_trainer = TraditionalMLTrainer(config)
    dataset_info = data_loader.get_dataset_info()
    class_names = dataset_info['class_names']
    
    metrics_calculator = MetricsCalculator(class_names)
    model_comparator = ModelComparator(class_names)
    viz_generator = VisualizationGenerator(config['output']['plots_dir'])
    
    results = {}
    
    # 1. Train a simple Random Forest (fast traditional ML model)
    logger.info("Training Random Forest...")
    try:
        rf_results = traditional_trainer.train_model(
            'randomforest', X_train[:5000], X_test[:1000], 
            y_train[:5000], y_test[:1000],
            {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
        )
        results['RandomForest'] = rf_results
        logger.info(f"Random Forest - Test Accuracy: {rf_results['test_accuracy']:.2f}%")
    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")
    
    # 2. Train a simple CNN (reduced epochs for demo)
    logger.info("Training Simple CNN...")
    try:
        simple_cnn = ModelFactory.create_pytorch_model('SimpleCNN', num_classes=10)
        cnn_results = pytorch_trainer.train_model(
            simple_cnn, train_loader, val_loader,
            learning_rate=0.001, epochs=5,  # Reduced epochs for demo
            experiment_name="SimpleCNN_demo"
        )
        
        # Evaluate on test set
        test_metrics = metrics_calculator.calculate_pytorch_metrics(
            simple_cnn, test_loader, pytorch_trainer.device
        )
        
        cnn_results.update(test_metrics)
        cnn_results['test_accuracy'] = test_metrics['accuracy'] * 100
        results['SimpleCNN'] = cnn_results
        
        # Generate visualizations
        if 'history' in cnn_results:
            viz_generator.plot_training_curves(cnn_results['history'], 'SimpleCNN')
        
        if 'confusion_matrix' in test_metrics:
            viz_generator.plot_confusion_matrix(
                np.array(test_metrics['confusion_matrix']), 
                class_names, 'SimpleCNN'
            )
        
        logger.info(f"Simple CNN - Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    except Exception as e:
        logger.error(f"Simple CNN training failed: {e}")
    
    # 3. Train ResNet18 (pretrained, reduced epochs)
    logger.info("Training ResNet18...")
    try:
        resnet18 = ModelFactory.create_pytorch_model('resnet18', num_classes=10, pretrained=True)
        resnet_results = pytorch_trainer.train_model(
            resnet18, train_loader, val_loader,
            learning_rate=0.001, epochs=3,  # Very reduced for demo
            experiment_name="ResNet18_demo"
        )
        
        # Evaluate on test set
        test_metrics = metrics_calculator.calculate_pytorch_metrics(
            resnet18, test_loader, pytorch_trainer.device
        )
        
        resnet_results.update(test_metrics)
        resnet_results['test_accuracy'] = test_metrics['accuracy'] * 100
        results['ResNet18'] = resnet_results
        
        # Generate visualizations
        if 'history' in resnet_results:
            viz_generator.plot_training_curves(resnet_results['history'], 'ResNet18')
        
        if 'confusion_matrix' in test_metrics:
            viz_generator.plot_confusion_matrix(
                np.array(test_metrics['confusion_matrix']), 
                class_names, 'ResNet18'
            )
        
        logger.info(f"ResNet18 - Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    except Exception as e:
        logger.error(f"ResNet18 training failed: {e}")
    
    # Add results to comparator and generate analysis
    for model_name, model_results in results.items():
        model_comparator.add_model_results(model_name, model_results)
    
    # Generate comparison table
    comparison_df = model_comparator.generate_comparison_table()
    ranked_df = model_comparator.rank_models('Test Accuracy')
    
    # Save results
    comparison_df.to_csv(os.path.join(config['output']['reports_dir'], 'model_comparison.csv'), index=False)
    ranked_df.to_csv(os.path.join(config['output']['reports_dir'], 'model_ranking.csv'), index=False)
    
    # Generate comparison visualization
    viz_generator.plot_model_comparison(comparison_df)
    
    # Generate summary report
    generate_demo_report(ranked_df, results, config)
    
    logger.info("Demo experiments completed successfully!")
    return results

def generate_demo_report(ranked_df, results, config):
    """Generate a summary report for the demo."""
    report_path = os.path.join(config['output']['reports_dir'], 'demo_summary.md')
    
    with open(report_path, 'w') as f:
        f.write("# ML Capstone Project - Demo Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Dataset:** CIFAR-10 (Image Classification)\n")
        f.write(f"**Models Tested:** {len(results)}\n\n")
        
        f.write("## Model Performance\n\n")
        f.write("| Rank | Model | Test Accuracy (%) | Training Time (s) | Parameters |\n")
        f.write("|------|-------|-------------------|-------------------|-----------|\n")
        
        for _, row in ranked_df.iterrows():
            f.write(f"| {row.get('Rank', 'N/A')} | {row['Model']} | {row['Test Accuracy']:.2f} | "
                   f"{row['Training Time (s)']:.1f} | {row['Parameters']} |\n")
        
        f.write("\n## Key Observations\n\n")
        
        if len(results) > 0:
            best_model = ranked_df.iloc[0]
            f.write(f"- **Best Model:** {best_model['Model']} with {best_model['Test Accuracy']:.2f}% accuracy\n")
            
            # Analyze overfitting
            for model_name, model_results in results.items():
                if 'history' in model_results:
                    history = model_results['history']
                    if len(history['train_acc']) > 0 and len(history['val_acc']) > 0:
                        final_train_acc = history['train_acc'][-1]
                        final_val_acc = history['val_acc'][-1]
                        gap = final_train_acc - final_val_acc
                        f.write(f"- **{model_name} Overfitting Analysis:** Training-Validation gap = {gap:.2f}%\n")
        
        f.write("\n## Files Generated\n\n")
        f.write("- `model_comparison.csv`: Detailed comparison of all models\n")
        f.write("- `model_ranking.csv`: Models ranked by performance\n")
        f.write("- `*_training_curves.png`: Training progress visualization\n")
        f.write("- `*_confusion_matrix.png`: Model prediction analysis\n")
        f.write("- `model_comparison.html`: Interactive comparison dashboard\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. **Full Experiment Suite:** Run complete experiments with all models\n")
        f.write("2. **Hyperparameter Tuning:** Optimize model parameters\n")
        f.write("3. **Ensemble Methods:** Combine best models for improved performance\n")
        f.write("4. **Cross-Validation:** Implement robust model validation\n")
        f.write("5. **Production Deployment:** Deploy best model for real-world use\n")

def main():
    """Main function."""
    print("="*80)
    print("ML ENGINEERING BOOTCAMP - CAPSTONE PROJECT STEP 7")
    print("Multi-Model Experiment Framework")
    print("="*80)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")
    
    print(f"PyTorch Version: {torch.__version__}")
    print("="*80)
    
    try:
        results = run_demo_experiments()
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Results saved in: ./results/")
        print("Models saved in: ./models/")
        print("Visualizations saved in: ./results/plots/")
        print("Reports saved in: ./results/reports/")
        print("="*80)
        
        # Display summary
        if results:
            print("\nMODEL PERFORMANCE SUMMARY:")
            print("-" * 50)
            for model_name, model_results in results.items():
                accuracy = model_results.get('test_accuracy', 0)
                time_taken = model_results.get('training_time', 0)
                print(f"{model_name:15} | Accuracy: {accuracy:6.2f}% | Time: {time_taken:6.1f}s")
            print("-" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()

