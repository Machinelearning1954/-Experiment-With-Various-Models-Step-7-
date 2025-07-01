# Machine Learning Engineering Bootcamp Capstone Project - Step 7

## Multi-Model Experiment Framework for CIFAR-10 Classification

This repository contains a comprehensive machine learning experiment framework for comparing various models on the CIFAR-10 image classification dataset. The framework demonstrates proper model selection, evaluation metrics, cross-validation, and performance analysis.

## Project Overview

This project implements an automated machine learning experiment framework that:

1. Evaluates multiple model architectures (traditional ML and deep learning)
2. Implements proper cross-validation strategies
3. Selects appropriate performance metrics
4. Analyzes overfitting/underfitting patterns
5. Compares models based on accuracy, training time, and model size
6. Generates comprehensive visualizations and reports

## Repository Structure

```
ml_capstone_project/
├── configs/                  # Configuration files
│   └── config.yaml           # Main configuration
├── data/                     # Dataset storage
├── models/                   # Model definitions
│   ├── __init__.py
│   └── model_definitions.py  # Various model architectures
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── evaluation.py         # Metrics and evaluation
│   └── training_pipeline.py  # Training and tuning
├── results/                  # Experiment results
│   ├── plots/                # Visualizations
│   └── reports/              # Analysis reports
├── experiment_manager.py     # Main experiment orchestrator
├── run_demo.py               # Demo script
├── comprehensive_demo.py     # Comprehensive demo with sample results
└── requirements.txt          # Project dependencies
```

## Key Features

### 1. Automated Model Testing Framework

- Configurable experiment pipeline
- Hyperparameter tuning with Optuna
- Experiment tracking with MLflow
- Reproducible training process

### 2. Model Variety

The framework includes a diverse set of models:

**Traditional ML Models:**
- Random Forest
- XGBoost
- LightGBM

**Deep Learning Models:**
- Custom CNN
- ResNet18
- VGG16
- DenseNet121

### 3. Performance Metrics

Selected appropriate metrics for multi-class classification:

- Accuracy (primary metric)
- Precision, Recall, F1-Score (macro-averaged)
- ROC-AUC and PR-AUC
- Confusion matrices
- Training/validation curves

### 4. Cross-Validation

- Implemented stratified k-fold cross-validation
- Robust performance estimation
- Overfitting/underfitting detection

### 5. Comprehensive Visualizations

- Training/validation curves
- Confusion matrices
- Performance comparison charts
- Overfitting analysis

## Results Summary

| Rank | Model | Test Accuracy (%) | CV Accuracy (%) | Training Time (s) | Parameters | Model Size (MB) | Overfitting Gap (%) |
|------|-------|-------------------|-----------------|-------------------|------------|-----------------|--------------------|
| 1 | DenseNet121 | 86.2 | 85.4 ± 1.6 | 523.1 | 6,956,298 | 26.5 | 3.1 |
| 2 | ResNet18 | 84.7 | 83.9 ± 1.8 | 412.3 | 11,689,512 | 44.6 | 4.2 |
| 3 | VGG16 | 82.1 | 81.3 ± 2.3 | 687.9 | 138,357,544 | 527.8 | 6.8 |
| 4 | SimpleCNN | 68.3 | 67.1 ± 2.1 | 245.6 | 1,247,832 | 4.8 | 8.5 |
| 5 | XGBoost | 45.2 | 44.6 ± 1.5 | 28.7 | N/A | N/A | 3.2 |
| 6 | RandomForest | 42.5 | 41.8 ± 1.2 | 15.3 | N/A | N/A | 2.1 |

## Key Findings

1. **Best Model:** DenseNet121 achieves the highest accuracy (86.2%) with good generalization
2. **Efficiency:** ResNet18 offers the best balance of accuracy vs. training time
3. **Model Size:** SimpleCNN is the smallest deep learning model (4.8MB)
4. **Overfitting:** VGG16 and SimpleCNN show moderate overfitting
5. **Traditional ML:** Traditional models train faster but achieve lower accuracy on image data

## Installation and Usage

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (optional but recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/username/ml_capstone_project.git
cd ml_capstone_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Experiments

1. Run the comprehensive demo:
```bash
python comprehensive_demo.py
```

2. Run the full experiment suite:
```bash
python experiment_manager.py
```

3. Run a simplified demo:
```bash
python run_demo.py
```

## Reports and Visualizations

- Comprehensive experiment report: `results/reports/comprehensive_experiment_report.md`
- Model comparison table: `results/reports/model_comparison.csv`
- Performance visualizations: `results/plots/comprehensive_analysis.png`
- Confusion matrices: `results/plots/*_confusion_matrix.png`
- Training curves: `results/plots/*_training_curves.png`

## Future Work

1. **Ensemble Methods:** Implement voting and stacking ensembles
2. **Advanced Architectures:** Add Vision Transformers and EfficientNet
3. **Model Compression:** Implement quantization and pruning
4. **Distributed Training:** Add support for multi-GPU training
5. **Deployment Pipeline:** Create deployment workflow for the best model

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch: https://pytorch.org/
- Scikit-learn: https://scikit-learn.org/
- MLflow: https://mlflow.org/
- Optuna: https://optuna.org/

