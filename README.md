# Adversarial Robustness Evaluation Pipeline

This project implements an automated pipeline to evaluate the adversarial robustness of image classification models using various adversarial attack techniques.

## Features

- **Pre-trained Model**: Uses ResNet-18 trained on CIFAR-10
- **Adversarial Attacks**: Implements FGSM, PGD, and DeepFool using the Adversarial Robustness Toolbox (ART)
- **Comprehensive Evaluation**: Compares model performance on clean vs. adversarial examples
- **Visualizations**: Generates plots showing accuracy comparisons and adversarial example samples
- **Mitigation Strategies**: Implements adversarial training for improved robustness
- **Transferability Analysis**: Evaluates how adversarial examples transfer between different models

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Main Adversarial Robustness Evaluation

Run the main pipeline to evaluate model robustness:

```bash
python adversarial_pipeline.py
```

This will:
1. Load a pre-trained ResNet-18 model and CIFAR-10 dataset
2. Generate adversarial examples using FGSM and PGD attacks
3. Evaluate model accuracy on clean and adversarial data
4. Create visualizations in the `results/` directory
5. Save a detailed analysis report

**Runtime**: ~5-15 minutes | **Outputs**: 4 files in `results/`

### 2. Adversarial Training Mitigation (Optional)

Test robustness improvements through adversarial training:

```bash
python adversarial_training.py
```

**Runtime**: ~30-90 minutes | **Outputs**: 3 files in `results/`

### 3. Transferability Analysis (Optional - Extra Credit)

Analyze how adversarial examples transfer between different model architectures:

```bash
python transferability_analysis.py
```

**Runtime**: ~15-30 minutes | **Outputs**: 5 files in `results/`

For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md)

## Project Structure

```
.
├── adversarial_pipeline.py      # Main robustness evaluation pipeline
├── adversarial_training.py      # Adversarial training mitigation (optional)
├── transferability_analysis.py  # Cross-model transferability analysis (optional)
├── requirements.txt             # Python dependencies
├── USAGE_GUIDE.md              # Detailed usage instructions
├── .gitignore                  # Git ignore patterns
├── results/                    # Generated visualizations and reports
│   ├── accuracy_comparison.png
│   ├── adversarial_examples.png
│   ├── perturbation_magnitudes.png
│   └── analysis_report.txt
└── README.md                   # This file
```

## Attack Methods

### FGSM (Fast Gradient Sign Method)
- Single-step attack that perturbs input in the direction of the gradient
- Fast but less effective than iterative methods
- Formula: x_adv = x + ε * sign(∇_x L(θ, x, y))

### PGD (Projected Gradient Descent)
- Iterative version of FGSM with projection step
- More powerful but computationally expensive
- Performs multiple small steps with projection to epsilon ball

### DeepFool (Optional)
- Finds minimal perturbation to cross decision boundary
- More sophisticated geometric approach
- Computes optimal adversarial perturbation

## Output Files

### Main Pipeline Outputs

1. **accuracy_comparison.png**: Bar chart comparing model accuracy on clean vs adversarial examples
2. **adversarial_examples.png**: Visual comparison of original and adversarially perturbed images
3. **perturbation_magnitudes.png**: Heatmaps showing perturbation patterns for different attacks
4. **analysis_report.txt**: Comprehensive text report with:
   - Attack effectiveness metrics
   - Conceptual explanations of attack methods
   - Observed vulnerabilities
   - Mitigation strategy recommendations

### Adversarial Training Outputs

1. **mitigation_comparison.png**: Comparison of standard vs adversarially trained models
2. **adversarial_training_history.png**: Training curves showing loss and accuracy evolution
3. **mitigation_report.txt**: Analysis of robustness improvements

### Transferability Analysis Outputs

1. **transferability_heatmap_fgsm.png**: FGSM attack transferability matrix
2. **transferability_heatmap_pgd.png**: PGD attack transferability matrix
3. **transferability_comparison.png**: Cross-model robustness comparison
4. **transferability_confusion.png**: Confusion matrix for prediction changes
5. **transferability_report.txt**: Detailed transferability analysis and security implications

## Key Insights

The pipeline provides comprehensive answers to:

- **How vulnerable is the model?** Quantitative accuracy drops under different attacks
- **Which attacks are most effective?** Comparison of FGSM vs PGD effectiveness
- **What makes attacks work differently?** Conceptual and empirical differences
- **How can we improve robustness?** Adversarial training and other mitigation strategies
- **Do attacks transfer between models?** Cross-architecture transferability analysis

## Technical Details

### Models Tested
- **Main Pipeline**: ResNet-18
- **Transferability**: ResNet-18, ResNet-34, VGG-11, MobileNetV2

### Attacks Implemented
- **FGSM**: Single-step gradient-based attack (ε=0.03)
- **PGD**: Iterative gradient-based attack (40 iterations, ε=0.03)
- **DeepFool**: Minimal perturbation attack (optional)

### Dataset
- **CIFAR-10**: 10-class image classification dataset
- **Test samples**: 1,000 (configurable for speed/accuracy tradeoff)
- **Image size**: 32×32×3 pixels

## References

- Adversarial Robustness Toolbox: https://github.com/Trusted-AI/adversarial-robustness-toolbox
- FGSM Paper: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2014)
- PGD Paper: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2017)
