# Adversarial Robustness Pipeline - Usage Guide

This guide provides detailed instructions on running the adversarial robustness evaluation pipeline.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Main Pipeline](#main-pipeline)
4. [Adversarial Training](#adversarial-training)
5. [Transferability Analysis](#transferability-analysis)
6. [Understanding the Results](#understanding-the-results)
7. [Configuration Options](#configuration-options)
8. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for faster execution)

### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch and torchvision
- Adversarial Robustness Toolbox (ART)
- NumPy, Matplotlib, scikit-learn
- Other required packages

## Quick Start

To run the complete adversarial robustness evaluation pipeline:

```bash
python adversarial_pipeline.py
```

This will:
1. Download CIFAR-10 dataset automatically
2. Load a pre-trained ResNet-18 model
3. Generate adversarial examples using FGSM and PGD
4. Evaluate model performance
5. Create visualizations in the `results/` directory
6. Generate a detailed analysis report

**Estimated runtime**: 5-15 minutes (depending on hardware)

## Main Pipeline

### What it does

The main pipeline (`adversarial_pipeline.py`) performs:

1. **Data Loading**: Downloads and prepares CIFAR-10 dataset
2. **Model Loading**: Initializes ResNet-18 for CIFAR-10
3. **Clean Evaluation**: Tests model on unmodified images
4. **Attack Generation**: Creates adversarial examples using:
   - FGSM (Fast Gradient Sign Method)
   - PGD (Projected Gradient Descent)
5. **Robustness Evaluation**: Tests model on adversarial examples
6. **Visualization**: Generates plots and comparisons
7. **Analysis**: Produces detailed insights and recommendations

### Outputs

All results are saved in the `results/` directory:

- `accuracy_comparison.png`: Bar chart comparing clean vs adversarial accuracy
- `adversarial_examples.png`: Visual examples of original and adversarial images
- `perturbation_magnitudes.png`: Heatmaps showing perturbation patterns
- `analysis_report.txt`: Detailed text report with findings and recommendations

### Example Output

```
Clean Data Accuracy: 85.30%

FGSM Attack:
  - Epsilon: 0.03
  - Adversarial Accuracy: 45.20%
  - Accuracy Drop: 40.10%

PGD Attack:
  - Epsilon: 0.03
  - Adversarial Accuracy: 18.50%
  - Accuracy Drop: 66.80%
```

## Adversarial Training

### Purpose

Adversarial training is a mitigation strategy that improves model robustness by training on both clean and adversarial examples.

### Running Adversarial Training

```bash
python adversarial_training.py
```

**Warning**: This script trains models from scratch and is computationally intensive!

**Estimated runtime**: 30-90 minutes (depending on hardware and epochs)

### What it does

1. Trains a standard model on clean data only
2. Trains a robust model using adversarial training
3. Compares both models against FGSM and PGD attacks
4. Visualizes the improvement in robustness

### Outputs

- `results/mitigation_comparison.png`: Bar chart comparing standard vs adversarial training
- `results/adversarial_training_history.png`: Training curves showing loss and accuracy
- `results/mitigation_report.txt`: Analysis of robustness improvements

### Configuration

You can modify training parameters in the script:

```python
# In adversarial_training.py
epochs=5          # Number of training epochs (increase for better results)
epsilon=0.03      # Perturbation magnitude for adversarial examples
alpha=0.5         # Weight for adversarial loss (0=clean only, 1=adversarial only)
```

## Transferability Analysis

### Purpose

Analyzes how adversarial examples crafted for one model transfer to other models with different architectures.

### Running Transferability Analysis

```bash
python transferability_analysis.py
```

**Estimated runtime**: 15-30 minutes (tests 4 different models)

### What it does

1. Loads multiple models: ResNet18, ResNet34, VGG11, MobileNetV2
2. Generates adversarial examples on each source model
3. Tests these examples on all target models
4. Creates transferability matrices showing attack success rates
5. Analyzes patterns and security implications

### Outputs

- `results/transferability_heatmap_fgsm.png`: FGSM transferability matrix
- `results/transferability_heatmap_pgd.png`: PGD transferability matrix
- `results/transferability_comparison.png`: Bar chart comparing all models
- `results/transferability_confusion.png`: Confusion matrix for prediction changes
- `results/transferability_report.txt`: Detailed transferability analysis

### Interpreting Transferability Matrices

In the heatmap:
- **Rows**: Source model (where attack was generated)
- **Columns**: Target model (where attack was tested)
- **Values**: Accuracy on adversarial examples
- **Lower values** (red): Better transferability (attack succeeded)
- **Higher values** (green): Worse transferability (model more robust)

## Understanding the Results

### Attack Effectiveness

**FGSM (Fast Gradient Sign Method)**
- Single-step attack
- Fast to compute
- Less effective but good baseline
- Formula: `x_adv = x + ε * sign(∇_x L(θ, x, y))`

**PGD (Projected Gradient Descent)**
- Iterative multi-step attack
- More powerful and effective
- Considered one of the strongest first-order attacks
- Uses projection to stay within epsilon ball

**Conceptual Differences:**
1. **Computation**: FGSM uses one gradient step, PGD uses multiple iterations
2. **Effectiveness**: PGD is generally more effective due to iterative refinement
3. **Speed**: FGSM is much faster, PGD is computationally expensive
4. **Use case**: FGSM for quick evaluation, PGD for robust evaluation and training

### Accuracy Metrics

- **Clean Accuracy**: Model performance on normal, unmodified images
- **Adversarial Accuracy**: Model performance on adversarially perturbed images
- **Accuracy Drop**: Difference between clean and adversarial accuracy
  - Small drop (<20%): Model shows some robustness
  - Medium drop (20-50%): Model is vulnerable
  - Large drop (>50%): Model is highly vulnerable

### Perturbation Magnitude

- **Average Perturbation**: Mean pixel-wise change across all images
- **Max Perturbation**: Maximum pixel-wise change
- **Epsilon (ε)**: Maximum allowed perturbation (default: 0.03)
  - Smaller ε: Harder to detect but less effective
  - Larger ε: More effective but more visible

## Configuration Options

### Adjusting Number of Test Samples

In any script, modify:

```python
NUM_TEST_SAMPLES = 1000  # Reduce for faster execution, increase for more accurate results
```

### Adjusting Epsilon

Change perturbation magnitude:

```python
EPSILON = 0.03  # Try: 0.01 (subtle), 0.05 (moderate), 0.1 (strong)
```

### Batch Size

Adjust based on available GPU memory:

```python
BATCH_SIZE = 128  # Reduce if out of memory (e.g., 64, 32)
```

### PGD Parameters

Fine-tune PGD attack:

```python
attack = ProjectedGradientDescent(
    estimator=classifier,
    eps=0.03,           # Maximum perturbation
    eps_step=0.003,     # Step size per iteration
    max_iter=40,        # Number of iterations (more = stronger)
    targeted=False      # Untargeted attack
)
```

## Troubleshooting

### Out of Memory Errors

**Problem**: CUDA out of memory or system RAM exhausted

**Solutions**:
1. Reduce batch size: `BATCH_SIZE = 64` or `BATCH_SIZE = 32`
2. Reduce number of samples: `NUM_TEST_SAMPLES = 500`
3. Use CPU instead of GPU: The code automatically handles this
4. Close other applications to free memory

### Slow Execution

**Problem**: Code runs too slowly

**Solutions**:
1. Reduce test samples for faster evaluation
2. Use GPU if available (check: `torch.cuda.is_available()`)
3. Reduce PGD iterations: `max_iter=10` instead of `max_iter=40`
4. Skip optional DeepFool attack (already handled in code)

### Download Errors

**Problem**: CIFAR-10 download fails

**Solutions**:
1. Check internet connection
2. Download manually from https://www.cs.toronto.edu/~kriz/cifar.html
3. Place in `./data/cifar-10-batches-py/` directory

### Import Errors

**Problem**: Module not found errors

**Solutions**:
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Upgrade pip: `pip install --upgrade pip`
3. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Visualization Issues

**Problem**: Plots don't display or save

**Solutions**:
1. Check that `results/` directory exists (created automatically)
2. Ensure matplotlib backend is compatible: `export MPLBACKEND=Agg`
3. Check file permissions in results directory

## Best Practices

1. **Start Small**: Run with fewer samples first to verify everything works
2. **Monitor Resources**: Check GPU/CPU usage and memory consumption
3. **Save Results**: Results are automatically saved to `results/` directory
4. **Compare Attacks**: Run both FGSM and PGD to understand different threat models
5. **Document Findings**: Review generated reports for insights and recommendations

## Advanced Usage

### Using Different Models

Modify the model loading function:

```python
# In adversarial_pipeline.py
def load_pretrained_model(num_classes=10):
    # Replace with your model choice
    model = torchvision.models.resnet34(pretrained=True)  # or vgg11, densenet, etc.
    # ... modify for CIFAR-10 as shown in code
    return model, device
```

### Custom Datasets

Replace CIFAR-10 with your own dataset:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)  # Your dataset's mean/std
])

custom_dataset = datasets.ImageFolder(root='path/to/data', transform=transform)
```

### Additional Attacks

Add more attacks from ART:

```python
from art.attacks.evasion import CarliniL2Method, DeepFool

# C&W L2 attack
cw_attack = CarliniL2Method(classifier=classifier, confidence=0.0)

# DeepFool attack
deepfool_attack = DeepFool(classifier=classifier, max_iter=50)
```

## Questions and Support

For issues or questions:
1. Check this usage guide
2. Review the code comments in each script
3. Consult the ART documentation: https://adversarial-robustness-toolbox.readthedocs.io/
4. Check PyTorch documentation: https://pytorch.org/docs/

## Summary

This pipeline provides a comprehensive evaluation of adversarial robustness:

✅ **Main Pipeline**: Evaluate model vulnerability to FGSM and PGD attacks
✅ **Adversarial Training**: Implement and test mitigation strategies
✅ **Transferability Analysis**: Understand cross-model attack transferability
✅ **Rich Visualizations**: Clear plots and heatmaps for analysis
✅ **Detailed Reports**: Text reports with insights and recommendations

Start with the main pipeline, then explore adversarial training and transferability as needed!
