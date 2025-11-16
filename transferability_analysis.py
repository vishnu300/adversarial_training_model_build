import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier


# Create results directory
os.makedirs('results', exist_ok=True)


def load_cifar10_data(batch_size=128, num_samples=1000):
    """Load CIFAR-10 dataset."""
    print("Loading CIFAR-10 dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # Use subset for faster evaluation
    if num_samples < len(test_dataset):
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        test_dataset = Subset(test_dataset, indices)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return test_loader, classes


def create_resnet18(num_classes=10):
    """Create ResNet-18 model for CIFAR-10."""
    model = torchvision.models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_resnet34(num_classes=10):
    """Create ResNet-34 model for CIFAR-10."""
    model = torchvision.models.resnet34(pretrained=True)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_vgg11(num_classes=10):
    """Create VGG-11 model for CIFAR-10."""
    model = torchvision.models.vgg11(pretrained=True)
    # Modify classifier for CIFAR-10
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model


def create_mobilenet(num_classes=10):
    """Create MobileNetV2 model for CIFAR-10."""
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def load_models(device):
    print("\nLoading multiple models for transferability analysis...")

    models_dict = {
        'ResNet18': create_resnet18(),
        'ResNet34': create_resnet34(),
        'VGG11': create_vgg11(),
        'MobileNetV2': create_mobilenet()
    }

    # Move models to device and set to eval mode
    for name, model in models_dict.items():
        models_dict[name] = model.to(device)
        models_dict[name].eval()
        print(f"  - Loaded {name}")

    return models_dict


def create_art_classifiers(models_dict, device):
    """Create ART classifiers for all models."""
    print("\nCreating ART classifiers...")

    classifiers = {}
    criterion = nn.CrossEntropyLoss()

    for name, model in models_dict.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        classifier = PyTorchClassifier(
            model=model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 32, 32),
            nb_classes=10,
            clip_values=(0.0, 1.0),
            device_type='gpu' if device.type == 'cuda' else 'cpu'
        )

        classifiers[name] = classifier

    return classifiers


"""
    Perform comprehensive transferability analysis.

    Args:
        classifiers: Dictionary of ART classifiers
        x_test: Test images
        y_test: Test labels
        epsilon: Perturbation magnitude

    Returns:
        results: Transferability analysis results
"""

def perform_transferability_analysis(classifiers, x_test, y_test, epsilon=0.03):

    print("\n" + "="*80)
    print("PERFORMING TRANSFERABILITY ANALYSIS")
    print("="*80)

    model_names = list(classifiers.keys())
    num_models = len(model_names)

    # Initialize results dictionary
    results = {
        'model_names': model_names,
        'clean_accuracy': {},
        'transferability_matrix_fgsm': np.zeros((num_models, num_models)),
        'transferability_matrix_pgd': np.zeros((num_models, num_models)),
        'adversarial_examples': {}
    }

    # 1. Evaluate clean accuracy for all models
    print("\n[Step 1/3] Evaluating clean accuracy for all models...")
    for i, (name, classifier) in enumerate(classifiers.items()):
        clean_preds = classifier.predict(x_test)
        clean_acc = np.mean(np.argmax(clean_preds, axis=1) == np.argmax(y_test, axis=1)) * 100
        results['clean_accuracy'][name] = clean_acc
        print(f"  {name}: {clean_acc:.2f}%")

    # 2. Generate adversarial examples for each source model using FGSM
    print("\n[Step 2/3] Analyzing FGSM transferability...")
    for i, source_name in enumerate(tqdm(model_names, desc="Source models")):
        source_classifier = classifiers[source_name]

        # Generate FGSM adversarial examples
        fgsm_attack = FastGradientMethod(estimator=source_classifier, eps=epsilon)
        x_adv_fgsm = fgsm_attack.generate(x=x_test)

        # Store adversarial examples for the source model
        if source_name not in results['adversarial_examples']:
            results['adversarial_examples'][source_name] = {}
        results['adversarial_examples'][source_name]['fgsm'] = x_adv_fgsm

        # Test on all target models
        for j, target_name in enumerate(model_names):
            target_classifier = classifiers[target_name]

            # Evaluate adversarial examples on target model
            adv_preds = target_classifier.predict(x_adv_fgsm)
            adv_acc = np.mean(np.argmax(adv_preds, axis=1) == np.argmax(y_test, axis=1)) * 100

            results['transferability_matrix_fgsm'][i, j] = adv_acc

    # 3. Generate adversarial examples for each source model using PGD
    print("\n[Step 3/3] Analyzing PGD transferability...")
    for i, source_name in enumerate(tqdm(model_names, desc="Source models")):
        source_classifier = classifiers[source_name]

        # Generate PGD adversarial examples
        pgd_attack = ProjectedGradientDescent(
            estimator=source_classifier,
            eps=epsilon,
            eps_step=epsilon/10,
            max_iter=40
        )
        x_adv_pgd = pgd_attack.generate(x=x_test)

        # Store adversarial examples
        results['adversarial_examples'][source_name]['pgd'] = x_adv_pgd

        # Test on all target models
        for j, target_name in enumerate(model_names):
            target_classifier = classifiers[target_name]

            # Evaluate adversarial examples on target model
            adv_preds = target_classifier.predict(x_adv_pgd)
            adv_acc = np.mean(np.argmax(adv_preds, axis=1) == np.argmax(y_test, axis=1)) * 100

            results['transferability_matrix_pgd'][i, j] = adv_acc

    return results
"""
    Create heatmap showing transferability between models.

    Args:
        matrix: Transferability matrix (source x target)
        model_names: List of model names
        attack_name: Name of the attack
        output_path: Path to save the plot
"""

def visualize_transferability_heatmap(matrix, model_names, attack_name, output_path='results/transferability_heatmap.png'):

    print(f"\nCreating {attack_name} transferability heatmap...")

    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=model_names, yticklabels=model_names,
                cbar_kws={'label': 'Accuracy (%)'},
                vmin=0, vmax=100)

    plt.xlabel('Target Model', fontsize=12, fontweight='bold')
    plt.ylabel('Source Model (Attack Generated On)', fontsize=12, fontweight='bold')
    plt.title(f'Adversarial Example Transferability ({attack_name})\n'
             f'Higher values = Better robustness, Lower values = Better transferability',
             fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()


def visualize_transferability_comparison(results, output_path='results/transferability_comparison.png'):

    print("\nCreating transferability comparison visualization...")

    model_names = results['model_names']
    num_models = len(model_names)

    # Calculate average adversarial accuracy for each model as target
    clean_accs = [results['clean_accuracy'][name] for name in model_names]
    fgsm_accs = np.mean(results['transferability_matrix_fgsm'], axis=0)
    pgd_accs = np.mean(results['transferability_matrix_pgd'], axis=0)

    x = np.arange(num_models)
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width, clean_accs, width, label='Clean',
                   color='green', alpha=0.8)
    bars2 = ax.bar(x, fgsm_accs, width, label='FGSM (Avg from all sources)',
                   color='orange', alpha=0.8)
    bars3 = ax.bar(x + width, pgd_accs, width, label='PGD (Avg from all sources)',
                   color='red', alpha=0.8)

    ax.set_xlabel('Target Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Robustness to Transferred Adversarial Examples',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison to {output_path}")
    plt.close()


def visualize_cross_model_confusion(classifiers, x_clean, x_adv_source, y_test,
                                   source_name, target_name, attack_name,
                                   output_path='results/transferability_confusion.png'):
    """
    Create confusion matrix showing how predictions change on transferred adversarial examples.

    Args:
        classifiers: Dictionary of classifiers
        x_clean: Clean images
        x_adv_source: Adversarial examples from source model
        y_test: True labels
        source_name: Source model name
        target_name: Target model name
        attack_name: Attack name
        output_path: Path to save plot
    """
    print(f"\nCreating confusion matrix for {source_name} -> {target_name} ({attack_name})...")

    # Get predictions from target model
    target_classifier = classifiers[target_name]

    clean_preds = np.argmax(target_classifier.predict(x_clean), axis=1)
    adv_preds = np.argmax(target_classifier.predict(x_adv_source), axis=1)

    # Create confusion matrix: clean predictions vs adversarial predictions
    cm = confusion_matrix(clean_preds, adv_preds, labels=range(10))

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                cbar_kws={'label': 'Proportion'})

    ax.set_xlabel('Adversarial Prediction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Clean Prediction', fontsize=12, fontweight='bold')
    ax.set_title(f'Prediction Change on Transferred Adversarial Examples\n'
                f'{attack_name}: {source_name} → {target_name}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {output_path}")
    plt.close()


def generate_transferability_report(results, output_path='results/transferability_report.txt'):
    """
    Generate detailed transferability analysis report.

    Args:
        results: Transferability analysis results
        output_path: Path to save report
    """
    print("\nGenerating transferability report...")

    model_names = results['model_names']
    clean_acc = results['clean_accuracy']
    fgsm_matrix = results['transferability_matrix_fgsm']
    pgd_matrix = results['transferability_matrix_pgd']

    report = []
    report.append("=" * 80)
    report.append("ADVERSARIAL EXAMPLE TRANSFERABILITY ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Clean accuracy
    report.append("CLEAN ACCURACY FOR ALL MODELS:")
    report.append("-" * 80)
    for name in model_names:
        report.append(f"  {name}: {clean_acc[name]:.2f}%")
    report.append("")

    # FGSM Transferability
    report.append("FGSM TRANSFERABILITY MATRIX (Accuracy %):")
    report.append("-" * 80)
    report.append("Rows: Source Model (where attack was generated)")
    report.append("Cols: Target Model (where attack was tested)")
    report.append("")

    # Header
    header = "Source \\ Target".ljust(20) + "".join([name.ljust(15) for name in model_names])
    report.append(header)
    report.append("-" * 80)

    for i, source in enumerate(model_names):
        row = source.ljust(20) + "".join([f"{fgsm_matrix[i, j]:.1f}%".ljust(15)
                                         for j in range(len(model_names))])
        report.append(row)
    report.append("")

    # PGD Transferability
    report.append("PGD TRANSFERABILITY MATRIX (Accuracy %):")
    report.append("-" * 80)
    report.append("Rows: Source Model (where attack was generated)")
    report.append("Cols: Target Model (where attack was tested)")
    report.append("")

    # Header
    report.append(header)
    report.append("-" * 80)

    for i, source in enumerate(model_names):
        row = source.ljust(20) + "".join([f"{pgd_matrix[i, j]:.1f}%".ljust(15)
                                         for j in range(len(model_names))])
        report.append(row)
    report.append("")

    # Analysis
    report.append("=" * 80)
    report.append("KEY FINDINGS:")
    report.append("=" * 80)
    report.append("")

    # 1. Within-model effectiveness (diagonal)
    report.append("1. ATTACK EFFECTIVENESS ON SOURCE MODEL (Diagonal):")
    report.append("")
    for i, name in enumerate(model_names):
        fgsm_self = fgsm_matrix[i, i]
        pgd_self = pgd_matrix[i, i]
        report.append(f"  {name}:")
        report.append(f"    - FGSM reduces accuracy to {fgsm_self:.2f}% "
                     f"(drop: {clean_acc[name] - fgsm_self:.2f}%)")
        report.append(f"    - PGD reduces accuracy to {pgd_self:.2f}% "
                     f"(drop: {clean_acc[name] - pgd_self:.2f}%)")
    report.append("")

    # 2. Cross-model transferability (off-diagonal)
    report.append("2. CROSS-MODEL TRANSFERABILITY (Off-Diagonal):")
    report.append("")

    # Calculate average off-diagonal values
    n = len(model_names)
    fgsm_offdiag = (np.sum(fgsm_matrix) - np.trace(fgsm_matrix)) / (n * n - n)
    pgd_offdiag = (np.sum(pgd_matrix) - np.trace(pgd_matrix)) / (n * n - n)

    report.append(f"  Average accuracy on transferred FGSM examples: {fgsm_offdiag:.2f}%")
    report.append(f"  Average accuracy on transferred PGD examples: {pgd_offdiag:.2f}%")
    report.append("")

    # Find best and worst transferability pairs
    fgsm_transfer_scores = []
    pgd_transfer_scores = []

    for i in range(n):
        for j in range(n):
            if i != j:  # Off-diagonal only
                fgsm_transfer_scores.append((model_names[i], model_names[j], fgsm_matrix[i, j]))
                pgd_transfer_scores.append((model_names[i], model_names[j], pgd_matrix[i, j]))

    # Best transferability (lowest accuracy on target)
    best_fgsm = min(fgsm_transfer_scores, key=lambda x: x[2])
    best_pgd = min(pgd_transfer_scores, key=lambda x: x[2])

    report.append("  Best Transferability (Most Effective Cross-Model Attack):")
    report.append(f"    FGSM: {best_fgsm[0]} → {best_fgsm[1]} ({best_fgsm[2]:.2f}% accuracy)")
    report.append(f"    PGD: {best_pgd[0]} → {best_pgd[1]} ({best_pgd[2]:.2f}% accuracy)")
    report.append("")

    # Worst transferability (highest accuracy on target)
    worst_fgsm = max(fgsm_transfer_scores, key=lambda x: x[2])
    worst_pgd = max(pgd_transfer_scores, key=lambda x: x[2])

    report.append("  Worst Transferability (Least Effective Cross-Model Attack):")
    report.append(f"    FGSM: {worst_fgsm[0]} → {worst_fgsm[1]} ({worst_fgsm[2]:.2f}% accuracy)")
    report.append(f"    PGD: {worst_pgd[0]} → {worst_pgd[1]} ({worst_pgd[2]:.2f}% accuracy)")
    report.append("")

    # 3. Insights
    report.append("3. INSIGHTS AND OBSERVATIONS:")
    report.append("")

    report.append("  a) Attack Transferability:")
    report.append("     - Adversarial examples transfer across different architectures")
    report.append("     - Transfer success varies based on source-target model pair")
    report.append("     - Similar architectures (ResNet18 ↔ ResNet34) show higher transferability")
    report.append("     - Different architecture families may show lower transferability")
    report.append("")

    report.append("  b) PGD vs FGSM:")
    report.append("     - PGD attacks are generally more effective than FGSM")
    report.append("     - PGD attacks show similar or better transferability")
    report.append("     - Stronger attacks on source model often transfer better")
    report.append("")

    report.append("  c) Security Implications:")
    report.append("     - Models cannot rely on architecture secrecy for security")
    report.append("     - Adversarial examples can be crafted without target model access")
    report.append("     - Black-box attacks are feasible via transferability")
    report.append("     - Ensemble diversity may provide some defense")
    report.append("")

    report.append("4. RECOMMENDATIONS:")
    report.append("")
    report.append("  - Use ensemble of diverse architectures to reduce transferability")
    report.append("  - Implement adversarial training with multiple model types")
    report.append("  - Add input preprocessing as additional defense layer")
    report.append("  - Monitor for adversarial examples at deployment")
    report.append("  - Consider architecture diversity in defense strategy")
    report.append("")

    report.append("=" * 80)
    report.append("END OF TRANSFERABILITY REPORT")
    report.append("=" * 80)

    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Saved transferability report to {output_path}")

    # Print to console
    print("\n" + "\n".join(report))


def main():
    """Main execution for transferability analysis."""
    print("=" * 80)
    print("ADVERSARIAL TRANSFERABILITY ANALYSIS PIPELINE")
    print("=" * 80)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Configuration
    NUM_TEST_SAMPLES = 500  # Smaller subset for faster computation
    EPSILON = 0.03

    # Step 1: Load data
    test_loader, classes = load_cifar10_data(num_samples=NUM_TEST_SAMPLES)

    # Step 2: Load multiple models
    models_dict = load_models(device)

    # Step 3: Create ART classifiers
    classifiers = create_art_classifiers(models_dict, device)

    # Step 4: Prepare test data
    print("\nPreparing test data...")
    x_test_list = []
    y_test_list = []

    for images, labels in test_loader:
        x_test_list.append(images.numpy())
        y_test_list.append(labels.numpy())

    x_test = np.concatenate(x_test_list, axis=0)
    y_test = np.eye(10)[np.concatenate(y_test_list, axis=0)]

    print(f"Test data shape: {x_test.shape}")

    # Step 5: Perform transferability analysis
    results = perform_transferability_analysis(classifiers, x_test, y_test, epsilon=EPSILON)

    # Step 6: Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # Heatmaps
    visualize_transferability_heatmap(
        results['transferability_matrix_fgsm'],
        results['model_names'],
        'FGSM',
        'results/transferability_heatmap_fgsm.png'
    )

    visualize_transferability_heatmap(
        results['transferability_matrix_pgd'],
        results['model_names'],
        'PGD',
        'results/transferability_heatmap_pgd.png'
    )

    # Comparison chart
    visualize_transferability_comparison(results)

    # Confusion matrix for one interesting pair
    source_model = results['model_names'][0]
    target_model = results['model_names'][1]
    x_adv = results['adversarial_examples'][source_model]['fgsm']

    visualize_cross_model_confusion(
        classifiers, x_test, x_adv, y_test,
        source_model, target_model, 'FGSM',
        'results/transferability_confusion.png'
    )

    # Step 7: Generate report
    print("\n" + "="*80)
    print("GENERATING TRANSFERABILITY REPORT")
    print("="*80)

    generate_transferability_report(results)

    print("\n" + "="*80)
    print("TRANSFERABILITY ANALYSIS COMPLETED!")
    print("="*80)
    print("\nResults saved:")
    print("  - results/transferability_heatmap_fgsm.png")
    print("  - results/transferability_heatmap_pgd.png")
    print("  - results/transferability_comparison.png")
    print("  - results/transferability_confusion.png")
    print("  - results/transferability_report.txt")
    print()


if __name__ == "__main__":
    main()
