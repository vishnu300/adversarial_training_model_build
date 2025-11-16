"""
Adversarial Robustness Evaluation Pipeline
This script evaluates the adversarial robustness of image classification models using FGSM and PGD attacks via the Adversarial Robustness Toolbox (ART).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool
from art.estimators.classification import PyTorchClassifier


# Create results directory
os.makedirs('results', exist_ok=True)


def load_cifar10_data(batch_size=128, num_samples=1000):
    """
    Load CIFAR-10 dataset with normalization.

    Args:
        batch_size: Batch size for data loader
        num_samples: Number of test samples to use (for faster evaluation)

    Returns:
        train_loader, test_loader, and dataset info
    """
    print("Loading CIFAR-10 dataset...")

    # CIFAR-10 normalization values
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load training and test datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # Use a subset for faster evaluation
    if num_samples < len(test_dataset):
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        test_dataset = Subset(test_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, test_loader, classes


def load_pretrained_model(num_classes=10):
    """
    Load a pre-trained ResNet-18 model for CIFAR-10.

    Args:
        num_classes: Number of output classes

    Returns:
        model: PyTorch model
        device: Device (cuda or cpu)
    """
    print("Loading pre-trained ResNet-18 model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained ResNet-18 and modify for CIFAR-10
    model = torchvision.models.resnet18(pretrained=True)

    # Modify first conv layer for CIFAR-10 (32x32 images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove max pooling

    # Modify final layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)
    model.eval()

    return model, device


def evaluate_model(model, data_loader, device):
    """
    Evaluate model accuracy on a dataset.

    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on

    Returns:
        accuracy: Classification accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def create_art_classifier(model, device):
    """
    Wrap PyTorch model with ART PyTorchClassifier.

    Args:
        model: PyTorch model
        device: Device

    Returns:
        classifier: ART classifier
    """
    # Define loss function and optimizer (not used for inference but required by ART)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create ART classifier
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type='gpu' if device.type == 'cuda' else 'cpu'
    )

    return classifier


def generate_adversarial_examples(classifier, attack_name, x_test, y_test, epsilon=0.03):
    """
    Generate adversarial examples using specified attack.

    Args:
        classifier: ART classifier
        attack_name: Name of attack ('fgsm', 'pgd', or 'deepfool')
        x_test: Test images
        y_test: Test labels
        epsilon: Perturbation magnitude

    Returns:
        x_adv: Adversarial examples
        attack: Attack object (for info)
    """
    print(f"\nGenerating adversarial examples using {attack_name.upper()}...")
    print(f"Epsilon: {epsilon}")

    if attack_name.lower() == 'fgsm':
        attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    elif attack_name.lower() == 'pgd':
        attack = ProjectedGradientDescent(
            estimator=classifier,
            eps=epsilon,
            eps_step=epsilon/10,
            max_iter=40,
            targeted=False
        )
    elif attack_name.lower() == 'deepfool':
        attack = DeepFool(classifier=classifier, max_iter=50, epsilon=1e-6)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    # Generate adversarial examples
    x_adv = attack.generate(x=x_test)

    return x_adv, attack


def evaluate_adversarial_robustness(classifier, x_test, y_test, attack_configs):
    """
    Evaluate model robustness against multiple adversarial attacks.

    Args:
        classifier: ART classifier
        x_test: Clean test images
        y_test: Test labels
        attack_configs: List of attack configurations

    Returns:
        results: Dictionary containing evaluation results
    """
    # Evaluate on clean data
    print("\nEvaluating on clean data...")
    clean_preds = classifier.predict(x_test)
    clean_accuracy = np.mean(np.argmax(clean_preds, axis=1) == np.argmax(y_test, axis=1)) * 100
    print(f"Clean Accuracy: {clean_accuracy:.2f}%")

    results = {
        'clean': {
            'accuracy': clean_accuracy,
            'predictions': clean_preds
        }
    }

    # Evaluate on adversarial data
    for attack_config in attack_configs:
        attack_name = attack_config['name']
        epsilon = attack_config.get('epsilon', 0.03)

        try:
            # Generate adversarial examples
            x_adv, attack = generate_adversarial_examples(
                classifier, attack_name, x_test, y_test, epsilon
            )

            # Evaluate on adversarial examples
            print(f"Evaluating on {attack_name.upper()} adversarial examples...")
            adv_preds = classifier.predict(x_adv)
            adv_accuracy = np.mean(np.argmax(adv_preds, axis=1) == np.argmax(y_test, axis=1)) * 100
            print(f"{attack_name.upper()} Adversarial Accuracy: {adv_accuracy:.2f}%")

            # Calculate perturbation magnitude
            perturbation = np.abs(x_adv - x_test)
            avg_perturbation = np.mean(perturbation)
            max_perturbation = np.max(perturbation)

            results[attack_name] = {
                'accuracy': adv_accuracy,
                'predictions': adv_preds,
                'adversarial_examples': x_adv,
                'avg_perturbation': avg_perturbation,
                'max_perturbation': max_perturbation,
                'epsilon': epsilon
            }

            print(f"Average perturbation: {avg_perturbation:.6f}")
            print(f"Max perturbation: {max_perturbation:.6f}")

        except Exception as e:
            print(f"\nWarning: {attack_name.upper()} attack failed: {str(e)}")
            print(f"Skipping {attack_name.upper()} attack and continuing with others...")
            continue

    return results


def visualize_accuracy_comparison(results, output_path='results/accuracy_comparison.png'):
    """
    Create bar chart comparing accuracy on clean vs adversarial examples.

    Args:
        results: Dictionary containing evaluation results
        output_path: Path to save the plot
    """
    print("\nCreating accuracy comparison plot...")

    attack_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in attack_names]

    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(attack_names)), accuracies, color=['green', 'red', 'orange', 'purple'][:len(attack_names)])

    # Customize plot
    plt.xlabel('Attack Type', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Model Accuracy: Clean vs Adversarial Examples', fontsize=14, fontweight='bold')
    plt.xticks(range(len(attack_names)), [name.upper() for name in attack_names])
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy comparison to {output_path}")
    plt.close()


def visualize_adversarial_examples(x_clean, x_adv_dict, y_true, classes,
                                   num_samples=5, output_path='results/adversarial_examples.png'):
    """
    Visualize original and adversarial examples side by side.

    Args:
        x_clean: Clean images
        x_adv_dict: Dictionary of adversarial examples for each attack
        y_true: True labels
        classes: Class names
        num_samples: Number of samples to visualize
        output_path: Path to save the plot
    """
    print("\nCreating adversarial examples visualization...")

    # Denormalize for visualization
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)

    def denormalize(x):
        x_denorm = x * std + mean
        return np.clip(x_denorm, 0, 1)

    # Select random samples
    indices = np.random.choice(len(x_clean), num_samples, replace=False)

    # Get attack names (excluding 'clean')
    attack_names = [name for name in x_adv_dict.keys() if name != 'clean']

    # Create subplots
    num_cols = len(attack_names) + 1  # +1 for original
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(4*num_cols, 4*num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        # Original image
        img_clean = denormalize(x_clean[idx:idx+1])[0].transpose(1, 2, 0)
        true_label = np.argmax(y_true[idx])

        axes[i, 0].imshow(img_clean)
        axes[i, 0].set_title(f'Original\nTrue: {classes[true_label]}', fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')

        # Adversarial images
        for j, attack_name in enumerate(attack_names):
            x_adv = x_adv_dict[attack_name]
            img_adv = denormalize(x_adv[idx:idx+1])[0].transpose(1, 2, 0)

            # Calculate perturbation
            perturbation = np.abs(img_adv - img_clean)
            avg_pert = np.mean(perturbation)

            axes[i, j+1].imshow(img_adv)
            axes[i, j+1].set_title(f'{attack_name.upper()}\nPert: {avg_pert:.4f}',
                                   fontsize=10, fontweight='bold')
            axes[i, j+1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved adversarial examples to {output_path}")
    plt.close()


def visualize_perturbations(x_clean, x_adv_dict, num_samples=5,
                           output_path='results/perturbation_magnitudes.png'):
    """
    Visualize perturbation magnitudes for different attacks.

    Args:
        x_clean: Clean images
        x_adv_dict: Dictionary of adversarial examples
        num_samples: Number of samples to visualize
        output_path: Path to save the plot
    """
    print("\nCreating perturbation visualization...")

    # Select random samples
    indices = np.random.choice(len(x_clean), num_samples, replace=False)

    # Get attack names
    attack_names = [name for name in x_adv_dict.keys() if name != 'clean']

    # Create subplots
    fig, axes = plt.subplots(num_samples, len(attack_names),
                            figsize=(4*len(attack_names), 4*num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if len(attack_names) == 1:
        axes = axes.reshape(-1, 1)

    for i, idx in enumerate(indices):
        for j, attack_name in enumerate(attack_names):
            x_adv = x_adv_dict[attack_name]

            # Calculate perturbation
            perturbation = np.abs(x_adv[idx] - x_clean[idx])
            perturbation = perturbation.transpose(1, 2, 0)
            perturbation = np.mean(perturbation, axis=2)  # Average across channels

            # Visualize
            im = axes[i, j].imshow(perturbation, cmap='hot')
            axes[i, j].set_title(f'{attack_name.upper()}\nMax: {np.max(perturbation):.4f}',
                               fontsize=10, fontweight='bold')
            axes[i, j].axis('off')
            plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved perturbation magnitudes to {output_path}")
    plt.close()


def generate_analysis_report(results, output_path='results/analysis_report.txt'):
    """
    Generate a detailed analysis report of the robustness evaluation.

    Args:
        results: Dictionary containing evaluation results
        output_path: Path to save the report
    """
    print("\nGenerating analysis report...")

    report = []
    report.append("=" * 80)
    report.append("ADVERSARIAL ROBUSTNESS EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")

    # Clean accuracy
    clean_acc = results['clean']['accuracy']
    report.append(f"Clean Data Accuracy: {clean_acc:.2f}%")
    report.append("")

    # Attack results
    report.append("Adversarial Attack Results:")
    report.append("-" * 80)

    for attack_name, data in results.items():
        if attack_name == 'clean':
            continue

        acc = data['accuracy']
        epsilon = data.get('epsilon', 'N/A')
        avg_pert = data.get('avg_perturbation', 0)
        max_pert = data.get('max_perturbation', 0)

        accuracy_drop = clean_acc - acc

        report.append(f"\n{attack_name.upper()} Attack:")
        report.append(f"  - Epsilon: {epsilon}")
        report.append(f"  - Adversarial Accuracy: {acc:.2f}%")
        report.append(f"  - Accuracy Drop: {accuracy_drop:.2f}%")
        report.append(f"  - Average Perturbation: {avg_pert:.6f}")
        report.append(f"  - Max Perturbation: {max_pert:.6f}")

    report.append("")
    report.append("=" * 80)
    report.append("ANALYSIS AND INSIGHTS")
    report.append("=" * 80)
    report.append("")

    # Analysis
    report.append("1. ATTACK EFFECTIVENESS:")
    report.append("")

    # Find most and least effective attacks
    attack_accs = {name: data['accuracy'] for name, data in results.items() if name != 'clean'}
    most_effective = min(attack_accs, key=attack_accs.get)
    least_effective = max(attack_accs, key=attack_accs.get)

    report.append(f"   - Most effective attack: {most_effective.upper()} "
                 f"(accuracy dropped to {attack_accs[most_effective]:.2f}%)")
    report.append(f"   - Least effective attack: {least_effective.upper()} "
                 f"(accuracy dropped to {attack_accs[least_effective]:.2f}%)")
    report.append("")

    report.append("2. CONCEPTUAL DIFFERENCES BETWEEN ATTACKS:")
    report.append("")
    report.append("   FGSM (Fast Gradient Sign Method):")
    report.append("   - Single-step attack using the sign of the gradient")
    report.append("   - Fast to compute but less sophisticated")
    report.append("   - Perturbation: x_adv = x + ε * sign(∇_x L(θ, x, y))")
    report.append("   - Pros: Very fast, good for adversarial training")
    report.append("   - Cons: Less effective than iterative methods")
    report.append("")

    report.append("   PGD (Projected Gradient Descent):")
    report.append("   - Iterative attack with multiple small steps")
    report.append("   - Projects perturbation back to epsilon ball after each step")
    report.append("   - Considered one of the strongest first-order attacks")
    report.append("   - Pros: More effective, better explores adversarial space")
    report.append("   - Cons: Computationally expensive (multiple iterations)")
    report.append("")

    if 'deepfool' in results:
        report.append("   DeepFool:")
        report.append("   - Finds minimal perturbation to cross decision boundary")
        report.append("   - Uses geometric approach to find optimal direction")
        report.append("   - Iteratively linearizes the classifier")
        report.append("   - Pros: Minimal perturbations, geometrically meaningful")
        report.append("   - Cons: More complex, computationally intensive")
        report.append("")

    report.append("3. OBSERVED VULNERABILITIES:")
    report.append("")

    # Calculate average accuracy drop
    avg_drop = np.mean([clean_acc - data['accuracy'] for name, data in results.items() if name != 'clean'])

    if avg_drop > 40:
        report.append(f"   - The model shows SIGNIFICANT vulnerability to adversarial attacks")
        report.append(f"   - Average accuracy drop: {avg_drop:.2f}%")
        report.append("   - This indicates the model relies heavily on non-robust features")
    elif avg_drop > 20:
        report.append(f"   - The model shows MODERATE vulnerability to adversarial attacks")
        report.append(f"   - Average accuracy drop: {avg_drop:.2f}%")
        report.append("   - There is substantial room for robustness improvement")
    else:
        report.append(f"   - The model shows RELATIVELY LOW vulnerability to adversarial attacks")
        report.append(f"   - Average accuracy drop: {avg_drop:.2f}%")
        report.append("   - The model exhibits some inherent robustness")

    report.append("")
    report.append("4. MITIGATION STRATEGIES:")
    report.append("")
    report.append("   Recommended approaches to improve adversarial robustness:")
    report.append("")
    report.append("   a) Adversarial Training:")
    report.append("      - Train on mix of clean and adversarial examples")
    report.append("      - Use PGD-generated adversarial examples during training")
    report.append("      - Most effective defense but computationally expensive")
    report.append("")
    report.append("   b) Input Preprocessing:")
    report.append("      - Apply transformations (JPEG compression, bit-depth reduction)")
    report.append("      - Use denoising autoencoders")
    report.append("      - Random resizing and padding")
    report.append("")
    report.append("   c) Certified Defenses:")
    report.append("      - Randomized smoothing")
    report.append("      - Provable robustness guarantees within epsilon ball")
    report.append("      - Trade-off between robustness and accuracy")
    report.append("")
    report.append("   d) Ensemble Methods:")
    report.append("      - Use multiple models with different architectures")
    report.append("      - Adversarial examples may not transfer well")
    report.append("      - Increases computational cost")
    report.append("")
    report.append("   e) Detection Methods:")
    report.append("      - Train adversarial example detectors")
    report.append("      - Reject suspicious inputs")
    report.append("      - Can be bypassed by adaptive attacks")
    report.append("")

    report.append("5. RECOMMENDATIONS:")
    report.append("")
    report.append("   Based on the evaluation results:")
    report.append(f"   - Priority: Implement adversarial training with PGD (ε={results.get('pgd', {}).get('epsilon', 0.03)})")
    report.append("   - Consider: Input preprocessing as additional defense layer")
    report.append("   - Evaluate: Trade-off between clean and robust accuracy")
    report.append("   - Monitor: Model performance on diverse adversarial attacks")
    report.append("")

    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Saved analysis report to {output_path}")

    # Also print to console
    print("\n" + "\n".join(report))


def main():
    """Main pipeline execution."""
    print("=" * 80)
    print("ADVERSARIAL ROBUSTNESS EVALUATION PIPELINE")
    print("=" * 80)
    print()

    # Configuration
    NUM_TEST_SAMPLES = 1000  # Use subset for faster evaluation
    BATCH_SIZE = 128
    EPSILON = 0.03  # Perturbation magnitude

    # Step 1: Load dataset
    train_loader, test_loader, classes = load_cifar10_data(
        batch_size=BATCH_SIZE,
        num_samples=NUM_TEST_SAMPLES
    )

    # Step 2: Load pre-trained model
    model, device = load_pretrained_model(num_classes=10)

    # Step 3: Evaluate clean accuracy
    print("\n" + "="*80)
    print("EVALUATING CLEAN ACCURACY")
    print("="*80)
    clean_accuracy = evaluate_model(model, test_loader, device)
    print(f"\nClean Test Accuracy: {clean_accuracy:.2f}%")

    # Step 4: Prepare data for ART
    print("\n" + "="*80)
    print("PREPARING DATA FOR ADVERSARIAL ATTACKS")
    print("="*80)

    # Get test data as numpy arrays
    x_test_list = []
    y_test_list = []

    for images, labels in test_loader:
        x_test_list.append(images.numpy())
        y_test_list.append(labels.numpy())

    x_test = np.concatenate(x_test_list, axis=0)
    y_test_labels = np.concatenate(y_test_list, axis=0)

    # Convert labels to one-hot encoding
    y_test = np.eye(10)[y_test_labels]

    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    # Step 5: Create ART classifier
    classifier = create_art_classifier(model, device)

    # Step 6: Generate and evaluate adversarial examples
    print("\n" + "="*80)
    print("GENERATING AND EVALUATING ADVERSARIAL EXAMPLES")
    print("="*80)

    attack_configs = [
        {'name': 'fgsm', 'epsilon': EPSILON},
        {'name': 'pgd', 'epsilon': EPSILON},
        # Note: DeepFool is optional and may be slow. It will be skipped if it fails.
        # {'name': 'deepfool'},
    ]

    results = evaluate_adversarial_robustness(
        classifier, x_test, y_test, attack_configs
    )

    # Step 7: Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # Accuracy comparison
    visualize_accuracy_comparison(results)

    # Adversarial examples
    x_adv_dict = {name: data.get('adversarial_examples', x_test)
                  for name, data in results.items()}
    visualize_adversarial_examples(x_test, x_adv_dict, y_test, classes)

    # Perturbation magnitudes
    visualize_perturbations(x_test, x_adv_dict)

    # Step 8: Generate analysis report
    print("\n" + "="*80)
    print("GENERATING ANALYSIS REPORT")
    print("="*80)

    generate_analysis_report(results)

    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved in 'results/' directory:")
    print("  - accuracy_comparison.png")
    print("  - adversarial_examples.png")
    print("  - perturbation_magnitudes.png")
    print("  - analysis_report.txt")
    print()


if __name__ == "__main__":
    main()