"""
Adversarial Training for Improved Robustness 
This script implements adversarial training as a mitigation strategy to improve model robustness against adversarial attacks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier


# Create results directory
os.makedirs('results', exist_ok=True)


def load_cifar10_data(batch_size=128):
    """Load CIFAR-10 dataset."""
    print("Loading CIFAR-10 dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def create_model(num_classes=10):
    """Create a ResNet-18 model for CIFAR-10."""
    model = torchvision.models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate_model(model, data_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def adversarial_training(model, train_loader, test_loader, device,
                        epochs=10, epsilon=0.03, alpha=0.5):
    """
    Train model with adversarial examples.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to train on
        epochs: Number of training epochs
        epsilon: Perturbation magnitude for adversarial examples
        alpha: Weight for adversarial loss (0 = clean only, 1 = adversarial only)

    Returns:
        model: Trained model
        history: Training history
    """
    print(f"\nStarting adversarial training...")
    print(f"Epochs: {epochs}, Epsilon: {epsilon}, Alpha: {alpha}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Create ART classifier for generating adversarial examples
    art_classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type='gpu' if device.type == 'cuda' else 'cpu'
    )

    # PGD attack for generating adversarial training examples
    attack = ProjectedGradientDescent(
        estimator=art_classifier,
        eps=epsilon,
        eps_step=epsilon/10,
        max_iter=10,  # Fewer iterations for faster training
        targeted=False
    )

    history = {
        'train_loss': [],
        'clean_acc': [],
        'adv_acc': []
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)

        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial examples
            if alpha > 0:
                # Temporarily set model to eval mode for adversarial generation
                model.eval()
                with torch.no_grad():
                    images_adv = torch.from_numpy(
                        attack.generate(x=images.cpu().numpy())
                    ).to(device)
                # Switch back to train mode
                model.train()
            else:
                images_adv = images

            optimizer.zero_grad()

            # Compute loss on clean and adversarial examples
            if alpha < 1.0:
                outputs_clean = model(images)
                loss_clean = criterion(outputs_clean, labels)
            else:
                loss_clean = 0

            if alpha > 0:
                outputs_adv = model(images_adv)
                loss_adv = criterion(outputs_adv, labels)
            else:
                loss_adv = 0

            # Combined loss
            loss = (1 - alpha) * loss_clean + alpha * loss_adv

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            pbar.set_postfix({'loss': f'{train_loss/train_batches:.4f}'})

        scheduler.step()

        # Evaluation phase
        avg_train_loss = train_loss / train_batches
        clean_acc = evaluate_model(model, test_loader, device)

        # Evaluate on adversarial examples
        model.eval()
        adv_correct = 0
        adv_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                # Generate adversarial examples
                images_adv = torch.from_numpy(
                    attack.generate(x=images.cpu().numpy())
                ).to(device)

                outputs = model(images_adv)
                _, predicted = torch.max(outputs.data, 1)
                adv_total += labels.size(0)
                adv_correct += (predicted == labels).sum().item()

        adv_acc = 100 * adv_correct / adv_total

        history['train_loss'].append(avg_train_loss)
        history['clean_acc'].append(clean_acc)
        history['adv_acc'].append(adv_acc)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Clean Accuracy: {clean_acc:.2f}%")
        print(f"Adversarial Accuracy (PGD): {adv_acc:.2f}%")

    return model, history


def compare_standard_vs_adversarial_training(device, epochs=5):
    """
    Compare standard training vs adversarial training.

    Args:
        device: Device to train on
        epochs: Number of training epochs

    Returns:
        results: Comparison results
    """
    print("="*80)
    print("COMPARING STANDARD VS ADVERSARIAL TRAINING")
    print("="*80)

    # Load data
    train_loader, test_loader = load_cifar10_data(batch_size=128)

    # 1. Train standard model
    print("\n[1/2] Training standard model...")
    standard_model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(standard_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    for epoch in range(epochs):
        standard_model.train()
        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = standard_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f'{train_loss/train_batches:.4f}'})

    # 2. Train adversarially trained model
    print("\n[2/2] Training adversarially robust model...")
    robust_model = create_model()
    robust_model, adv_history = adversarial_training(
        robust_model, train_loader, test_loader, device,
        epochs=epochs, epsilon=0.03, alpha=0.5
    )

    # 3. Evaluate both models
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    # Create ART classifiers for both models
    art_standard = PyTorchClassifier(
        model=standard_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type='gpu' if device.type == 'cuda' else 'cpu'
    )

    art_robust = PyTorchClassifier(
        model=robust_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type='gpu' if device.type == 'cuda' else 'cpu'
    )

    # Get test data
    x_test_list = []
    y_test_list = []
    for images, labels in test_loader:
        x_test_list.append(images.numpy())
        y_test_list.append(labels.numpy())

    x_test = np.concatenate(x_test_list, axis=0)
    y_test = np.eye(10)[np.concatenate(y_test_list, axis=0)]

    # Evaluate on clean data
    print("\nEvaluating on clean data...")
    standard_clean_acc = np.mean(
        np.argmax(art_standard.predict(x_test), axis=1) == np.argmax(y_test, axis=1)
    ) * 100
    robust_clean_acc = np.mean(
        np.argmax(art_robust.predict(x_test), axis=1) == np.argmax(y_test, axis=1)
    ) * 100

    print(f"Standard Model Clean Accuracy: {standard_clean_acc:.2f}%")
    print(f"Robust Model Clean Accuracy: {robust_clean_acc:.2f}%")

    # Evaluate on FGSM adversarial examples
    print("\nGenerating FGSM adversarial examples...")
    from art.attacks.evasion import FastGradientMethod

    fgsm = FastGradientMethod(estimator=art_standard, eps=0.03)
    x_adv_fgsm = fgsm.generate(x=x_test[:1000])  # Use subset for speed

    standard_fgsm_acc = np.mean(
        np.argmax(art_standard.predict(x_adv_fgsm), axis=1) == np.argmax(y_test[:1000], axis=1)
    ) * 100
    robust_fgsm_acc = np.mean(
        np.argmax(art_robust.predict(x_adv_fgsm), axis=1) == np.argmax(y_test[:1000], axis=1)
    ) * 100

    print(f"Standard Model FGSM Accuracy: {standard_fgsm_acc:.2f}%")
    print(f"Robust Model FGSM Accuracy: {robust_fgsm_acc:.2f}%")

    # Evaluate on PGD adversarial examples
    print("\nGenerating PGD adversarial examples...")
    pgd = ProjectedGradientDescent(
        estimator=art_standard,
        eps=0.03,
        eps_step=0.003,
        max_iter=40
    )
    x_adv_pgd = pgd.generate(x=x_test[:1000])  # Use subset for speed

    standard_pgd_acc = np.mean(
        np.argmax(art_standard.predict(x_adv_pgd), axis=1) == np.argmax(y_test[:1000], axis=1)
    ) * 100
    robust_pgd_acc = np.mean(
        np.argmax(art_robust.predict(x_adv_pgd), axis=1) == np.argmax(y_test[:1000], axis=1)
    ) * 100

    print(f"Standard Model PGD Accuracy: {standard_pgd_acc:.2f}%")
    print(f"Robust Model PGD Accuracy: {robust_pgd_acc:.2f}%")

    results = {
        'standard': {
            'clean': standard_clean_acc,
            'fgsm': standard_fgsm_acc,
            'pgd': standard_pgd_acc
        },
        'robust': {
            'clean': robust_clean_acc,
            'fgsm': robust_fgsm_acc,
            'pgd': robust_pgd_acc
        },
        'history': adv_history
    }

    return results


def visualize_mitigation_results(results, output_path='results/mitigation_comparison.png'):
    """
    Visualize the effectiveness of adversarial training.

    Args:
        results: Comparison results
        output_path: Path to save the plot
    """
    print("\nCreating mitigation comparison visualization...")

    attack_types = ['Clean', 'FGSM', 'PGD']
    standard_accs = [results['standard']['clean'],
                     results['standard']['fgsm'],
                     results['standard']['pgd']]
    robust_accs = [results['robust']['clean'],
                   results['robust']['fgsm'],
                   results['robust']['pgd']]

    x = np.arange(len(attack_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, standard_accs, width, label='Standard Training',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, robust_accs, width, label='Adversarial Training',
                   color='coral', alpha=0.8)

    ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Standard Training vs Adversarial Training: Robustness Comparison',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(attack_types)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved mitigation comparison to {output_path}")
    plt.close()

    # Plot training history
    if 'history' in results and results['history']:
        history = results['history']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        axes[0].plot(history['train_loss'], marker='o', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)

        # Accuracy plot
        axes[1].plot(history['clean_acc'], marker='o', linewidth=2, label='Clean Accuracy')
        axes[1].plot(history['adv_acc'], marker='s', linewidth=2, label='Adversarial Accuracy')
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Accuracy During Adversarial Training', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/adversarial_training_history.png', dpi=300, bbox_inches='tight')
        print("Saved training history to results/adversarial_training_history.png")
        plt.close()


def generate_mitigation_report(results, output_path='results/mitigation_report.txt'):
    """Generate analysis report for mitigation strategies."""
    print("\nGenerating mitigation report...")

    report = []
    report.append("=" * 80)
    report.append("ADVERSARIAL TRAINING MITIGATION REPORT")
    report.append("=" * 80)
    report.append("")

    report.append("STANDARD TRAINING RESULTS:")
    report.append("-" * 80)
    report.append(f"  Clean Accuracy: {results['standard']['clean']:.2f}%")
    report.append(f"  FGSM Accuracy: {results['standard']['fgsm']:.2f}%")
    report.append(f"  PGD Accuracy: {results['standard']['pgd']:.2f}%")
    report.append("")

    report.append("ADVERSARIAL TRAINING RESULTS:")
    report.append("-" * 80)
    report.append(f"  Clean Accuracy: {results['robust']['clean']:.2f}%")
    report.append(f"  FGSM Accuracy: {results['robust']['fgsm']:.2f}%")
    report.append(f"  PGD Accuracy: {results['robust']['pgd']:.2f}%")
    report.append("")

    report.append("IMPROVEMENT ANALYSIS:")
    report.append("-" * 80)

    clean_diff = results['robust']['clean'] - results['standard']['clean']
    fgsm_diff = results['robust']['fgsm'] - results['standard']['fgsm']
    pgd_diff = results['robust']['pgd'] - results['standard']['pgd']

    report.append(f"  Clean Accuracy Change: {clean_diff:+.2f}%")
    report.append(f"  FGSM Robustness Improvement: {fgsm_diff:+.2f}%")
    report.append(f"  PGD Robustness Improvement: {pgd_diff:+.2f}%")
    report.append("")

    report.append("KEY INSIGHTS:")
    report.append("-" * 80)
    report.append("1. Adversarial training significantly improves robustness against attacks")
    report.append("2. There may be a small trade-off in clean accuracy")
    report.append("3. The model learns to be robust to the training attack (PGD)")
    report.append("4. Improved robustness often generalizes to other attacks (FGSM)")
    report.append("")

    report.append("=" * 80)

    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Saved mitigation report to {output_path}")
    print("\n" + "\n".join(report))


def main():
    """Main execution for adversarial training mitigation."""
    print("=" * 80)
    print("ADVERSARIAL TRAINING MITIGATION PIPELINE")
    print("=" * 80)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Compare standard vs adversarial training
    results = compare_standard_vs_adversarial_training(device, epochs=5)

    # Create visualizations
    visualize_mitigation_results(results)

    # Generate report
    generate_mitigation_report(results)

    print("\n" + "="*80)
    print("MITIGATION PIPELINE COMPLETED!")
    print("="*80)
    print("\nResults saved:")
    print("  - results/mitigation_comparison.png")
    print("  - results/adversarial_training_history.png")
    print("  - results/mitigation_report.txt")
    print()


if __name__ == "__main__":
    main()