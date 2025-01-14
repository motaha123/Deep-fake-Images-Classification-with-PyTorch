import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedKFold
import optuna
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm


# =============================================================================
# BLOCK 1: get_model
#   - Supports alexnet, resnet50, densenet121, mobilenet_v2
# =============================================================================
def get_model(model_name='alexnet', num_classes=2):
    """
    Returns a pretrained model with its final classification layer
    adjusted for the given number of classes.

    Supported:
      - alexnet
      - resnet50
      - densenet121
      - mobilenet_v2
    """
    print(f"Initializing model: {model_name}")

    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    return model


# =============================================================================
# BLOCK 2: Get data loader from the dataset folder
# =============================================================================
def get_data_loader(data_dir, batch_size=32):
    """
    Loads images using ImageFolder from a directory structure:
      data_dir/
        class_0/
        class_1/
        ...
    Returns a DataLoader and the class names.
    """
    print(f"Loading data from: {data_dir}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset.classes


# =============================================================================
# BLOCK 3: Prepare dataset for k-fold cross-validation (optional usage)
# =============================================================================
def prepare_dataset(dataset_path, output_path, num_folds=3):
    """
    Copies data from 'dataset_path/Train' into multiple folds for cross-validation.
    """
    print("\nPreparing dataset with 3-fold cross-validation...")
    dataset = datasets.ImageFolder(os.path.join(dataset_path, 'Train'))
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset.samples, dataset.targets)):
        fold_path = os.path.join(output_path, f"fold_{fold + 1}")
        os.makedirs(fold_path, exist_ok=True)

        train_path = os.path.join(fold_path, "Train")
        val_path   = os.path.join(fold_path, "Validation")
        test_path  = os.path.join(fold_path, "Test")

        for target_path in [train_path, val_path, test_path]:
            os.makedirs(target_path, exist_ok=True)

        # Split
        train_samples = [dataset.samples[i] for i in train_idx]
        test_samples  = [dataset.samples[i] for i in test_idx]
        split_idx = int(len(train_samples) * 0.8)
        train_samples, val_samples = train_samples[:split_idx], train_samples[split_idx:]

        # Copy files
        for sample_list, dest in [
            (train_samples, train_path),
            (val_samples,   val_path),
            (test_samples,  test_path),
        ]:
            for img_path, label in tqdm(sample_list, desc=f"Processing Fold {fold + 1}", leave=False):
                class_dir = os.path.join(dest, dataset.classes[label])
                os.makedirs(class_dir, exist_ok=True)
                shutil.copy(img_path, class_dir)

        print(f"Fold {fold + 1}: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples prepared.")


# =============================================================================
# BLOCK 4: Training and evaluation function (prints performance measures)
# =============================================================================
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device='cpu'):
    """
    Train the model for 1 epoch (as per original code) 
    and then evaluate on the validation set.

    Prints:
      - Average Training Loss
      - Validation Loss
      - Validation Accuracy
      - Confusion Matrix

    Returns (val_loss, accuracy, confusion_matrix).
    """
    print(f"\nTraining and evaluating on {device}...\n")
    model.to(device)

    # Training phase
    model.train()
    total_train_loss = 0
    print("\nTraining started...")
    for epoch in range(1):  # original code trains for 1 epoch
        print(f"\nEpoch {epoch + 1}...")
        for images, labels in tqdm(train_loader, desc="Training Progress"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

    # Evaluation phase
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    print("\nEvaluating on validation set...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation Progress"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    print(f"\nValidation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    return avg_val_loss, accuracy, cm


# =============================================================================
# BLOCK 5: Fine-tuning helpers (freeze/unfreeze layers)
# =============================================================================
def freeze_all_layers(model):
    """Freeze all parameters in the entire model."""
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_layers(model, layers_to_unfreeze):
    """
    Unfreeze layers in 'layers_to_unfreeze'. 
    Typically you match partial layer names (like 'layer4', 'fc' for ResNet).
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layers_to_unfreeze):
            param.requires_grad = True

def get_fine_tuned_model(model_name: str, num_classes: int = 2, layers_to_unfreeze=None):
    """
    Returns a model where all layers are frozen except for the ones 
    specified in 'layers_to_unfreeze'.
    """
    if layers_to_unfreeze is None:
        layers_to_unfreeze = []

    model = get_model(model_name, num_classes)
    freeze_all_layers(model)
    unfreeze_layers(model, layers_to_unfreeze)

    return model


# =============================================================================
# BLOCK 6: Adding an Extra Convolutional Layer to Each Model
# =============================================================================
class AlexNetWithExtraConv(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.classifier[6] = nn.Linear(self.alexnet.classifier[6].in_features, num_classes)

        self.features = self.alexnet.features
        self.extra_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.classifier = self.alexnet.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.extra_conv(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNetWithExtraConv(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        self.extra_conv = nn.Conv2d(2048, 2048, kernel_size=3, padding=1)
        self.avgpool = self.resnet.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.extra_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DenseNetWithExtraConv(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.densenet = models.densenet121(pretrained=True)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()

        self.features = self.densenet.features
        self.extra_conv = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.extra_conv(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MobileNetWithExtraConv(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.features = self.mobilenet.features
        self.mobilenet.classifier[1] = nn.Identity()

        self.extra_conv = nn.Conv2d(1280, 1280, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.extra_conv(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# =============================================================================
# BLOCK 7: Vision Transformer (Requires torchvision >= 0.13)
# =============================================================================
def get_pretrained_vit(num_classes=2):
    """
    Returns a Vision Transformer (vit_b_16) pretrained on ImageNet, 
    with a new linear head for num_classes.
    """
    vit_model = models.vit_b_16(pretrained=True)
    in_features = vit_model.heads.head.in_features
    vit_model.heads.head = nn.Linear(in_features, num_classes)
    return vit_model


# =============================================================================
# BLOCK 8: Hyperparameter Optimization (Optuna)
#   - Chooses among the 4 models and loguniform lr
# =============================================================================
def objective(trial):
    """
    Optuna objective function:
      1) Picks a model among {alexnet, resnet50, densenet121, mobilenet_v2}.
      2) Picks a learning rate via loguniform.
      3) Trains for 1 epoch, evaluates on validation.
      4) Returns validation accuracy to maximize.
    """
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    model_name = trial.suggest_categorical('model', [
        'alexnet', 'resnet50', 'densenet121', 'mobilenet_v2'
    ])

    model = get_model(model_name=model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\nHyperparameter optimization started...")
    train_loader, _ = get_data_loader("dataset/Train", batch_size=32)
    val_loader, _   = get_data_loader("dataset/Validation", batch_size=32)

    _, accuracy, _ = train_and_evaluate(
        model, train_loader, val_loader, 
        criterion, optimizer, device='cpu'
    )
    return accuracy

def optimize_hyperparameters():
    """
    Runs the Optuna study to maximize accuracy. 
    We let it run for n_trials, returning the best (lr, model) combination.
    """
    print("\nOptimizing hyperparameters with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)  # increase if you want a deeper search
    optimal_lr = study.best_params['lr']
    optimal_model = study.best_params['model']
    print(f"Optimal Learning Rate: {optimal_lr}, Optimal Model: {optimal_model}")
    return optimal_lr, optimal_model


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Prepare dataset for cross-validation
    dataset_path = "dataset"
    output_path = "output_dataset"
    prepare_dataset(dataset_path, output_path, num_folds=3)

    # --- 1) OPTUNA: find best (model, lr) across AlexNet, ResNet50, DenseNet121, MobileNetV2
    optimal_lr, optimal_model = optimize_hyperparameters()
    print(f"\nUsing optimal model: {optimal_model} with learning rate: {optimal_lr}")

    # Instantiate best model
    model = get_model(model_name=optimal_model, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=optimal_lr)

    # Load data
    train_loader, _ = get_data_loader("dataset/Train", batch_size=32)
    val_loader, _   = get_data_loader("dataset/Validation", batch_size=32)

    # Train & evaluate final
    train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device='cpu')

    # -------------------------------------------------------------------------
    # EXAMPLES OF TRANSFER LEARNING (WITHOUT EXTRA CONV) AND EXTRA CONV MODELS
    # -------------------------------------------------------------------------
    print("\n--- Transfer Learning Example: ResNet-50 ---")
    model_resnet = get_model('resnet50', num_classes=2)
    opt_resnet = optim.Adam(model_resnet.parameters(), lr=1e-3)
    train_and_evaluate(model_resnet, train_loader, val_loader, criterion, opt_resnet, device='cpu')

    # -------------------------------------------------------------------------
    # FINE-TUNING THE LAST LAYERS FOR ALL FOUR MODELS
    # -------------------------------------------------------------------------
    print("\n--- Fine-tuning the last layers of AlexNet ---")
    # Unfreeze the final linear layer (classifier.6). 
    # Add more layers if you want to unfreeze more of the classifier.
    model_ft_alex = get_fine_tuned_model(
        'alexnet', 
        num_classes=2, 
        layers_to_unfreeze=['classifier.6']
    )
    opt_ft_alex = optim.Adam(
        filter(lambda p: p.requires_grad, model_ft_alex.parameters()), 
        lr=1e-4
    )
    train_and_evaluate(model_ft_alex, train_loader, val_loader, criterion, opt_ft_alex, device='cpu')

    print("\n--- Fine-tuning the last layers of ResNet-50 ---")
    # Already in your code, but now it's in the same block as the others
    model_ft_res = get_fine_tuned_model(
        'resnet50', 
        num_classes=2, 
        layers_to_unfreeze=['layer4', 'fc']
    )
    opt_ft_res = optim.Adam(
        filter(lambda p: p.requires_grad, model_ft_res.parameters()), 
        lr=1e-4
    )
    train_and_evaluate(model_ft_res, train_loader, val_loader, criterion, opt_ft_res, device='cpu')

    print("\n--- Fine-tuning the last layers of DenseNet121 ---")
    # Typically unfreeze 'denseblock4' + 'classifier' for final portion
    model_ft_den = get_fine_tuned_model(
        'densenet121', 
        num_classes=2, 
        layers_to_unfreeze=['denseblock4', 'classifier']
    )
    opt_ft_den = optim.Adam(
        filter(lambda p: p.requires_grad, model_ft_den.parameters()), 
        lr=1e-4
    )
    train_and_evaluate(model_ft_den, train_loader, val_loader, criterion, opt_ft_den, device='cpu')

    print("\n--- Fine-tuning the last layers of MobileNetV2 ---")
    # Typically final blocks around 'features.18' / 'features.19' & the final classifier
    model_ft_mob = get_fine_tuned_model(
        'mobilenet_v2', 
        num_classes=2, 
        layers_to_unfreeze=['features.18', 'classifier.1']
    )
    opt_ft_mob = optim.Adam(
        filter(lambda p: p.requires_grad, model_ft_mob.parameters()), 
        lr=1e-4
    )
    train_and_evaluate(model_ft_mob, train_loader, val_loader, criterion, opt_ft_mob, device='cpu')

    # -------------------------------------------------------------------------
    # EXTRA CONVOLUTION MODELS
    # -------------------------------------------------------------------------
    print("\n--- Extra Conv: AlexNet ---")
    model_alex_extra = AlexNetWithExtraConv(num_classes=2)
    opt_alex_extra = optim.Adam(model_alex_extra.parameters(), lr=1e-4)
    train_and_evaluate(model_alex_extra, train_loader, val_loader, criterion, opt_alex_extra, device='cpu')

    print("\n--- Extra Conv: ResNet-50 ---")
    model_res_extra = ResNetWithExtraConv(num_classes=2)
    opt_res_extra = optim.Adam(model_res_extra.parameters(), lr=1e-4)
    train_and_evaluate(model_res_extra, train_loader, val_loader, criterion, opt_res_extra, device='cpu')

    print("\n--- Extra Conv: DenseNet-121 ---")
    model_den_extra = DenseNetWithExtraConv(num_classes=2)
    opt_den_extra = optim.Adam(model_den_extra.parameters(), lr=1e-4)
    train_and_evaluate(model_den_extra, train_loader, val_loader, criterion, opt_den_extra, device='cpu')

    print("\n--- Extra Conv: MobileNetV2 ---")
    model_mob_extra = MobileNetWithExtraConv(num_classes=2)
    opt_mob_extra = optim.Adam(model_mob_extra.parameters(), lr=1e-4)
    train_and_evaluate(model_mob_extra, train_loader, val_loader, criterion, opt_mob_extra, device='cpu')

    # -------------------------------------------------------------------------
    # VISION TRANSFORMER (OPTIONAL)
    # -------------------------------------------------------------------------
    print("\n--- Vision Transformer ---")
    vit_model = get_pretrained_vit(num_classes=2)
    opt_vit = optim.Adam(vit_model.parameters(), lr=1e-4)
    train_and_evaluate(vit_model, train_loader, val_loader, criterion, opt_vit, device='cpu')