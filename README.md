# Deep-fake-Images-Classification-with-PyTorch
This repository contains a comprehensive implementation of deep fake models for image classification using PyTorch.
The project supports several popular architectures, including AlexNet, ResNet50, DenseNet121, MobileNetV2, and Vision Transformer (ViT). It also includes utilities for hyperparameter optimization, fine-tuning, and cross-validation.

Features

Model Support:
AlexNet
ResNet50
DenseNet121
MobileNetV2
Vision Transformer (ViT)

Data Loading:
Loads images from a directory structure using torchvision.datasets.ImageFolder.
Supports data augmentation and transformation.

Cross-Validation:
Prepares the dataset for k-fold cross-validation.
Splits data into training, validation, and test sets.

Training and Evaluation:
Trains models for a specified number of epochs.
Evaluates models on validation data, providing loss, accuracy, and confusion matrix.

Fine-Tuning:
Freezes and unfreezes specific layers for fine-tuning.
Supports fine-tuning of the last layers of the model.

Hyperparameter Optimization:
Uses Optuna for hyperparameter tuning.
Optimizes learning rate and model selection.

Extra Convolutional Layers:
Adds an extra convolutional layer to each model for enhanced feature extraction.
