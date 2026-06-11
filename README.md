# Binary Image Classification using CNN

A 3-layer Convolutional Neural Network (CNN) built from scratch in PyTorch for binary image classification (cats vs. dogs).

## Overview

This project implements a custom CNN architecture trained on the PetImages dataset. The model pipeline includes data preprocessing, convolutional feature extraction, and fully connected classification layers, evaluated on a held-out test set.

## Tech Stack

- Python
- PyTorch
- torchvision
- scikit-learn

## Model Architecture

- 3 convolutional layers with ReLU activations and max pooling
- 3 fully connected layers
- CrossEntropyLoss with Adam optimizer
- 80/20 train/test split
- Input images resized to 100x100 and normalized

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 72.75% |
| Precision | 73.4% |
| Recall | 71.2% |

## How to Run

```bash
pip install torch torchvision scikit-learn
python cnn.py
```

> **Note:** The PetImages dataset is required to run this project. Download it from [Microsoft's Kaggle dataset](https://www.kaggle.com/c/dogs-vs-cats/data) and place it in a `./petimages` folder.

## Author

Aleah Hassabo 
