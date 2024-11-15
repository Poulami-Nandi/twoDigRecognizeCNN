# Two consecutive Digit Recognition Using Convolutional Neural Network (CNN)

In this repo, we solve the Kaggle-MNIST Digit Recognizer competition using Convolutional Neural Networks (CNNs).
This approach leverages the power of CNNs to classify handwritten digit images from the MNIST dataset.
We'll cover data visualization, preprocessing, model building, training, and evaluation.
- We take the MNIST single digit dataset and convert that to two digit dataset by horizontal concatenation and run CNN model on that

## Data Loading
We first load and inspect the training and test datasets.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Training](#training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
The project uses a CNN model built with TensorFlow/Keras to classify digits in the MNIST dataset. This is a standard deep learning task that provides a solid foundation for understanding neural network models applied to image classification.

## Dataset
The MNIST dataset consists of 70,000 grayscale images of handwritten digits, where each image is 28x56 pixels in size:
- **Training set**: 99,000 images
- **Test set**: 20,000 images

Each image is labeled from 0 to 99, corresponding to the digit it represents.

## Model Architecture
The model is a Convolutional Neural Network (CNN) with the following structure:
1. **Input Layer**: Accepts 28x56 grayscale images
2. **Convolutional Layers**: Extracts features from the images
3. **Pooling Layers**: Reduces spatial dimensions
4. **Fully Connected Layers**: Maps features to digit classes
5. **Output Layer**: Uses softmax activation for multiclass classification (digits 0-9)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Poulami-Nandi/twoDigRecognizeCNN.git
   cd twoDigRecognizeCNN

## Results
Final Training Accuracy: 0.9984

Final Validation Accuracy: 0.9804

Final Training Loss: 0.0051

Final Validation Loss: 0.0917
