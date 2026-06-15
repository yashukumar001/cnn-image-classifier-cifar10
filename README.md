# CNN Image Classifier using PyTorch

## Overview

This project implements a Convolutional Neural Network (CNN) from scratch using PyTorch to classify images from the CIFAR-10 dataset. The model learns visual features through convolution and pooling operations and predicts one of ten object categories.

The project demonstrates fundamental deep learning concepts including convolutional layers, activation functions, pooling, forward propagation, backpropagation, optimization, and model evaluation.

---

## Dataset

Dataset: CIFAR-10

The CIFAR-10 dataset contains:

* 60,000 color images
* Image size: 32 × 32 pixels
* 10 classes
* 50,000 training images
* 10,000 testing images

Classes:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

The dataset is automatically downloaded using Torchvision.

---

## Project Structure

```text
cnn-image-classifier/
│
├── model.py
├── train.py
├── requirements.txt
├── README.md
├── cnn_model.pth
└── data/
```

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib

---

## Model Architecture

The CNN consists of:

### Convolution Layer 1

* Input Channels: 3 (RGB)
* Output Channels: 32
* Kernel Size: 3×3
* ReLU Activation
* Max Pooling

### Convolution Layer 2

* Input Channels: 32
* Output Channels: 64
* Kernel Size: 3×3
* ReLU Activation
* Max Pooling

### Fully Connected Layers

* Flatten Layer
* Dense Layer (128 neurons)
* Output Layer (10 classes)

---

## Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Optimizer     | Adam             |
| Loss Function | CrossEntropyLoss |
| Learning Rate | 0.001            |
| Batch Size    | 64               |
| Epochs        | 5                |
| Dataset       | CIFAR-10         |

---

## Results

### Test Accuracy

**68.60%**

The model achieved 68.60% accuracy on the CIFAR-10 test dataset after 5 training epochs.

### Training Loss

| Epoch | Loss    |
| ----- | ------- |
| 1     | 1193.49 |
| 2     | 894.90  |
| 3     | 778.68  |
| 4     | 706.03  |
| 5     | 647.95  |

The decreasing loss demonstrates successful learning and convergence during training.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/cnn-image-classifier-cifar10.git
cd cnn-image-classifier-cifar10
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Train the model:

```bash
python train.py
```

The script will:

1. Download CIFAR-10
2. Train the CNN
3. Evaluate performance
4. Save the trained model

Output example:

```text
Epoch 1/5 Loss: 1193.4914
Epoch 2/5 Loss: 894.9049
Epoch 3/5 Loss: 778.6752
Epoch 4/5 Loss: 706.0250
Epoch 5/5 Loss: 647.9458

Test Accuracy: 68.60%
Model saved.
```

---

## Key Concepts Learned

### Convolutional Neural Networks (CNNs)

CNNs are specialized neural networks designed for image processing tasks. They automatically learn hierarchical visual features such as edges, textures, shapes, and objects.

### ReLU Activation

Introduces non-linearity into the network and helps improve training efficiency.

### Max Pooling

Reduces spatial dimensions while preserving important features, lowering computational requirements.

### Cross Entropy Loss

Measures prediction error for multi-class classification problems.

### Adam Optimizer

Adaptive optimization algorithm that updates model parameters efficiently during training.

---

## Challenges Encountered

* Understanding image tensor dimensions
* Designing a suitable CNN architecture
* Managing overfitting and underfitting
* Selecting appropriate hyperparameters
* Interpreting model performance metrics

---

## Future Improvements

* Increase training epochs
* Add Batch Normalization
* Add Dropout layers
* Implement Data Augmentation
* Experiment with deeper architectures
* Improve accuracy beyond 75%
* Add confusion matrix visualization
* Deploy as a web application

---

## Author

Yashu Kumar

B.Tech Computer Science Engineering

Interested in Artificial Intelligence, Machine Learning, Deep Learning, Computer Vision, and Software Engineering.

---

## License

This project is intended for educational and portfolio purposes.
