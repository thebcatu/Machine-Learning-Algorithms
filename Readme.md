# Handwritten Digit Classification using ANN 🧠

This project implements a **Neural Network (ANN)** using **TensorFlow/Keras** to classify **handwritten digits** from the **Digits dataset**.

## 📌 Features
- Uses **Artificial Neural Network (ANN)**
- **Digits dataset** (8x8 grayscale images)
- **StandardScaler** for feature scaling
- **Categorical Cross-Entropy** for multi-class classification
- **Adam Optimizer** for efficient training
- Predicts digit (0-9) with high accuracy

## 🚀 Installation
1. Install dependencies:
   ```bash
   pip install numpy scikit-learn tensorflow
   ```
2. Run the script:
   ```bash
   python neural_network.py
   ```

## 📊 Model Architecture
- Input Layer: 64 neurons (flattened images)
- Hidden Layers: 128, 64 neurons (ReLU activation)
- Output Layer: 10 neurons (Softmax activation)

## 🏆 Results
The trained model achieves high accuracy on the test dataset.

## 🔮 Future Improvements
- Implement Convolutional Neural Networks (CNN)
- Train on a larger dataset like MNIST
- Optimize hyperparameters for better performance

🔹 Happy Coding! 🚀