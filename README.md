# Deepfake Detection System

This repository contains a **Deepfake Detection System** developed using a Convolutional Neural Network (CNN) model. The system is trained on the **FaceForensics++ dataset**, a benchmark dataset for detecting manipulated media. It achieves **100% accuracy** in detecting fake images, showcasing its robustness and effectiveness.

## Project Overview

Deepfakes are synthetic media where a person in an image or video is replaced with someone else's likeness, often for malicious purposes. This project addresses the growing concern of deepfake media by providing an accurate and efficient detection system. 

### Key Features:
- **Dataset**: Utilizes the FaceForensics++ dataset, containing real and fake images.
- **Model**: A custom CNN designed to classify images as either *real* or *fake*.
- **Performance**: Achieved a test accuracy of 100% and a test loss of 0.0000.
- **Functionality**: Includes preprocessing of images, training the model, and real-time prediction capabilities.

---

## Dataset

The system uses the **FaceForensics++ dataset**, which contains folders with images categorized into:
- **Real**: Genuine, unaltered images.
- **Fake**: Images manipulated using deepfake generation techniques.

To download the dataset, visit [FaceForensics++](https://www.kaggle.com/datasets/greatgamedota/faceforensics).

---

## System Architecture

The project follows these steps:

1. **Dataset Preprocessing**:
   - Resizes images to a consistent dimension (64x64).
   - Normalizes image pixel values to the range [0, 1].
   - Splits the dataset into training (80%) and testing (20%) sets.

2. **Model Architecture**:
   - Built using TensorFlow/Keras.
   - Includes convolutional, pooling, and fully connected layers with dropout for regularization.
   - Binary classification output (real or fake).

3. **Training**:
   - Trains the CNN for 10 epochs using the Adam optimizer.
   - Monitors performance with training and validation accuracy/loss.

4. **Evaluation**:
   - Evaluates the model on the test set, achieving exceptional results.
   - Visualizes training history with accuracy and loss plots.

5. **Prediction**:
   - Accepts new image inputs for classification as *real* or *fake* with confidence scores.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmdmohamedd/Deepfake-Detection-System.git
   cd Deepfake-Detection-System
   ```

2. Install the required Python libraries:
   ```bash
   pip install tensorflow opencv-python numpy matplotlib scikit-learn
   ```

3. Update the `dataset_path` variable in the script with the path to your dataset.

---

## Usage

1. **Train the Model**:
   Run the provided notebook to preprocess the dataset, train the model, and evaluate its performance.

2. **Make Predictions**:
   Use the `predict_deepfake` function to classify new images:
   ```python
   label, confidence = predict_deepfake(model, '/path/to/image.jpg')
   print(f"Prediction: {label}, Confidence: {confidence[0][0]:.4f}")
   ```

3. **Visualize Results**:
   Visualize sample predictions and training history directly in the notebook.

---

## Results

- **Test Accuracy**: 100.00%
- **Test Loss**: 0.0000
- Successfully detected a fake image during evaluation.

---

## Future Work

- Extend the system to work with video input by analyzing multiple frames.
- Test on additional datasets like Celeb-DF and DFDC for broader validation.
- Explore advanced architectures like XceptionNet for improved generalization.

---

## Acknowledgments

- **FaceForensics++ Dataset**: [GitHub Repository](https://github.com/ondyari/FaceForensics)
- TensorFlow and Keras for deep learning tools.
- The open-source community for providing essential resources and datasets. 

--- 
