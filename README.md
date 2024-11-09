
# Cat vs Dog Classifier using Deep Convolutional Neural Networks

## Overview

This project focuses on building and training a deep convolutional neural network (CNN) model to classify images as either cats or dogs. The project utilizes the Keras library with a TensorFlow backend, working with a dataset containing 25,000 labeled images of cats and dogs.

## Project Objective

The primary goal is to create a CNN model with a high level of accuracy (target precision: 90-95%) to reliably distinguish between cat and dog images. The resulting classifier can be used as a feature to enhance engagement in a product by allowing users to upload images and receive predictions on whether they resemble a "cat" or a "dog."

## Dataset

The dataset consists of:
- 25,000 images in total
  - 12,500 images of cats
  - 12,500 images of dogs

The dataset is split into training, validation, and test sets to evaluate model performance accurately. The images are preprocessed to 150x150 pixels to match the input requirements of the CNN model.

## Model Architecture

The CNN model was built using Keras and consists of the following key layers:
1. Convolutional and MaxPooling layers for feature extraction
2. Batch Normalization to stabilize learning
3. Dropout layers to reduce overfitting
4. Fully connected layers to classify images

The final output layer uses a sigmoid activation for binary classification (cat or dog).

## Training Process

Training included:
1. Data augmentation (rotation, shift, zoom, and flip) to increase dataset diversity.
2. Using callbacks like Early Stopping and ReduceLROnPlateau to improve training efficiency.
3. The Adam optimizer with a custom learning rate for effective optimization.

The model was trained over 30 epochs, with early stopping based on validation loss.

## Results

The model achieved:
- **Training accuracy:** ~92%
- **Validation accuracy:** ~93%
- **Test accuracy:** ~93%

Confusion matrix and classification reports confirmed strong performance in both precision and recall across classes.

## Evaluation and Visualization

Several functions were implemented for evaluation and visualization:
- **Accuracy and loss plots** for both training and validation sets.
- **Confusion matrix and classification reports** to summarize performance metrics.
- **Visualizations of correctly and incorrectly classified images** to understand model predictions better.

## How to Run the Model

1. Clone the repository.
2. Download and unzip the dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).
3. Run the Jupyter Notebook file , train, and evaluate the model.

## Conclusion

The Cat vs Dog classifier successfully classifies images with high accuracy and could be deployed as a feature in a product to engage users. The project demonstrates the effectiveness of deep CNNs for image classification tasks and provides a framework for further enhancements, such as using transfer learning or fine-tuning pre-trained models.

--- 
