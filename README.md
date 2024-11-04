1. Importing the necessary libraries
python
Copy code
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
tensorflow is a powerful machine learning library, and we use Keras (built into TensorFlow) to construct the neural network.
datasets module is used to load the MNIST dataset.
layers module provides layers for building neural networks (e.g., convolutional, pooling, dense layers).
models is used to create the sequential model.
matplotlib.pyplot is used for visualizing the results.
2. Loading the MNIST dataset
python
Copy code
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
MNIST is a dataset of 70,000 grayscale images of handwritten digits (0–9) of size 28x28 pixels.
train_images and train_labels: Training data with 60,000 samples.
test_images and test_labels: Test data with 10,000 samples.
3. Preprocessing the data
python
Copy code
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
We reshape the images from (28, 28) to (28, 28, 1) to make them compatible with the CNN input format (28x28 with 1 color channel, since they're grayscale).
We normalize the pixel values by dividing by 255 so that each pixel is in the range [0, 1], which helps the neural network converge faster during training.
4. Building the CNN model
python
Copy code
model = models.Sequential([
    # First convolutional layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third convolutional layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten and fully connected layer
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
Sequential model: This type of model lets us stack layers one after the other.

Convolutional layers: These layers scan the input image and learn filters to detect patterns such as edges and shapes.

Conv2D(32, (3, 3)): 32 filters of size 3x3 are applied to the input image.
activation='relu': ReLU (Rectified Linear Unit) activation function introduces non-linearity.
input_shape=(28, 28, 1): This specifies the input shape for the first layer.
MaxPooling: Reduces the dimensionality of the data by taking the maximum value in each 2x2 block, reducing the image size while retaining important features.

MaxPooling2D((2, 2)): A pooling operation that halves the spatial dimensions.
This structure repeats for the second and third convolutional layers, with increasing filter sizes to learn more complex features.

Flatten: Flattens the 2D output from the convolutional layers into a 1D vector to be fed into the fully connected (dense) layers.

Dense layers:

Dense(64): A fully connected layer with 64 neurons and ReLU activation.
Dense(10): The output layer with 10 neurons (one for each digit 0-9), using softmax activation to predict the probability of each class.
5. Compiling the model
python
Copy code
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
Optimizer: The adam optimizer is an adaptive learning rate optimizer commonly used for deep learning tasks.
Loss function: sparse_categorical_crossentropy is used because we are dealing with multi-class classification, and our labels are integers (not one-hot encoded).
Metrics: We track accuracy during training and evaluation.
6. Training the model
python
Copy code
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)
Training: The model is trained using the training data for 5 epochs (iterations over the entire dataset).
Batch size: We use a batch size of 64, meaning the model processes 64 samples at a time before updating the weights.
Validation split: 20% of the training data is used for validation during training to monitor performance.
7. Evaluating the model
python
Copy code
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')
After training, the model's performance is evaluated on the test set.
model.evaluate returns the test loss and test accuracy.
The test accuracy is printed to show how well the model generalizes to new data.
8. Plotting sample predictions
python
Copy code
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(f"Label: {test_labels[i]}, Predicted: {model.predict_classes(test_images[i].reshape(1, 28, 28, 1))[0]}")
plt.show()
This code block visualizes 25 sample images from the test set.
It shows the actual label of each image and the predicted class by the model.
The imshow function is used to display the images, and model.predict_classes outputs the predicted digit for each image.
Summary:
The code defines a CNN to classify handwritten digits using the MNIST dataset.
The CNN consists of 3 convolutional layers followed by a fully connected network.
After training for 5 epochs, the model achieves over 90% accuracy on the test data.
A visualization of predicted digits is provided to check the model's predictions.
This script is a basic CNN model that provides a strong foundation for more complex image classification tasks.