# MNIST Autoencoder with TensorFlow/Keras

This repository contains an implementation of a simple autoencoder using TensorFlow/Keras to compress and reconstruct MNIST digits.

## Overview

An autoencoder is a type of neural network that learns to compress data into a lower-dimensional representation and then reconstruct it. This implementation uses:

- Input dimension: 784 (28x28 pixels)
- Bottleneck dimension: 32
- Output dimension: 784 (28x28 pixels)


## Requirements

```shellscript
pip install tensorflow
pip install matplotlib
```

## Code Structure

```python
# Import required libraries
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

# Load and preprocess MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Build the autoencoder
input_layer = Input(shape=(784,))
bottleneck = Dense(32, activation='relu', name='bottleneck')(input_layer)
output = Dense(784, activation='sigmoid', name='output')(bottleneck)

autoencoder = Model(input_layer, output)

# Compile and train
autoencoder.compile(loss='mse', optimizer='adam')
history = autoencoder.fit(X_train, X_train, epochs=20, batch_size=128)
```

## Model Architecture

The autoencoder consists of three main layers:

1. Input Layer: 784 neurons (28x28 flattened image)
2. Bottleneck Layer: 32 neurons (compressed representation)
3. Output Layer: 784 neurons (reconstructed image)


Total parameters: 50,992

- Bottleneck layer: 25,120 parameters (784 × 32 + 32 biases)
- Output layer: 25,872 parameters (32 × 784 + 784 biases)


## Training

The model is trained using:

- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Epochs: 20
- Batch size: 128


## Usage

1. Clone the repository
2. Install dependencies
3. Run the code:


```python
# Train the model
history = autoencoder.fit(X_train, X_train, epochs=20, batch_size=128)

# Get the model summary
autoencoder.summary()
```

## Results

The autoencoder learns to:

- Compress 784-dimensional MNIST digits into 32 dimensions
- Reconstruct the original images from the compressed representation
- Minimize the reconstruction error using MSE loss


## License

MIT License

## Contributing

Feel free to open issues or submit pull requests for improvements.

## Contact
mahdielaimani@gmail.com
