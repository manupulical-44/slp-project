import numpy as np

# XOR Dataset

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

# Initialize weights and bias

weights = np.random.randn(2)
bias = np.random.randn()

learning_rate = 0.1
epochs = 20

# Activation function

def activation(z):

    if z >= 0:
        return 1
    else:
        return 0

# Training

for epoch in range(epochs):

    total_error = 0

    for i in range(len(X)):

        z = np.dot(X[i], weights) + bias

        prediction = activation(z)

        error = y[i] - prediction

        weights += learning_rate * error * X[i]

        bias += learning_rate * error

        total_error += abs(error)

    print(f"Epoch {epoch+1} Error: {total_error}")

# Final Predictions

print("\nFinal Predictions:\n")

for i in range(len(X)):

    z = np.dot(X[i], weights) + bias

    prediction = activation(z)

    print(
        f"Input: {X[i]} "
        f"Predicted: {prediction} "
        f"Actual: {y[i]}"
    )