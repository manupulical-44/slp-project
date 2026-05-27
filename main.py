import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# DATASET
# --------------------------------

# Inputs:
# [Study Hours, Attendance]

X = np.array([
    [1, 20],
    [2, 30],
    [3, 40],
    [4, 50],
    [5, 60],
    [6, 70],
    [7, 80],
    [8, 90]
])

# Outputs:
# 0 = Fail
# 1 = Pass

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# --------------------------------
# NORMALIZATION
# --------------------------------

X = X / np.max(X, axis=0)

# --------------------------------
# INITIALIZE WEIGHTS AND BIAS
# --------------------------------

weights = np.random.randn(2)

bias = np.random.randn()

# --------------------------------
# HYPERPARAMETERS
# --------------------------------

learning_rate = 0.1
epochs = 20

# --------------------------------
# ACTIVATION FUNCTION
# --------------------------------

def activation(z):

    if z >= 0:
        return 1
    else:
        return 0

# --------------------------------
# TRAINING
# --------------------------------

for epoch in range(epochs):

    print(f"\nEpoch {epoch+1}")

    total_error = 0

    for i in range(len(X)):

        # Weighted Sum
        z = np.dot(X[i], weights) + bias

        # Prediction
        prediction = activation(z)

        # Error
        error = y[i] - prediction

        # Update Weights
        weights += learning_rate * error * X[i]

        # Update Bias
        bias += learning_rate * error

        # Count Errors
        total_error += abs(error)

        print(
            f"Input: {X[i]} | "
            f"Prediction: {prediction} | "
            f"Actual: {y[i]}"
        )

    print("Weights:", weights)
    print("Bias:", bias)
    print("Total Error:", total_error)

# --------------------------------
# FINAL PREDICTIONS
# --------------------------------

print("\nFINAL PREDICTIONS\n")

for i in range(len(X)):

    z = np.dot(X[i], weights) + bias

    prediction = activation(z)

    print(
        f"Input: {X[i]} "
        f"Predicted: {prediction}"
    )

# --------------------------------
# VISUALIZATION
# --------------------------------

for i in range(len(X)):

    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], marker='x')

    else:
        plt.scatter(X[i][0], X[i][1], marker='o')

# Decision Boundary

x_values = np.linspace(0, 1, 100)

y_values = -(
    weights[0] * x_values + bias
) / weights[1]

plt.plot(x_values, y_values)

plt.xlabel("Study Hours")
plt.ylabel("Attendance")

plt.title("Single Layer Perceptron")

plt.show()