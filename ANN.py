import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import os

# Define the paths for the dataset
train_csv_path = "datasets/mnist/train.csv"
test_csv_path = "datasets/mnist/test.csv"

# Check if the dataset files exist
if not os.path.exists(train_csv_path):
    raise FileNotFoundError(f"Training dataset not found at {train_csv_path}")
if not os.path.exists(test_csv_path):
    raise FileNotFoundError(f"Test dataset not found at {test_csv_path}")

# Load data
mnist_train = pd.read_csv(train_csv_path)
mnist_test = pd.read_csv(test_csv_path)

# Visualize the image represented by the first row of the train data and the test data
train_data_digit1 = np.asarray(mnist_train.iloc[0, 1:]).reshape(28, 28)
test_data_digit1 = np.asarray(mnist_test.iloc[0, :]).reshape(28, 28)

# Plotting the first digit in the train data
plt.subplot(1, 2, 1)
plt.imshow(train_data_digit1, cmap=plt.cm.gray_r)
plt.title("First digit in train data")

# Plotting the first digit in the test data
plt.subplot(1, 2, 2)
plt.imshow(test_data_digit1, cmap=plt.cm.gray_r)
plt.title("First digit in test data")

plt.show()

# Assign the label column value to Y_train and the remaining column values to X_train
X_train = mnist_train.iloc[:, 1:]
Y_train = mnist_train.iloc[:, 0]

# Create a neural network model to learn from train data
nn_model = MLPClassifier(hidden_layer_sizes=(50), max_iter=300, random_state=42)
nn_model.fit(X_train, Y_train)

# Predict the value of the first digit in the test data
predicted_digit = nn_model.predict(mnist_test.iloc[0:1, :])
print(f"Predicted digit for the first test sample: {predicted_digit[0]}")

# Evaluate the model on the training data
train_predictions = nn_model.predict(X_train)
print("Classification Report on Training Data:")
print(classification_report(Y_train, train_predictions))

# Uncomment the following lines if you have the true labels for the test set and want to evaluate the test data
# Y_test = mnist_test_labels  # You need to replace this with your actual test labels if available
# test_predictions = nn_model.predict(mnist_test)
# print("Classification Report on Test Data:")
# print(classification_report(Y_test, test_predictions))
