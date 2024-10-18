import numpy as np
from sklearn.model_selection import train_test_split

# Example data (limited prototype with 500 images, 10 classes)
# X -> shape (number_of_images, height, width, channels)
# y -> shape (number_of_images,)

# Load or define your dataset
X = np.random.randn(500, 28, 28, 1)  # Example images (500 grayscale images, 28x28)
y = np.random.randint(0, 10, 500)    # Example labels (500 labels in range [0, 9])

# Split dataset into training + validation and testing (70% train/val, 30% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Further split training + validation into separate sets (85% training, 15% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15 / 0.85, random_state=42, stratify=y_train_val)

# Check the sizes of the splits to ensure correct proportions
print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
