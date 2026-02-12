# Task 1 — Convolutional Neural Network (CNN) for Cybersecurity

## 1. What is a CNN? (Conceptual Overview)


### Key building blocks
- **Convolution layer**: learns local patterns via filters/kernels.
- **ReLU**: introduces non-linearity and improves gradient flow.
- **Pooling**: downsamples feature maps (MaxPool) to reduce computation and increase invariance.
- **Batch Normalization** (optional): stabilizes training by normalizing activations.
- **Dropout**: regularization; reduces overfitting by randomly dropping neurons.
- **Fully-connected layer**: combines extracted features for final decision.
- **Sigmoid/Softmax output**: outputs probability for binary/multi-class prediction.

### Why CNNs work

CNNs are effective because they exploit the spatial structure of data. Instead of treating inputs as flat vectors, they preserve the relative position of features. This is important when patterns depend on neighboring values, which is often the case in images and structured cybersecurity telemetry.

A key concept is **local receptive fields**. Each neuron in a convolutional layer only looks at a small region of the input at a time. This allows the network to detect local patterns such as edges, shapes, or unusual feature combinations. In cybersecurity, this can help identify suspicious clusters of activity across related metrics.

Another important idea is **weight sharing**. The same filter (set of weights) is applied across the entire input. This drastically reduces the number of parameters and allows the model to detect the same pattern anywhere in the data. For example, a malicious behavior pattern can be recognized regardless of where it appears in the feature grid.

CNNs also use **hierarchical feature learning**. Early layers detect simple patterns, while deeper layers combine them into more complex representations. This layered learning enables CNNs to capture subtle and high-level structures, which is valuable for distinguishing benign from malicious behavior in cybersecurity monitoring systems.


---

## 2. Practical Cybersecurity Example — Benign vs Malicious Telemetry as Images

We model cybersecurity telemetry as a 4×4 image (16 features). The CNN learns discriminative patterns for classification.

### 2.1 Embedded dataset (CSV)

```csv
label,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16
PASTE REAL ROWS HERE (not "...")
```

---

### 2.2 Python Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras import layers, models

os.makedirs("images", exist_ok=True)


np.random.seed(42)

# ----- Create synthetic cybersecurity dataset -----
n = 200
X = np.random.rand(n, 16)

# benign vs malicious labels (0=benign, 1=malicious)
y = np.array([0] * 100 + [1] * 100)

# reshape to 4x4 "images" with 1 channel
X = X.reshape(n, 4, 4, 1)

# ----- Class distribution plot -----
plt.figure()
plt.bar(["Benign", "Malicious"], [np.sum(y == 0), np.sum(y == 1)])
plt.title("Class Distribution")
plt.ylabel("Count")
plt.savefig("images/class_distribution.png")
plt.close()

# ----- Split data (stratified keeps class balance) -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----- CNN model -----
model = models.Sequential([
    layers.Conv2D(16, (2, 2), activation="relu", input_shape=(4, 4, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ----- Train -----
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    verbose=1
)

# ----- Training curves -----
plt.figure()
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.legend()
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("images/training.png")
plt.close()

# ----- Confusion Matrix -----
y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int).ravel()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malicious"])
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix.png")
plt.close()

