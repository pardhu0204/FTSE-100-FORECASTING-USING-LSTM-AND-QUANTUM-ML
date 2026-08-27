import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Correct imports for Qiskit ML
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler

from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss

# Set a fixed random seed
algorithm_globals.random_seed = 42

# Step 1: Download data
print("Downloading FTSE 100 data...")
df = yf.download("^FTSE", start="2010-01-01", end="2024-12-31")
df.dropna(inplace=True)

# Step 2: Create the target column
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df.dropna(inplace=True)

# Step 3: Feature engineering
df["MA5"] = df["Close"].rolling(window=5).mean()
df["MA10"] = df["Close"].rolling(window=10).mean()
df.dropna(inplace=True)

features = df[["MA5", "MA10"]]
labels = df["Target"]

scaler = StandardScaler()
X = scaler.fit_transform(features)
y = np.array(labels)  # Ensure it's a NumPy array to avoid reshape issues

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 5: VQC setup
feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
ansatz = RealAmplitudes(num_qubits=2, reps=1)
sampler = StatevectorSampler()

optimizer = COBYLA(maxiter=100)
loss = CrossEntropyLoss()

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    loss=loss,
    sampler=sampler
)

# Step 6: Train
print("Training Quantum Classifier...")
vqc.fit(X_train, y_train)

# Step 7: Predict & Evaluate
y_pred = vqc.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Plot results
plt.figure(figsize=(10, 4))
plt.plot(np.array(y_test), label="True")
plt.plot(y_pred, label="Predicted", alpha=0.7)
plt.legend()
plt.title("FTSE 100 Direction Prediction (VQC)")
plt.xlabel("Test Samples")
plt.ylabel("Direction (1=Up, 0=Down)")
plt.grid()
plt.tight_layout()
plt.show()

# Prepare features for tomorrow (example: last row from your dataset)
X_tomorrow = X[-1].reshape(1, -1)  # Make sure shape is correct

# Predict tomorrow's direction
tomorrow_pred = vqc.predict(X_tomorrow)
direction = "Up" if tomorrow_pred[0] == 1 else "Down"
print(f"Predicted FTSE direction for tomorrow: {direction}")

# Evaluate accuracy on test set
accuracy = vqc.score(X_test, y_test)
print(f"Accuracy on test data: {accuracy:.2%}")




