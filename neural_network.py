import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
tf.random.set_seed(42)
np.random.seed(42)

# Create directory for saved models
os.makedirs('saved_models', exist_ok=True)

print("Loading data...")

# Read data
with open('banknotes.csv') as f:
    reader = csv.reader(f)
    next(reader)
    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": 0 if row[4] == "0" else 1  # 0 = Authentic, 1 = Counterfeit
        })

# Split data
holdout = int(0.40 * len(data))
testing = data[:holdout]
training = data[holdout:]

X_train = np.array([row["evidence"] for row in training], dtype=np.float32)
y_train = np.array([row["label"] for row in training], dtype=np.float32)
X_test = np.array([row["evidence"] for row in testing], dtype=np.float32)
y_test = np.array([row["label"] for row in testing], dtype=np.float32)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print()


# ============================================
# PYTORCH NEURAL NETWORK
# ============================================

class NeuralNetwork(nn.Module):
    """Simple neural network with one hidden layer"""
    
    def __init__(self, input_size=4, hidden_size=8):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x


def train_pytorch_model(epochs=100, learning_rate=0.01):
    """Train PyTorch neural network"""
    print("Training PyTorch Neural Network...")
    
    # Convert to PyTorch tensors
    X_train_torch = torch.from_numpy(X_train)
    y_train_torch = torch.from_numpy(y_train).reshape(-1, 1)
    X_test_torch = torch.from_numpy(X_test)
    
    # Create model
    model = NeuralNetwork()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train_torch)
        loss = criterion(outputs, y_train_torch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_torch)
        predictions = (predictions.numpy() > 0.5).astype(int).flatten()
    
    # Calculate accuracy
    correct = (predictions == y_test).sum()
    total = len(y_test)
    accuracy = 100 * correct / total
    
    print(f"\nPyTorch Results:")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    
    # Save model
    torch.save(model.state_dict(), 'saved_models/pytorch_nn.pth')
    print("Saved: saved_models/pytorch_nn.pth")
    
    return predictions, accuracy


# ============================================
# TENSORFLOW NEURAL NETWORK
# ============================================

def create_tensorflow_model():
    """Create TensorFlow neural network"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_tensorflow_model(epochs=100):
    """Train TensorFlow neural network"""
    print("\nTraining TensorFlow Neural Network...")
    
    # Create and train model
    model = create_tensorflow_model()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        verbose=0,
        validation_split=0.1
    )
    
    # Print training progress
    for i in range(0, epochs, 20):
        if i < len(history.history['loss']):
            print(f"Epoch [{i+1}/{epochs}], Loss: {history.history['loss'][i]:.4f}")
    
    # Evaluation
    predictions = model.predict(X_test, verbose=0)
    predictions = (predictions > 0.5).astype(int).flatten()
    
    # Calculate accuracy
    correct = (predictions == y_test).sum()
    total = len(y_test)
    accuracy = 100 * correct / total
    
    print(f"\nTensorFlow Results:")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    
    # Save model
    model.save('saved_models/tensorflow_nn.keras')
    print("Saved: saved_models/tensorflow_nn.keras")
    
    return predictions, accuracy, history


# ============================================
# TRAINING AND VISUALIZATION
# ============================================

# Train both models
pytorch_predictions, pytorch_accuracy = train_pytorch_model(epochs=100)
tensorflow_predictions, tensorflow_accuracy, tf_history = train_tensorflow_model(epochs=100)

# Visualization 1: Accuracy Comparison
plt.figure(figsize=(10, 6))
models = ['PyTorch', 'TensorFlow']
accuracies = [pytorch_accuracy, tensorflow_accuracy]
colors = ['#e74c3c', '#3498db']

bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5, width=0.5)
plt.ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
plt.xlabel('Neural Network Framework', fontsize=13, fontweight='bold')
plt.title('Neural Network Performance Comparison', fontsize=15, fontweight='bold', pad=20)
plt.ylim(0, 105)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('nn_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved: nn_comparison.png")
plt.close()

# Visualization 2: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Convert to string labels for confusion matrix display
y_test_labels = ["Authentic" if y == 0 else "Counterfeit" for y in y_test]
pytorch_pred_labels = ["Authentic" if y == 0 else "Counterfeit" for y in pytorch_predictions]
tensorflow_pred_labels = ["Authentic" if y == 0 else "Counterfeit" for y in tensorflow_predictions]

# PyTorch confusion matrix
cm_pytorch = confusion_matrix(y_test_labels, pytorch_pred_labels, 
                              labels=["Authentic", "Counterfeit"])
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_pytorch,
                               display_labels=["Authentic", "Counterfeit"])
disp1.plot(ax=axes[0], cmap='Reds', colorbar=True)
axes[0].set_title(f'PyTorch Neural Network\nAccuracy: {pytorch_accuracy:.2f}%',
                  fontweight='bold', fontsize=12)
axes[0].grid(False)

# TensorFlow confusion matrix
cm_tensorflow = confusion_matrix(y_test_labels, tensorflow_pred_labels,
                                 labels=["Authentic", "Counterfeit"])
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_tensorflow,
                               display_labels=["Authentic", "Counterfeit"])
disp2.plot(ax=axes[1], cmap='Blues', colorbar=True)
axes[1].set_title(f'TensorFlow Neural Network\nAccuracy: {tensorflow_accuracy:.2f}%',
                  fontweight='bold', fontsize=12)
axes[1].grid(False)

plt.suptitle('Neural Network Confusion Matrices', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('nn_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: nn_confusion_matrices.png")
plt.close()

# Visualization 3: Training Loss
plt.figure(figsize=(10, 6))
plt.plot(tf_history.history['loss'], color='#3498db', linewidth=2, label='Training Loss')
if 'val_loss' in tf_history.history:
    plt.plot(tf_history.history['val_loss'], color='#e74c3c', linewidth=2, label='Validation Loss')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('TensorFlow Training Progress', fontsize=14, fontweight='bold', pad=20)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
print("Saved: training_loss.png")
plt.close()

print("\n✓ All visualizations saved successfully!")
print(f"\nFinal Results:")
print(f"PyTorch:    {pytorch_accuracy:.2f}% accuracy")
print(f"TensorFlow: {tensorflow_accuracy:.2f}% accuracy")