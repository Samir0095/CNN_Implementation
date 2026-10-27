import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ============================================================
# 2. LOAD DATASET
# ============================================================

train_path = "dataset/Training"

train_dataset = datasets.ImageFolder(
    train_path,
    transform=transform
)
test_path = "dataset/Testing"

test_dataset = datasets.ImageFolder(
    test_path,
    transform=transform
)

print("Number of testing images:", len(test_dataset))
print("Testing classes:", test_dataset.classes)

print("Number of training images:", len(train_dataset))
print("Classes:", train_dataset.classes)
print("Class to index:", train_dataset.class_to_idx)


# ============================================================
# 3. CREATE DATALOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# ============================================================
# 4. DEFINE CNN
# ============================================================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Second convolution
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layer
        self.fc = nn.Linear(
            64 * 32 * 32,
            4
        )


    def forward(self, x):

        # First convolution block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Second convolution block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layer
        x = self.fc(x)

        return x


# ============================================================
# 5. CREATE MODEL
# ============================================================

model = CNN()

print(model)


# ============================================================
# 6. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 7. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 8. TRAIN THE CNN
# ============================================================

num_epochs = 5
epoch_losses = []
epoch_accuracies = []

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # -------------------------
        # Forward pass
        # -------------------------

        outputs = model(images)

        # -------------------------
        # Calculate loss
        # -------------------------

        loss = criterion(outputs, labels)

        # -------------------------
        # Clear old gradients
        # -------------------------

        optimizer.zero_grad()

        # -------------------------
        # Backpropagation
        # -------------------------

        loss.backward()

        # -------------------------
        # Update weights
        # -------------------------

        optimizer.step()

        # -------------------------
        # Calculate accuracy
        # -------------------------

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


    accuracy = 100 * correct / total
    epoch_losses.append(running_loss)
    epoch_accuracies.append(accuracy)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )

# ============================================================
# 9. TEST THE CNN ON TESTING DATASET
# ============================================================

model.eval()

correct = 0
total = 0

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Store actual and predicted labels
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())


# Calculate accuracy
test_accuracy = 100 * correct / total

print("\nTest Accuracy:", f"{test_accuracy:.2f}%")


# ============================================================
# 9.1 CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 9.2 CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes
    )
)

# ============================================================
# 10. TEST ONE IMAGE
# ============================================================

image, label = test_dataset[0]

# Add batch dimension
image = image.unsqueeze(0)

print("\nSingle image shape:", image.shape)

# Turn off gradient calculation for prediction
with torch.no_grad():

    output = model(image)

print("CNN output:", output)

print("Output shape:", output.shape)

# Find the class with the highest score
predicted_class = torch.argmax(output, dim=1)

print("Predicted class:", predicted_class.item())
print("Actual class:", label)

print("Predicted class name:",
      test_dataset.classes[predicted_class.item()])

print("Actual class name:",
      test_dataset.classes[label])


# ============================================================
# 11. PLOT TRAINING & EVALUATION RESULTS
# ============================================================

epochs_range = range(1, num_epochs + 1)
class_names = test_dataset.classes

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    zero_division=0
)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Brain Tumor CNN — Training & Evaluation Results", fontsize=16, fontweight="bold")

# 11.1 Training Loss Curve
ax1 = axes[0, 0]
ax1.plot(epochs_range, epoch_losses, marker="o", color="#e11d48", linewidth=2)
ax1.set_title("Training Loss per Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Running Loss")
ax1.set_xticks(list(epochs_range))
ax1.grid(True, linestyle="--", alpha=0.5)

# 11.2 Training vs Testing Accuracy
ax2 = axes[0, 1]
ax2.plot(epochs_range, epoch_accuracies, marker="s", color="#10b981", linewidth=2, label="Train Accuracy")
ax2.axhline(y=test_accuracy, color="#0284c7", linestyle="--", linewidth=2, label=f"Test Accuracy ({test_accuracy:.2f}%)")
ax2.set_title("Training vs. Testing Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_xticks(list(epochs_range))
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

# 11.3 Confusion Matrix Heatmap
ax3 = axes[1, 0]
im = ax3.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax3.set_title("Confusion Matrix")
fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
tick_marks = np.arange(len(class_names))
ax3.set_xticks(tick_marks)
ax3.set_xticklabels(class_names, rotation=30, ha="right")
ax3.set_yticks(tick_marks)
ax3.set_yticklabels(class_names)
ax3.set_ylabel("Actual Class")
ax3.set_xlabel("Predicted Class")

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax3.text(
            j, i, format(cm[i, j], "d"),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontweight="bold"
        )

# 11.4 Per-Class Precision, Recall, F1-Score
ax4 = axes[1, 1]
x = np.arange(len(class_names))
width = 0.25
ax4.bar(x - width, precision * 100, width, label="Precision", color="#0ea5e9")
ax4.bar(x, recall * 100, width, label="Recall", color="#10b981")
ax4.bar(x + width, f1 * 100, width, label="F1-Score", color="#8b5cf6")
ax4.set_title("Per-Class Evaluation Metrics (%)")
ax4.set_xticks(x)
ax4.set_xticklabels(class_names)
ax4.set_ylim(0, 110)
ax4.set_ylabel("Score (%)")
ax4.legend()
ax4.grid(True, axis="y", linestyle="--", alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
output_plot_path = "training_results.png"
plt.savefig(output_plot_path, dpi=150)
print(f"\nSaved graphical results to: {output_plot_path}")
plt.show()