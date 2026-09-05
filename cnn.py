import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report


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