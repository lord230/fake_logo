# Created By LORD 
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from dataset import train_loader, test_loader, train_dataset  # Make sure you're using BinaryLogoDataset

# Use pretrained EfficientNet
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)

# Binary classification: 2 classes (real and fake)
num_classes = 2
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  # use BCEWithLogitsLoss for sigmoid, but we're using softmax
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training
from tqdm import tqdm
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}", leave=False)
    for images, labels in train_loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        acc = 100 * correct / total
        train_loop.set_postfix(loss=total_loss / (total / labels.size(0)), accuracy=f"{acc:.2f}%")

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {total_loss/len(train_loader):.4f}, Train Accuracy: {acc:.2f}%")

    # Evaluation
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        test_loop = tqdm(test_loader, desc=f"Test Epoch {epoch+1}/{num_epochs}", leave=False)
        for images, labels in test_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            acc = 100 * correct / total
            test_loop.set_postfix(loss=test_loss / (total / labels.size(0)), accuracy=f"{acc:.2f}%")

    print(f"Epoch [{epoch+1}/{num_epochs}] Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {acc:.2f}%")

# Save the model
torch.save(model.state_dict(), "Fake_Logo/results/efficientnet_fake_logo.pth")
print("Model saved as Fake_Logo/results/efficientnet_fake_logo.pth")
