import os
import torch
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model import get_model

# create results folder
os.makedirs("results/plots", exist_ok=True)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Data
train_data = datasets.ImageFolder("data/train", transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

val_data = datasets.ImageFolder("data/val", transform=transform)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Model
model = get_model()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
losses = []

for epoch in range(3):
    epoch_loss = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Validation
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")

# Save accuracy
with open("results/metrics.txt", "w") as f:
    f.write(f"Validation Accuracy: {accuracy:.2f}%")

# Save predictions images
images, labels = next(iter(val_loader))
outputs = model(images)
_, preds = torch.max(outputs, 1)

for i in range(5):
    plt.imshow(images[i].permute(1, 2, 0).cpu())
    plt.title(f"Pred: {preds[i].item()} | True: {labels[i].item()}")
    plt.axis('off')

    plt.savefig(f"results/plots/pred_{i}.png")
    plt.close()

# Save loss plot
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("results/plots/loss.png")
plt.close()