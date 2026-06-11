"""
Binary Image Classification using CNN
Author: Aleah Hassabo

This project implements a 3-layer Convolutional Neural Network (CNN) for binary
image classification (cats vs. dogs) using PyTorch. The model is trained on the
PetImages dataset and evaluated using accuracy, precision, and recall.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn import metrics

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Data Preprocessing ──
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = datasets.ImageFolder("./petimages", transform=transform)

test_set, train_set = torch.utils.data.random_split(
    dataset, [0.2, 0.8], generator=torch.Generator().manual_seed(42)
)

# ── Hyperparameters ──
learning_rate = 0.0001
batch_size = 32
epoch_size = 10

trainloader = torch.utils.data.DataLoader(train_set, batch_size)
testloader = torch.utils.data.DataLoader(test_set, batch_size)

# ── Model Architecture ──
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.lin1 = nn.Linear(9216, 512)  # 64 * 12 * 12
        self.lin2 = nn.Linear(512, 128)
        self.lin3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        x = self.lin3(x)
        return x

# ── Training ──
cnn = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

cnn.train()
for epoch in range(epoch_size):
    loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = cnn(inputs)
        lossX = criterion(outputs, labels)
        lossX.backward()
        optimizer.step()

        loss += lossX.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {loss / 100:.3f}")
            loss = 0.0

print("Finished Training")

# ── Evaluation ──
ground_truth = []
prediction = []

cnn.eval()
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        ground_truth += labels.cpu().tolist()
        outputs = cnn(inputs)
        _, predicted = torch.max(outputs, 1)
        prediction += predicted.cpu().tolist()

# ── Results ──
accuracy = metrics.accuracy_score(ground_truth, prediction)
recall = metrics.recall_score(ground_truth, prediction)
precision = metrics.precision_score(ground_truth, prediction)

print("=" * 40)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Precision: {precision:.4f}")
print("=" * 40)
