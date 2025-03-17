from torchvision import datasets, transforms
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms



# transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),  # PIL image to PyTorch tensor
    transforms.Normalize(mean=(0.5,), std=(0.5,)),  # normalize
])

# load dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FCNN(nn.Module):
  def __init__(self):
    super(FCNN, self).__init__()
    self.fc1 = nn.Linear(28 * 28, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 10)
    self.act = nn.ReLU()

  def forward(self, x):
    x = x.view(-1, 28 * 28)  # flatten
    x = self.fc1(x)
    x = self.act(x)
    x = self.fc2(x)
    x = self.act(x)
    x = self.fc3(x)
    return x

model = FCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 10
for i in range(epochs):
  for image, label in train_loader:
    image, label = image.to(device), label.to(device)
    optimizer.zero_grad()
    out = model(image)
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()
torch.save(model.state_dict(), "fcnn_mnist_state_dict.pth")


# evaluate

# load test dataset
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# evaluate model on test data
model.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():  
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# avg test loss
test_loss /= len(test_loader)
accuracy = 100 * correct / total

print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
