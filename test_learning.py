import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from model import ConvNet2
import numpy as np

def get_pretrain_loader(data_path, batch_size=16, num_samples=200):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
    ])
    
    site1_train_path = os.path.join(data_path, "site1", "train")
    if not os.path.exists(site1_train_path):
        print(f"Error: Path {site1_train_path} does not exist")
        return None
        
    full_dataset = datasets.ImageFolder(root=site1_train_path, transform=transform)
    
    indices = np.random.choice(len(full_dataset), min(num_samples, len(full_dataset)), replace=False)
    subset = Subset(full_dataset, indices)
    
    return DataLoader(subset, batch_size=batch_size, shuffle=True)

def check_learning():
    data_path = os.path.abspath("chest_xray")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loader = get_pretrain_loader(data_path)
    if loader is None: return
    
    model = ConvNet2().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Testing learning on a small subset...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}, Accuracy = {100*correct/total:.2f}%")

if __name__ == "__main__":
    check_learning()
