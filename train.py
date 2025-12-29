import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from medical_resnet import MedicalCNN
from utils import get_class_weights, evaluate_model, save_model, plot_confusion_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_path = r"chest_xray\train"
val_path   = r"chest_xray\val"
test_path  = r"chest_xray\test"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


train_ds = datasets.ImageFolder(train_path, transform=transform)
val_ds   = datasets.ImageFolder(val_path, transform=transform)
test_ds  = datasets.ImageFolder(test_path, transform=transform)

class_weights = get_class_weights(train_ds, device)


targets = [train_ds[i][1] for i in range(len(train_ds))]
weights = [class_weights[t].item() for t in targets]


sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
val_loader   = DataLoader(val_ds, batch_size=32)
test_loader  = DataLoader(test_ds, batch_size=32)



model = MedicalCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)



best_loss = float("inf")
patience, counter = 7, 0
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1} | Val Acc: {val_acc*100:.2f}% | Val Loss: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        save_model(model, "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

test_loss, test_acc, labels, preds = evaluate_model(model, test_loader, criterion, device)
print(f"Test Accuracy: {test_acc*100:.2f}%")
plot_confusion_matrix(labels, preds, ["NORMAL", "PNEUMONIA"], save_path="confusion_matrix.png")
