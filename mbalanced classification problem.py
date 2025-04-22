# train_efficientnet_imbalanced.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from imbalanced_dataset_tools import (
    ImbalancedImageFolder, get_weighted_sampler,
    get_class_weights, get_class_distribution, get_minority_classes
)

# ==== 参数 ====
data_dir = r"D:\\archive\\Aerial_Landscapes_Split"
train_root = os.path.join(data_dir, 'train')
test_root = os.path.join(data_dir, 'test')
batch_size = 32
num_epochs = 10

# ==== 分析类别分布并识别少数类 ====
class_counts = get_class_distribution(train_root)
minority_classes = get_minority_classes(class_counts, threshold=300)

# ==== 准备训练数据（少数类增强 + 加权采样） ====
train_data = ImbalancedImageFolder(root=train_root, minority_classes=minority_classes)
sampler = get_weighted_sampler(train_data)
train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)

# ==== 准备测试数据（普通 ImageFolder） ====
from torchvision import transforms, datasets

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_data = datasets.ImageFolder(root=test_root, transform=test_transforms)
test_loader = DataLoader(test_data, batch_size=batch_size)

# ==== 模型 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_data.classes))
model = model.to(device)

# ==== 损失函数 + 优化器（含类别权重） ====
class_weights = get_class_weights(train_data).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ==== 训练 ====
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ==== 测试 ====
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds, target_names=test_data.classes, zero_division=0))
