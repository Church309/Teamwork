"""
Fashion MNIST 多模型分类对比
- Logistic Regression (sklearn)
- Random Forest (sklearn)
- K-Nearest Neighbors (sklearn)
- CNN (PyTorch)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# ─── 全局设置 ───
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# 设备检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# ─── 1. 加载数据 ───
print("\n" + "=" * 60)
print("1. 加载 Fashion MNIST 数据")
print("=" * 60)

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(root='./数据集', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./数据集', train=False, download=True, transform=transform)

X_train = train_dataset.data.numpy().astype(np.float32)
y_train = train_dataset.targets.numpy()
X_test = test_dataset.data.numpy().astype(np.float32)
y_test = test_dataset.targets.numpy()

# 归一化 [0,1]
X_train /= 255.0
X_test /= 255.0

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"类别: {len(np.unique(y_train))}, {CLASS_NAMES}")

# ─── 2. 可视化样本 ───
print("\n" + "=" * 60)
print("2. 样本可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()
for i in range(10):
    idx = np.where(y_train == i)[0][0]
    axes[i].imshow(X_train[idx], cmap='gray')
    axes[i].set_title(CLASS_NAMES[i])
    axes[i].axis('off')
fig.suptitle('Fashion MNIST — 每类样本示例', fontsize=14)
plt.tight_layout()
plt.savefig('01_samples.png', dpi=150)
plt.close()
print("已保存: 01_samples.png")

# ─── 3. 准备 sklearn 数据（展平） ───
print("\n" + "=" * 60)
print("3. 数据预处理（展平 + 标准化）")
print("=" * 60)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# 标准化（对 LR/SVM/KNN 很重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)
print(f"展平后: {X_train_scaled.shape}")

# ─── 4. 训练 sklearn 模型 ───
results = {}

def evaluate_model(name, y_true, y_pred, y_pred_proba=None):
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    results[name] = {
        'accuracy': acc,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'cm': cm,
        'pred': y_pred
    }
    print(f"  {name:25s} | Accuracy: {acc:.4f} | F1: {f1_macro:.4f}")

# 由于数据集较大（6万训练），用部分数据提速
sample_size = 20000
indices = np.random.choice(X_train_scaled.shape[0], sample_size, replace=False)
X_train_sub = X_train_scaled[indices]
y_train_sub = y_train[indices]
print(f"\n使用 {sample_size} 个训练样本用于 sklearn 模型")

# ─── 4a. Logistic Regression ───
print("\n>>> [1/4] Logistic Regression ...")
start = time.time()
lr = LogisticRegression(max_iter=200, multi_class='multinomial',
                         solver='lbfgs', random_state=RANDOM_STATE, n_jobs=-1)
lr.fit(X_train_sub, y_train_sub)
y_pred_lr = lr.predict(X_test_scaled)
evaluate_model('Logistic Regression', y_test, y_pred_lr)
print(f"    耗时: {time.time() - start:.1f}s")

# ─── 4b. Random Forest ───
print("\n>>> [2/4] Random Forest ...")
start = time.time()
rf = RandomForestClassifier(n_estimators=150, max_depth=20,
                             random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_sub, y_train_sub)
y_pred_rf = rf.predict(X_test_scaled)
evaluate_model('Random Forest', y_test, y_pred_rf)
print(f"    耗时: {time.time() - start:.1f}s")

# ─── 4c. K-Nearest Neighbors ───
print("\n>>> [3/4] K-Nearest Neighbors ...")
start = time.time()
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_sub, y_train_sub)
y_pred_knn = knn.predict(X_test_scaled)
evaluate_model('K-Nearest Neighbors', y_test, y_pred_knn)
print(f"    耗时: {time.time() - start:.1f}s")

# ─── 5. PyTorch CNN ───
print("\n>>> [4/4] CNN (PyTorch) ...")
print("=" * 60)

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# CNN 需要保留 2D 结构
X_train_cnn = torch.FloatTensor(X_train).unsqueeze(1)  # [N, 1, 28, 28]
y_train_cnn = torch.LongTensor(y_train)
X_test_cnn = torch.FloatTensor(X_test).unsqueeze(1)
y_test_cnn = torch.LongTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_cnn, y_train_cnn),
                           batch_size=128, shuffle=True, num_workers=0)

model = FashionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
n_epochs = 10
start = time.time()
train_losses = []
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"  Epoch [{epoch+1}/{n_epochs}] Loss: {avg_loss:.4f}")

train_time = time.time() - start
print(f"    训练耗时: {train_time:.1f}s")

# 评估 CNN
model.eval()
with torch.no_grad():
    outputs = model(X_test_cnn.to(device))
    _, y_pred_cnn = torch.max(outputs, 1)
    y_pred_cnn = y_pred_cnn.cpu().numpy()

evaluate_model('CNN (PyTorch)', y_test, y_pred_cnn)

# 绘制 CNN 训练损失
plt.figure(figsize=(8, 4))
plt.plot(range(1, n_epochs + 1), train_losses, 'b-o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN 训练损失曲线')
plt.grid(True)
plt.savefig('02_cnn_loss.png', dpi=150)
plt.close()
print("已保存: 02_cnn_loss.png")

# ─── 6. 可视化混淆矩阵 ───
print("\n" + "=" * 60)
print("6. 混淆矩阵可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()
model_names = ['Logistic Regression', 'Random Forest', 'K-Nearest Neighbors', 'CNN (PyTorch)']

for ax, name in zip(axes, model_names):
    cm = results[name]['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
    ax.set_title(f'{name}  — Confusion Matrix', fontsize=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('03_confusion_matrices.png', dpi=150)
plt.close()
print("已保存: 03_confusion_matrices.png")

# ─── 7. 模型对比柱状图 ───
print("\n" + "=" * 60)
print("7. 模型性能对比")
print("=" * 60)

metrics = ['accuracy', 'precision', 'recall', 'f1']
metric_labels = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-score (macro)']
x = np.arange(len(model_names))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
for i, (m, label) in enumerate(zip(metrics, metric_labels)):
    values = [results[name][m] for name in model_names]
    bars = ax.bar(x + i * width, values, width, label=label)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel('Score')
ax.set_title('Fashion MNIST — 4种模型性能对比')
ax.set_ylim(0.7, 1.0)
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('04_comparison_bar.png', dpi=150)
plt.close()
print("已保存: 04_comparison_bar.png")

# ─── 8. 每个类别的 Accuracy 对比 ───
print("\n" + "=" * 60)
print("8. 各类别 Accuracy 对比")
print("=" * 60)

fig, ax = plt.subplots(figsize=(12, 6))
for name in model_names:
    cm = results[name]['cm']
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    ax.plot(CLASS_NAMES, per_class_acc, 'o-', label=name, markersize=6)

ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.set_ylabel('Per-class Accuracy')
ax.set_title('各类别 Accuracy 对比')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('05_per_class_accuracy.png', dpi=150)
plt.close()
print("已保存: 05_per_class_accuracy.png")

# ─── 9. 预测错误样本展示 ───
print("\n" + "=" * 60)
print("9. 预测错误样本展示（以 CNN 为例）")
print("=" * 60)

errors = np.where(y_pred_cnn != y_test)[0]
fig, axes = plt.subplots(3, 5, figsize=(14, 9))
axes = axes.flatten()
for i, idx in enumerate(errors[:15]):
    axes[i].imshow(X_test[idx], cmap='gray')
    axes[i].set_title(f'True: {CLASS_NAMES[y_test[idx]]}\nPred: {CLASS_NAMES[y_pred_cnn[idx]]}',
                      fontsize=9, color='red')
    axes[i].axis('off')
fig.suptitle('CNN 预测错误示例', fontsize=14)
plt.tight_layout()
plt.savefig('06_cnn_errors.png', dpi=150)
plt.close()
print("已保存: 06_cnn_errors.png")

# ─── 10. 汇总报告 ───
print("\n" + "=" * 60)
print("10. 综合对比汇总")
print("=" * 60)

print(f"\n{'Model':25s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1-score':>10s}")
print("-" * 65)
for name in model_names:
    r = results[name]
    print(f"{name:25s} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
          f"{r['recall']:>10.4f} {r['f1']:>10.4f}")

best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n*** 最佳模型: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

print("\n*** 所有结果已保存为 01_samples.png ~ 06_cnn_errors.png")
