# ----------------------------------------
# MoCo v2 Implementation for FER Dataset
# ----------------------------------------
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import deque
from copy import deepcopy
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score , confusion_matrix
from sklearn.preprocessing import label_binarize



# ------------------------- Configuration -------------------------
torch.set_default_dtype(torch.float32)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

data_root = 'C:/Users/parth/code_master/masters2026/cs512/Project/Project/raf_db/DATASET'  
embedding_dim = 256
temperature = 0.1
momentum = 0.999
batch_size = 64

# ------------------------- Data Transform -------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'test'), transform=transform)
class_names = train_dataset.classes
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------------- Synthetic Heatmap Generator -------------------------
def generate_synthetic_heatmap(size=(48, 48), sigma=5):
    x = np.linspace(0, size[1] - 1, size[1])
    y = np.linspace(0, size[0] - 1, size[0])
    x, y = np.meshgrid(x, y)
    center_x, center_y = size[1] // 2, size[0] // 2
    heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    return heatmap.astype(np.float32)

def preprocess_input(image_tensor):
    image_np = image_tensor.squeeze().numpy() * 255
    heatmap = generate_synthetic_heatmap()
    combined = np.stack([image_np, heatmap], axis=0)
    return torch.tensor(combined / 255.0, dtype=torch.float32)

# ------------------------- MoCo v2 Encoder -------------------------
class MoCoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.wide_resnet50_2(weights=None)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.encoder[0] = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, embedding_dim)
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = feat.view(feat.size(0), -1)
        return self.projector(feat)

# ------------------------- Classifier -------------------------
class MoCoClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, len(class_names))
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            x = x.view(x.size(0), -1)  
        return self.classifier(x)

# ------------------------- Loss & Momentum -------------------------
def contrastive_loss(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    logits = torch.mm(x, y.T) / temperature
    labels = torch.arange(x.size(0)).to(x.device)
    return F.cross_entropy(logits, labels)

def update_momentum_encoder(model_q, model_k):
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data = momentum * param_k.data + (1. - momentum) * param_q.data

# ------------------------- Pretraining -------------------------
model_q = MoCoEncoder().to(device)
model_k = deepcopy(model_q).to(device)
for p in model_k.parameters(): p.requires_grad = False

optimizer = torch.optim.SGD(model_q.parameters(), lr=0.015, momentum=0.9, weight_decay=1e-6)
scheduler = CosineAnnealingLR(optimizer, T_max=1000)

print("Starting MoCo v2 pretraining...")
train_losses = []

for epoch in range(70):
    model_q.train()
    total_loss = 0
    for images, _ in train_loader:
        x_q = torch.stack([preprocess_input(img) for img in images]).to(device)
        x_k = torch.stack([preprocess_input(img) for img in images]).to(device)

        z_q = model_q(x_q)
        with torch.no_grad():
            z_k = model_k(x_k)

        loss = contrastive_loss(z_q, z_k)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        update_momentum_encoder(model_q, model_k)

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}, MoCo Loss: {avg_loss:.4f}")

# ------------------------- Classifier Training -------------------------
classifier = MoCoClassifier(model_q.encoder).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=0.001)

def evaluate(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            inputs = torch.stack([preprocess_input(img) for img in images]).to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted.cpu() == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

print("\nTraining classifier...")
best_acc = 0
patience = 5
no_improve = 0
val_accuracies = []

for epoch in range(70):
    classifier.train()
    running_loss = 0
    for images, labels in train_loader:
        inputs = torch.stack([preprocess_input(img) for img in images]).to(device)
        labels = labels.to(device)

        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    acc = evaluate(classifier)
    val_accuracies.append(acc)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        no_improve = 0
        torch.save(classifier.state_dict(), "moco_classifier_best.pth")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping triggered.")
            break
# Save accuracy plot
plt.figure()
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("MoCo Classifier Accuracy Over Epochs")
plt.grid()
plt.tight_layout()
plt.savefig("MoCo_accuracy_plot.png")
plt.close()
print("✅ Saved accuracy plot as MoCo_accuracy_plot.png")
#####################  Martix ###############

def plot_confusion_matrix(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            inputs = torch.stack([preprocess_input(img) for img in images]).to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("MoCo_confusion_matrix.png")
    plt.close()
    print("Saved confusion matrix as MoCo_confusion_matrix.png")

plot_confusion_matrix(classifier, test_loader, class_names)
# ------------------------- Visualization -------------------------
def prediction_grid(model, dataset, grid_size=4):
    model.eval()
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = random.randint(0, len(dataset)-1)
            img, label = dataset[idx]
            input_tensor = preprocess_input(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(input_tensor)
                pred_label = torch.argmax(pred, dim=1).item()
            axs[i, j].imshow(img.squeeze(), cmap='gray')
            axs[i, j].set_title(f"True: {class_names[label]}\nPred: {class_names[pred_label]}")
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.savefig("MoCo_prediction_grid.png")
    plt.close()
    print("Saved prediction grid as MoCo_prediction_grid.png")

prediction_grid(classifier, test_dataset)

def evaluate_per_emotion_metrics(model, dataloader, class_names, mode='classify'):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            inputs = torch.stack([preprocess_input(img) for img in images]).to(device)
            outputs = model(inputs, mode=mode)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_labels_bin = label_binarize(all_labels, classes=list(range(len(class_names))))

    
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)

    
    try:
        auc = roc_auc_score(all_labels_bin, all_probs, average=None, multi_class='ovr')
    except Exception as e:
        print(f"⚠️ AUC computation failed: {e}")
        auc = [None] * len(class_names)

    
    results_df = pd.DataFrame({
        'Emotion': class_names,
        'Precision': precision,
        'Recall': recall,
        'AUC': auc
    })

    print("\nEmotion-wise Evaluation Metrics:")
    print(results_df.round(4))
    return results_df

encoder = MoCoEncoder().encoder  
model = MoCoClassifier(encoder).to(device)
model.load_state_dict(torch.load("moco_classifier_best.pth"))
evaluate_per_emotion_metrics(model, test_loader, class_names, mode='classify')