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
from sklearn.metrics import precision_score, recall_score, roc_auc_score , confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns
import pandas as pd

# ------------------------- Configuration -------------------------
torch.set_default_dtype(torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "hncl_checkpoint.pth"
classifier_checkpoint_path = "hncl_classifier.pth"
data_root = 'C:/Users/parth/code_master/masters2026/cs512/Project/Project/raf_db/DATASET'
embedding_dim = 256
queue_size = 8192
temperature = 0.1
momentum = 0.999
batch_size = 64

# ------------------------- Logger Function -------------------------
log_list = []

def log_message(message):
    log_list.append(message)

    return log_list

def log_it(logs, lg_file="training_logs_hncl.txt"):

    with open(lg_file, "w") as f:
        f.write("\n".join(logs))

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
from collections import Counter
print("Class distribution in training data:")
log_message("Class distribution in training data:")
print(Counter([label for _, label in train_dataset]))
log_message(Counter([label for _, label in train_dataset]))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = train_dataset.classes

# ------------------------- Heatmap Generator -------------------------
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
    combined = torch.tensor(combined / 255.0, dtype=torch.float32)
    return combined

# ------------------------- Encoder Modification -------------------------
class ModifiedResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.wide_resnet50_2(weights=None)  # Wider ResNet-50
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.features[0] = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

# ------------------------- Projector and Predictor -------------------------
class ProjectorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, embedding_dim), nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class PredictorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4096), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, embedding_dim)
        )

    def forward(self, x):
        return self.mlp(x)
    
# ------------------------- Contrastive Loss -------------------------
def contrastive_loss(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    logits = torch.mm(x, y.T) / temperature
    labels = torch.arange(x.size(0)).to(x.device)
    return F.cross_entropy(logits, labels)

# ------------------------- HNCL Model -------------------------
class HNCL(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ModifiedResNet50()
        self.projector = ProjectorMLP()
        self.predictor = PredictorMLP()
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

    def forward(self, x, mode='contrastive'):
        features = self.encoder(x)
        if mode == 'contrastive':
            z = self.projector(features)
            p = self.predictor(z)
            return z.detach(), p
        elif mode == 'classify':
            return self.classifier(features)

# ------------------------- Momentum Encoder -------------------------
def update_momentum_encoder(model_q, model_k):
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data = momentum * param_k.data + (1. - momentum) * param_q.data

# ------------------------- Visualizations -------------------------
def prediction_grid(model, dataset, class_names, grid_size=4):
    model.eval()
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = random.randint(0, len(dataset) - 1)
            image_tensor, label = dataset[idx]
            input_tensor = preprocess_input(image_tensor).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor, mode='classify')
                predicted = torch.argmax(output, dim=1).item()
            axs[i, j].imshow(image_tensor.squeeze(), cmap='gray')
            axs[i, j].set_title(f"True: {class_names[label]}\nPred: {class_names[predicted]}")
            axs[i, j].axis('off')
    plt.suptitle("Prediction Grid")
    plt.tight_layout()
    plt.show()

def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            inputs = torch.stack([preprocess_input(img) for img in images]).to(device)
            outputs = model(inputs, mode='classify')
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    log_message(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def plot_confusion_matrix(model, dataloader, class_names, mode='classify'):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            inputs = torch.stack([preprocess_input(img) for img in images]).to(device)
            outputs = model(inputs, mode=mode)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'HNCL Confusion Matrix ({mode})')
    plt.tight_layout()
    plt.savefig(f'HNCL_confusion_matrix_{mode}.png')
    plt.close()
    print(f"✅ Saved confusion matrix as: HNCL_confusion_matrix_{mode}.png")
    log_message(f"✅ Saved confusion matrix as: HNCL_confusion_matrix_{mode}.png")

def plot_accuracy_curve(val_accuracies, label):
    plt.figure()
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title(f"Validation Accuracy over Epochs ({label})")
    plt.grid()
    plt.tight_layout()
    filename = f'HNCL_accuracy_plot_{label}.png'
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved accuracy plot as: {filename}")
    log_message(f"✅ Saved accuracy plot as: {filename}")
# ------------------------- Train Classifier -------------------------
def train_classifier(model, patience=3, min_delta=0.01):
    print("\nTraining classifier on labeled data with early stopping...")
    log_message("\nTraining classifier on labeled data with early stopping...")
    model.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    best_state = None
    epochs_no_improve = 0
    val_accuracies = [] 

    model.train()
    for epoch in range(70):
        running_loss = 0
        for images, labels in train_loader:
            inputs = torch.stack([preprocess_input(img) for img in images]).to(device)
            labels = labels.to(device)

            outputs = model(inputs, mode='classify')
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Classifier Loss: {avg_loss:.4f}")
        log_message(f"Epoch {epoch+1}, Classifier Loss: {avg_loss:.4f}")

        val_acc = evaluate_model(model, test_loader)
        val_accuracies.append(val_acc)
        if val_acc - best_accuracy > min_delta:
            best_accuracy = val_acc
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"New best model in memory with accuracy: {val_acc:.2f}%")
            log_message(f"New best model in memory with accuracy: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            log_message(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            log_message("Early stopping triggered.")
            break

    if best_state:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), 'hncl_final_classifier.pth')  # ✅ Save final classifier model

    plot_accuracy_curve(val_accuracies, label='classifier') 

# ------------------------- Fine-Tune Model -------------------------
def finetune_model(model, patience=3, min_delta=0.01):
    print("\nFine-tuning entire model with early stopping...")
    log_message("\nFine-tuning entire model with early stopping...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    best_state = None
    epochs_no_improve = 0
    val_accuracies = []

    model.train()
    for epoch in range(70):
        running_loss = 0
        for images, labels in train_loader:
            inputs = torch.stack([preprocess_input(img) for img in images]).to(device)
            labels = labels.to(device)

            outputs = model(inputs, mode='classify')
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Fine-tune Loss: {avg_loss:.4f}")
        log_message(f"Epoch {epoch+1}, Fine-tune Loss: {avg_loss:.4f}")

        val_acc = evaluate_model(model, test_loader)
        val_accuracies.append(val_acc)
        if val_acc - best_accuracy > min_delta:
            best_accuracy = val_acc
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"New best fine-tuned model in memory with accuracy: {val_acc:.2f}%")
            log_message(f"New best fine-tuned model in memory with accuracy: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            log_message(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            log_message("Early stopping triggered.")
            break

    if best_state:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), 'hncl_final_finetuned.pth') 
    
    plot_accuracy_curve(val_accuracies, label='finetune')  

model_q = HNCL().to(device)
model_k = deepcopy(model_q).to(device)

queue = deque(maxlen=queue_size)
optimizer = torch.optim.SGD(model_q.parameters(), lr=0.015, momentum=0.9, weight_decay=1e-6)
scheduler = CosineAnnealingLR(optimizer, T_max=1000)

start_epoch = 0

print("\nStarting self-supervised HNCL pretraining...")
log_message("\nStarting self-supervised HNCL pretraining...")
model_q.train()
model_k.train()
train_losses = []

for epoch in range(start_epoch, start_epoch + 70):
    total_loss = 0
    for images, _ in train_loader:
        inputs = torch.stack([preprocess_input(img) for img in images]).to(device)

        with torch.no_grad():
            z_k, _ = model_k(inputs)

        _, p_q = model_q(inputs)
        loss = contrastive_loss(p_q, z_k)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        update_momentum_encoder(model_q, model_k)

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Contrastive Loss: {avg_loss:.4f}")
    log_message(f"Epoch {epoch+1}, Contrastive Loss: {avg_loss:.4f}")

    torch.save({
        'epoch': epoch,
        'model_q': model_q.state_dict(),
        'model_k': model_k.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, checkpoint_path)

plt.figure()
plt.plot(range(start_epoch + 1, start_epoch + 1 + len(train_losses)), train_losses, label='Contrastive Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid()
plt.legend()
plt.show()

train_classifier(model_q)
evaluate_model(model_q, test_loader)
prediction_grid(model_q, test_dataset, class_names, grid_size=4)
plot_confusion_matrix(model_q, test_loader, class_names, mode='classify')

finetune_model(model_q)
evaluate_model(model_q, test_loader)
prediction_grid(model_q, test_dataset, class_names, grid_size=4)
plot_confusion_matrix(model_q, test_loader, class_names, mode='finetune')

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
        print("⚠️ AUC will be set to None for all classes.")
        auc = [None] * len(class_names)

    
    results_df = pd.DataFrame({
        'Emotion': class_names,
        'Precision': precision,
        'Recall': recall,
        'AUC': auc
    })

    print("\nEmotion-wise Evaluation Metrics:")
    log_message("\nEmotion-wise Evaluation Metrics:")
    print(results_df.round(4))
    log_message(results_df.round(4).to_string())
    return results_df

encoder = ModifiedResNet50()
model = HNCL().to(device) 
model.load_state_dict(torch.load("hncl_best_classifier.pth"))
evaluate_per_emotion_metrics(model, test_loader, class_names, mode='classify')

logs = log_message("Last Message")
log_it(logs)