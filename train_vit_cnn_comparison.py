import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import random
import time
from multiprocessing import freeze_support

# ====================== CONFIGURATION COMMUNE ======================
parser = argparse.ArgumentParser(description="Compare un ViT entraîné de zéro et un ResNet18 sur Intel Image Classification.")
parser.add_argument("--data-dir", default="data", help="Dossier qui contient seg_train et seg_test.")
parser.add_argument("--output-dir", default="resultats", help="Dossier des figures, modèles et prédictions.")
parser.add_argument("--epochs", type=int, default=20)
ARGS, _ = parser.parse_known_args()
DATA_ROOT = ARGS.data_dir
OUTPUT_DIR = ARGS.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
BATCH_SIZE = 64
EPOCHS = ARGS.epochs
LEARNING_RATE = 3e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0 if device == 'cpu' else min(4, os.cpu_count() or 1)
PIN_MEMORY = device == 'cuda'

torch.manual_seed(42)
random.seed(42)

CLASSES_DICT = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}
CLASSES_LIST = list(CLASSES_DICT.keys())
NUM_CLASSES = len(CLASSES_LIST)

print(f" Utilisation de : {device} | Entraînement des DEUX modèles (ViT + CNN)")

# ====================== FONCTIONS COMMUNES ======================
def data_loader(path):
    # L'archive Kaggle contient un sous-dossier du même nom (seg_train/seg_train)
    imbrique = os.path.join(path, os.path.basename(path))
    if os.path.isdir(imbrique):
        path = imbrique
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        if os.path.isdir(label_path):
            for image in os.listdir(label_path):
                data.append({'Image_path': os.path.join(label_path, image), 'Class': label})
    return pd.DataFrame(data)

class IntelDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        label_str = self.data.iloc[idx, 1]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = CLASSES_DICT[label_str]
        return image, label

# ====================== FONCTIONS ENTRAÎNEMENT ======================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in tqdm(loader, desc="Train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# ====================== MAIN ======================
if __name__ == "__main__":
    freeze_support()

    # CHARGEMENT DONNÉES (une seule fois)
    df_train = data_loader(os.path.join(DATA_ROOT, 'seg_train'))
    df_test  = data_loader(os.path.join(DATA_ROOT, 'seg_test'))
    print(f" Train images : {len(df_train)} | Test images : {len(df_test)}")

    # ====================== PARTIE 1 : ViT ======================
    print("\n" + "="*60)
    print(" ENTRAÎNEMENT ViT (from scratch)")
    print("="*60)

    IMAGE_SIZE_VIT = 256
    train_transform_vit = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE_VIT, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(15), 
        transforms.ColorJitter(0.2, 0.2, 0.2), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    test_transform_vit  = transforms.Compose([
        transforms.Resize((IMAGE_SIZE_VIT, IMAGE_SIZE_VIT)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    full_train_vit = IntelDataset(df_train, transform=train_transform_vit)
    test_vit = IntelDataset(df_test, transform=test_transform_vit)
    train_size = int(0.8 * len(full_train_vit))
    train_dataset_vit, val_dataset_vit = random_split(full_train_vit, [train_size, len(full_train_vit)-train_size])

    train_loader_vit = DataLoader(train_dataset_vit, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader_vit   = DataLoader(val_dataset_vit,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader_vit  = DataLoader(test_vit,          batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Classes ViT
    class PatchEmbedding(nn.Module):
        def __init__(self, image_size, patch_size, in_channels, embed_dim):
            super().__init__()
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
            num_patches = (image_size // patch_size) ** 2
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        def forward(self, x):
            B = x.shape[0]
            x = self.proj(x).flatten(2).transpose(1, 2)
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.pos_embedding
            return x

    class MLP(nn.Module):
        def __init__(self, in_features, hidden_features, drop_rate):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, in_features)
            self.dropout = nn.Dropout(drop_rate)
        def forward(self, x):
            x = self.dropout(F.gelu(self.fc1(x)))
            x = self.dropout(self.fc2(x))
            return x

    class TransformerEncoderLayer(nn.Module):
        def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate):
            super().__init__()
            self.norm1 = nn.LayerNorm(embed_dim)
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.mlp = MLP(embed_dim, mlp_dim, drop_rate)
        def forward(self, x):
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            x = x + self.mlp(self.norm2(x))
            return x

    class ViT(nn.Module):
        def __init__(self, image_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, drop_rate):
            super().__init__()
            self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
            self.encoder = nn.Sequential(*[TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, drop_rate) for _ in range(depth)])
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, num_classes)
        def forward(self, x):
            x = self.patch_embedding(x)
            x = self.encoder(x)
            x = self.norm(x)
            cls_token = x[:, 0]
            return self.head(cls_token)

    model_vit = ViT(IMAGE_SIZE_VIT, 16, 3, NUM_CLASSES, 384, 6, 6, 1536, 0.4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_vit = optim.Adam(model_vit.parameters(), lr=LEARNING_RATE)

    t0_vit = time.time()
    train_losses_vit, train_acc_vit = [], []
    val_losses_vit, val_acc_vit = [], []

    for epoch in range(EPOCHS):
        t_loss, t_acc = train_epoch(model_vit, train_loader_vit, optimizer_vit, criterion)
        v_loss, v_acc = eval_epoch(model_vit, val_loader_vit, criterion)
        train_losses_vit.append(t_loss); train_acc_vit.append(t_acc)
        val_losses_vit.append(v_loss); val_acc_vit.append(v_acc)
        print(f"ViT Epoch {epoch+1}/{EPOCHS} | Train Acc: {t_acc*100:.2f}% | Val Acc: {v_acc*100:.2f}%")

    duree_vit = time.time() - t0_vit

    # Courbes ViT
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1); plt.plot(train_acc_vit, label='Train ViT'); plt.plot(val_acc_vit, label='Val ViT'); plt.title("Accuracy ViT"); plt.legend(); plt.grid()
    plt.subplot(1, 2, 2); plt.plot(train_losses_vit, label='Train ViT'); plt.plot(val_losses_vit, label='Val ViT'); plt.title("Loss ViT"); plt.legend(); plt.grid()
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "courbes_vit.png"), dpi=120); plt.close()

    # ====================== PARTIE 2 : CNN ResNet18 ======================
    print("\n" + "="*60)
    print(" ENTRAÎNEMENT CNN ResNet18 (Transfer Learning)")
    print("="*60)

    IMAGE_SIZE_CNN = 224
    train_transform_cnn = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE_CNN, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(15), 
        transforms.ColorJitter(0.2, 0.2, 0.2), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform_cnn  = transforms.Compose([
        transforms.Resize((IMAGE_SIZE_CNN, IMAGE_SIZE_CNN)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_train_cnn = IntelDataset(df_train, transform=train_transform_cnn)
    test_cnn = IntelDataset(df_test, transform=test_transform_cnn)
    train_dataset_cnn, val_dataset_cnn = random_split(full_train_cnn, [train_size, len(full_train_cnn)-train_size])

    train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader_cnn   = DataLoader(val_dataset_cnn,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader_cnn  = DataLoader(test_cnn,          batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model_cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model_cnn.fc = nn.Linear(model_cnn.fc.in_features, NUM_CLASSES)
    model_cnn = model_cnn.to(device)
    optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=LEARNING_RATE)

    t0_cnn = time.time()
    train_losses_cnn, train_acc_cnn = [], []
    val_losses_cnn, val_acc_cnn = [], []

    for epoch in range(EPOCHS):
        t_loss, t_acc = train_epoch(model_cnn, train_loader_cnn, optimizer_cnn, criterion)
        v_loss, v_acc = eval_epoch(model_cnn, val_loader_cnn, criterion)
        train_losses_cnn.append(t_loss); train_acc_cnn.append(t_acc)
        val_losses_cnn.append(v_loss); val_acc_cnn.append(v_acc)
        print(f"CNN Epoch {epoch+1}/{EPOCHS} | Train Acc: {t_acc*100:.2f}% | Val Acc: {v_acc*100:.2f}%")

    duree_cnn = time.time() - t0_cnn

    # Courbes CNN
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1); plt.plot(train_acc_cnn, label='Train CNN'); plt.plot(val_acc_cnn, label='Val CNN'); plt.title("Accuracy CNN"); plt.legend(); plt.grid()
    plt.subplot(1, 2, 2); plt.plot(train_losses_cnn, label='Train CNN'); plt.plot(val_losses_cnn, label='Val CNN'); plt.title("Loss CNN"); plt.legend(); plt.grid()
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "courbes_resnet18.png"), dpi=120); plt.close()

    # ====================== ÉVALUATION FINALE & COMPARAISON ======================
    print("\n" + "="*80)
    print(" RAPPORTS FINAUX ET COMPARAISON ViT vs CNN")
    print("="*80)

    # ViT
    print("\n=== ViT sur seg_test ===")
    model_vit.eval()
    all_preds_vit, all_labels = [], []
    vit_image_paths = []
    
    with torch.no_grad():
        for images, labels in test_loader_vit:
            images = images.to(device)
            outputs = model_vit(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds_vit.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Récupérer les chemins pour le CSV (Optionnel, lié à l'ordre du dataset)
    vit_image_paths = df_test['Image_path'].tolist()

    print(classification_report(all_labels, all_preds_vit, target_names=CLASSES_LIST, digits=4))

    # CNN
    print("\n=== ResNet18 (CNN) sur seg_test ===")
    model_cnn.eval()
    all_preds_cnn = []
    
    with torch.no_grad():
        for images, labels in test_loader_cnn:
            images = images.to(device)
            outputs = model_cnn(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds_cnn.extend(preds)
            
    print(classification_report(all_labels, all_preds_cnn, target_names=CLASSES_LIST, digits=4))

    # Matrice de confusion pour les deux
    cm_vit = confusion_matrix(all_labels, all_preds_vit)
    cm_cnn = confusion_matrix(all_labels, all_preds_cnn)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm_vit, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST, ax=ax[0])
    ax[0].set_title('Matrice de confusion - ViT')
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST, ax=ax[1])
    ax[1].set_title('Matrice de confusion - ResNet18 CNN')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, 'matrices_confusion.png'), dpi=120); plt.close()

    # Tableau de comparaison (Calcul des métriques finales)
    acc_vit_final = (np.array(all_preds_vit) == np.array(all_labels)).mean() * 100
    acc_cnn_final = (np.array(all_preds_cnn) == np.array(all_labels)).mean() * 100

    print("\n=== TABLEAU DE COMPARAISON ViT vs CNN ===")
    print(f"| Critère                | ViT from scratch      | ResNet18 CNN         |")
    print(f"|------------------------|-----------------------|----------------------|")
    print(f"| Accuracy Test          | {acc_vit_final:6.2f} %              | {acc_cnn_final:6.2f} %             |")
    print(f"| Temps entraînement     | {duree_vit/60:6.1f} min ({device})   | {duree_cnn/60:6.1f} min ({device})  |")

    # ====================== SAUVEGARDE DES FICHIERS ======================
    print("\n Sauvegarde des modèles et prédictions...")
    
    # 1. Sauvegarde des poids
    torch.save(model_vit.state_dict(), os.path.join(OUTPUT_DIR, 'vit_scene_classification.pth'))
    torch.save(model_cnn.state_dict(), os.path.join(OUTPUT_DIR, 'resnet_scene_classification.pth'))
    
    # 2. Sauvegarde des prédictions (CSV)
    df_vit_preds = pd.DataFrame({'Image_path': vit_image_paths, 'True_Label': [CLASSES_LIST[i] for i in all_labels], 'Predicted_Label': [CLASSES_LIST[i] for i in all_preds_vit]})
    df_cnn_preds = pd.DataFrame({'Image_path': vit_image_paths, 'True_Label': [CLASSES_LIST[i] for i in all_labels], 'Predicted_Label': [CLASSES_LIST[i] for i in all_preds_cnn]})
    
    df_vit_preds.to_csv(os.path.join(OUTPUT_DIR, 'predictions_seg_test_ViT.csv'), index=False)
    df_cnn_preds.to_csv(os.path.join(OUTPUT_DIR, 'predictions_seg_test_CNN.csv'), index=False)

    print("\n Tout est terminé !")
    print(f"   Résultats enregistrés dans : {OUTPUT_DIR}")
