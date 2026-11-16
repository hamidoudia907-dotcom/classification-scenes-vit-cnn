"""Affine un ViT préentraîné (timm) sur Intel Image Classification et sauvegarde model_vit.pth."""
import argparse, os, glob, time, random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import timm

parser = argparse.ArgumentParser(description="Affinage d'un ViT préentraîné sur Intel Image Classification.")
parser.add_argument("--data-dir", default="data", help="Dossier qui contient seg_train et seg_test.")
parser.add_argument("--output-dir", default="resultats", help="Dossier où sauvegarder le modèle.")
ARGS = parser.parse_args()
BASE_DIR, RESULT_DIR = ARGS.data_dir, ARGS.output_dir

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE, BATCH_SIZE, VAL_SPLIT, EPOCHS_VIT, SEED = (224, 224), 32, 0.2, 8, 42
def resoudre(nom):
    # L'archive Kaggle contient un sous-dossier du même nom (seg_train/seg_train)
    d = os.path.join(BASE_DIR, nom)
    return os.path.join(d, nom) if os.path.isdir(os.path.join(d, nom)) else d
TRAIN_DIR, TEST_DIR = resoudre("seg_train"), resoudre("seg_test")
os.makedirs(RESULT_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilise : {device}")
for d in (TRAIN_DIR, TEST_DIR):
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Dossier introuvable : {d} -> verifie --data-dir.")

class IntelDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform; self.samples = []
        for label, cls in enumerate(CLASS_NAMES):
            for img in glob.glob(os.path.join(root, cls, "*.*")):
                self.samples.append((img, label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, y = self.samples[i]; img = Image.open(p).convert("RGB")
        return (self.transform(img) if self.transform else img), y

MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
tf_train = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
tf_eval = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)])

ds_aug = IntelDataset(TRAIN_DIR, tf_train); ds_clean = IntelDataset(TRAIN_DIR, tf_eval)
test_ds = IntelDataset(TEST_DIR, tf_eval)
if len(ds_aug) == 0 or len(test_ds) == 0:
    raise RuntimeError("Aucune image trouvee : verifie que TRAIN_DIR/TEST_DIR contiennent bien les sous-dossiers de classes.")
n = len(ds_aug); split = int((1 - VAL_SPLIT) * n)
perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED)).tolist()
train_ds = Subset(ds_aug, perm[:split]); val_ds = Subset(ds_clean, perm[split:])
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0)
print(f"ViT : {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=NUM_CLASSES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_VIT)
criterion = nn.CrossEntropyLoss()

def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    tot, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            if train: optimizer.zero_grad()
            out = model(x); loss = criterion(out, y)
            if train: loss.backward(); optimizer.step()
            tot += loss.item(); correct += (out.argmax(1) == y).sum().item(); total += y.size(0)
            if train and i % 20 == 0:
                print(f"    batch {i}/{len(loader)}  loss={loss.item():.3f}  acc={correct/total:.3f}", flush=True)
    return tot/len(loader), correct/total

best_val, best_state = 0.0, None
t0 = time.time()
for e in range(EPOCHS_VIT):
    print(f"\n--- Epoque {e+1}/{EPOCHS_VIT} ---", flush=True)
    tl, ta = run_epoch(train_loader, True)
    vl, va = run_epoch(val_loader, False)
    scheduler.step()
    print(f"Epoque {e+1} : train_acc={ta:.4f}  val_acc={va:.4f}", flush=True)
    if va > best_val: best_val = va; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

print(f"\nTermine en {time.time()-t0:.0f}s | meilleure val_acc = {best_val:.4f}")
model.load_state_dict(best_state)
_, test_acc = run_epoch(test_loader, False)
print(f"Precision test ViT : {test_acc:.4f}")
out = os.path.join(RESULT_DIR, "model_vit.pth")
torch.save(model.state_dict(), out)
print(f"\nModele sauvegarde : {out}")
