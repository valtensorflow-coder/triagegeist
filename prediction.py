import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Ajouter le chemin parent pour les imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from analyze_data.data_train import df_train
from analyze_data.data_test import df_test
from model import TriageModel

os.makedirs("models", exist_ok=True)

# ──────────────────────────────────────────
# DATA PREPARATION
# ──────────────────────────────────────────

X_train = torch.tensor(df_train.values, dtype=torch.float32)
y_train = torch.tensor(df_train["triage_acuity"].values, dtype=torch.float32).unsqueeze(1)

print("  \n- - - - Input/Output Train- - - -\n  ")
print(f"X.shape : {X_train.shape} \n y.shape : {y_train.shape}")

dataset_train = TensorDataset(X_train, y_train)
trainloader = DataLoader(dataset_train, batch_size=16, shuffle=True)

X_test = torch.tensor(df_test.values, dtype=torch.float32)
missing_cols = set(df_train.columns) - set(df_test.columns)
for col in missing_cols:
    df_test[col] = 0
df_test = df_test[df_train.columns]
X_test = torch.tensor(df_test.values, dtype=torch.float32)

print(f"X_test.shape : {X_test.shape}")

missing = set(df_train.columns) - set(df_test.columns)
extra   = set(df_test.columns) - set(df_train.columns)

print("Manquantes dans test :", missing)
print("En trop dans test    :", extra)

dataset_test = TensorDataset(X_test)
testloader = DataLoader(dataset_test, batch_size=16, shuffle=False)

print("  \n- - - - Input/Output Test- - - -\n  ")
print(f"X.shape : {X_test.shape} \n")

# ──────────────────────────────────────────
# EARLY STOPPING
# ──────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

# ──────────────────────────────────────────
# PLOT FUNCTION
# ──────────────────────────────────────────
def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Train Loss", color="red", linewidth=2)
    best_epoch = losses.index(min(losses))
    best_loss = min(losses)
    plt.scatter(best_epoch, best_loss, color="green", zorder=5, s=100)
    plt.axvline(x=best_epoch, color="green", linestyle=":", 
                label=f"Meilleur epoch {best_epoch} ({best_loss:.4f})")
    plt.title("Évolution de la Train Loss", fontsize=16, fontweight="bold")
    plt.xlabel("Époques", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()

# ──────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TriageModel(n_features=40).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    losses = []

    for epoch in range(25):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for X_batch, y_batch in trainloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.squeeze(1).long().to(device) - 1
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            batch = X_batch.size(0)
            total_loss += loss.item() * batch
            n_samples += batch

        avg_loss = total_loss / n_samples
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        early_stopping(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d} — Loss : {avg_loss:.6f} — LR : {current_lr:.6f}")

        if early_stopping.stop:
            print(f"Early stopping à l'epoch {epoch}")
            break

    # Sauvegarder le modèle
    torch.save(model.state_dict(), "models/triage_model.pth")
    print("Modèle sauvegardé dans models/triage_model.pth")

    # PRÉDICTIONS
    model.eval()
    all_preds = []

    with torch.no_grad():
        for (X_batch,) in testloader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probas = torch.softmax(logits, dim=1)
            preds = probas.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())

    # Reconvertir en ESI 1-5
    all_preds = [p + 1 for p in all_preds]

    # Afficher la courbe de loss
    plot_loss(losses)

    print(f"Nombre de prédictions : {len(all_preds)}")
    print(f"Distribution : {pd.Series(all_preds).value_counts().sort_index()}")

    # Créer fichier de soumission
    data_submission = pd.read_csv("data/sample_submission.csv")
    data_submission["triage_acuity"] = all_preds
    data_submission.to_csv("submission.csv", index=False)
    print("✅ Fichier submission.csv créé !")