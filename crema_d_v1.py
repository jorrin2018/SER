# -*- coding: utf-8 -*-
"""
Created on Wed May 28 17:19:18 2025

@author: jjorrinc2100
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from speechbrain.pretrained import EncoderClassifier

# ======================
# Configuration flags
# ======================
ENABLE_SPECTRAL = True
ENABLE_SPECTRAL_ATTENTION = False
ENABLE_MELSPECTROGRAM = False
ENABLE_PROSODY = True
ENABLE_PROSODY_ATTENTION = False
ENABLE_XVECTOR = False
ENABLE_SELF_ATTENTION = True
ENABLE_MULTIHEAD_ATTENTION = False
ENABLE_TEMPORAL_ATTENTION = True
ENABLE_CLASS_BALANCING = False
ENABLE_CROSS_VALIDATION = False

# ======================
# Hyperparameters you can tweak
# ======================
NUM_LSTM_LAYERS = 2      # <-- change this to experiment
HIDDEN_DIM = 128
EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-2

# ======================
# Feature dims for dynamic calculation
# ======================
N_MFCC = 40
N_MELS = 64
PROSODIC_FEAT_DIM = 5  # [rms, pitch, zcr, centroid, bandwidth]

# ======================
# CREMA-D labels (from filename part[2])
# ======================
LABEL_LIST = ['ang', 'dis', 'fea', 'hap', 'neu', 'sad']
LABEL2IDX = {lab: idx for idx, lab in enumerate(LABEL_LIST)}


class SERDataset(Dataset):
    def __init__(self, file_list: pd.DataFrame, audio_dir: str,
                 sample_rate=16000, n_mfcc=N_MFCC, n_mels=N_MELS, max_len=200):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.max_len = max_len
        self.metadata = file_list.reset_index(drop=True)
        if ENABLE_XVECTOR:
            self.spk_enc = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        fn = row['path']
        emo = row['emotion']
        label = torch.tensor(LABEL2IDX[emo], dtype=torch.long)

        wav, sr = librosa.load(os.path.join(self.audio_dir, fn), sr=self.sample_rate)

        # Data augmentation
        rate = random.uniform(0.9, 1.1)
        wav = librosa.effects.time_stretch(y=wav, rate=rate)
        wav = wav + 0.005 * np.random.randn(len(wav))

        # No acoustic features in this config
        acoustic = torch.zeros(0, 0, dtype=torch.float)

        # x-vector
        if ENABLE_XVECTOR:
            wav_tensor = torch.tensor(wav, dtype=torch.float).unsqueeze(0)
            emb = self.spk_enc.encode_batch(wav_tensor).squeeze(0)
            xvec = emb.mean(dim=-1) if emb.dim() > 1 else emb
        else:
            xvec = torch.zeros(0, dtype=torch.float)

        return acoustic, xvec, label


class LSTMAttentionWithXVector(nn.Module):
    def __init__(self, acoustic_dim, xvec_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 2-layer BiLSTM (or NUM_LSTM_LAYERS) with dropout between layers
        if acoustic_dim > 0:
            self.lstm = nn.LSTM(
                input_size=acoustic_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.3
            )

        # Temporal attention on the final hidden state
        if ENABLE_TEMPORAL_ATTENTION and acoustic_dim > 0:
            self.attn_fc = nn.Linear(hidden_dim * 2, 1)

        # Final classifier
        fc_in = (hidden_dim * 2 if acoustic_dim > 0 else 0) + xvec_dim
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(fc_in, num_classes)

    def forward(self, acoustic, xvec):
        if acoustic.numel() == 0:
            context = xvec
        else:
            # Run multi-layer BiLSTM
            _, (h, _) = self.lstm(acoustic)
            # h: [num_layers*2, batch, hidden_dim]
            # pick the last layer's forward & backward
            layer_idx = (NUM_LSTM_LAYERS - 1) * 2
            h_fwd = h[layer_idx]
            h_bwd = h[layer_idx + 1]
            h_cat = torch.cat([h_fwd, h_bwd], dim=1)  # [batch, 2*H]

            if ENABLE_TEMPORAL_ATTENTION:
                e = torch.tanh(self.attn_fc(h_cat.unsqueeze(1)))  # [batch,1,1]
                w = torch.softmax(e, dim=1)                       # [batch,1,1]
                context = (w * h_cat.unsqueeze(1)).sum(dim=1)     # [batch,2H]
            else:
                context = h_cat

            if xvec.numel() != 0:
                context = torch.cat([context, xvec], dim=1)

        context = self.dropout(context)
        return self.fc(context)


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    for acoustic, xvec, labels in loader:
        acoustic, xvec, labels = acoustic.to(device), xvec.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(acoustic, xvec)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_true = [], []
    with torch.no_grad():
        for acoustic, xvec, labels in loader:
            acoustic, xvec, labels = acoustic.to(device), xvec.to(device), labels.to(device)
            outputs = model(acoustic, xvec)
            total_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.cpu().tolist())
    return total_loss / len(loader), correct / total, all_true, all_preds


def main():
    audio_dir = "F:/DOCTORADO/DATASETS/CREMA D/AudioWAV"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build metadata
    files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]
    records = []
    for fn in files:
        parts = fn.split("_")
        if len(parts) >= 3:
            emo = parts[2].lower()
            if emo in LABEL_LIST:
                records.append({"path": fn, "emotion": emo})
    df = pd.DataFrame(records)
    print(f"Total samples: {len(df)}")

    # Split
    train_df, val_df = train_test_split(df, test_size=0.25,
                                        stratify=df["emotion"], random_state=42)

    # Datasets & loaders
    train_ds = SERDataset(train_df, audio_dir)
    val_ds   = SERDataset(val_df, audio_dir)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Compute dims
    acoustic_dim = 0
    if ENABLE_SPECTRAL:        acoustic_dim += N_MFCC
    if ENABLE_MELSPECTROGRAM:  acoustic_dim += N_MELS
    if ENABLE_PROSODY:         acoustic_dim += PROSODIC_FEAT_DIM
    xvec_dim = train_ds[0][1].shape[0] if ENABLE_XVECTOR else 0

    # Model
    model = LSTMAttentionWithXVector(
        acoustic_dim, xvec_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LSTM_LAYERS,
        num_classes=len(LABEL_LIST)
    ).to(device)

    # Optimizer, scheduler, loss
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-2,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, patience, no_improve = 0.0, 5, 0
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS+1):
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_acc, y_true, y_pred = eval_epoch(model, val_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if val_acc > best_acc:
            best_acc, no_improve = val_acc, 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            no_improve += 1

        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        if no_improve >= patience:
            print("Early stopping.")
            break

    # Plot losses
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig("loss_curve.png"); plt.show()

    # Report & confusion
    print(classification_report(y_true, y_pred, target_names=LABEL_LIST))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_LIST))))
    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    ticks = range(len(LABEL_LIST))
    plt.xticks(ticks, LABEL_LIST, rotation=45)
    plt.yticks(ticks, LABEL_LIST)
    thresh = cm.max()/2
    for i in ticks:
        for j in ticks:
            plt.text(j, i, cm[i,j], ha="center", va="center",
                     color="white" if cm[i,j]>thresh else "black")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig("confusion_matrix.png"); plt.show()

    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
