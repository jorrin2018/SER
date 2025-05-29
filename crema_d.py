# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:30:52 2025

@author: jjorrinc2100
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
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
ENABLE_MELSPECTROGRAM = True
ENABLE_PROSODY = True
ENABLE_PROSODY_ATTENTION = False
ENABLE_XVECTOR = False
ENABLE_SELF_ATTENTION = True        # Self-attention before LSTM
ENABLE_MULTIHEAD_ATTENTION = False   # Multi-head attention after LSTM
ENABLE_TEMPORAL_ATTENTION = True   # Single-head temporal attention
ENABLE_CLASS_BALANCING = False
ENABLE_CROSS_VALIDATION = False

# ======================
# Feature dims (for dynamic acoustic_dim calculation)
# ======================
N_MFCC = 40
N_MELS = 64
PROSODIC_FEAT_DIM = 5  # [rms, pitch, zcr, spec_cent, spec_bw]

# ======================
# CREMA-D labels (from filename part[2]), no 'xxx' or 'oth'
# ======================
LABEL_LIST = ['ang', 'dis', 'fea', 'hap', 'neu', 'sad']
LABEL2IDX = {lab: idx for idx, lab in enumerate(LABEL_LIST)}


class SERDataset(Dataset):
    """
    Dataset for SER on CREMA-D where filenames encode emotion:
      e.g. "1001_DFA_ANG_XX.wav"
    Expects metadata DataFrame with columns ['path','emotion'].
    """
    def __init__(self, metadata: pd.DataFrame, audio_dir: str,
                 sample_rate=16000, n_mfcc=N_MFCC, n_mels=N_MELS, max_len=200):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.max_len = max_len
        self.metadata = metadata.reset_index(drop=True)
        # Pretrained x-vector encoder (CPU)
        self.spk_enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device":"cpu"},
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = row['path']
        label_str = row['emotion']
        label = torch.tensor(LABEL2IDX[label_str], dtype=torch.long)

        # load waveform
        wav_path = os.path.join(self.audio_dir, filename)
        wav, sr = librosa.load(wav_path, sr=self.sample_rate)

        feats = []

        # 1) Spectral: MFCC
        if ENABLE_SPECTRAL:
            mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=self.n_mfcc).T  # [T, n_mfcc]
            if ENABLE_SPECTRAL_ATTENTION:
                attn_w = torch.softmax(
                    nn.Linear(self.n_mfcc, 1)(torch.tensor(mfcc, dtype=torch.float)),
                    dim=0
                ).detach().numpy()
                mfcc = mfcc * attn_w
            feats.append(mfcc)

        # 2) Mel-Spectrogram
        if ENABLE_MELSPECTROGRAM:
            melspec = librosa.feature.melspectrogram(
                y=wav, sr=sr, n_mels=self.n_mels, fmax=sr//2
            ).T                                            # [T, n_mels]
            melspec_db = librosa.power_to_db(melspec)
            feats.append(melspec_db)

        # 3) Prosodic features
        if ENABLE_PROSODY:
            rms = librosa.feature.rms(y=wav).T                             # [T,1]
            pitches, _ = librosa.piptrack(y=wav, sr=sr)
            pitch = np.mean(pitches, axis=0)[None, :].T                    # [T,1]
            zcr = librosa.feature.zero_crossing_rate(y=wav).T             # [T,1]
            cent = librosa.feature.spectral_centroid(y=wav, sr=sr).T      # [T,1]
            bw = librosa.feature.spectral_bandwidth(y=wav, sr=sr).T        # [T,1]
            pros = np.concatenate([rms, pitch, zcr, cent, bw], axis=1)    # [T,5]
            if ENABLE_PROSODY_ATTENTION:
                attn_w = torch.softmax(
                    nn.Linear(pros.shape[1], 1)(torch.tensor(pros, dtype=torch.float)),
                    dim=0
                ).detach().numpy()
                pros = pros * attn_w
            feats.append(pros)

        # concatenate frame-wise features and pad/truncate to max_len
        if feats:
            feat = np.concatenate(feats, axis=1)  # [T, total_dim]
            if feat.shape[0] < self.max_len:
                pad = np.zeros((self.max_len - feat.shape[0], feat.shape[1]))
                feat = np.vstack([feat, pad])
            else:
                feat = feat[:self.max_len]
            acoustic = torch.tensor(feat, dtype=torch.float)
        else:
            acoustic = torch.zeros(0, 0, dtype=torch.float)

        # x-vector embedding
        if ENABLE_XVECTOR:
            wav_tensor = torch.tensor(wav, dtype=torch.float).unsqueeze(0)
            emb = self.spk_enc.encode_batch(wav_tensor)  # [1, xdim] or [1, xdim,1]
            xvec = emb.squeeze(0)
            if xvec.dim() > 1:
                xvec = xvec.mean(dim=-1)
        else:
            xvec = torch.zeros(0, dtype=torch.float)

        return acoustic, xvec, label


class LSTMAttentionWithXVector(nn.Module):
    """
    SER model: optional self-attn, BiLSTM, temporal or multi-head attn,
    plus x-vector concatenation.
    """
    def __init__(self, acoustic_dim, xvec_dim, hidden_dim, num_classes):
        super().__init__()
        self.acoustic_dim = acoustic_dim
        self.xvec_dim = xvec_dim
        self.hidden_dim = hidden_dim

        # self-attention pre-LSTM
        if ENABLE_SELF_ATTENTION and acoustic_dim > 0:
            heads = 4
            while acoustic_dim % heads != 0 and heads > 1:
                heads -= 1
            self.self_attn = nn.MultiheadAttention(acoustic_dim, num_heads=heads, batch_first=True)

        # BiLSTM
        if acoustic_dim > 0:
            self.lstm = nn.LSTM(acoustic_dim, hidden_dim, batch_first=True, bidirectional=True)

        # post-LSTM attention
        if ENABLE_MULTIHEAD_ATTENTION and acoustic_dim > 0:
            heads = 4
            total_dim = hidden_dim * 2
            while total_dim % heads != 0 and heads > 1:
                heads -= 1
            self.post_attn = nn.MultiheadAttention(total_dim, num_heads=heads, batch_first=True)
        elif ENABLE_TEMPORAL_ATTENTION and acoustic_dim > 0:
            self.attn_fc = nn.Linear(hidden_dim * 2, 1)

        # final classifier
        fc_in = (hidden_dim * 2 if acoustic_dim > 0 else 0) + (xvec_dim if ENABLE_XVECTOR else 0)
        self.fc = nn.Linear(fc_in, num_classes)

    def forward(self, acoustic, xvec):
        # bypass acoustic branch
        if acoustic.numel() == 0:
            context = xvec
        else:
            x = acoustic  # [B, T, D]

            if ENABLE_SELF_ATTENTION:
                x, _ = self.self_attn(x, x, x)

            hseq, _ = self.lstm(x)  # [B, T, 2H]

            if ENABLE_MULTIHEAD_ATTENTION:
                hseq, _ = self.post_attn(hseq, hseq, hseq)
                context = hseq.mean(dim=1)
            else:
                e = torch.tanh(self.attn_fc(hseq))  # [B, T, 1]
                w = torch.softmax(e, dim=1)
                context = (w * hseq).sum(dim=1)     # [B, 2H]

            if ENABLE_XVECTOR:
                context = torch.cat([context, xvec], dim=1)

        return self.fc(context)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for acoustic, xvec, labels in loader:
        acoustic, xvec, labels = acoustic.to(device), xvec.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(acoustic, xvec)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
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
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.cpu().tolist())
    return total_loss / len(loader), correct / total, all_true, all_preds


def main():
    audio_dir = "F:/DOCTORADO/DATASETS/CREMA D/AudioWAV"  # adjust to your CREMA-D folder path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # scan directory and build metadata DataFrame
    files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]
    data = []
    for fn in files:
        parts = fn.split("_")
        if len(parts) >= 3:
            emo = parts[2].lower()
            if emo in LABEL_LIST:
                data.append({"path": fn, "emotion": emo})
    df = pd.DataFrame(data)
    print(f"Total CREMA-D samples loaded: {len(df)}")

    # train/val split
    train_df, val_df = train_test_split(df, test_size=0.25,
                                        stratify=df["emotion"], random_state=42)

    # optional class balancing
    sampler = None
    if ENABLE_CLASS_BALANCING:
        counts = train_df["emotion"].value_counts().to_dict()
        weights = [1.0 / counts[e] for e in train_df["emotion"]]
        sampler = WeightedRandomSampler(weights, len(weights))

    # datasets & loaders
    train_ds = SERDataset(train_df, audio_dir)
    val_ds   = SERDataset(val_df, audio_dir)
    train_loader = DataLoader(train_ds, batch_size=16,
                              shuffle=(sampler is None), sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=16)

    # dynamic acoustic_dim
    acoustic_dim = 0
    if ENABLE_SPECTRAL:        acoustic_dim += N_MFCC
    if ENABLE_MELSPECTROGRAM:  acoustic_dim += N_MELS
    if ENABLE_PROSODY:         acoustic_dim += PROSODIC_FEAT_DIM
    xvec_dim = 0  # x-vector disabled in this config

    model = LSTMAttentionWithXVector(
        acoustic_dim, xvec_dim, hidden_dim=128, num_classes=len(LABEL_LIST)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # training loop
    best_acc = 0.0
    train_losses, val_losses = [], []
    patience = 5
    no_improve = 0

    for epoch in range(1, 31):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
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

    # loss curves
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig("loss_curve.png"); plt.show()

    # report & confusion
    print(classification_report(y_true, y_pred, target_names=LABEL_LIST))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_LIST))))
    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.xticks(range(len(LABEL_LIST)), LABEL_LIST, rotation=45)
    plt.yticks(range(len(LABEL_LIST)), LABEL_LIST)
    thresh = cm.max() / 2
    for i in range(len(LABEL_LIST)):
        for j in range(len(LABEL_LIST)):
            plt.text(j, i, cm[i,j],
                     ha="center", va="center",
                     color="white" if cm[i,j] > thresh else "black")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig("confusion_matrix.png"); plt.show()

    print(f"Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
