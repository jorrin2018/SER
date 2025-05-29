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
ENABLE_SPECTRAL            = True
ENABLE_SPECTRAL_ATTENTION  = False
ENABLE_MELSPECTROGRAM      = False
ENABLE_PROSODY             = True
ENABLE_PROSODY_ATTENTION   = False
ENABLE_XVECTOR             = False
ENABLE_SELF_ATTENTION      = True
ENABLE_MULTIHEAD_ATTENTION = False
ENABLE_TEMPORAL_ATTENTION  = True
ENABLE_CLASS_BALANCING     = False
ENABLE_CROSS_VALIDATION    = False  # not implemented in this script

# ======================
# Hyperparameters
# ======================
NUM_LSTM_LAYERS = 2      # number of stacked LSTM layers
HIDDEN_DIM      = 128    # hidden size per direction
EPOCHS          = 30
BATCH_SIZE      = 64
LR              = 1e-3
WEIGHT_DECAY    = 1e-2

# ======================
# Feature dims (for dynamic acoustic_dim)
# ======================
N_MFCC            = 40
N_MELS            = 64
PROSODIC_FEAT_DIM = 2    # rms + pitch

# ======================
# CREMA-D labels (third token in filename)
# ======================
LABEL_LIST = ['ang','dis','fea','hap','neu','sad']
LABEL2IDX  = {lab: i for i, lab in enumerate(LABEL_LIST)}

class SERDataset(Dataset):
    """Dataset for CREMA-D: filename encodes emotion as the 3rd underscore token."""
    def __init__(self, records: pd.DataFrame, audio_dir: str,
                 sample_rate=16000, n_mfcc=N_MFCC, n_mels=N_MELS, max_len=200):
        self.audio_dir   = audio_dir
        self.sample_rate = sample_rate
        self.n_mfcc      = n_mfcc
        self.n_mels      = n_mels
        self.max_len     = max_len
        self.meta        = records.reset_index(drop=True)
        if ENABLE_XVECTOR:
            self.spk_enc = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device":"cpu"},
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        fn  = self.meta.iloc[idx]['path']
        emo = self.meta.iloc[idx]['emotion']
        label = torch.tensor(LABEL2IDX[emo], dtype=torch.long)

        # load and augment
        wav, sr = librosa.load(os.path.join(self.audio_dir, fn), sr=self.sample_rate)
        rate = random.uniform(0.9, 1.1)
        wav = librosa.effects.time_stretch(y=wav, rate=rate)
        wav = wav + 0.005 * np.random.randn(len(wav))

        feats = []
        # spectral (MFCC)
        if ENABLE_SPECTRAL:
            mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=self.n_mfcc).T  # [T, n_mfcc]
            feats.append(mfcc)
        # mel-spectrogram
        if ENABLE_MELSPECTROGRAM:
            mels = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=self.n_mels).T
            mels_db = librosa.power_to_db(mels)
            feats.append(mels_db)
        # prosodic (rms + pitch)
        if ENABLE_PROSODY:
            rms      = librosa.feature.rms(y=wav).T                        # [T,1]
            pitches,_= librosa.piptrack(y=wav, sr=sr)
            pitch    = np.mean(pitches, axis=0)[None,:].T                  # [T,1]
            pros = np.concatenate([rms, pitch], axis=1)                    # [T,2]
            feats.append(pros)

        # pad/truncate
        if feats:
            feat = np.concatenate(feats, axis=1)                           # [T, D]
            if feat.shape[0] < self.max_len:
                pad = np.zeros((self.max_len - feat.shape[0], feat.shape[1]))
                feat = np.vstack([feat, pad])
            else:
                feat = feat[:self.max_len]
            acoustic = torch.tensor(feat, dtype=torch.float)               # [max_len, D]
        else:
            acoustic = torch.zeros(0, 0, dtype=torch.float)

        # x-vector
        if ENABLE_XVECTOR:
            wav_t = torch.tensor(wav, dtype=torch.float).unsqueeze(0)
            emb   = self.spk_enc.encode_batch(wav_t).squeeze(0)
            xvec  = emb.mean(dim=-1) if emb.dim()>1 else emb
        else:
            xvec = torch.zeros(0, dtype=torch.float)

        return acoustic, xvec, label

class LSTMAttentionWithXVector(nn.Module):
    """Model: optional self-attn, BiLSTM, attention, x-vector concat."""
    def __init__(self, acoustic_dim, xvec_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        # self-attention pre-LSTM
        if ENABLE_SELF_ATTENTION and acoustic_dim>0:
            heads = 4
            while acoustic_dim % heads != 0 and heads>1:
                heads -= 1
            self.self_attn = nn.MultiheadAttention(acoustic_dim, num_heads=heads, batch_first=True)
        # BiLSTM
        if acoustic_dim>0:
            self.lstm = nn.LSTM(
                input_size=acoustic_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.3
            )
        # post-LSTM attention
        if ENABLE_MULTIHEAD_ATTENTION and acoustic_dim>0:
            heads = 4
            total = hidden_dim*2
            while total % heads != 0 and heads>1:
                heads -= 1
            self.post_attn = nn.MultiheadAttention(total, num_heads=heads, batch_first=True)
        elif ENABLE_TEMPORAL_ATTENTION and acoustic_dim>0:
            self.attn_fc = nn.Linear(hidden_dim*2, 1)
        # final classifier
        fc_in = (hidden_dim*2 if acoustic_dim>0 else 0) + xvec_dim
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(fc_in, num_classes)

    def forward(self, acoustic, xvec):
        # bypass if no acoustic
        if acoustic.numel()==0:
            context = xvec
        else:
            x = acoustic.unsqueeze(0) if acoustic.ndim==2 else acoustic  # [B,T,D] or [T,D]
            if ENABLE_SELF_ATTENTION:
                x, _ = self.self_attn(x, x, x)
            hseq, _ = self.lstm(x)
            if ENABLE_MULTIHEAD_ATTENTION:
                hseq, _ = self.post_attn(hseq, hseq, hseq)
                context = hseq.mean(dim=1)
            elif ENABLE_TEMPORAL_ATTENTION:
                e = torch.tanh(self.attn_fc(hseq))    # [B,T,1]
                w = torch.softmax(e, dim=1)
                context = (w * hseq).sum(dim=1)
            else:
                context = hseq.mean(dim=1)
            if xvec.numel()!=0:
                context = torch.cat([context, xvec], dim=1)
        context = self.dropout(context)
        return self.fc(context)

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total = 0.0
    for acoustic, xvec, labels in loader:
        acoustic, xvec, labels = acoustic.to(device), xvec.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(acoustic, xvec)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / len(loader)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total = correct = count = 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for acoustic, xvec, labels in loader:
            acoustic, xvec, labels = acoustic.to(device), xvec.to(device), labels.to(device)
            out = model(acoustic, xvec)
            total += criterion(out, labels).item()
            preds = out.argmax(dim=1)
            correct += (preds==labels).sum().item()
            count += labels.size(0)
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    return total/len(loader), correct/count, all_true, all_pred

def main():
    audio_dir = "F:/DOCTORADO/DATASETS/CREMA D/AudioWAV"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build metadata
    files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    recs  = [{"path":f,"emotion":f.split("_")[2].lower()} 
             for f in files if f.split("_")[2].lower() in LABEL_LIST]
    df    = pd.DataFrame(recs)
    print(f"Total files: {len(df)}")

    # train/val split
    train_df, val_df = train_test_split(
        df, test_size=0.25, stratify=df["emotion"], random_state=42
    )

    # optional class balancing
    sampler = None
    if ENABLE_CLASS_BALANCING:
        counts = train_df["emotion"].value_counts().to_dict()
        weights = [1.0/counts[e] for e in train_df["emotion"]]
        sampler = WeightedRandomSampler(weights, len(weights))

    train_ds = SERDataset(train_df, audio_dir)
    val_ds   = SERDataset(val_df, audio_dir)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=(sampler is None), sampler=sampler
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # compute input dims
    acoustic_dim = 0
    if ENABLE_SPECTRAL:       acoustic_dim += N_MFCC
    if ENABLE_MELSPECTROGRAM: acoustic_dim += N_MELS
    if ENABLE_PROSODY:        acoustic_dim += PROSODIC_FEAT_DIM
    xvec_dim = train_ds[0][1].shape[0] if ENABLE_XVECTOR else 0

    # model, optimizer, scheduler, loss
    model = LSTMAttentionWithXVector(
        acoustic_dim, xvec_dim, HIDDEN_DIM, NUM_LSTM_LAYERS, len(LABEL_LIST)
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer, max_lr=1e-2,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS, pct_start=0.1,
        div_factor=10, final_div_factor=100
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS+1):
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        vl_loss, vl_acc, y_true, y_pred = eval_epoch(model, val_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), "best_model.pt")
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, val_loss={vl_loss:.4f}, val_acc={vl_acc:.4f}")

    # plot losses
    plt.figure(); plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig("loss_curve.png"); plt.show()

    # report & confusion
    print(classification_report(y_true, y_pred, target_names=LABEL_LIST))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_LIST))))
    plt.figure(figsize=(6,6)); plt.imshow(cm, cmap="Blues", interpolation="nearest")
    ticks = range(len(LABEL_LIST))
    plt.xticks(ticks, LABEL_LIST, rotation=45); plt.yticks(ticks, LABEL_LIST)
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
