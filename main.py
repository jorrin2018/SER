# -*- coding: utf-8 -*-
"""
Created on Thu May 22 16:58:29 2025

@author: jjorrinc2100
"""
"""
Speech Emotion Recognition (SER) pipeline with modular features,
configurable attention (single vs. multi-head, self vs. temporal),
class balancing, and optional cross-validation.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from speechbrain.pretrained import EncoderClassifier

# ======================
# Configuration flags
# ======================
ENABLE_SPECTRAL = True
ENABLE_SPECTRAL_ATTENTION = True
ENABLE_PROSODY = True
ENABLE_PROSODY_ATTENTION = True
ENABLE_XVECTOR = False
ENABLE_SELF_ATTENTION = False       # Self-attention before LSTM
ENABLE_MULTIHEAD_ATTENTION = True  # Multi-head attention after LSTM
ENABLE_TEMPORAL_ATTENTION = True    # Single-head temporal attention
ENABLE_CLASS_BALANCING = True      # Weighted sampling for classes
ENABLE_CROSS_VALIDATION = True     # K-fold cross-validation

class SERDataset(Dataset):
    """
    Speech Emotion Recognition dataset with modular feature extraction.
    """
    def __init__(self, metadata, audio_dir, label2idx,
                 sample_rate=16000, n_mfcc=40, max_len=200):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.audio_dir = audio_dir
        self.label2idx = label2idx
        self.num_classes = len(label2idx)
        self.file_list = metadata['path'].tolist()
        self.labels = metadata['emotion'].tolist()
        if ENABLE_XVECTOR:
            self.spk_enc = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load waveform
        path = os.path.join(self.audio_dir, self.file_list[idx])
        wav, _ = librosa.load(path, sr=self.sample_rate)
        feats = []
        # Spectral features (MFCC)
        if ENABLE_SPECTRAL:
            spec = librosa.feature.mfcc(y=wav, sr=self.sample_rate, n_mfcc=self.n_mfcc).T
            if ENABLE_SPECTRAL_ATTENTION:
                att = F.softmax(torch.tensor(spec), dim=-1)
                spec = spec * att.numpy()
            feats.append(spec)
        # Prosodic features
        if ENABLE_PROSODY:
            rms = librosa.feature.rms(y=wav).T
            pitches, _ = librosa.piptrack(y=wav, sr=self.sample_rate)
            pitch = np.mean(pitches, axis=0)[None, :].T
            pro = np.concatenate([rms, pitch], axis=1)
            if ENABLE_PROSODY_ATTENTION:
                att = F.softmax(torch.tensor(pro), dim=-1)
                pro = pro * att.numpy()
            feats.append(pro)
        # Combine features
        feat = np.concatenate(feats, axis=1)
        T, D = feat.shape
        if T < self.max_len:
            pad = np.zeros((self.max_len - T, D))
            feat = np.vstack([feat, pad])
        else:
            feat = feat[:self.max_len]
        acoustic = torch.tensor(feat, dtype=torch.float)
        # x-vector
        if ENABLE_XVECTOR:
            w = torch.tensor(wav, dtype=torch.float).unsqueeze(0)
            emb = self.spk_enc.encode_batch(w).squeeze(0)
            if emb.dim() > 1:
                emb = emb.mean(-1)
            xvec = emb
        else:
            xvec = torch.zeros(0)
        # Label one-hot
        idx_label = self.label2idx[self.labels[idx]]
        label = F.one_hot(torch.tensor(idx_label), num_classes=self.num_classes).float()
        return acoustic, xvec, label

class LSTMAttentionSER(nn.Module):
    """
    BiLSTM-based SER model with optional self-attention before LSTM,
    and multi-head or single-head temporal attention after LSTM,
    plus optional x-vector integration.
    """
    def __init__(self, input_dim, xvec_dim, hidden_dim, num_classes):
        super().__init__()
        # Self-attention before LSTM
        if ENABLE_SELF_ATTENTION:
            # choose heads based on divisibility
            num_heads_self = 4 if input_dim % 4 == 0 else 1
            self.self_attn = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads_self,
                batch_first=True
            )
        else:
            self.self_attn = None
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        # Temporal attention: multi-head or single-head
        if ENABLE_MULTIHEAD_ATTENTION:
            temp_dim = hidden_dim * 2
            num_heads_temp = 4 if temp_dim % 4 == 0 else 1
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=temp_dim,
                num_heads=num_heads_temp,
                batch_first=True
            )
            self.attn_fc = None
        elif ENABLE_TEMPORAL_ATTENTION:
            self.temporal_attn = None
            self.attn_fc = nn.Linear(hidden_dim * 2, 1)
        else:
            self.temporal_attn = None
            self.attn_fc = None
        # Final classifier
        total_dim = hidden_dim * 2 + (xvec_dim if ENABLE_XVECTOR else 0)
        self.fc = nn.Linear(total_dim, num_classes)

    def forward(self, x, xvec=None):
        # x: [B, T, D]
        # Apply self-attention if configured
        if self.self_attn is not None:
            x, _ = self.self_attn(x, x, x)  # [B, T, D]
        # Pass through BiLSTM
        h, _ = self.lstm(x)  # [B, T, 2H]
        # Apply temporal attention
        if self.temporal_attn is not None:
            h_att, _ = self.temporal_attn(h, h, h)  # [B, T, 2H]
            context = h_att.mean(dim=1)            # [B, 2H]
        elif self.attn_fc is not None:
            scores = torch.tanh(self.attn_fc(h))   # [B, T, 1]
            weights = F.softmax(scores, dim=1)     # [B, T, 1]
            context = (weights * h).sum(dim=1)     # [B, 2H]
        else:
            # No attention: average pooling
            context = h.mean(dim=1)               # [B, 2H]
        # Concatenate x-vector if used
        if ENABLE_XVECTOR and xvec is not None and xvec.numel() > 0:
            if xvec.dim() == 1:
                xvec = xvec.unsqueeze(0).expand(context.size(0), -1)
            context = torch.cat([context, xvec], dim=1)
        # Final classification
        return self.fc(context)

# Training and evaluation routines (unchanged signatures)
def train_epoch(model, loader, optim, crit, dev):
    model.train(); total=0
    for acoustic, xvec, labels in loader:
        acoustic, labels = acoustic.to(dev), labels.to(dev)
        if ENABLE_XVECTOR:
            xvec = xvec.to(dev)
            outputs = model(acoustic, xvec)
        else:
            outputs = model(acoustic)
        loss = crit(outputs, labels)
        optim.zero_grad(); loss.backward(); optim.step()
        total += loss.item()
    return total/len(loader)

def eval_epoch(model, loader, crit, dev):
    model.eval(); total=0; corr=0; count=0
    with torch.no_grad():
        for acoustic, xvec, labels in loader:
            acoustic, labels = acoustic.to(dev), labels.to(dev)
            if ENABLE_XVECTOR:
                xvec = xvec.to(dev)
                outputs = model(acoustic, xvec)
            else:
                outputs = model(acoustic)
            loss = crit(outputs, labels)
            total += loss.item()
            preds = outputs.argmax(dim=1)
            true = labels.argmax(dim=1)
            corr += (preds == true).sum().item()
            count += labels.size(0)
    return total/len(loader), corr/count

# Main pipeline with optional class balancing and cross-validation

def main():
    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fixed data label list and display names
    data_labels = ['neu', 'fru', 'xxx', 'sur', 'ang', 'hap', 'sad', 'exc', 'fea']
    display_labels = ['neu', 'fru', 'other', 'sur', 'ang', 'hap', 'sad', 'exc', 'fea']
    label2idx = {label: idx for idx, label in enumerate(data_labels)}

    # Load and filter metadata
    df = pd.read_csv(
        "F:/DOCTORADO/DATASETS/IEMOCAP/archive/iemocap_full_dataset.csv"
    )
    df = df[df['emotion'].isin(data_labels)].reset_index(drop=True)
    # Print total number of audio files after filtering
    print(f"Total audio files detected: {len(df)}")

    # Function to train and evaluate one train/val split
    def run_fold(train_df, val_df):
        # Prepare datasets
        train_ds = SERDataset(
            train_df,
            "F:/DOCTORADO/DATASETS/IEMOCAP/archive/IEMOCAP_full_release",
            label2idx
        )
        val_ds = SERDataset(
            val_df,
            "F:/DOCTORADO/DATASETS/IEMOCAP/archive/IEMOCAP_full_release",
            label2idx
        )
        # DataLoader with optional class balancing
        if ENABLE_CLASS_BALANCING:
            counts = train_df['emotion'].value_counts().to_dict()
            weights = [1.0/counts[e] for e in train_df['emotion']]
            sampler = WeightedRandomSampler(weights, len(weights))
            train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
        else:
            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16)

        # Model, optimizer, scheduler, loss
        input_dim = train_ds[0][0].shape[1]
        xvec_dim = train_ds[0][1].shape[0] if ENABLE_XVECTOR else 0
        model = LSTMAttentionSER(
            input_dim, xvec_dim, hidden_dim=128,
            num_classes=len(data_labels)
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        criterion = nn.BCEWithLogitsLoss()

        train_losses, val_losses = [], []
        best_acc, patience = 0.0, 0
        for epoch in range(1, 31):
            tr_loss = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            vl_loss, vl_acc = eval_epoch(
                model, val_loader, criterion, device
            )
            train_losses.append(tr_loss)
            val_losses.append(vl_loss)
            scheduler.step(vl_loss)
            if vl_acc > best_acc:
                best_acc, patience = vl_acc, 0
            else:
                patience += 1
            if patience >= 5:
                break
        return best_acc, train_losses, val_losses, model, val_loader

    # Cross-validation or single split
    if ENABLE_CROSS_VALIDATION:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for train_idx, val_idx in kf.split(df):
            tdf, vdf = df.iloc[train_idx], df.iloc[val_idx]
            acc, *_ = run_fold(tdf, vdf)
            accs.append(acc)
        print("CV accuracies:", accs)
        print("Mean acc:", np.mean(accs))
    else:
        # Single train/val split
        train_df, val_df = train_test_split(
            df,
            test_size=0.25,
            stratify=df['emotion'],
            random_state=42
        )
        best_acc, tr_l, val_l, model, val_loader = run_fold(train_df, val_df)
        print(f"Validation accuracy: {best_acc:.4f}")

        # Plot loss curves
        plt.figure()
        plt.plot(tr_l, label='Train Loss')
        plt.plot(val_l, label='Val Loss')
        plt.legend()
        plt.show()

        # Classification report and confusion matrix
        from sklearn.metrics import classification_report, confusion_matrix
        y_true, y_pred = [], []
        for acoustic, xvec, labels in val_loader:
            acoustic = acoustic.to(device)
            outputs = (
                model(acoustic, xvec.to(device))
                if ENABLE_XVECTOR else model(acoustic)
            )
            preds = outputs.argmax(dim=1).cpu().numpy()
            trg = labels.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(trg)
        print(classification_report(
            y_true,
            y_pred,
            labels=list(range(len(display_labels))),
            target_names=display_labels
        ))
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8,6))
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        ticks = np.arange(len(display_labels))
        plt.xticks(ticks, display_labels, rotation=45)
        plt.yticks(ticks, display_labels)
        plt.show()

if __name__ == '__main__':
    main()
