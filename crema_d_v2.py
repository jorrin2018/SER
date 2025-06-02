# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:43:16 2025

@author: jjorrinc2100
"""
"""
Speech Emotion Recognition with x‑vector sequences + BiLSTM + Attention
---------------------------------------------------------------------
This version of the original script implements the architecture we
commented on:
  • The audio is segmented into 1‑second windows (0.5‑sec hop).
  • A SpeechBrain x‑vector extractor yields a 512‑d embedding per window.
  • The resulting [T,512] sequence is fed to a BiLSTM.
  • Optional self‑attention (over the 512 dims) and temporal attention.
  • Padding + pack_padded_sequence lets us batch variable‑length utterances.
"""
import os
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from speechbrain.inference import EncoderClassifier

# ────────────────────────────────────────────────────────────────────
# Config flags
# ────────────────────────────────────────────────────────────────────
ENABLE_XVECTOR            = True   # usamos secuencia de x‑vectors
ENABLE_SELF_ATTENTION     = True   # atención sobre la dimensión 512
ENABLE_TEMPORAL_ATTENTION = True   # atención después del LSTM

# Hand‑crafted features deshabilitadas para centrarnos en los x‑vectors
ENABLE_SPECTRAL           = True
ENABLE_MELSPECTROGRAM     = True
ENABLE_PROSODY            = True

# ────────────────────────────────────────────────────────────────────
# Audio → x‑vector hiperparámetros
# ────────────────────────────────────────────────────────────────────
WIN_LEN_SEC = 1.0   # duración de cada ventana en segundos
HOP_SEC     = 0.5   # paso entre ventanas
SAMPLE_RATE = 16000

# ────────────────────────────────────────────────────────────────────
# Training hiperparámetros
# ────────────────────────────────────────────────────────────────────
NUM_LSTM_LAYERS = 3
HIDDEN_DIM      = 512
EPOCHS          = 100
BATCH_SIZE      = 128
LR              = 3e-4
WEIGHT_DECAY    = 1e-2

# ────────────────────────────────────────────────────────────────────
# Etiquetas CREMA‑D
# ────────────────────────────────────────────────────────────────────
LABEL_LIST = ['ang', 'dis', 'fea', 'hap', 'neu', 'sad']
LABEL2IDX  = {lab: i for i, lab in enumerate(LABEL_LIST)}

# ===================================================================
# Dataset que devuelve la SECUENCIA de x‑vectors por locución
# ===================================================================
class SERDataset(Dataset):
    """Secuencia de x‑vectors por locución CREMA‑D."""
    def __init__(self, records: pd.DataFrame, audio_dir: str,
                 sample_rate=SAMPLE_RATE,
                 win_len_sec=WIN_LEN_SEC, hop_sec=HOP_SEC):
        self.audio_dir   = audio_dir
        self.sample_rate = sample_rate
        self.win_len     = int(win_len_sec * sample_rate)
        self.hop         = int(hop_sec * sample_rate)
        self.meta        = records.reset_index(drop=True)
        # extractor de x‑vectors
        self.spk_enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            run_opts={"device": "cpu"},
            savedir="pretrained_models/xvector"
        )

    def __len__(self):
        return len(self.meta)

    def _window_starts(self, n_samples: int):
        """Genera índices de inicio para las ventanas."""
        if n_samples <= self.win_len:
            yield 0
        else:
            for s in range(0, n_samples - self.win_len + 1, self.hop):
                yield s

    def __getitem__(self, idx):
        row   = self.meta.iloc[idx]
        label = torch.tensor(LABEL2IDX[row['emotion']], dtype=torch.long)

        wav, sr = torchaudio.load(os.path.join(self.audio_dir, row['path']))
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.squeeze()  # [N]

        # ── Augment: ligero time‑stretch + ruido blanco ──
        rate = random.uniform(0.95, 1.05)
        wav  = torchaudio.functional.phase_vocoder(wav.unsqueeze(0), rate, torch.zeros(1)).squeeze()
        wav  = wav + 0.003 * torch.randn_like(wav)

        # ── Extraer x‑vector por ventana ──
        xvec_seq = []
        with torch.no_grad():
            for s in self._window_starts(len(wav)):
                frame = wav[s:s + self.win_len]
                if frame.numel() < self.win_len:  # pad última
                    pad = torch.zeros(self.win_len - frame.numel(), device=frame.device)
                    frame = torch.cat([frame, pad])
                emb = self.spk_enc.encode_batch(frame.unsqueeze(0)).squeeze()  # → [512] sin dimensiones extras  # [512]
                xvec_seq.append(emb)
        xvec_seq = torch.stack(xvec_seq)  # [T, 512]

        return xvec_seq, label

# ===================================================================
# Collate: padding y longitudes para pack_padded_sequence
# ===================================================================

def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    padded  = nn.utils.rnn.pad_sequence(seqs, batch_first=True)  # [B,T_max,512]
    labels  = torch.stack(labels)
    return padded, lengths, labels

# ===================================================================
# Modelo: Self‑Attn (opcional) → BiLSTM → Temporal Attn → Softmax
# ===================================================================
class XvecLSTMAttn(nn.Module):
    def __init__(self, in_dim=512, hidden=HIDDEN_DIM,
                 num_layers=NUM_LSTM_LAYERS, n_classes=len(LABEL_LIST)):
        super().__init__()
        self.self_attn = (nn.MultiheadAttention(in_dim, num_heads=4, batch_first=True)
                          if ENABLE_SELF_ATTENTION else None)
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.temp_attn = nn.Linear(hidden * 2, 1) if ENABLE_TEMPORAL_ATTENTION else None
        self.cls = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden * 2, n_classes))

    def forward(self, x, lengths):
        # x: [B,T,D]
        if self.self_attn:
            x, _ = self.self_attn(x, x, x, need_weights=False)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        hseq, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B,T,2H]

        # ── Temporal pooling ──
        if self.temp_attn:
            mask = (torch.arange(hseq.size(1), device=lengths.device).unsqueeze(0) <
                    lengths.unsqueeze(1))
            e = torch.tanh(self.temp_attn(hseq)).squeeze(-1)  # [B,T]
            e = e.masked_fill(~mask, -1e4)                       # evitar in‑place
            w = torch.softmax(e, dim=1).unsqueeze(-1)
            context = (w * hseq).sum(dim=1)                  # [B,2H]
        else:
            mask = (torch.arange(hseq.size(1), device=lengths.device).unsqueeze(0) <
                    lengths.unsqueeze(1)).unsqueeze(-1)
            context = (hseq * mask).sum(dim=1) / lengths.unsqueeze(1)

        return self.cls(context)

# ===================================================================
# Funciones train / eval
# ===================================================================

def train_epoch(model, loader, optim, sched, criterion, device):
    model.train(); total = 0
    for x, lens, y in loader:
        x, lens, y = x.to(device), lens.to(device), y.to(device)
        optim.zero_grad()
        out = model(x, lens)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optim.step(); sched.step(); total += loss.item()
    return total / len(loader)

def eval_epoch(model, loader, criterion, device):
    model.eval(); total = 0; true_all = []; pred_all = []
    with torch.no_grad():
        for x, lens, y in loader:
            x, lens, y = x.to(device), lens.to(device), y.to(device)
            out = model(x, lens)
            total += criterion(out, y).item()
            pred = out.argmax(1)
            true_all.extend(y.cpu()); pred_all.extend(pred.cpu())
    acc = (torch.tensor(true_all) == torch.tensor(pred_all)).float().mean().item()
    return total / len(loader), acc, true_all, pred_all

# ===================================================================
# Main
# ===================================================================

def main():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    audio_dir = "F:/DOCTORADO/DATASETS/CREMA D/AudioWAV"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    df = pd.DataFrame([
        {"path": f, "emotion": f.split("_")[2].lower()}
        for f in files if f.split("_")[2].lower() in LABEL_LIST
    ])
    train_df, val_df = train_test_split(df, test_size=0.25,
                                        stratify=df["emotion"], random_state=42)

    train_ds = SERDataset(train_df, audio_dir)
    val_ds   = SERDataset(val_df, audio_dir)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = XvecLSTMAttn().to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(optimizer, max_lr=LR * 10, steps_per_epoch=len(train_loader),
                           epochs=EPOCHS, pct_start=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        vl_loss, vl_acc, y_true, y_pred = eval_epoch(model, val_loader, criterion, device)
        train_losses.append(tr_loss); val_losses.append(vl_loss)
        if vl_acc > best:
            best = vl_acc; torch.save(model.state_dict(), "best_model.pt")
        print(f"Epoch {epoch:02d}: train={tr_loss:.4f} | val={vl_loss:.4f} | acc={vl_acc:.4f}")

    print("Best validation accuracy:", best)
    print(classification_report(y_true, y_pred, target_names=LABEL_LIST, zero_division=0))

    # ── Gráfica de pérdidas ──
    plt.figure(); plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=120)

    # ── Matriz de confusión ──
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_LIST))))
    plt.figure(figsize=(6,6)); plt.imshow(cm, cmap="Blues", interpolation="nearest")
    ticks = range(len(LABEL_LIST))
    plt.xticks(ticks, LABEL_LIST, rotation=45); plt.yticks(ticks, LABEL_LIST)
    thresh = cm.max() / 2
    for i in ticks:
        for j in ticks:
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.xlabel("Predicho"); plt.ylabel("Real"); plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)

if __name__ == "__main__":
    main()
    main()
