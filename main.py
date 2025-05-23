# -*- coding: utf-8 -*-
"""
Created on Thu May 22 16:58:29 2025

@author: jjorrinc2100
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from speechbrain.pretrained import EncoderClassifier

class SERDataset(Dataset):
    """
    Dataset para Speech Emotion Recognition usando metadata CSV y directorio de audio.

    Args:
        metadata (pd.DataFrame): DataFrame con columnas ['path', 'emotion'] relativas a audio_dir.
        audio_dir (str): Directorio raíz donde se encuentran los archivos WAV.
        sample_rate (int): Frecuencia de muestreo para cargar audio.
        n_mfcc (int): Número de coeficientes MFCC a extraer.
        max_len (int): Número máximo de frames para padding/truncamiento.
    """
    def __init__(self, metadata, audio_dir, sample_rate=16000, n_mfcc=40, max_len=200):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.audio_dir = audio_dir
        self.file_list = metadata['path'].tolist()
        self.labels = metadata['emotion'].tolist()
        self.label2idx = {lab: idx for idx, lab in enumerate(sorted(set(self.labels)))}
        # Cargador preentrenado de x-vectors (speaker embeddings)
        self.spk_enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rel_path = self.file_list[idx]
        label = self.labels[idx]
        path = os.path.join(self.audio_dir, rel_path)
        wav, sr = librosa.load(path, sr=self.sample_rate)
        mfcc = librosa.feature.mfcc(wav, sr, n_mfcc=self.n_mfcc).T
        energy = librosa.feature.rms(wav).T
        pitches, _ = librosa.piptrack(wav, sr)
        pitch = np.mean(pitches, axis=0)[None, :].T
        prosody = np.concatenate([energy, pitch], axis=1)
        feat = np.concatenate([mfcc, prosody], axis=1)
        if feat.shape[0] < self.max_len:
            pad = np.zeros((self.max_len - feat.shape[0], feat.shape[1]))
            feat = np.vstack([feat, pad])
        else:
            feat = feat[:self.max_len]
        acoustic = torch.tensor(feat, dtype=torch.float)
        wav_tensor = torch.tensor(wav, dtype=torch.float).unsqueeze(0)
        embeddings = self.spk_enc.encode_batch(wav_tensor)
        embeddings = embeddings.squeeze(0)
        if embeddings.dim() > 1:
            embeddings = embeddings.mean(dim=-1)
        xvector = embeddings
        label_idx = torch.tensor(self.label2idx[label], dtype=torch.long)
        return acoustic, xvector, label_idx

class LSTMAttentionWithXVector(nn.Module):
    """
    Modelo SER combinando BiLSTM con atención temporal y x-vector.
    """
    def __init__(self, input_dim, xvector_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2 + xvector_dim, num_classes)

    def forward(self, acoustic_seq, xvector):
        lstm_out, _ = self.lstm(acoustic_seq)
        energy = torch.tanh(self.attn_fc(lstm_out))
        weights = torch.softmax(energy, dim=1)
        context = (weights * lstm_out).sum(dim=1)
        if xvector.dim() == 1:
            xvector = xvector.unsqueeze(0).expand(context.size(0), -1)
        combined = torch.cat([context, xvector], dim=1)
        logits = self.fc(combined)
        return logits


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for acoustic, xvector, labels in loader:
        acoustic, xvector, labels = acoustic.to(device), xvector.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(acoustic, xvector)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for acoustic, xvector, labels in loader:
            acoustic, xvector, labels = acoustic.to(device), xvector.to(device), labels.to(device)
            outputs = model(acoustic, xvector)
            total_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def main():
    # Configuraciones y paths actualizados por el usuario
    metadata_csv = "F:/DOCTORADO/DATASETS/IEMOCAP/archive/iemocap_full_dataset.csv"
    audio_dir = "F:/DOCTORADO/DATASETS/IEMOCAP/archive/IEMOCAP_full_release"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(metadata_csv)
    train_df, val_df = train_test_split(df, test_size=0.25, stratify=df['emotion'], random_state=42)
    train_ds = SERDataset(train_df, audio_dir)
    val_ds = SERDataset(val_df, audio_dir)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    input_dim = train_ds[0][0].shape[1]
    xvector_dim = train_ds[0][1].shape[0]
    hidden_dim = 128
    num_classes = len(train_ds.label2idx)
    model = LSTMAttentionWithXVector(input_dim, xvector_dim, hidden_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()
    best_acc, no_improve, start_epoch = 0.0, 0, 1
    patience, checkpoint_path = 5, 'checkpoint.pth'
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['opt_state'])
        scheduler.load_state_dict(ckpt['sched_state'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        no_improve = ckpt['no_improve']
        train_losses, val_losses = ckpt['train_losses'], ckpt['val_losses']
    else:
        train_losses, val_losses = [], []
    epochs = 30
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict(),
            'best_acc': best_acc,
            'no_improve': no_improve,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
            torch.save(ckpt, checkpoint_path)
        else:
            no_improve += 1
            torch.save(ckpt, checkpoint_path)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        if no_improve >= patience:
            print("Early stopping ejecutado.")
            break
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()
    print(f'Mejor accuracy de validación: {best_acc:.4f}')

if __name__ == "__main__":
    main()
