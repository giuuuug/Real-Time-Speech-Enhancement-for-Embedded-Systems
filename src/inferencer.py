import os
import sys
from pathlib import Path

import torch
import torchaudio

from src.model.crn import CRN
from src.utils.torch import get_torch_device

# --- CONFIGURAZIONE ---
N_FFT = 320
HOP_LENGTH = 160
WIN_LENGTH = 320
SAMPLE_RATE = 16000
DEVICE = get_torch_device()

CHECKPOINT_PATH = "checkpoints/crn_best.pth"
INFERENCE_INPUT_DIR = "inference"
INPUT_FILENAME = "input_1.wav"
OUTPUT_FILENAME = "output_1_denoised.wav"

# Parametri Noise Gate
NOISE_GATE_THRESHOLD = 0.03  # Soglia energetica (da tarare a orecchio: 0.01 - 0.05)
MIN_MASK_VALUE = 0.0         # A quanto portare il silenzio (0.0 = muto assoluto)


def load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    """Carica, resample e normalizza l'audio."""
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
        sr = SAMPLE_RATE
    
    # --- MODIFICA 1: NORMALIZZAZIONE INPUT ---
    # Fondamentale affinché la soglia del Noise Gate funzioni ugualmente su tutti i file
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val
        
    return waveform, sr


def compute_stft(waveform: torch.Tensor) -> torch.Tensor:
    """Calcola STFT complessa."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    window = torch.hann_window(WIN_LENGTH, device=waveform.device)
    
    # Nota: Usiamo la versione vettorizzata se possibile, altrimenti loop sui canali
    # Qui manteniamo la tua logica originale per compatibilità
    stft_list = []
    for ch in range(waveform.shape[0]):
        stft_ch = torch.stft(
            waveform[ch],
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
            return_complex=True,
        )
        stft_list.append(stft_ch)
    
    return torch.stack(stft_list) if len(stft_list) > 1 else stft_list[0].unsqueeze(0)


def reconstruct_waveform(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Ricostruisce waveform da Mag e Fase."""
    complex_spec = magnitude * torch.exp(1j * phase)
    window = torch.hann_window(WIN_LENGTH, device=magnitude.device)
    
    num_channels = complex_spec.shape[0]
    waveforms = []
    
    for ch in range(num_channels):
        waveform = torch.istft(
            complex_spec[ch],
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
            center=True, # Importante: deve matchare la STFT
        )
        waveforms.append(waveform)
    
    return torch.stack(waveforms) if len(waveforms) > 1 else waveforms[0].unsqueeze(0)


def apply_noise_gate(magnitude_linear: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Applica un noise gate semplice basato sull'energia media del frame.
    """
    # Calcola l'energia media per ogni frame temporale (media su frequenze)
    # magnitude_linear shape: [Batch, Freq, Time] -> Energy: [Batch, 1, Time]
    energy = magnitude_linear.mean(dim=1, keepdim=True)
    
    # Crea maschera binaria (1 se > soglia, 0 altrimenti)
    mask = (energy > threshold).float()
    
    # Applica maschera (Opzionale: smoothing temporale sulla maschera per ridurre 'click')
    gated_magnitude = magnitude_linear * mask
    
    return gated_magnitude


def infer():
    """Esegue l'inference."""
    
    # 1. Carica Checkpoint
    if not os.path.isfile(CHECKPOINT_PATH):
        print(f"❌ Checkpoint non trovato: {CHECKPOINT_PATH}")
        return
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = CRN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✓ Checkpoint caricato da {CHECKPOINT_PATH}")
    
    # 2. Carica Audio
    input_path = os.path.join(INFERENCE_INPUT_DIR, INPUT_FILENAME)
    if not os.path.isfile(input_path):
        print(f"❌ File input mancante: {input_path}")
        return
    
    waveform, sr = load_audio(input_path)
    print(f"✓ Audio caricato: {INPUT_FILENAME} (SR: {sr})")
    
    if waveform.shape[0] > 1:
        waveform = waveform[0:1]
        print("  (stereo -> mono)")
    
    # 3. STFT
    stft = compute_stft(waveform)
    stft = stft.to(DEVICE)
    
    noisy_mag = stft.abs()
    noisy_phase = torch.angle(stft)
    
    # --- MODIFICA 2: COMPRESSIONE (Come nel training) ---
    noisy_mag_compressed = torch.pow(noisy_mag, 0.5)
    # ----------------------------------------------------
    
    print(f"✓ STFT calcolata. Input rete shape: {noisy_mag_compressed.shape}")
    
    # 4. Forward Pass
    with torch.no_grad():
        enhanced_mag_compressed = model(noisy_mag_compressed)
    
    # --- MODIFICA 3: DECOMPRESSIONE & CLAMP ---
    enhanced_mag_compressed = torch.clamp(enhanced_mag_compressed, min=0.0)
    enhanced_mag_linear = torch.pow(enhanced_mag_compressed, 2.0)
    # ------------------------------------------
    
    # --- MODIFICA 4: NOISE GATE ---
    print(f"  Applico Noise Gate (Soglia: {NOISE_GATE_THRESHOLD})...")
    enhanced_mag_final = apply_noise_gate(enhanced_mag_linear, NOISE_GATE_THRESHOLD)
    # ------------------------------

    # 5. Ricostruzione
    # Usiamo la magnitudo post-gate e la fase originale (rumorosa)
    enhanced_waveform = reconstruct_waveform(enhanced_mag_final, noisy_phase)
    enhanced_waveform = enhanced_waveform.cpu()
    
    # 6. Salvataggio
    output_path = os.path.join(INFERENCE_INPUT_DIR, OUTPUT_FILENAME)
    torchaudio.save(output_path, enhanced_waveform, SAMPLE_RATE)
    print(f"✓ Output salvato: {output_path}")
    print("="*60)


if __name__ == "__main__":
    infer()