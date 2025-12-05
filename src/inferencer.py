import os
import sys
from pathlib import Path

import torch
import torchaudio

from model.crn import CRN
from utils.torch import get_torch_device

# --- CONFIGURAZIONE (deve corrispondere a trainer.py) ---
N_FFT = 320
HOP_LENGTH = 160
WIN_LENGTH = 320
SAMPLE_RATE = 16000
DEVICE = get_torch_device()
CHECKPOINT_PATH = "checkpoints/crn_best.pth"
INFERENCE_INPUT_DIR = "inference"
INPUT_FILENAME = "input_2.wav"
OUTPUT_FILENAME = "output_2_denoised.wav"


def load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    """
    Carica un file audio e lo resample a SAMPLE_RATE se necessario.
    
    Args:
        audio_path: Percorso al file audio
        
    Returns:
        waveform: Tensor [channels, samples]
        sr: Sample rate del file caricato
    """
    waveform, sr = torchaudio.load(audio_path)
    
    # Se il sample rate è diverso, resample
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
        sr = SAMPLE_RATE
    
    return waveform, sr


def compute_stft(waveform: torch.Tensor) -> torch.Tensor:
    """
    Calcola la STFT complessa del segnale.
    
    Args:
        waveform: [channels, samples]
        
    Returns:
        stft: STFT complessa [channels, freq, time]
    """
    # Se mono, aggiungi dimensione di batch fittizia
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Calcola STFT per ogni canale
    window = torch.hann_window(WIN_LENGTH, device=waveform.device)
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


def reconstruct_waveform(
    magnitude: torch.Tensor, phase: torch.Tensor
) -> torch.Tensor:
    """
    Ricostruisce la waveform dalla magnitude e fase usando iSTFT.
    
    Args:
        magnitude: Magnitude [channels, freq, time]
        phase: Fase [channels, freq, time]
        
    Returns:
        waveform: Segnale ricostruito [channels, samples]
    """
    # Combina magnitude e fase per creare spettrogramma complesso
    complex_spec = magnitude * torch.exp(1j * phase)
    
    # Crea finestra di Hann sul device corretto
    window = torch.hann_window(WIN_LENGTH, device=magnitude.device)
    
    # Applica iSTFT per ogni canale
    num_channels = complex_spec.shape[0]
    waveforms = []
    
    for ch in range(num_channels):
        waveform = torch.istft(
            complex_spec[ch],
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
        )
        waveforms.append(waveform)
    
    return torch.stack(waveforms) if len(waveforms) > 1 else waveforms[0].unsqueeze(0)


def infer():
    """Esegue l'inference su un file audio."""
    
    # 1. Carica il checkpoint
    if not os.path.isfile(CHECKPOINT_PATH):
        print(f"❌ Checkpoint non trovato: {CHECKPOINT_PATH}")
        return
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = CRN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✓ Checkpoint caricato da {CHECKPOINT_PATH}")
    
    # 2. Carica il file audio di input
    input_path = os.path.join(INFERENCE_INPUT_DIR, INPUT_FILENAME)
    if not os.path.isfile(input_path):
        print(f"❌ File di input non trovato: {input_path}")
        return
    
    waveform, sr = load_audio(input_path)
    print(f"✓ Audio caricato: {INPUT_FILENAME} (sample rate: {sr} Hz)")
    
    # Se stereo, prendi solo il primo canale
    if waveform.shape[0] > 1:
        waveform = waveform[0:1]
        print("  (convertito a mono)")
    
    # 3. Calcola STFT
    stft = compute_stft(waveform)
    stft = stft.to(DEVICE)
    
    # Estrai magnitude e fase
    noisy_mag = stft.abs().contiguous()
    noisy_phase = torch.angle(stft)
    
    print(f"✓ STFT calcolata: shape={noisy_mag.shape}")
    
    # 4. Forward pass attraverso la rete
    with torch.no_grad():
        enhanced_mag = model(noisy_mag)
    
    print(f"✓ Forward pass completato")
    
    # 5. Ricostruisci il segnale audio
    enhanced_waveform = reconstruct_waveform(enhanced_mag, noisy_phase)
    enhanced_waveform = enhanced_waveform.cpu()
    
    print(f"✓ Waveform ricostruita: shape={enhanced_waveform.shape}")
    
    # 6. Salva l'output
    output_path = os.path.join(INFERENCE_INPUT_DIR, OUTPUT_FILENAME)
    torchaudio.save(output_path, enhanced_waveform, SAMPLE_RATE)
    print(f"✓ Output salvato: {output_path}")
    
    print("\n" + "="*60)
    print("✅ Inference completato!")
    print("="*60)


if __name__ == "__main__":
    infer()