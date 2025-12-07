"""
Metriche per la valutazione di Speech Enhancement.

Implementa PESQ, STOI e SI-SDR usando torchmetrics per valutare
la qualità del segnale audio enhanced rispetto al segnale pulito.
"""

import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

# Sample rate del dataset VoiceBank-DEMAND
SAMPLE_RATE = 16000


def create_metrics(device: str) -> dict:
    """
    Crea e inizializza le metriche audio su device specificato.

    Args:
        device: 'cuda' o 'cpu'

    Returns:
        Dict con le metriche inizializzate:
        - pesq: Perceptual Evaluation of Speech Quality (range: -0.5 a 4.5)
        - stoi: Short-Time Objective Intelligibility (range: 0 a 1)
        - si_sdr: Scale-Invariant Signal-to-Distortion Ratio (dB, più alto = meglio)
    """
    return {
        "pesq": PerceptualEvaluationSpeechQuality(fs=SAMPLE_RATE, mode="wb").to(device),
        "stoi": ShortTimeObjectiveIntelligibility(fs=SAMPLE_RATE).to(device),
        "si_sdr": ScaleInvariantSignalDistortionRatio().to(device),
    }


def compute_metrics(
    enhanced_wav: torch.Tensor,
    clean_wav: torch.Tensor,
    metrics: dict,
) -> dict:
    """
    Calcola le metriche tra segnale enhanced e segnale pulito.

    Args:
        enhanced_wav: Waveform enhanced dal modello [batch, samples] o [batch, 1, samples]
        clean_wav: Waveform pulita di riferimento [batch, samples] o [batch, 1, samples]
        metrics: Dict delle metriche da create_metrics()

    Returns:
        Dict con i valori medi delle metriche per il batch
    """
    # Assicurati che le waveform siano 2D [batch, samples]
    if enhanced_wav.dim() == 3:
        enhanced_wav = enhanced_wav.squeeze(1)
    if clean_wav.dim() == 3:
        clean_wav = clean_wav.squeeze(1)

    # Assicurati che siano sullo stesso device delle metriche
    device = next(iter(metrics.values())).device
    enhanced_wav = enhanced_wav.to(device)
    clean_wav = clean_wav.to(device)

    results = {}

    # PESQ - può fallire su alcuni campioni, gestiamo l'eccezione
    try:
        pesq_value = metrics["pesq"](enhanced_wav, clean_wav)
        results["pesq"] = pesq_value.item()
    except Exception:
        results["pesq"] = float("nan")

    # STOI
    try:
        stoi_value = metrics["stoi"](enhanced_wav, clean_wav)
        results["stoi"] = stoi_value.item()
    except Exception:
        results["stoi"] = float("nan")

    # SI-SDR
    try:
        si_sdr_value = metrics["si_sdr"](enhanced_wav, clean_wav)
        results["si_sdr"] = si_sdr_value.item()
    except Exception:
        results["si_sdr"] = float("nan")

    return results


def reset_metrics(metrics: dict) -> None:
    """Resetta lo stato interno delle metriche (per nuova epoca)."""
    for metric in metrics.values():
        metric.reset()


def print_metrics_comparison(train_loss, eval_loss, noisy_results: dict, enhanced_results: dict, epoch: int) -> None:
    """
    Stampa confronto tra metriche noisy e enhanced.

    Args:
        noisy_results: Metriche del segnale noisy (baseline)
        enhanced_results: Metriche del segnale enhanced (output modello)
        epoch: Numero dell'epoca corrente
    """
    def delta_str(enhanced: float, noisy: float) -> str:
        """Calcola delta e formatta con + o -."""
        delta = enhanced - noisy
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.4f}"
    

    print(f"\nEpoch {epoch}:")
    print(f"  📉 Train Loss: {train_loss:.5f}")
    print(f"  📉 Eval Loss:  {eval_loss:.5f}")


    print(f"\n{'='*70}")
    print(f"📊 METRICHE EPOCH {epoch}")
    print(f"{'='*70}")
    print(f"{'Metrica':<10} {'Noisy':>12} {'Enhanced':>12} {'Delta':>12}")
    print(f"{'-'*70}")

    pesq_noisy = noisy_results.get('pesq', float('nan'))
    pesq_enh = enhanced_results.get('pesq', float('nan'))
    print(f"{'PESQ':<10} {pesq_noisy:>12.4f} {pesq_enh:>12.4f} {delta_str(pesq_enh, pesq_noisy):>12}")

    stoi_noisy = noisy_results.get('stoi', float('nan'))
    stoi_enh = enhanced_results.get('stoi', float('nan'))
    print(f"{'STOI':<10} {stoi_noisy:>12.4f} {stoi_enh:>12.4f} {delta_str(stoi_enh, stoi_noisy):>12}")

    sisdr_noisy = noisy_results.get('si_sdr', float('nan'))
    sisdr_enh = enhanced_results.get('si_sdr', float('nan'))
    print(f"{'SI-SDR':<10} {sisdr_noisy:>10.2f} dB {sisdr_enh:>10.2f} dB {delta_str(sisdr_enh, sisdr_noisy):>10} dB")

    print(f"{'='*70}\n")


def verify_reconstruction(
    original_wav: torch.Tensor,
    reconstructed_wav: torch.Tensor,
    threshold: float = 1e-4,
) -> tuple[bool, float]:
    """
    Verifica che la ricostruzione iSTFT sia corretta.

    Confronta la waveform originale con quella ricostruita tramite
    STFT → magnitude + phase → iSTFT per assicurarsi che la pipeline
    funzioni correttamente.

    Args:
        original_wav: Waveform originale [batch, samples] o [batch, 1, samples]
        reconstructed_wav: Waveform ricostruita [batch, samples] o [batch, 1, samples]
        threshold: Errore massimo accettabile (default: 1e-4)

    Returns:
        Tuple (passed: bool, error: float)
    """
    # Assicurati che siano 2D
    if original_wav.dim() == 3:
        original_wav = original_wav.squeeze(1)
    if reconstructed_wav.dim() == 3:
        reconstructed_wav = reconstructed_wav.squeeze(1)

    # Gestisci possibili differenze di lunghezza (iSTFT può variare di qualche sample)
    min_len = min(original_wav.shape[-1], reconstructed_wav.shape[-1])
    original_wav = original_wav[..., :min_len]
    reconstructed_wav = reconstructed_wav[..., :min_len]

    # Calcola errore medio assoluto
    error = torch.mean(torch.abs(original_wav - reconstructed_wav)).item()

    passed = error < threshold
    return passed, error
