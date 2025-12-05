"""
Utility per la gestione di file: salvataggio metriche JSON e audio debug.
"""

import json
import os
from datetime import datetime
from typing import Optional

import torch
import torchaudio


# --- CONFIGURAZIONE PATH ---
METRICS_DIR = "metrics"
METRICS_JSON_PATH = os.path.join(METRICS_DIR, "crn_metrics.json")
DEBUG_AUDIO_DIR = os.path.join(METRICS_DIR, "debug_audio")
SAMPLE_RATE = 16000


def _ensure_dir(path: str) -> None:
    """Crea la directory se non esiste."""
    os.makedirs(path, exist_ok=True)


def save_metrics_json(
    epoch_metrics: dict,
    epoch: int,
    filepath: str = METRICS_JSON_PATH,
) -> None:
    """
    Salva/aggiorna le metriche in formato JSON.

    Il file contiene uno storico di tutte le epoche con timestamp.

    Args:
        epoch_metrics: Dict con metriche dell'epoca (pesq, stoi, si_sdr, loss, etc.)
        epoch: Numero dell'epoca
        filepath: Path del file JSON
    """
    _ensure_dir(os.path.dirname(filepath))

    # Carica storico esistente o crea nuovo
    history = load_metrics_json(filepath)

    # Aggiungi entry per questa epoca
    entry = {
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        **epoch_metrics,
    }
    history["epochs"].append(entry)

    # Aggiorna best metrics se migliori
    if "enhanced" in epoch_metrics:
        enh = epoch_metrics["enhanced"]
        best = history.get("best", {})

        if enh.get("pesq", float("-inf")) > best.get("pesq", float("-inf")):
            best["pesq"] = enh["pesq"]
            best["pesq_epoch"] = epoch

        if enh.get("stoi", float("-inf")) > best.get("stoi", float("-inf")):
            best["stoi"] = enh["stoi"]
            best["stoi_epoch"] = epoch

        if enh.get("si_sdr", float("-inf")) > best.get("si_sdr", float("-inf")):
            best["si_sdr"] = enh["si_sdr"]
            best["si_sdr_epoch"] = epoch

        history["best"] = best

    # Salva su file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def load_metrics_json(filepath: str = METRICS_JSON_PATH) -> dict:
    """
    Carica lo storico delle metriche da file JSON.

    Args:
        filepath: Path del file JSON

    Returns:
        Dict con storico metriche, o struttura vuota se file non esiste
    """
    if os.path.isfile(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Struttura base se file non esiste o è corrotto
    return {
        "model": "CRN",
        "created": datetime.now().isoformat(),
        "sample_rate": SAMPLE_RATE,
        "epochs": [],
        "best": {},
    }


def save_debug_audio(
    noisy_wav: torch.Tensor,
    enhanced_wav: torch.Tensor,
    clean_wav: torch.Tensor,
    sample_idx: int,
    epoch: int,
    output_dir: str = DEBUG_AUDIO_DIR,
) -> None:
    """
    Salva campioni audio per verifica manuale.

    Salva 3 file WAV: noisy, enhanced, clean per permettere
    di ascoltare e confrontare manualmente la qualità.

    Args:
        noisy_wav: Waveform noisy [samples] o [1, samples]
        enhanced_wav: Waveform enhanced [samples] o [1, samples]
        clean_wav: Waveform clean [samples] o [1, samples]
        sample_idx: Indice del campione nel batch
        epoch: Numero dell'epoca
        output_dir: Directory di output
    """
    _ensure_dir(output_dir)

    # Assicurati che siano 2D [1, samples] per torchaudio.save
    def prepare_wav(wav: torch.Tensor) -> torch.Tensor:
        wav = wav.cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 3:
            wav = wav.squeeze(0)
        # Normalizza per evitare clipping
        max_val = wav.abs().max()
        if max_val > 1.0:
            wav = wav / max_val
        return wav

    prefix = f"epoch{epoch:03d}_sample{sample_idx:02d}"

    noisy_path = os.path.join(output_dir, f"{prefix}_noisy.wav")
    enhanced_path = os.path.join(output_dir, f"{prefix}_enhanced.wav")
    clean_path = os.path.join(output_dir, f"{prefix}_clean.wav")

    torchaudio.save(noisy_path, prepare_wav(noisy_wav), SAMPLE_RATE)
    torchaudio.save(enhanced_path, prepare_wav(enhanced_wav), SAMPLE_RATE)
    torchaudio.save(clean_path, prepare_wav(clean_wav), SAMPLE_RATE)

    print(f"  💾 Audio debug salvati in: {output_dir}/")
    print(f"     - {prefix}_noisy.wav")
    print(f"     - {prefix}_enhanced.wav")
    print(f"     - {prefix}_clean.wav")


def save_baseline_metrics(
    noisy_metrics: dict,
    filepath: str = METRICS_JSON_PATH,
) -> None:
    """
    Salva le metriche baseline (noisy vs clean) nel JSON.

    Queste metriche servono come riferimento per misurare
    il miglioramento del modello.

    Args:
        noisy_metrics: Dict con metriche del segnale noisy
        filepath: Path del file JSON
    """
    _ensure_dir(os.path.dirname(filepath))

    history = load_metrics_json(filepath)
    history["baseline_noisy"] = {
        "pesq": noisy_metrics.get("pesq"),
        "stoi": noisy_metrics.get("stoi"),
        "si_sdr": noisy_metrics.get("si_sdr"),
        "timestamp": datetime.now().isoformat(),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"  📁 Baseline noisy salvato in: {filepath}")
