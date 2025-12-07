"""
Utility per la gestione di file: salvataggio metriche JSON e audio debug.
"""

import json
import os
from datetime import datetime
from src.utils import directories as dir_helper

import torch
import torchaudio


# --- CONFIGURAZIONE PATH ---
METRICS_DIR = "metrics"
METRICS_JSON_PATH = os.path.join(METRICS_DIR, "crn_metrics.json")
DEBUG_AUDIO_DIR = os.path.join(METRICS_DIR, "debug_audio")
SAMPLE_RATE = 16000


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
    dir_helper.validate_dir(os.path.dirname(filepath))

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
